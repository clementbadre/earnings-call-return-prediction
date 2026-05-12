"""
FinBERT sentiment inference on Earnings Calls.

Input  : ../sm-calls_with_connectors.parquet
Output : ../outputs/finbert_scores.parquet

Steps:
  1. Filter 2008-2023, drop empty transcripts
  2. Extract Q&A section from each transcript
  3. Chunk text into 512-token segments
  4. Run FinBERT (ProsusAI/finbert) on each chunk — MPS backend on M4
  5. Weighted average of scores across chunks → [p_pos, p_neg, p_neu] per call
  6. Save to parquet

Runtime: ~2-4h on MacBook Pro M4 (MPS backend).
Run once and save — never needs to be re-executed.
"""

import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

RAW_PATH = "../sm-calls_with_connectors.parquet"
OUT_PATH = "../outputs/finbert_scores.parquet"

MODEL_NAME  = "ProsusAI/finbert"
CHUNK_SIZE  = 512
MIN_WORDS   = 100
START       = "2008-01-01"
END         = "2023-12-31"

QA_PATTERNS = [
    r"question.and.answer",
    r"q&a session",
    r"q\s*&\s*a",
    r"questions? and answers?",
    r"open.*questions?",
    r"we will now begin.*question",
]
QA_REGEX = re.compile("|".join(QA_PATTERNS), re.IGNORECASE)


def extract_qa_section(text: str) -> str:
    """Return Q&A section if found, else full text."""
    match = QA_REGEX.search(text)
    if match:
        return text[match.start():]
    return text


def chunk_tokens(input_ids: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> list[torch.Tensor]:
    """Split a 1D token tensor into overlapping chunks of chunk_size."""
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk = input_ids[i : i + chunk_size]
        if len(chunk) < 10:
            continue
        chunks.append(chunk)
    return chunks


def score_text(text: str, tokenizer, model, device) -> np.ndarray | None:
    """Return [p_pos, p_neg, p_neu] averaged across all chunks."""
    section = extract_qa_section(text)
    tokens  = tokenizer(
        section,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"][0]

    chunks = chunk_tokens(tokens)
    if not chunks:
        return None

    scores  = []
    weights = []

    for chunk in chunks:
        input_ids = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        scores.append(probs)
        weights.append(len(chunk))

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.average(scores, axis=0, weights=weights)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("Loading Earnings Calls...")
    df = pd.read_parquet(RAW_PATH, columns=[
        "permno", "mostimportantdateutc", "text", "word_count"
    ])
    df = df.rename(columns={"mostimportantdateutc": "call_date"})
    df["call_date"] = pd.to_datetime(df["call_date"])
    df = df[(df["call_date"] >= START) & (df["call_date"] <= END)]
    df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)
    print(f"  Calls to process: {len(df)}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="FinBERT"):
        scores = score_text(row["text"], tokenizer, model, device)
        if scores is None:
            continue
        results.append({
            "permno":    int(row["permno"]),
            "call_date": row["call_date"],
            "p_pos":     float(scores[0]),
            "p_neg":     float(scores[1]),
            "p_neu":     float(scores[2]),
        })

    out = pd.DataFrame(results).sort_values(["permno", "call_date"]).reset_index(drop=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nDone. {len(out)} calls scored.")
    print(f"Saved → {OUT_PATH}")
    print(out[["p_pos", "p_neg", "p_neu"]].describe().round(3))


if __name__ == "__main__":
    main()
