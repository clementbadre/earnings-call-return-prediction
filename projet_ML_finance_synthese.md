# Projet Machine Learning in Finance — Synthèse complète

---

## 0. Original Subject (English — verbatim)

### Machine-Learning in Finance — Project Instructions
**Deadline : 29th of May 23:59**

#### 1 Introduction

You are hired as external consultants for a hedge fund that wants to create a new machine-learning-based fund. Your task is as follows: you are given two types of datasets, return datasets and predictor datasets. The Hedge Fund wants you to find a Machine Learning model to predict one of the returns datasets using one or many of the predictor datasets, and build investment strategies using your model(s). The projects account for 30% of your final grade so we expect you to produce a rigorous, well-structured, and insightful analysis, combining sound data preprocessing, thoughtful model design, and careful evaluation of results.

In particular, your work should demonstrate a clear understanding of the limitations of your approach and the economic relevance of your findings. Beyond predictive performance, special attention should be given to the practical implementation of your strategy, including considerations such as transaction costs, robustness, and out-of-sample performance. Creativity, critical thinking, and clarity of presentation will be key components of a successful project.

#### 2 Guidelines

You are free to use any machine learning approaches that we have studied in the class. You can also use other approaches if you wish, but at least one of the approaches in the report has to use deep learning. For each dataset, you can either use the following.

**Regression Task:** Let R_{i,t+1} be the return on stock i = 1, .., P at time t + 1. Using some predictors X_t, available at time t, build a model such that:

R_{i,t+1} ≈ f(X_t; θ)

Note that you could also want to predict the whole cross-section of returns directly. Building R_{t+1} ≈ f(X_t; θ) ∈ R^P

**Classification task:** Let R_{i,t+1} be the return on the stock i = 1, .., P at time t + 1. Create c = 1, ..., C bins of returns and, using some predictors X_t, available at time t, build a model such that:

P(R_{i,t+1} ∈ c) ≈ f(X_t; c)

#### 3 Datasets

We provide many different datasets to use to carry out your projects. They can be downloaded from this Google drive. You are not allowed to use external datasets.

##### 3.1 Returns dataset

First, here are the five return datasets that you can use. Those would be the targets you are trying to predict.

- **Monthly CRSP:** This dataset contains US monthly stock returns from December 1925 to December 2024. It contains 10 important columns:
  - PERMNO: Unique permanent identifier for a security assigned by CRSP.
  - HdrCUSIP: Header CUSIP code identifying the issuer of the security.
  - CUSIP: 8-character identifier for the specific security, often includes issuer and issue.
  - Ticker: The ticker symbol of the stock representing the security on the exchanges.
  - TradingSymbol: Trading symbol used in CRSP, sometimes more precise than the standard ticker.
  - PERMCO: Permanent company identifier in CRSP, it links together all PERMNOs belonging to the same firm.
  - SICCD: Standard Industrial Classification (SIC) code used to classify the firm's industry.
  - NAICS: North American Industry Classification System code, a modern classification for industries.
  - MthCalDt: Calendar month-end date for the observation in format YYYY-MM-DD. Note that the date corresponds to the return for the same month. For example, the return associated with 1986-12-31 is from the end of November to the end of December.
  - MthRet: Monthly return for the security (not adjusted for dividends or splits unless specified).
  - sprtrn: S&P 500 return for the same calendar month, used as a market benchmark.

- **Daily CRSP:** This dataset contains US daily stock returns from January 2000 to December 2024. It contains 10 important columns:
  - PERMNO: Unique permanent identifier for a security assigned by CRSP.
  - HdrCUSIP: Header CUSIP code identifying the issuer of the security.
  - CUSIP: 8-character identifier for the specific security, often includes issuer and issue.
  - Ticker: Stock ticker symbol representing the security on exchanges.
  - TradingSymbol: Trading symbol used in CRSP, sometimes more precise than the standard ticker.
  - PERMCO: Permanent company identifier in CRSP, links together all PERMNOs belonging to the same firm.
  - SICCD: Standard Industrial Classification (SIC) code used to classify the firm's industry.
  - NAICS: North American Industry Classification System code, a modern classification for industries.
  - DlyCalDt: Calendar day-end date for the observation in format YYYY-MM-DD.
  - MthRet: Daily return for the security (not adjusted for dividends or splits unless specified).
  - sprtrn: S&P 500 return for the same calendar month, used as a market benchmark.

- **10-minute frequency stock returns:** This dataset contains one month of intraday stock prices at the 10-minute frequency. It contains 4 columns that are important for your project:
  - DATE: Date in format YYYYMMDD.
  - SYMBOL: Stock symbol to which the other data are associated.
  - TIME: Intraday time of the data observation in format HH:MM:SS.
  - MID OPEN: Mid-price (average of the best bid and best ask) at TIME

- **Daily Futures returns:** This dataset contains daily prices of various futures contracts. The date column is in format YYYY-MM-DD and is the date of the associated price (at close). As you might know, futures expire. This dataset circumvents this problem by creating a continuous price using the front-month contract returns to create a "synthetic" price history.

##### 3.2 Predictor datasets

Then, there are plenty of predictors that you can use to try to achieve your prediction task:

1. **Quarterly Compustat Firm Characteristics:** This dataset contains firm characteristics from income statements, balance sheets, cash-flow statements, and so on. The dataset has 256 columns, but not all of them are stock characteristics. As you learned, some datasets have many missing values, so you will have to create a way to pick the columns you believe to be meaningful and to take care of duplicates and NaNs accordingly. Also, you might want to investigate the dtypes of each column, as some are strings. Here is a list of the important columns you might need:
   - cusip: This is a stock identifier that you can use to merge with the others datasets.
   - datadate: This is the date in format DD.MM.YYYY. The data on a given line are available from this date on.
   You can find the description of each column in the PDF on the Google Drive.

2. **Jensen, Kelly, and Pedersen (JKP) factors:** This dataset contains the returns of the 153 JKP factors at each date. It contains 3 important columns:
   - date: The date to which the other data are associated. The date has a "same line" approach. This means, for example, if the date is 31.05.1926 and the associated return is 5%, then it means that the return of the factor from the 30.04.1926 that ended on the 31.05.1926 had a 5% return.
   - name: This is the name of the given factor.
   - n stocks: This represents the number of stock in the factor portfolio.
   - ret: This is the return of the given factor.

3. **Chen-Zimmerman data:** This dataset contains monthly long-short return for 205 predictors. You can get the data from Monthly long-short returns for 205 predictors and get the relevant informations using the data dictionary. Additional data might be found on the Fed's website.

4. **Earnings Calls:** Earnings calls are quarterly conference calls hosted by publicly traded companies to discuss their financial results with investors and analysts. They usually follow the release of earnings reports (10-Q or 10-K) and include management commentary on performance, strategy, and guidance, followed by a Q&A session.

5. **10-K Reports:** A 10-K is an annual report filed by public companies with the SEC, providing a comprehensive summary of the firm's financial performance, business operations, risk factors, and financial performance. It includes audited financial statements and is more detailed than quarterly 10-Q filings. The part we are most interested in for this project is the Management Discussion and Analysis (MD&A) section, where executives provide qualitative insights into the company's past performance, strategic outlook, and perceived risks.

6. **Bonus: 8K Report:** An 8-K is a report that publicly traded companies in the United States must file with the Securities and Exchange Commission (SEC) to announce major events that shareholders should know about. Unlike quarterly (10-Q) or annual (10-K) filings, the 8-K is an unscheduled report, filed as needed, usually within four business days of the event. If you want to use those reports, you will have to scrape them yourself. You can get them from the SEC EDGAR website.

#### 4 Report instruction

Each group, consisting of at most 2 people, is required to submit a concise and well-structured report in PDF format, detailing the experimental setup, methodology, results, and conclusions of your project. The report must adhere to the following guidelines:

- The report must not exceed 10 pages, excluding references and appendices (if any). Exceeding this limit will result in penalties.
- The document should be clear and logically organized, including (but not limited to) the following components:
  - Introduction and problem statement
  - Description of data preprocessing and feature engineering
  - Explanation of the predictive models used
  - Evaluation methodology and performance metrics
  - Discussion of results, including tables and figures where appropriate
  - Conclusion and possible extensions
- Figures and tables should be used effectively to illustrate key findings, and must be properly labeled and referenced in the text.
- Any external libraries or tools used must be clearly stated.

In addition to the report, students must submit a well-structured and documented codebase. The code should be clean, modular, and reproducible. It should be possible for the teaching team to run your experiments using minimal setup instructions (ideally from a README file). The quality of the codebase will be taken into account during grading and will have a non-negligible impact on the final evaluation.

#### 5 Small projects examples

##### 5.1 10-minute frequency stock returns

For this high-frequency dataset, the low-frequency (monthly/quarterly) predictors that we provide will not be of great help. Hence, you are forced to use past returns to try to predict future ones. One way could be to build a deep neural network (DNN) that uses the whole cross-section of past returns R_t to predict the whole section of future returns R_{t+1}. You would train your model by fixing a lookback window T and use R_{t−τ}, τ = 1, ..., T as training data. Now, even if your model works, it might very well be that it can not generate any real-life money because of transaction costs or the bid-ask spread. Hence, one cool idea would be to implement a custom loss function that would refrain from excessive trading when using the model predictions.

##### 5.2 Textual embeddings as predictors

Financial documents such as 10-K reports or earnings call transcripts contain valuable forward-looking information that traditional numerical predictors do not always capture. In this project, you could extract textual data from these documents and use natural language processing (NLP) techniques to obtain document-level embeddings. Embeddings are numerical vector representations of text that capture its semantic meaning. Unlike simple keyword counts, embeddings place similar texts close together in vector space, allowing machine learning models to understand context and tone better. For instance, two earnings call segments that express optimism may have similar embeddings even if they use different wording. A simple version could involve regression or classification with embeddings at the firm-month level, but you could also explore how these textual signals interact with known risk factors or whether traditional variables subsume them.

#### 6 Miscellaneous Advice

Here are some recommendations to think about before starting:

- Investigate your data before training your models. Are there outliers? Are there values that do not make sense? Are all of your predictors of the same magnitude? Try to get a deep grasp of how you can make your data as clean as possible before training your model.
- Your goal is to provide predictions that would actually work. In that sense, data mining should be avoided as much as possible: please always split the data into train and test components. Implement techniques that can help you assess whether your model is robust. Being critical of your own work is the best way to avoid catastrophic outcomes when using your models on live data!
- Recall that if something is easy to discover, then it is likely that people might have started trading on it and made the anomaly disappear. So be creative in your approach, it can be a real asset when looking for trading strategies.
- Some of those datasets are very large. You might want to try your methods on a subset of the data before launching your model for a big run. Moreover, when you use Python and load some data, it is loaded into your RAM (Readily Available Memory). This is a part of your computer that can be accessed rapidly to perform operations. Because some of those datasets are large, it might be that your RAM is overloaded (the dataset size is larger than your RAM). If this is the case you can use a subset of your whole dataset for your analysis. The function pd.read_csv() has an argument nrows that you can use to load the first nrows of the CSV file.
- An alternative would be to use Parquet files instead of CSV. Parquet is a columnar storage format that is much more efficient for large datasets, especially when you're only interested in a few columns or rows. It allows for faster reading and smaller memory usage compared to CSV. You can convert a CSV to Parquet using pandas with df.to_parquet() and read it back using pd.read_parquet(). If your dataset is large and you're facing memory constraints, this can significantly affect performance.
- The following Pandas function might be super useful when merging different datasets: pd.merge_asof(). It allows you to merge dataframes of various frequencies by appending the most up-to-date data from the right dataset to the left dataset.
- To merge the different datasets, you will need to use appropriate linking tables. For example, to connect SEC filings (such as 10-K reports) to Compustat data, you can use the Central Index Key (CIK), which is the unique identifier assigned by the SEC to each company. To link Compustat with CRSP data, you should use the CRSP/Compustat Merged (CCM) linking table available on WRDS.

#### 7 Submission Guidelines

To submit your project, please create a private github repo and invite DjoFE2021 to the repo. The repo should include:

- Your source code
- Your report

The repo should have the following name ML For Finance Project–FullName1-Sciper1-FullName2-Sciper2 (Important: do not use space. If your name is Johannes Schwab, write JohannesSchwab). We will not grade your project if you do not respect the submission guidelines. Moreover, I will grade the last commit before the deadline.

---

## 1. Décisions stratégiques — choix arrêtés

### Vue d'ensemble

| Élément | Décision | Justification |
|---|---|---|
| **Dataset cible** | CRSP mensuel | Standard académique, alignement naturel avec les autres datasets |
| **Variable cible** | `MthRet - sprtrn` (excès de rendement) | Convention académique, mesure de l'alpha par rapport au marché |
| **Prédicteur quantitatif** | JKP Factors (153 facteurs) | Prêt à l'emploi, bien cité, 0% NaN sur 2008–2024 |
| **Prédicteur textuel** | Earnings Calls via FinBERT | Information douce non capturée par les chiffres |
| **Usage de FinBERT** | Mode sentiment (3 scores) | Plus simple que les embeddings 768-dim, académiquement solide |
| **Type de tâche** | Régression | Plus adapté à la finance, préserve les magnitudes |
| **Modèle de prédiction** | XGBoost (+ MLP en extension) | Performance, interprétabilité, robustesse |
| **Deep learning** | FinBERT (Transformer) | Satisfait la contrainte obligatoire |
| **Environnement** | MacBook Pro M4 local (MPS backend) | Pas besoin de Colab, suffisant pour l'inférence FinBERT |
| **Stratégies d'investissement** | Multiples, testées en dernière étape | Long-short décile, quintile, long-only, equal/value-weighted |

---

### Décision 1 — Variable cible : excès de rendement

**Choix** : `target = MthRet - sprtrn`

**Justification** :
- `sprtrn` est explicitement fourni et labelisé "market benchmark" dans l'énoncé — signal implicite du prof pour son utilisation
- Convention standard de la littérature (Gu, Kelly & Xiu 2020)
- Ce qu'un hedge fund mesure vraiment (alpha sur le benchmark)
- Stabilise l'entraînement en retirant le mouvement commun du marché

**Validation empirique** (sur le sous-ensemble avec Earnings Calls, 2008–2023) :
- Excès de rendement moyen : **-0.019%/mois** ≈ 0 → distribution centrée
- 48.2% des observations ont un excès positif → quasi-symétrique
- Aucun biais systématique à corriger

---

### Décision 2 — FinBERT en mode sentiment, pas embeddings

**Choix** : extraire les 3 scores de probabilité `[p_positive, p_negative, p_neutral]`, pas le vecteur 768-dim

**Justification** :
- Académiquement solide et citée dans la littérature (Huang et al. 2023, JFE)
- Beaucoup plus simple à intégrer avec les JKP (3 colonnes au lieu de 768)
- Le modèle de prédiction final (XGBoost) gère mieux des features compactes
- Interprétable dans le rapport

**Contrainte deep learning** : FinBERT est un Transformer → la contrainte "au moins un modèle deep learning" est satisfaite par FinBERT lui-même. Le modèle de prédiction final (XGBoost) n'a pas besoin d'être du deep learning.

**Note sur les modèles externes** : la restriction "no external datasets" concerne les données financières, pas les modèles. FinBERT est un modèle pré-entraîné (pas un dataset) → autorisé. Le dictionnaire Loughran-McDonald (LM) est écarté par précaution (zone grise dataset/outil) et parce que FinBERT le rend redondant.

---

### Décision 3 — Régression (pas classification)

**Justification** :
- La finance travaille avec des magnitudes, pas seulement des directions
- Le backtest long-short repose sur le ranking des prédictions → nécessite des valeurs continues
- Standard de la littérature (Gu, Kelly & Xiu 2020)
- La classification peut être mentionnée comme extension dans le rapport

---

### Décision 4 — JKP : colonne `direction` ignorée

**Analyse empirique** :
- Sans multiplication par `direction` : 88.9% des facteurs ont une moyenne positive
- Avec multiplication par `direction` : tombe à 54.2% → dégrade la qualité
- Les deux groupes (direction=1 et direction=-1) ont déjà des rendements positifs en moyenne (+0.24% et +0.17%/mois)

**Conclusion** : `ret` est déjà sign-corrigé dans le dataset. La colonne `direction` est du metadata (indique comment le portefeuille a été construit), pas un coefficient à appliquer. **À ignorer dans le pipeline.**

---

### Décision 5 — Stratégies d'investissement : multiples, testées en fin de projet

**Approche** : le modèle produit un score de prédiction par action par mois. Les stratégies sont des règles de conversion de ces scores en positions, testées après la modélisation.

**Stratégies à tester** :

| Stratégie | Description |
|---|---|
| Long-short décile | Long top 10%, short bottom 10% |
| Long-short quintile | Long top 20%, short bottom 20% |
| Long-only top décile | Long top 10% uniquement |
| Equal-weighted | Pondération identique pour toutes les positions |
| Value-weighted | Pondération par capitalisation boursière |

**Métriques d'évaluation** :
- Sharpe ratio (métrique principale)
- Rendement annualisé
- Maximum drawdown
- Alpha vs S&P500
- Impact des coûts de transaction (0.1–0.3% aller-retour estimé)

---

### Décision 6 — Plan d'ablation (3 modèles)

Pour prouver la valeur ajoutée de chaque composante :

| Modèle | Features | Objectif |
|---|---|---|
| Baseline quantitatif | 153 JKP Factors seuls | Benchmark classique, signal factoriel pur |
| Baseline textuel | 3 scores FinBERT seuls | Signal NLP pur, sans facteurs |
| Modèle combiné | 153 JKP + 3 scores FinBERT | Valeur ajoutée de la combinaison |

**Question de recherche centrale** :
> *"Est-ce que le signal textuel des Earnings Calls apporte une valeur prédictive incrémentale au-delà des facteurs de risque classiques (JKP) ?"*

---

## 2. Analyse des données

### 2.1 CRSP mensuel (`monthly_crsp.csv`, 369 MB)

**Structure générale** :
- 5 179 742 lignes × 11 colonnes
- Période : décembre 1925 – décembre 2024
- 38 853 PERMNO uniques, 1 189 dates mensuelles

**Colonnes** :

| Colonne | Type | Utilité |
|---|---|---|
| `PERMNO` | int | Identifiant unique de l'action → clé de jointure principale |
| `MthCalDt` | date | Date de fin de mois → clé temporelle |
| `MthRet` | float | Rendement mensuel brut → base de la variable cible |
| `sprtrn` | float | Rendement S&P500 → sert à calculer l'excès de rendement |
| `SICCD` | int | Code industrie → potentiellement utile pour analyses sectorielles |
| `HdrCUSIP`, `CUSIP` | str | Identifiants secondaires → non nécessaires |
| `Ticker`, `TradingSymbol` | str | Lisibilité humaine → non nécessaires (NaN massifs) |
| `PERMCO`, `NAICS` | int | Identifiants alternatifs → non nécessaires |

**Colonnes retenues pour le projet** : `PERMNO`, `MthCalDt`, `MthRet`, `sprtrn`, `SICCD`

**Qualité des données** :
- `MthRet` : 1.61% de NaN → à supprimer (dropna)
- `TradingSymbol` : 40% de NaN → colonne inutile, à dropper dès le chargement
- `Ticker` : 10% de NaN → non critique, PERMNO suffit

**Outliers sur MthRet** :
- Maximum observé : **+3 900%** (penny stocks, actions très spéculatives)
- Retours > 100% : **11 031 lignes** (0.2% du total)
- Retours = -1.0 exactement : **204 lignes** → delistings légitimes (faillite, rachat), à conserver
- **Action requise** : winsorisation à 1%–99% avant toute modélisation

```python
p1, p99 = df['MthRet'].quantile([0.01, 0.99])
df['MthRet'] = df['MthRet'].clip(p1, p99)
```

**Période et périmètre utiles** (overlap avec Earnings Calls) :
- Période : 2008–2023
- Lignes : 1 463 164
- PERMNO uniques : 17 251
- PERMNO matchés avec Earnings Calls : **5 377** (quasi-totalité des 5 393 calls)

---

### 2.2 Earnings Calls (`sm-calls_with_connectors.parquet`, 775 MB)

**Structure générale** :
- 35 271 calls × 11 colonnes
- Période utile : 2008–2023 (2005–2007 quasi vide : 612 calls seulement)
- 5 377 PERMNO uniques matchés avec CRSP

**Colonnes** :

| Colonne | Type | Utilité |
|---|---|---|
| `permno` | float | Identifiant CRSP → clé de jointure directe (déjà présent, pas de CCM table nécessaire) |
| `mostimportantdateutc` | datetime | Date du call → clé temporelle pour alignement |
| `text` | str | Transcription complète du call → input FinBERT |
| `word_count` | int | Longueur en mots → utile pour le chunking |
| `text_length` | int | Longueur en caractères |
| `audiolengthsec` | float | Durée audio (5 482 NaN → non critique) |
| `headline` | str | Titre du call (ex: "Albemarle Corp., Q4 2007 Earnings Call") |
| `companyname` | str | Nom de l'entreprise |
| `transcriptid`, `companyid` | float | Identifiants internes → non nécessaires |
| `mostimportanttimeutc` | str | Heure du call → non nécessaire |

**Avantage clé** : le fichier s'appelle "with_connectors" car il contient déjà le `permno` CRSP. Aucune CCM linking table nécessaire.

**Statistiques sur les textes** :
- Longueur moyenne : **7 548 mots** (~44 000 caractères, ~54 min d'audio)
- Minimum : 1 mot (calls vides/corrompus à filtrer)
- Maximum : 42 274 mots

**Problème central — limite FinBERT (512 tokens)** :
- FinBERT accepte 512 tokens maximum en entrée (~380 mots)
- 7 548 mots ÷ 380 ≈ **~20 chunks par call**
- 35 271 calls × 20 chunks ≈ **700 000 passes forward**

**Stratégie de chunking retenue** :
1. Extraire la section Q&A (après le discours formel) → réduit à ~5–8 chunks par call
2. Tokeniser et découper en chunks de 512 tokens
3. Passer chaque chunk dans FinBERT → 3 scores `[p_pos, p_neg, p_neu]`
4. Moyenne pondérée par taille de chunk → 1 triplet de scores par call
5. Sauvegarder en Parquet → utiliser tout au long du projet sans re-exécuter

**Environnement d'exécution** :
- MacBook Pro M4, backend PyTorch **MPS** (Metal Performance Shaders)
- Temps estimé : **2–4 heures** (run unique, à faire la nuit)
- Après sauvegarde : FinBERT n'intervient plus dans le pipeline

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
# → sauvegarder finbert_scores.parquet (3 colonnes : permno, date, p_pos, p_neg, p_neu)
```

**Distribution temporelle** :
- 2005–2007 : 612 calls → ignorés
- 2008–2023 : ~35 000 calls → période utile
- 2024 : 14 calls → ignorés (test set tronqué à 2023)

---

### 2.3 JKP Factors (`[usa]_[all_factors]_[monthly]_[vw_cap].csv`, 9.9 MB)

**Structure générale** :
- 144 620 lignes × 9 colonnes
- Format **LONG** : 1 ligne = 1 facteur × 1 date
- 153 facteurs × 1 188 mois (1926–2024)
- Filtré : `location=usa`, `freq=monthly`, `weighting=vw_cap`

**Colonnes** :

| Colonne | Utilité |
|---|---|
| `name` | Nom du facteur (ex: `mom_12m_1m`, `be_me`, `gp_at`...) |
| `date` | Date de fin de mois |
| `ret` | Rendement du portefeuille long-short du facteur |
| `direction` | -1 ou 1 → **à ignorer** (voir analyse ci-dessous) |
| `n_stocks` | Nombre d'actions dans le portefeuille (info, non utilisée) |
| `location`, `freq`, `weighting`, `n_stocks_min` | Metadata → à dropper |

**Analyse de la colonne `direction`** :
- Analyse empirique : multiplier `ret × direction` **dégrade** les résultats
  - Sans multiplication : 88.9% des facteurs ont une moyenne positive
  - Avec multiplication : seulement 54.2%
- Conclusion : `ret` est déjà sign-corrigé. `direction` = metadata pur → **ignoré**

**Analyse des valeurs manquantes** :

| Période | % NaN | Cause |
|---|---|---|
| 1926–1950 | 11.0% | Seulement 49 facteurs disponibles |
| 1950–1970 | 12.8% | Montée progressive vers 153 facteurs |
| 1970–1990 | 0.2% | Quasi-complet |
| 1990–2008 | 0.0% | Complet |
| **2008–2024** | **0.0%** | **100% de couverture** |

**Vérification** : 31 212 paires (date × facteur) attendues = 31 212 présentes → **couverture exacte 100%** sur 2008–2024. Aucune imputation nécessaire.

**Transformation requise** : pivot long → wide

```python
jkp_wide = df[df['date'] >= '2008-01-01'].pivot(
    index='date', columns='name', values='ret'
)
# Résultat : 204 dates × 153 colonnes, 0 NaN
```

---

## 3. Pipeline complet du projet

```
ÉTAPE 1 — PREPROCESSING (local, ~30 min)
│
├── 1.1 CRSP
│     ├─ Charger monthly_crsp.csv
│     ├─ Garder colonnes : [PERMNO, MthCalDt, MthRet, sprtrn, SICCD]
│     ├─ Filtrer 2008–2023
│     ├─ Dropna sur MthRet
│     ├─ Winsoriser MthRet à [1%, 99%]
│     ├─ Calculer : excess_ret = MthRet - sprtrn
│     └─ Sauvegarder → crsp_clean.parquet
│
├── 1.2 JKP Factors
│     ├─ Charger [usa]_[all_factors]_[monthly]_[vw_cap].csv
│     ├─ Filtrer 2008–2023
│     ├─ Pivot long → wide : index=date, columns=name, values=ret
│     ├─ Résultat : 204 × 153, 0 NaN
│     └─ Sauvegarder → jkp_wide.parquet
│
└── 1.3 FinBERT sur Earnings Calls (M4 MPS, run unique ~3h)
      ├─ Charger sm-calls_with_connectors.parquet
      ├─ Filtrer 2008–2023 (et word_count > 100)
      ├─ Pour chaque call :
      │     ├─ Extraire section Q&A (texte après marqueur Q&A)
      │     ├─ Tokeniser → chunks de 512 tokens
      │     ├─ Passer chaque chunk dans FinBERT (MPS)
      │     └─ Moyenne pondérée → [p_pos, p_neg, p_neu]
      └─ Sauvegarder → finbert_scores.parquet
            Colonnes : [permno, mostimportantdateutc, p_pos, p_neg, p_neu]


ÉTAPE 2 — FUSION DES DATASETS (local, ~10 min)
│
├── Base : crsp_clean.parquet  (PERMNO, MthCalDt, excess_ret, SICCD)
│
├── LEFT JOIN jkp_wide sur MthCalDt = date
│     → Ajoute 153 colonnes JKP (mêmes valeurs pour toutes les actions d'un même mois)
│
├── LEFT JOIN finbert_scores sur (PERMNO, mostimportantdateutc)
│     → Utiliser pd.merge_asof() : aligner le call le plus récent AVANT la date de prédiction
│     → Règle stricte anti look-ahead : call_date < MthCalDt
│     → Ajoute 3 colonnes [p_pos, p_neg, p_neu]
│
└── Résultat final : dataset_final.parquet
      Colonnes : PERMNO, MthCalDt, excess_ret (target), SICCD,
                 [153 JKP factors], p_pos, p_neg, p_neu
      Lignes : ~600 000 (actions × mois avec calls disponibles)


ÉTAPE 3 — SPLIT TEMPOREL (jamais aléatoire)
│
├── Train      : 2008–2014  (~7 ans)
├── Validation : 2015–2018  (~3 ans, pour tuning hyperparamètres)
└── Test       : 2019–2023  (~5 ans, out-of-sample strict)


ÉTAPE 4 — MODÉLISATION — ABLATION STUDY
│
├── Modèle 1 : XGBoost sur JKP seuls (153 features)
│     → Baseline quantitatif
│
├── Modèle 2 : XGBoost sur FinBERT seuls (3 features)
│     → Baseline NLP pur
│
└── Modèle 3 : XGBoost sur JKP + FinBERT (156 features)
      → Modèle combiné principal

Pour chaque modèle :
  - Entraîner sur Train
  - Tuner hyperparamètres sur Validation (grid search ou Optuna)
  - Évaluer sur Test (out-of-sample)

Métriques de prédiction :
  - R² out-of-sample
  - IC (Information Coefficient = corrélation rang entre prédictions et réalisations)
  - Rank IC (plus robuste aux outliers)


ÉTAPE 5 — STRATÉGIES D'INVESTISSEMENT (à partir des prédictions du meilleur modèle)
│
├── Pour chaque mois t dans le Test set :
│     ├─ Récupérer les scores prédits pour toutes les actions disponibles
│     ├─ Ranger les actions par score prédit
│     └─ Construire le portefeuille selon la stratégie choisie
│
├── Stratégie A : Long-short décile
│     → Long top 10%, Short bottom 10%, equal-weighted
│
├── Stratégie B : Long-short quintile
│     → Long top 20%, Short bottom 20%, equal-weighted
│
├── Stratégie C : Long-only top décile
│     → Long top 10%, no short leg
│
└── Stratégie D : Value-weighted
      → Long top 10% pondéré par market cap

Pour chaque stratégie, calculer :
  - Rendement mensuel moyen
  - Rendement annualisé
  - Sharpe ratio
  - Maximum drawdown
  - Alpha et Beta vs S&P500 (régression sur sprtrn)
  - Turnover mensuel
  - Rendement net après coûts de transaction (0.1–0.3% aller-retour)


ÉTAPE 6 — RAPPORT ET CODE (≤ 10 pages PDF + GitHub)
│
├── Sections du rapport :
│     1. Introduction et question de recherche
│     2. Données et preprocessing
│     3. Modèles (FinBERT + XGBoost, ablation study)
│     4. Résultats de prédiction (tableau comparatif des 3 modèles)
│     5. Stratégies d'investissement (courbes de performance, Sharpe)
│     6. Conclusion et extensions
│
└── Code GitHub privé :
      - Modulaire : un script par étape
      - README avec instructions d'exécution
      - Inviter DjoFE2021
```

---

## 4. Points de vigilance techniques

### Look-ahead bias — priorité absolue

Le risque principal : utiliser une information qui n'était pas encore disponible à la date de prédiction.

**Règle stricte** : un Earnings Call daté `call_date` ne peut être utilisé que pour prédire des rendements des mois **strictement postérieurs** à `call_date`.

Implémentation avec `merge_asof` :

```python
# Trier les deux datasets par date
crsp = crsp.sort_values('MthCalDt')
scores = finbert_scores.sort_values('mostimportantdateutc')

# Merger : pour chaque ligne CRSP, prendre le call le plus récent STRICTEMENT avant MthCalDt
merged = pd.merge_asof(
    crsp,
    scores,
    left_on='MthCalDt',
    right_on='mostimportantdateutc',
    by='permno',
    direction='backward'   # prend le call le plus récent AVANT la date CRSP
)
```

### Split temporel — jamais aléatoire

Sur des données de séries temporelles financières, un split aléatoire introduit du data leakage (le modèle voit le futur pendant l'entraînement).

```python
train = df[df['MthCalDt'] < '2015-01-01']
val   = df[(df['MthCalDt'] >= '2015-01-01') & (df['MthCalDt'] < '2019-01-01')]
test  = df[df['MthCalDt'] >= '2019-01-01']
```

### Standardisation des features

JKP et FinBERT scores ont des échelles très différentes. Standardiser sur le train set, appliquer la même transformation sur val/test.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit sur train uniquement
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

### Coûts de transaction

À fréquence mensuelle, modéliser les coûts comme suit :
- Turnover mensuel = % du portefeuille rebalancé
- Coût estimé : 0.1–0.3% aller-retour par transaction
- Rendement net = Rendement brut - (Turnover × coût unitaire)

---

## 5. Justification économique

### Pourquoi CRSP mensuel

Standard académique (Gu, Kelly & Xiu 2020). La fréquence mensuelle permet d'aligner naturellement les prédicteurs trimestriels (Earnings Calls) et mensuels (JKP) sans mismatch de fréquence.

### Pourquoi Earnings Calls

Les Earnings Calls contiennent de l'information **douce** — le ton du CEO, l'hésitation dans les réponses aux analystes, le degré de confiance sur les projections — que les chiffres comptables ne capturent pas. La session Q&A est particulièrement révélatrice (moins préparée, plus spontanée).

Versus 10-K : fréquence trimestrielle vs annuelle, texte plus spontané, un seul pipeline NLP.

### Pourquoi JKP plutôt que Compustat ou Chen-Zimmerman

- **vs Compustat** : JKP prêt à l'emploi, Compustat nécessite un feature engineering lourd sur 256 colonnes avec NaN massifs + CCM linking table
- **vs Chen-Zimmerman** : JKP plus cité dans la littérature récente, couverture identique
- **Effort technique** : pivot + merge par date, 0% NaN sur la période utile

### Pourquoi combiner les deux

Deux types d'inefficience orthogonaux :
1. **Informationnelle douce** (Earnings Calls) : réaction lente et hétérogène des investisseurs aux signaux qualitatifs
2. **Structurelle** (JKP) : anomalies persistantes dues aux contraintes des institutionnels

Ces deux inefficiences ne s'expliquent pas mutuellement → la combinaison devrait donner un signal plus robuste.

---

## 6. Timeline

```
Semaine 1 — 6 au 11 mai 2026   PREPROCESSING
  ├─ Script preprocessing CRSP (filtre, winsorise, excess_ret)
  ├─ Script pivot JKP (long → wide)
  ├─ Parser section Q&A des Earnings Calls
  └─ Lancer FinBERT sur M4 (run overnight, sauvegarder parquet)

Semaine 2 — 12 au 18 mai 2026  FUSION ET MODÉLISATION
  ├─ Script de fusion des 3 datasets (merge_asof)
  ├─ Vérification du dataset final (stats, NaN, look-ahead check)
  ├─ Modèle 1 : XGBoost JKP seuls
  ├─ Modèle 2 : XGBoost FinBERT seuls
  └─ Modèle 3 : XGBoost combiné + tuning

Semaine 3 — 19 au 24 mai 2026  STRATÉGIES ET ÉVALUATION
  ├─ Backtest des 4 stratégies d'investissement
  ├─ Calcul des métriques (Sharpe, drawdown, alpha)
  ├─ Impact des coûts de transaction
  └─ Génération des figures et tableaux pour le rapport

Semaine 4 — 25 au 29 mai 2026  RAPPORT ET CODE
  ├─ Rédaction du rapport PDF (≤ 10 pages)
  ├─ Nettoyage et modularisation du code
  ├─ README et instructions de reproduction
  └─ Push GitHub + invitation DjoFE2021
                                              ↑
                                        Deadline 29 mai 23:59
```

---

## 7. Livrables

1. **Rapport PDF** ≤ 10 pages (hors références et annexes)
2. **Code GitHub privé** — repo `ML For Finance Project–FullName1-Sciper1-FullName2-Sciper2`
   - Inviter `DjoFE2021`
   - Code modulaire, reproductible, avec README
   - Dernier commit avant le 29 mai 23:59
