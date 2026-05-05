# Projet Machine Learning in Finance — Synthèse complète

---

## 1. Énoncé complet du projet

### Contexte et objectif

Le projet représente 30% de la note finale. Il simule une mission de conseil externe pour un hedge fund souhaitant créer un fonds basé sur le machine learning. La tâche consiste à :
1. Trouver un modèle ML pour prédire un dataset de rendements à partir d'un ou plusieurs datasets de prédicteurs
2. Construire des stratégies d'investissement basées sur ces modèles

Le projet attend une analyse rigoureuse et structurée, combinant un preprocessing solide, une conception réfléchie des modèles, et une évaluation soigneuse des résultats. Une attention particulière doit être portée aux limitations de l'approche, à la pertinence économique des résultats, aux coûts de transaction, à la robustesse, et à la performance out-of-sample.

**Deadline : 29 mai 23:59**  
**Groupes : 2 personnes maximum**

---

### Tâches de modélisation possibles

#### Régression
Prédire le rendement futur d'un titre :

$$R_{i,t+1} \approx f(X_t; \theta)$$

Ou prédire la coupe transversale entière des rendements :

$$R_{t+1} \approx f(X_t; \theta) \in \mathbb{R}^P$$

#### Classification
Créer C bins de rendements et prédire la probabilité d'appartenance :

$$P(R_{i,t+1} \in c) \approx f(X_t; c)$$

**Contrainte obligatoire : au moins un modèle doit utiliser le deep learning.**

---

### Datasets disponibles

#### Datasets de rendements (targets)

| Dataset | Période | Fréquence | Contenu clé |
|---|---|---|---|
| CRSP mensuel | Déc. 1925 – Déc. 2024 | Mensuel | Rendements mensuels US, PERMNO, ticker, SIC, S&P500 benchmark |
| CRSP journalier | Jan. 2000 – Déc. 2024 | Journalier | Rendements journaliers US, mêmes identifiants |
| Intraday 10-min | 1 mois | 10 minutes | DATE, SYMBOL, TIME, MID_OPEN (mid-price bid/ask) |
| Futures journaliers | — | Journalier | Prix de clôture de contrats futures (synthétiques, front-month) |

#### Datasets prédicteurs

| Dataset | Fréquence | Description | Contenu |
|---|---|---|---|
| Compustat Firm Characteristics | Trimestriel | Données comptables des entreprises cotées US | 256 colonnes : revenus, actifs, dettes, R&D, dividendes… Identifiant CUSIP. Beaucoup de NaN et colonnes string. |
| JKP Factors | Mensuel | 153 facteurs de risque (Jensen, Kelly & Pedersen 2023) | Rendement par facteur, nom, nombre de titres. Exemples : momentum, value, profitability, low volatility. |
| Chen-Zimmerman | Mensuel | Rendements long-short de 205 anomalies académiques | Par prédicteur : rendement du portefeuille long-short. Couvre accruals, asset growth, earnings surprise… |
| Earnings Calls | Trimestriel | Transcriptions des conférences résultats | Texte brut : discours CEO/CFO + Q&A avec analystes. Contient ton, confiance, projections qualitatives. |
| 10-K Reports (MD&A) | Annuel | Rapport annuel SEC, section analyse de la direction | Texte long, formel et audité : performance, stratégie, risques. Lien Compustat via CIK. |
| 8-K Reports *(bonus)* | Événementiel | Événements majeurs (fusion, litige, changement direction) | À scraper sur EDGAR. Non structuré, très lourd à traiter. |

---

### Instructions pour le rapport

- **Format** : PDF, maximum 10 pages hors références et annexes (pénalité si dépassement)
- **Structure attendue** :
  - Introduction et formulation du problème
  - Description du preprocessing et du feature engineering
  - Explication des modèles prédictifs utilisés
  - Méthodologie d'évaluation et métriques de performance
  - Discussion des résultats avec tableaux et figures
  - Conclusion et extensions possibles
- Figures et tableaux doivent être proprement labelisés et référencés dans le texte
- Toute librairie externe doit être clairement mentionnée

### Instructions pour le code

- Dépôt GitHub **privé**, inviter `DjoFE2021`
- Nom du repo : `ML For Finance Project–FullName1-Sciper1-FullName2-Sciper2`
- Code propre, modulaire, reproductible avec un README
- La qualité du code impacte la note finale

---

### Conseils de l'énoncé

- **Explorer les données avant de modéliser** : outliers, valeurs aberrantes, magnitudes des features
- **Éviter le data mining** : toujours splitter train/test, tester la robustesse
- **Créativité** : si une anomalie est facile à trouver, elle est probablement déjà arbitragée
- **Gestion mémoire** : utiliser Parquet plutôt que CSV pour les grands datasets, `pd.read_csv(nrows=...)`
- **Merging multi-fréquence** : utiliser `pd.merge_asof()` pour aligner des datasets de fréquences différentes
- **Linking tables** : utiliser CIK pour connecter SEC filings à Compustat, CCM linking table pour Compustat-CRSP

---

## 2. Décisions prises — Option B

### Setup général

| Élément | Choix |
|---|---|
| **Target (rendements)** | CRSP mensuel |
| **Prédicteur textuel** | Earnings Calls (embeddings NLP) |
| **Prédicteur quantitatif** | JKP Factors (153 facteurs) |
| **Type de tâche** | Régression (+ éventuellement classification) |
| **Deep learning requis** | Transformer (FinBERT) pour le NLP + MLP/LSTM pour la combinaison |

---

### Justification économique du choix

#### Pourquoi CRSP mensuel

C'est le combo classique de la littérature académique (référence : Gu, Kelly & Xiu 2020). La fréquence mensuelle permet d'aligner naturellement les prédicteurs trimestriels (Earnings Calls) et mensuels (JKP) sans mismatch de fréquence. L'infrastructure est standard et bien documentée.

#### Pourquoi Earnings Calls

Les Earnings Calls contiennent de l'information **douce** — le ton du CEO, l'hésitation dans les réponses aux analystes, le degré de confiance sur les projections — que les chiffres comptables ne capturent pas. C'est une information que les investisseurs traitent lentement et de façon hétérogène, créant une inefficience exploitable. La session Q&A est particulièrement révélatrice car elle est moins préparée et contrôlée que le discours formel.

Les Earnings Calls seuls sont suffisants par rapport aux 10-K (qu'on n'inclut pas) car :
- Ils couvrent les mêmes informations à fréquence plus élevée (trimestriel vs annuel)
- Le texte est plus spontané et donc plus informatif sur l'état d'esprit réel du management
- Un seul pipeline NLP à construire au lieu de deux

#### Pourquoi JKP plutôt que Compustat ou Chen-Zimmerman

- **vs Compustat** : JKP est directement utilisable (merge par date), alors que Compustat requiert un feature engineering lourd sur 256 colonnes avec NaN massifs
- **vs Chen-Zimmerman** : JKP est plus cité dans la littérature récente et suffisant pour la justification quantitative
- **Effort technique** : ajouter JKP revient à un `pd.merge()` supplémentaire par date — coût quasi nul

#### Pourquoi combiner les deux

Les marchés sont inefficients de deux façons **orthogonales** :

1. **Inefficience informationnelle douce** (capturée par les Earnings Calls) : les investisseurs réagissent lentement et émotionnellement aux signaux qualitatifs du management
2. **Inefficience structurelle** (capturée par JKP) : les anomalies quantitatives persistent car les investisseurs institutionnels sont contraints (limites de risque, horizons courts)

Ces deux types d'inefficience ne s'expliquent pas mutuellement. Les combiner dans un même modèle devrait donner un signal plus robuste que chacun séparément.

**Question de recherche centrale du projet :**
> *"Est-ce que le signal textuel des Earnings Calls apporte une valeur prédictive incrémentale au-delà des facteurs de risque classiques (JKP) ?"*

Cette question est ouverte dans la littérature, ce qui rend le projet original.

---

### Architecture envisagée

```
Earnings Calls (texte)
        |
   FinBERT / BERT
        |
  Embedding vectoriel        JKP Factors (numérique)
  (dimension ~768)                    |
        |                    Normalisation / scaling
        └──────────┬─────────────────┘
                   |
           Couche de fusion
         (concatenation ou attention)
                   |
              MLP / LSTM
                   |
           Prédiction R_{i,t+1}
```

---

### Plan d'ablation recommandé

Pour justifier la combinaison dans le rapport, tester les 3 configurations et comparer :

| Modèle | Features | Objectif |
|---|---|---|
| Baseline quantitatif | JKP seuls | Benchmark classique |
| Baseline textuel | Embeddings Earnings Calls seuls | Signal NLP pur |
| Modèle combiné | JKP + Embeddings | Valeur ajoutée de la combinaison |

---

### Points de vigilance techniques

#### Look-ahead bias (priorité absolue)
Le risque principal avec les Earnings Calls est d'utiliser une information publiée après la date de prédiction. Il faut aligner précisément :
- Date de publication du call (souvent quelques jours après la fin du trimestre)
- Date à partir de laquelle l'information est disponible pour le marché
- Rendement mensuel CRSP correspondant

**Règle stricte** : n'utiliser un Earnings Call pour prédire un rendement que s'il était publié *avant* la période de ce rendement.

#### Coûts de transaction
À fréquence mensuelle, les coûts de transaction sont moins critiques qu'en intraday, mais restent à modéliser dans le backtest (typiquement 0.1–0.3% aller-retour).

#### Train / test split
Utiliser un split temporel strict (jamais aléatoire sur des données de séries temporelles). Exemple : train jusqu'en 2015, validation 2015–2018, test 2018–2024.

---

### Livrables attendus

1. **Rapport PDF** ≤ 10 pages structuré selon les consignes
2. **Code** propre, modulaire, avec README sur GitHub privé (`DjoFE2021` invité)

