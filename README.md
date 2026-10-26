[README.md](https://github.com/user-attachments/files/33980968/README.md)
# NBA Game Outcome Prediction and Analytics

**An end-to-end NBA machine learning and predictive analytics project**  
**Research component:** *Predicting NBA Game Outcomes Using Logistic Regression: An Analysis of Historical Team and Player Performance*

**Project status:** Historical data collection and primary cleaning completed; exploratory data analysis in progress  
**Data coverage:** 2015–16 through 2024–25 NBA regular seasons  
**Environment:** Python · Google Colab · pandas · NumPy · Matplotlib · scikit-learn (modeling planned)

## Project overview

This repository documents an end-to-end analytical workflow for understanding and predicting NBA regular-season game outcomes. It includes historical data collection, data-quality auditing, exploratory data analysis (EDA), pregame feature engineering, predictive modeling, and reproducible evaluation.

The repository also supports a **Predictive Modeling course culminating project**. That academic component has a deliberately narrower scope: **develop and evaluate one binary logistic regression model** to estimate the probability of a home-team win using historical team and player indicators.

The wider repository may later be extended with other machine learning methods, player-strength analysis, and **Monte Carlo simulation of NBA regular-season standings**. These extensions are **not part of the current course study** and are not presented as completed work.

### Two scopes, one repository

| Area | End-to-end GitHub project | Predictive Modeling course |
|---|---|---|
| Purpose | Showcase the complete analytics and machine learning workflow | Develop and evaluate a focused predictive research study |
| Data collection and preparation | Included | Included |
| EDA | Broad team, player, and game-context analysis | Analysis relevant to the research questions and predictors |
| Predictive modeling | Extensible framework | **One algorithm: binary logistic regression** |
| Evaluation | Reproducibility, performance, interpretation | Classification, probability quality, and model interpretation |
| NBA standings simulation | Possible later extension | Outside the current study |

## Culminating research study

### Research title

**Predicting NBA Game Outcomes Using Logistic Regression: An Analysis of Historical Team and Player Performance**

### Research question

How effectively can historical pregame team and player performance indicators predict the probability of a home-team victory in an NBA regular-season game using logistic regression?

### General objective

Develop and evaluate a logistic regression model for predicting NBA regular-season game outcomes using historical team- and player-level performance information.

### Specific objectives

1. Describe the historical distribution of NBA game outcomes and relevant team/player performance characteristics.
2. Construct candidate **pregame** features from previous results and previous player performances, including recent form, scoring trends, and estimated starter/bench contributions.
3. Fit a binary logistic regression model with `HOME_WIN` as the response variable.
4. Assess predictive performance on chronologically held-out games using discrimination, classification, and probability metrics.
5. Interpret estimated coefficients and odds ratios while discussing data-quality constraints and the distinction between association and causation.

### Target and study unit

- **Unit of observation:** One NBA regular-season **game** (not one player-game record).
- **Dependent variable:** `HOME_WIN`, with `1` for a home-team victory and `0` for an away-team victory.
- **Algorithm:** Binary logistic regression.
- **Predictors:** Historical features available before tipoff. **Same-game final scores, player box scores, and realized minutes cannot be used to predict that same game.**

A player-level table is useful for summarizing historical performance, but player statistics must be aggregated into appropriately lagged, game-level features before fitting the classifier.

## Data sources and coverage

Data were collected using Python from ESPN's public-facing basketball endpoints:

| Data source | Endpoint | Main fields |
|---|---|---|
| Game scoreboard | [`/nba/scoreboard`](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard) | Game and season IDs, date, home/away teams, official scores, result |
| Game summary | [`/nba/summary`](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary) | Player/ESPN IDs, team, starter and DNP flags, minutes, shooting, counting statistics |

These are public-facing endpoints; they are not guaranteed to remain stable. Availability and response formats may change.

### Regular-season game coverage

| Season | Games |
|---|---:|
| 2015–16 | 1,230 |
| 2016–17 | 1,230 |
| 2017–18 | 1,230 |
| 2018–19 | 1,230 |
| 2019–20 | 1,059 |
| 2020–21 | 1,080 |
| 2021–22 | 1,230 |
| 2022–23 | 1,230 |
| 2023–24 | 1,230 |
| 2024–25 | 1,230 |
| **Total** | **11,979** |

The 2019–20 and 2020–21 seasons had atypical schedules and must be handled carefully in season-to-season comparisons.

## Completed work

### Phase 1 — Historical data collection ✅

- Collected **11,979** NBA regular-season game results across **10 seasons**.
- Downloaded **308,263** original ESPN player-game records representing **1,488 unique source player IDs**.
- Retrieved game results and player summaries using ESPN's scoreboard and summary endpoints.
- Merged game metadata with player records by `GAME_ID` for auditing and preparation.
- Retained source data rather than overwriting the raw extracts.

### Phase 2 — Data preparation and quality validation ✅ (with documented limitations)

- Investigated missing box-score statistics: **53,499** original player records with entirely absent box-score statistics were flagged `DID_NOT_PLAY = True`.
- Parsed made/attempted shooting values (e.g. `FG = "8-15"`) into numerical components.
- Identified **36** repeated-name, same-team/same-game cases associated with suspected duplicate ESPN player identities and removed the secondary records using controlled ID-pair rules.
- Corrected five identified starter flags across three games; the **starter-count** check then yielded exactly five starters in **23,958 / 23,958** team-games.
- Validated aggregated player points against the recorded official team scores and investigated three isolated non-placeholder discrepancies.
- Flagged **487 games** whose player box scores contained placeholder data despite positive official team scores. These games occur in the **2015–16 through 2017–18** seasons.
- Preserved valid game results from those 487 games while excluding unreliable player box scores from player-performance analysis.

### Data quality summary

| Prepared asset or check | Result |
|---|---:|
| Regular-season games retained | **11,979** |
| Games passing both team-score consistency checks | **11,492** |
| Games with placeholder player box scores | **487** |
| Cleaned player records | **308,226** |
| Player-performance records (score-validated games, non-DNP) | **244,588** |
| Team-games passing five-starter **count** check | **23,958 / 23,958** |
| Player-performance records with unavailable/non-numeric minutes | **245** |

**Important distinctions:** A score match does not validate every player statistic, and five recorded starters does not independently verify their identities. The 245 unresolved minutes records must be addressed or excluded from **per-minute** feature calculations, without misclassifying legitimate zero-point games.

### Dataset filtering rules

`players_performance` retains player records only when both conditions hold:

1. The game passes the player-scoring consistency check (`PLAYER_STATS_VALID = True`).
2. The player is not flagged DNP (`DNP_FLAG = False`).

A valid recorded `0` (for example, zero points by a player who played) is **not** treated as missing. Unavailable minutes such as `"--"` are flagged rather than silently converted to zero.

## Phase 3 — Exploratory data analysis 🔄

EDA is designed to support both descriptive insights and the selection of predictors for the academic model.

| Part | Planned analyses | Modeling relevance |
|---|---|---|
| 1. Dataset and outcome overview | Game/season coverage, completeness, home-win class distribution, scoring trends | Sample characteristics and class balance |
| 2. Team performance | Home/away results, offense/defense, rolling team form, head-to-head history | Candidate pregame team predictors |
| 3. Player performance | Productivity, shooting, starters vs. bench, historical efficiency | Candidate historical player-strength predictors |
| 4. Game context | Rest days, back-to-backs, opponent strength | Additional pregame conditions |
| 5. Predictor assessment | Feature-outcome patterns, collinearity, approximate logit-linearity, stability by season | Logistic regression suitability and specification |

**EDA is currently in progress; no EDA findings or predictive results are claimed here until computed.** Completed-game statistics may be used for descriptive analysis, but model predictors must be calculated strictly from prior information.

## Planned feature engineering

The modeling table will have **one row per game**, with `HOME_WIN` as its target. Candidate features include:

| Group | Example pregame variables |
|---|---|
| Recent team strength | Previous 5/10-game win percentage, rolling scoring margin |
| Offensive and defensive form | Lagged points scored and points allowed |
| Relative matchup strength | Differences between home and away historical ratings/features |
| Player contribution | Lagged expected starter strength and reserve contribution, where roster information permits |
| Schedule context | Rest-day difference, back-to-back indicators |
| Opponent history | Prior head-to-head and pregame opponent strength |

**Leakage safeguards:** Sort games chronologically, calculate rolling indicators from *previous* games only, and handle missing early-season histories and incomplete player box scores explicitly. Historical starters do not automatically identify future expected starters; lineup availability needs separate pregame evidence.

## Course model: binary logistic regression (planned)

The selected model estimates the probability of a home-team win:

\[
P(\mathrm{HOME\_WIN}=1\mid X)=\frac{1}{1+\exp[-(\beta_0+\beta_1X_1+\cdots+\beta_pX_p)]}.
\]

Planned assessment:

- **Temporal training, validation, and testing** to evaluate later games without training on future results.
- **Classification metrics:** accuracy, precision, recall, F1-score, and ROC-AUC when applicable.
- **Probability metrics:** log loss, Brier score, and calibration diagnostics.
- **Interpretation:** coefficients, odds ratios, direction of associations, and multicollinearity checks.
- **Reference baseline:** a simple non-trained prediction (such as historical home-win frequency) for context; **no second machine learning algorithm** is required for the culminating study.

**Model status:** Not yet trained. This repository does not yet report coefficients, statistical significance, test-set accuracy, or forecasting performance.

## Broader project: possible future extensions

After the culminating project, the same repository may expand into:

- Alternative machine learning approaches and comparisons against the initial logistic model.
- Further historical player-impact and expected-lineup modeling.
- Monte Carlo simulation of the remaining NBA schedule using pregame win probabilities.
- Projected wins, standings uncertainty, and playoff probability estimates.

These items are future possibilities, **not completed deliverables and not requirements of the current course study**.

## Repository organization

```text
nba-standings-prediction/
├── README.md
├── notebooks/
│   ├── 01_nba_data_collection.ipynb
│   ├── 02_nba_data_preparation.ipynb
│   ├── 03_nba_exploratory_analysis.ipynb       # in progress
│   ├── 04_nba_feature_engineering.ipynb       # planned
│   └── 05_logistic_regression.ipynb           # planned
├── reports/
│   └── predictive_modeling_course_report.pdf # planned course deliverable
├── data/                                       # optional samples and data dictionary
├── requirements.txt                           # planned or update as needed
├── .gitignore
└── LICENSE
```

The tree shows the intended organization; planned files should not be interpreted as already committed. Large CSVs, downloaded batches, and temporary outputs can stay in Google Drive rather than being committed to GitHub.

### Cleaned data files (Google Drive)

```text
Colab Notebooks/Dataset/NBA Dataset/cleaned/
├── nba_games_cleaned.csv
├── nba_players_cleaned.csv
├── nba_players_performance.csv
├── nba_game_quality_report.csv
└── nba_correction_audit.csv
```

## Progress and next steps

| Phase | Status |
|---|---|
| 1. Data collection | ✅ Completed |
| 2. Cleaning and validation | ✅ Primary checks completed; documented caveats |
| 3. Exploratory data analysis | 🔄 In progress |
| 4. Pregame feature engineering | ⏳ Planned |
| 5. Logistic regression and holdout evaluation | ⏳ Planned — course culmination |
| 6. NBA standings simulation | ⏳ Optional future portfolio extension |

**Immediate next step:** Complete EDA Part 1 (dataset overview, data quality, outcome distribution, and scoring trends), then move to team/player EDA and leakage-free feature engineering.

## Limitations and reproducibility

- ESPN endpoints used here may change, and some older player summaries contain unavailable or placeholder statistics.
- 487 games lack reliable player box scores, although their game outcomes remain available.
- 245 retained player-performance records have unavailable/non-numeric minutes; avoid assuming these are zero-minute or confirmed participating records.
- Source player IDs, starter flags, and selected player-game statistics required targeted corrections; retain the relevant correction/audit files.
- Observing a relationship does not establish causation; model results must be evaluated out of sample.
- Historical player availability, injuries, roster changes, and other information may limit genuine pregame predictions if not explicitly collected.
- Keep original inputs immutable, record transformations, and clearly distinguish actual completed results from future model plans.

---

*This README describes one overarching NBA analytics project and its focused Predictive Modeling course component. The current completed work is historical data collection and primary preparation; EDA, predictive findings, and standings forecasts are forthcoming.*
