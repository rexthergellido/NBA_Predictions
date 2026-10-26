[README.md](https://github.com/user-attachments/files/33978316/README.md)
# Predicting NBA Game Outcomes Using Logistic Regression

### An Analysis of Historical Team and Player Performance

**Project type:** Predictive Analytics course project  
**Project status:** Data collection and preparation completed; exploratory data analysis and model development next  
**Tools:** Python · Google Colab · pandas · scikit-learn (planned) · Matplotlib (planned)  
**Historical coverage:** 2015–16 through 2024–25 NBA regular seasons

## 1. Project overview

This project investigates whether historical NBA team and player performance can be used to **predict the probability that the home team wins a regular-season game**. The primary predictive method is **binary logistic regression**. The project emphasizes careful data validation, pregame feature construction, model interpretation, and evaluation on later games or seasons.

This is a **single-model research project**: logistic regression is the predictive algorithm to be developed and evaluated. Different feature sets may be examined within logistic regression, but comparisons among multiple machine-learning algorithms are **not** the research objective.

**Future extension (outside the core course scope):** Use predicted game-win probabilities in a Monte Carlo simulation of remaining games to estimate NBA regular-season standings.

## 2. Research question and objectives

**Research question:** To what extent can historical team-level and player-level indicators predict NBA regular-season game outcomes using logistic regression?

The objectives are to:

1. Explore NBA game results, team performance, and player statistics for potential predictive relationships.
2. Engineer features that would have been known **before each game's tipoff**, including recent team performance and historical player contributions.
3. Develop a **logistic regression** classifier with `HOME_WIN` as the target variable.
4. Evaluate its classification performance and the quality of its predicted win probabilities on chronologically held-out games.
5. Interpret estimated predictor effects while acknowledging data and modeling limitations.

## 3. Data sources and study period

Historical NBA data were collected with Python from ESPN public-facing endpoints:

| Dataset | Endpoint | Key information |
|---|---|---|
| Games | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard` | Date, season, team IDs, home and away teams, final scores, home-team win |
| Player box scores | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary` | Players, teams, starter/DNP indicators, minutes, shooting, points, rebounds, assists, other statistics |

The collection covers ten seasons, **2015–16 to 2024–25**. The ESPN interfaces used here are public-facing endpoints, not a guaranteed stable data contract; availability and response formats may change.

### Historical game coverage

| Season | Regular-season games |
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

The 2019–20 and 2020–21 seasons had unusual schedules, so their game counts should not be compared with 82-game seasons without accounting for schedule differences.

## 4. Completed workflow

### Phase 1 — Data collection

- Collected and combined **11,979 regular-season game records** across ten NBA seasons.
- Downloaded **308,263 original player box-score records** involving **1,488 unique ESPN player IDs**.
- Used game IDs to join game metadata with player box-score records.
- Retained the raw information for auditing and later cleaning.

### Phase 2 — Data preparation and validation

- Investigated missingness: **53,499 original player records** with missing box-score statistics were marked `DID_NOT_PLAY = True`.
- Converted shooting entries such as `FG = "8-15"` into made/attempted numeric components for analysis.
- Identified and removed **36 suspected duplicate player-identity records** using reviewed, same-team/same-game ID-pair rules. These were controlled corrections rather than name-only deduplication.
- Investigated and corrected five starter flags across three exceptional team-games; the resulting starter-count check passed for **23,958 of 23,958 team-games**. This checks counts, not every lineup's independent historical accuracy.
- Checked the sum of player points against official team scores. Found **487 games with placeholder or unavailable player box scores**, all in the 2015–16 to 2017–18 seasons.
- Investigated and applied targeted corrections to the other three team-score mismatches.
- Preserved all game outcomes for team-level analysis, while excluding placeholder box-score games from player-performance analysis.

### Latest prepared datasets

| Data asset | Reported count | Intended use |
|---|---:|---|
| Cleaned game results | **11,979 games** | Game outcomes and team-level features |
| Games passing both team-score checks | **11,492 games** | Eligible historical player box scores |
| Games with placeholder box scores | **487 games** | Retain outcomes; exclude unreliable player performances |
| Cleaned player table | **308,226 records** | Audit-ready player records, including DNP and flagged games |
| Player-performance subset | **244,588 records** | Player-level descriptive analysis and historical features |
| Team-games with five starter flags | **23,958 / 23,958** | Starting-five count validation |

**Outstanding caveat:** Of the 244,588 player-performance records, **245 have unavailable or nonnumeric minutes**. These should not be used directly for per-minute features until participation and minutes are resolved. Matching team-point totals is a useful consistency check but does **not** prove that every individual player statistic is correct.

## 5. Data processing rules

- The original downloaded files are retained separately from cleaned analysis data.
- **DNP player records** are excluded from the `players_performance` analysis subset, but kept in the comprehensive player table.
- **Placeholder-game player records** are excluded from the player-performance analysis subset; their **game outcomes remain available** in the game-level dataset.
- Legitimate zero values (for example, a player who scored zero points) are **not** automatically treated as missing.
- Duplicate-identity and selected box-score corrections are documented separately.
- A game or player record that passes one validation check may still require other checks before modeling.

## 6. Planned exploratory data analysis

The EDA notebook will examine:

- Game counts, scoring distributions, margins, and season trends.
- Home-court advantage and home-versus-away performance.
- Team scoring and points allowed, point differential, and win percentage.
- Pregame rolling team form (for example, previous 5 and 10 games).
- Prior head-to-head performance, rest days, and opponent strength.
- Historical player scoring, shooting, participation, starting-five versus bench contributions, and relevant missing-data patterns.
- Associations between **pregame** features and game outcomes, including feature distributions and correlations.

Descriptive analyses may summarize finished games; however, **same-game scores and player statistics cannot be used as predictors of that same game's result**.

## 7. Predictive modeling plan

### Outcome variable

`HOME_WIN` is the binary response:

- `1` = the home team won;
- `0` = the away team won.

### Chosen predictive model: logistic regression

The model estimates a probability of home-team victory:

\[
P(\text{HOME\_WIN}=1\mid X)=\frac{1}{1+\exp[-(\beta_0+\beta_1X_1+\cdots+\beta_pX_p)]}.
\]

**Candidate predictors** (to be engineered and assessed, not yet final):

| Feature group | Pregame examples |
|---|---|
| Team form | Previous-game win percentage, rolling point differential, rolling points scored/allowed |
| Home/away | Home-team indicator or historical home/away splits |
| Game context | Rest days, back-to-back status, relative rest |
| Opposition | Pregame opponent strength, prior head-to-head record |
| Player contribution | Lagged expected starter/bench strength, historical player production, expected minutes/role |

**Important:** Player features must use information available before tipoff. The actual starters, player minutes, and box scores of the game being predicted cannot be used as pregame predictors unless they were independently known at prediction time.

### Evaluation approach

- **Chronological training/validation/test splits** to prevent future games leaking into earlier predictions.
- Classification metrics: accuracy, precision, recall, F1-score, and ROC-AUC, where appropriate.
- Probability metrics: **log loss and Brier score**, with calibration assessment.
- Compare the fitted model to a simple non-trained reference (for example, a constant historical home-win probability) for context; this does not introduce a second predictive algorithm to the study.
- Use relevant logistic regression coefficients and odds ratios to interpret associations, with care about collinearity and the limits of causal claims.

**No model has been trained or evaluated yet.** Model coefficients, statistical significance, and performance metrics will be reported after implementation.

## 8. Repository structure

```text
nba-standings-prediction/
├── README.md
├── notebooks/
│   ├── 01_nba_data_collection.ipynb
│   ├── 02_nba_data_preparation.ipynb
│   └── 03_nba_exploratory_analysis.ipynb       # planned
├── data/                                        # optional samples / data dictionary
├── reports/                                     # planned charts and results
├── requirements.txt                            # add/update as dependencies are finalized
├── .gitignore
└── LICENSE
```

Only notebooks and files actually created should be uploaded; planned entries illustrate the intended organization. Large raw datasets and batch-download outputs may be kept outside GitHub with reproducible collection instructions.

### Main cleaned files on Google Drive

```text
Colab Notebooks/Dataset/NBA Dataset/cleaned/
├── nba_games_cleaned.csv
├── nba_players_cleaned.csv
├── nba_players_performance.csv
├── nba_game_quality_report.csv
└── nba_correction_audit.csv
```

## 9. Project status and next steps

| Phase | Status |
|---|---|
| 1. Historical data collection | Completed |
| 2. Data preparation and validation | Completed for primary datasets; minutes caveat remains |
| 3. Exploratory data analysis | Next |
| 4. Pregame feature engineering | Planned |
| 5. Logistic regression training and evaluation | Planned |
| 6. Standings simulation | Optional future extension |

**Next milestone:** Complete EDA, resolve or explicitly flag the remaining unavailable-minute cases as needed, engineer leakage-free pregame features, and fit/evaluate logistic regression using a time-based split.

## 10. Limitations and reproducibility

- Historical coverage ends at 2024–25 and does not by itself establish current NBA rosters or availability.
- **487 games** have placeholder player box scores; those games are usable for results-based analyses but not for direct player-performance summaries.
- Player IDs and starter indicators from the source required targeted auditing; corrected records should remain traceable via an audit file.
- Player-experience measures calculated only from this ten-season extract are proxies for actual NBA career tenure.
- Schedule disruptions, team changes, and incomplete player availability may influence historical comparisons.
- For reproducibility, notebooks should run in a documented order, keep raw data immutable, and clearly separate pregame features from postgame outcomes.

---

*Documentation describes the observed data preparation results and the planned single-model research design. Predictive findings have not yet been produced.*
