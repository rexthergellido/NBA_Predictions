[NBA_Standings_Prediction_README.md](https://github.com/user-attachments/files/33956574/NBA_Standings_Prediction_README.md)

# NBA Regular-Season Standings Prediction — Data Engineering and Modeling Project

**Status:** Historical game results validated; player box-score ingestion and merge scripts developed; modeling not yet started.

## 1. Project overview

This project aims to forecast NBA regular-season standings by estimating team strength and game-win probabilities from historical game results and player-level performance. The planned modeling approach will consider:

- Historical team performance and home-court advantage
- Player-level box-score statistics and experience
- Expected roster and lineup strength, including starting five versus bench
- Opponent and head-to-head history
- Game-by-game probabilities and season schedule simulation

**Current scope:** Data engineering and exploratory preparation. No trained prediction model, accuracy metrics, or simulated standings are available yet.

## 2. Technical environment

- **Notebook:** Google Colab / Python
- **Storage:** Google Drive
- **Libraries used:** `pandas`, `requests`, `pathlib`, `time`, and `tqdm`
- **Game data source:** ESPN public NBA scoreboard endpoint
- **Player data source:** ESPN public NBA game-summary endpoint

We initially explored the `nba_api` NBA Stats endpoints, but requests timed out in Colab. We therefore switched to ESPN's public-facing endpoints. These are not a guaranteed or versioned data contract; availability, field names, and access conditions may change.

```python
from pathlib import Path
BASE_DIR = Path('/content/drive/MyDrive/Colab Notebooks/Dataset/NBA Dataset')
```

## 3. Data sources and extraction

### 3.1 Historical game results

ESPN scoreboard endpoint:

```text
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
```

Typical request parameters:

```python
params = {'dates': '20241022', 'limit': 100}
```

The workflow queried game dates across ten NBA seasons, extracted game identifiers, dates, teams, scores, and home-team win indicators, and saved raw and cleaned season-level CSV files.

### 3.2 Player box scores

ESPN game-summary endpoint:

```text
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary
```

Example:

```python
import requests
payload = requests.get(
    'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary',
    params={'event': '400827888'},
    timeout=30,
).json()
```

The extractor iterates through `boxscore.players`, the nested `statistics` groups, and their `athletes`. It maps each statistics group's `names` to the player's corresponding `stats` values.

Extracted player fields include:

`SEASON`, `GAME_ID`, `GAME_DATE`, `TEAM_ID`, `TEAM`, `PLAYER_ID`, `PLAYER`, `POSITION`, `STARTER`, `DID_NOT_PLAY`, `MIN`, `PTS`, `FG`, `3PT`, `FT`, `REB`, `AST`, `TO`, `STL`, `BLK`, `OREB`, `DREB`, `PF`, and `+/-`.

**Parsing note:** `FG`, `3PT`, and `FT` are made/attempted strings, not numeric percentages; `MIN` and `+/-` also require normalization before modeling.

## 4. Historical game data cleaning

The initial scoreboard extraction contained events outside the regular season. The cleaning pipeline:

1. Selected ESPN regular-season events (`season.type == 2`).
2. Restricted matchups to the 30 recognized NBA franchises, allowing the `LA Clippers` name variant.
3. Excluded All-Star mini-tournament games accidentally returned with the 2024–25 schedule.
4. Excluded NBA Cup championship games that do **not** count toward regular-season standings (2023–24 and 2024–25).
5. Deduplicated game records and checked season-level counts.

### Validated season totals

| Season | Games | NBA teams | Expected games | Count check |
|---|---:|---:|---:|---|
| 2015–16 | 1,230 | 30 | 1,230 | Pass |
| 2016–17 | 1,230 | 30 | 1,230 | Pass |
| 2017–18 | 1,230 | 30 | 1,230 | Pass |
| 2018–19 | 1,230 | 30 | 1,230 | Pass |
| 2019–20 | 1,059 | 30 | 1,059 | Pass |
| 2020–21 | 1,080 | 30 | 1,080 | Pass |
| 2021–22 | 1,230 | 30 | 1,230 | Pass |
| 2022–23 | 1,230 | 30 | 1,230 | Pass |
| 2023–24 | 1,230 | 30 | 1,230 | Pass |
| 2024–25 | 1,230 | 30 | 1,230 | Pass |
| **Total** | **12,009** | | **12,009** | **Pass** |

The 2019–20 season was disrupted by COVID-19; 2020–21 used a shortened 72-game-per-team schedule. **Passing game-count checks does not by itself prove every individual game record is correct.**

## 5. Player box-score extraction: testing and batching

### Pilot test

Five game summaries were tested, one from each of 2015–16 through 2019–20:

| Season | Example game ID | Player records |
|---|---|---:|
| 2015–16 | 400827888 | 26 |
| 2016–17 | 400899375 | 26 |
| 2017–18 | 400974437 | 25 |
| 2018–19 | 401070213 | 26 |
| 2019–20 | 401160623 | 26 |
| **Total** | | **129** |

In this pilot, all ten team-game groups had exactly five starters, and the 129 extracted records included 50 starters. This validates the **sample**, not the entire ten-season extraction.

### Full extraction workflow developed

- Process the **12,009** known game IDs.
- Download summaries with `requests.Session()`.
- Retry transient failures up to three times.
- Wait approximately 0.8 seconds between requests.
- Write a checkpoint CSV for every 100 games under `player_boxscores/`.
- Record failed requests in `failed_batch_XXXX.csv`.
- Concatenate saved batch files and remove duplicate player-game-team keys.

**Important limitation of the initial checkpoint implementation:** It skips any batch with an existing `players_batch_XXXX.csv`, even if that batch had failed game requests. Therefore, saved batches must be reconciled with `failed_batch_XXXX.csv` and missing game IDs must be retried before claiming complete coverage. Long Colab sessions may also disconnect.

## 6. Combining game and player data

There are two levels of observation:

- **Game table:** one row per `GAME_ID`.
- **Player table:** multiple rows per `GAME_ID` (one row per player and team).

This is a **many-to-one merge** from player rows to game metadata. The resulting table remains **player-game level**, not team-game level.

```python
import pandas as pd
from pathlib import Path

BASE_DIR = Path('/content/drive/MyDrive/Colab Notebooks/Dataset/NBA Dataset')
games = pd.read_csv(BASE_DIR / 'nba_games_2015_2025_combined.csv',
                    dtype={'GAME_ID': str})
players = pd.read_csv(BASE_DIR / 'nba_player_boxscores_2015_2025.csv',
                      dtype={'GAME_ID': str, 'PLAYER_ID': str, 'TEAM_ID': str})

assert games['GAME_ID'].is_unique, 'Game IDs must be unique in games table'

merged = players.merge(
    games.drop(columns=['SEASON', 'GAME_DATE'], errors='ignore'),
    on='GAME_ID',
    how='left',
    validate='many_to_one',
    indicator=True,
)

print(merged['_merge'].value_counts())
assert merged['_merge'].eq('both').all(), 'Unmatched player rows detected'
merged = merged.drop(columns='_merge')

output = BASE_DIR / 'nba_complete_dataset_2015_2025.csv'
merged.to_csv(output, index=False)
print('Saved:', output)
```

The merge script was provided; **successful execution and final output counts have not been independently confirmed in this documentation**.

## 7. Quality assurance checks

Recommended checks after downloading and merging:

1. Compare unique player-data `GAME_ID`s against all **12,009** historical games.
2. Report missing game IDs and retry them.
3. Check exactly two NBA teams per game.
4. Audit exactly five starters per team-game, investigating exceptions rather than automatically deleting them.
5. Check duplicates on (`GAME_ID`, `TEAM_ID`, `PLAYER_ID`).
6. Confirm that player `TEAM_ID` belongs to the game's home or away team.
7. Inspect null values, DNPs, minutes, and box-score field formats.
8. Cross-check selected player/team point totals with final game scores.
9. Ensure `GAME_DATE` and `SEASON` agree between source tables.

A prior discussion mentioned a player-data coverage figure of **11,979 games** versus **12,009** expected, a possible gap of **30 games**. This figure needs to be rechecked against actual saved files; it should **not** be presented as a validated final count.

## 8. Proposed repository structure

```text
nba-standings-prediction/
├── README.md
├── notebooks/
│   ├── 01_collect_games.ipynb
│   ├── 02_clean_validate_games.ipynb
│   ├── 03_collect_player_boxscores.ipynb
│   ├── 04_merge_validate_datasets.ipynb
│   └── 05_feature_engineering.ipynb      # planned
├── data/
│   ├── raw/                            # optional local samples only
│   ├── processed/                      # optional local samples only
│   └── validation_report.csv
├── src/                                 # planned reusable scripts
├── requirements.txt
└── .gitignore
```

This is a **recommended structure**, not a claim that all these files already exist. Export the actual Colab notebooks and rename them consistently before publishing. Large data files can remain on Google Drive, with instructions for reproducing the downloads; verify source terms and privacy before sharing.

Suggested `.gitignore`:

```gitignore
.ipynb_checkpoints/
__pycache__/
.env
*.log
data/raw/
data/processed/
player_boxscores/
```

## 9. Modeling roadmap

**Phase 1 — Completed:** Historical game extraction, cleaning, and season-level count validation.

**Phase 2 — In progress:** Download all player game logs, reconcile missing games, normalize statistics, and validate the merged player-game dataset.

**Phase 3 — Planned:** Engineer pregame features: rolling team win rate, recent scoring margin, home/away splits, rest days, opponent strength, roster availability, starter/bench contributions, and prior head-to-head outcomes. Experience features will require a separate historical roster or career-history source.

**Phase 4 — Planned:** Build chronological train/validation/test splits and train baseline and more advanced game-win probability models. Compare against simple baselines and assess calibration as well as discrimination.

**Phase 5 — Planned:** Obtain the target season's actual schedule and updated rosters; predict individual matchups and simulate remaining games repeatedly to estimate win totals, standings, and uncertainty.

### Avoiding data leakage

For a game on date *t*, all features must be computed only from information available **before tipoff on date t**. Do not use full-season player averages or final standings as predictors for games earlier in that season. Evaluate on future seasons or chronological holdouts rather than randomly mixing games from the same season.

## 10. Reproducibility and caveats

- Historical window currently ends at **2024–25**; forecasting a later season requires current rosters, player movement, injuries/availability, and schedule data.
- The ESPN endpoints are public-facing but unofficial for this project; handle request failures and schema changes.
- The current dataset is descriptive and historical; it does not yet include verified current roster membership or player experience.
- A `SyntaxError` encountered during notebook execution came from pasting Markdown triple-backtick fences into a Python cell. Only Python code, not the surrounding ``` fences, should be pasted into executable cells.
- Checkpoint files are not proof of complete game coverage; use explicit game-ID reconciliation.

## 11. Files created or targeted in Google Drive

| Path relative to `NBA Dataset/` | Purpose | Status |
|---|---|---|
| `raw/nba_games_{season}_raw.csv` | Raw season-level game downloads | Created in game collection workflow |
| `clean/nba_games_{season}_clean.csv` | Filtered regular-season games | Created and count-validated |
| `nba_games_2015_2025_combined.csv` | Combined historical game table | Validated: 12,009 games |
| `validation_report.csv` | Season-level game count checks | Created |
| `player_boxscores/players_batch_XXXX.csv` | Checkpointed player box-score batches | Download workflow supplied; completeness unverified |
| `player_boxscores/failed_batch_XXXX.csv` | Failed game download logs | Created when failures occur |
| `nba_player_boxscores_2015_2025.csv` | Concatenated player-game rows | Combining code supplied; final completeness unverified |
| `nba_complete_dataset_2015_2025.csv` | Merged player and game data | Merge code supplied; successful creation unverified |

## 12. Next immediate actions

1. Reconcile the downloaded player `GAME_ID`s against all 12,009 expected games.
2. Retry missing or failed games and rebuild the combined player file.
3. Run merge integrity and box-score quality checks.
4. Export and commit the actual Colab notebooks alongside this README.
5. Begin pregame feature engineering, then develop and validate a baseline prediction model.

---

*Project documentation reflects the work discussed and reported so far. Results that have not been verified are explicitly labeled as such.*
