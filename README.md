# EPL Predictor V2.1

A mobile-first Flask application that produces explainable English Premier League 1X2 probabilities. V2.1 is an operational reliability release: the statistical model remains **Enhanced Dixon–Coles V2, model version 2.0**.

The application returns:

- home win, draw and away win probabilities
- expected goals and the most likely score
- the top five scorelines
- contributing model inputs, explicitly labelled as indicators rather than causal explanations
- separate upcoming-fixture, completed-result and model-diagnostics views

It does **not** provide betting recommendations, picks, stakes or guarantees.

## Versioning

| Component | Version | Meaning |
|---|---:|---|
| Application | 2.1 | stale-while-revalidate, live tracking, CI and operations |
| Live statistical model | 2.0 | unchanged V2 Elo/form/Dixon–Coles model |
| V1 comparison model | 1.0 | retained Elo + independent Poisson baseline |

Prediction metadata exposes both `app_version` and `model_version`, so infrastructure releases are not mistaken for modelling changes.

## Prediction architecture

| Capability | V1 baseline | V2 live model |
|---|---|---|
| Rating | Elo | Elo |
| Goals | Basic independent Poisson | Dixon–Coles-adjusted Poisson |
| Team strength | Time-decayed home/away goals | Venue attack/defence with regression to league averages |
| Form | Aggregate weighted goals | Recent 5 and 10 GF/GA, shots and shots on target |
| Context | Fixed home advantage | Rest days and EPL-only 7/14-day congestion |
| xG | No | Real HxG/AxG only when the source exposes it |
| Calibration | No | Chronological temperature scaling |
| Evaluation | Comparison baseline | Walk-forward plus live prospective tracking |

V1 remains available through `?model=v1`. V2.1 does not alter the V2 expected-goals, Elo, form or Dixon–Coles logic.

## Data and leakage prevention

The baseline source is [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). The committed six-season snapshot contains fields that are actually present in each CSV:

- date, time, teams, full-time goals and result
- shots and shots on target
- `HxG` / `AxG` when published
- decimal 1X2 odds for an optional benchmark

Optional columns are detected safely. Bookmaker odds are normalized to remove overround and are used only as an evaluation benchmark.

Historical predictions use state ending before the match date. Same-day fixtures are predicted as one group before any result from that date is added. The pipeline never uses future Elo, later matches, final standings or target-match statistics.

## Startup refresh and stale-while-revalidate

`DATA_MAX_AGE_HOURS` defaults to 12 hours.

At startup, the committed/cached snapshot is loaded first. If stale, one guarded refresh is attempted. A failure never prevents the app from starting.

During normal web and API traffic:

1. a cheap freshness check runs at most once every five minutes;
2. fresh data causes no download;
3. stale data immediately continues serving the current valid model;
4. one daemon thread downloads fresh Football-Data CSV files;
5. that thread builds a complete repository plus V1 and V2 predictors away from live traffic;
6. only after every step succeeds is the complete runtime state atomically swapped;
7. failure preserves the previous state and records `last_refresh_error`.

A lock and `refresh_in_progress` flag prevent duplicate downloads. There is no retry loop and no refresh on every request. The Render configuration intentionally uses one Gunicorn worker with four threads, matching this process-local coordination model.

`/health`, `/api/model` and prediction metadata expose:

```json
{
  "status": "ok",
  "app_version": "2.1",
  "model_version": "2.0",
  "data_through": "2026-09-20",
  "last_refresh_attempt": "...",
  "last_refresh_success": "...",
  "data_stale": false,
  "refresh_in_progress": false,
  "last_refresh_error": null,
  "matches": 1950
}
```

`POST /api/refresh` intentionally does not exist.

## Historical backtest

Run:

```bash
python manage.py backtest
```

The published walk-forward report uses 2023/24 as the first out-of-sample calibration season, then evaluates 2024/25, 2025/26 and available 2026/27 matches chronologically. Calibration is fitted only on earlier predictions.

Generated reports:

- `reports/backtest_v2.json`
- `reports/backtest_v2.csv`

Current historical report, generated from 810 actual evaluation matches through 2026-09-20:

| Model | Log Loss ↓ | Brier ↓ | RPS ↓ | Accuracy |
|---|---:|---:|---:|---:|
| V1 Elo + Poisson | 1.0232 | 0.2044 | 0.2088 | 50.25% |
| V2 Dixon–Coles | **1.0082** | **0.2014** | **0.2056** | 49.75% |
| Normalized bookmaker benchmark | 0.9982 | 0.1992 | 0.2018 | 50.99% |

V2 improved Log Loss, Brier and RPS over V1 in this backtest, but not accuracy, and it did not beat the bookmaker benchmark.

## Live prospective prediction tracking

Historical backtests and live performance are deliberately separate.

Record predictions before kickoff:

```bash
python manage.py snapshot_predictions
```

The command reads only reliable future fixtures already present in the source, generates model-version 2.0 predictions and appends them under:

```text
reports/live_predictions/YYYY-MM-DD.json
```

Records include creation time, fixture/kickoff, 1X2 probabilities, expected goals, likely score, model version, data-through date and xG status. A six-hour fixture/model window prevents duplicates. If no reliable future fixtures exist, the command exits successfully without inventing dates.

Settle recorded predictions after results arrive:

```bash
python manage.py score_predictions
```

Actual results are written separately to `reports/live_results.json`; original prediction files are never modified during scoring. `reports/live_performance.json` contains:

- overall live accuracy, Log Loss, Brier and RPS
- rolling last-20 and last-50 metrics
- season-to-date metrics
- sample-size-aware descriptive status

Until 50 predictions are settled, diagnostics display:

> Small sample — live performance is not yet statistically reliable.

## Persistence design

Render's local filesystem is ephemeral. Runtime files created only inside a Render instance are **not** described as durable.

Durable automation is provided by the scheduled GitHub Actions workflow. It records prediction/report changes and commits only those files back to the repository with the built-in `GITHUB_TOKEN`. No paid database, personal access token or API secret is required.

For local use, JSON files remain local until explicitly committed. If repository settings or branch protection prevent GitHub Actions from pushing to `main`, scheduled predictions will run but will not persist; allow the workflow's built-in token to write repository contents or adopt an approved pull-request workflow.

## GitHub Actions

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests:

1. install dependencies;
2. run `pytest -q`;
3. run `python manage.py smoke_test`;
4. verify V1/V2 probability normalization and non-negative expected goals;
5. run a lightweight chronological leakage check;
6. import the production WSGI app.

The full 810-match backtest is not repeated on every commit.

`.github/workflows/live_predictions.yml` runs every six hours and on manual dispatch. It refreshes data where available, snapshots only genuine future fixtures, scores older predictions and makes no commit when files are unchanged. Scheduled commits use `[skip ci]`, and the scheduled workflow itself is not triggered by pushes, preventing loops.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py run
```

Open <http://127.0.0.1:5000>.

Validation commands:

```bash
pytest -q
python manage.py smoke_test
python manage.py backtest
```

Local maintenance can explicitly run `python manage.py refresh`; this is a CLI action, not a public endpoint.

## API

```text
GET /api/predict?home=Arsenal&away=Chelsea
GET /api/predict?home=Arsenal&away=Chelsea&model=v1
GET /api/fixtures?limit=20
GET /api/model
GET /health
```

## Docker and Render

```bash
docker build -t epl-predictor .
docker run --rm -p 8000:8000 -e PORT=8000 epl-predictor
```

`render.yaml` preserves the Docker deployment, Gunicorn startup and `/health` check. It configures a 12-hour freshness threshold and five-minute request check interval. No paid service is required.

## Limitations

- Football-Data CSVs do not always contain future fixtures; snapshots safely do nothing in that case.
- EPL-only congestion excludes cups and European competitions.
- Real xG coverage remains sparse and source-dependent; xG is never fabricated.
- Injuries, line-ups, transfers and tactical changes are not modelled.
- Process-level background coordination assumes the configured single Gunicorn worker.
- GitHub repository commits are the durable free-tier store; uncommitted Render runtime files are ephemeral.
- Live metrics are noisy at small sample sizes and are never combined with historical backtest metrics.
- Backtest or live performance does not guarantee future performance.

This is a probabilistic model. A 54% home-win probability means that outcome is estimated to occur about 54 times in 100 comparable situations—not that the home team will definitely win.
