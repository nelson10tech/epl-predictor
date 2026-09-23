# EPL Predictor V2

A mobile-first Flask application that produces explainable English Premier League 1X2 probabilities. The live V2 model combines Elo, time-decayed team form, venue-specific attack/defence, rest and EPL fixture congestion with a Dixon–Coles low-score correction and chronological probability calibration.

The output includes:

- home win / draw / away win probabilities
- expected goals for each team
- the most likely score and top five scorelines
- contributing model inputs (not causal explanations)
- separate upcoming-fixture and completed-result sections
- model diagnostics and walk-forward backtest metrics

This project does **not** provide betting recommendations, picks, stakes or guarantees.

## V1 versus V2

| Capability | V1 baseline | V2 live model |
|---|---|---|
| Rating | Elo | Elo |
| Goals | Basic independent Poisson | Dixon–Coles-adjusted Poisson |
| Team strength | Time-decayed home/away goals | Venue attack/defence with regression to league averages |
| Form | Aggregate weighted goals | Recent 5 and 10 GF/GA, shots and shots on target |
| Context | Fixed home advantage | Rest days and EPL-only 7/14-day congestion |
| xG | No | Real HxG/AxG only when the source exposes it |
| Calibration | No | Chronological temperature scaling |
| Evaluation | None | Walk-forward by match date |

V1 remains available through `?model=v1` for comparison and regression testing.

## Data and leakage prevention

The baseline source is [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). The committed six-season snapshot contains results plus fields actually present in each CSV:

- date, time, teams, full-time goals and result
- shots and shots on target
- `HxG` / `AxG` when present
- decimal 1X2 odds for the optional benchmark

The loader detects columns safely; it never assumes optional fields exist. Bookmaker odds are normalized to remove overround and are used only as a benchmark—not as the live model and never for recommendations.

Every backtest prediction is generated from state ending before that match date. Matches on the same calendar date are predicted as a group and only then added to model state. The feature pipeline does not use future Elo, later matches, final standings or post-match values from the target fixture. Ratings are not hard-reset between seasons; promoted or low-sample teams regress strongly toward league averages.

## Optional xG support

`app/xg.py` defines the `XGProvider` interface:

```text
load(matches)
available()
get_match_xg(match)
metadata()
```

The bundled provider reads only real `HxG`/`AxG` columns published in Football-Data.co.uk rows. In the current snapshot, 50 matches contain these fields. Earlier seasons do not, so V2 falls back to goals, shots and shots on target. The application never fabricates xG or labels an estimate as real xG. `/api/predict`, `/api/model` and the diagnostics page expose whether xG is available.

## Walk-forward backtest

Run:

```bash
python manage.py backtest
```

The process uses 2023/24 as the first out-of-sample calibration season, then evaluates 2024/25, 2025/26 and available 2026/27 matches in order. Calibration is fit only on earlier predictions and updated between seasons. It never uses a random split.

Generated reports:

- `reports/backtest_v2.json`
- `reports/backtest_v2.csv`

Current report, generated from 810 actual evaluation matches through 2026-09-20:

| Model | Log Loss ↓ | Brier ↓ | RPS ↓ | Accuracy |
|---|---:|---:|---:|---:|
| V1 Elo + Poisson | 1.0232 | 0.2044 | 0.2088 | 50.25% |
| V2 Dixon–Coles | **1.0082** | **0.2014** | **0.2056** | 49.75% |
| Normalized bookmaker benchmark | 0.9982 | 0.1992 | 0.2018 | 50.99% |

V2 improved the primary metrics (Log Loss, Brier and RPS) over V1 on this backtest. It did **not** improve accuracy, and it did not beat the bookmaker benchmark overall. Accuracy is secondary because it ignores probability quality. Calibration bins with predicted probability, observed frequency and sample count are included in the JSON report.

## Automatic data freshness

Normal production refresh is automatic; there is no public refresh endpoint and no refresh token.

At application startup:

1. the committed/cached dataset is loaded first;
2. its last successful refresh timestamp is checked;
3. if older than 12 hours, one process-level guarded refresh is attempted;
4. after success, data is reloaded before live models are built;
5. after failure, startup continues with the last valid snapshot and reports `data_stale: true`.

The check does not run on page requests, does not retry in a loop and uses a process lock to prevent duplicate thread-triggered startup work. The committed snapshot is the Render cold-start fallback. Freshness fields are exposed by `/health`, `/api/model` and prediction metadata:

```json
{
  "data_through": "2026-09-20",
  "last_refresh_attempt": "...",
  "last_refresh_success": "...",
  "data_stale": false,
  "refresh_source": "Football-Data.co.uk"
}
```

For local maintenance only, `python manage.py refresh` remains available. It is not exposed as an HTTP route.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py run
```

Open <http://127.0.0.1:5000>.

Run tests and regenerate the backtest:

```bash
pytest -q
python manage.py backtest
```

## API

```text
GET /api/predict?home=Arsenal&away=Chelsea
GET /api/predict?home=Arsenal&away=Chelsea&model=v1
GET /api/fixtures?limit=20
GET /api/model
GET /health
```

`POST /api/refresh` intentionally does not exist.

## Docker

```bash
docker build -t epl-predictor .
docker run --rm -p 8000:8000 -e PORT=8000 epl-predictor
```

Gunicorn uses one worker with four threads so a successful startup refresh and rebuilt model are shared consistently without extra memory-heavy processes.

## Deploy to Render

The existing Docker/Render deployment remains intact:

1. Push or merge the repository to GitHub.
2. In Render, choose **New → Blueprint**.
3. Connect `nelson10tech/epl-predictor` and approve `render.yaml`.
4. Wait for the Docker build and `/health` check to pass.

`DATA_MAX_AGE_HOURS=12` is configured in `render.yaml`. No paid database, background worker or external API key is required. Render filesystems may be ephemeral, so any downloaded update can disappear after a restart; the committed snapshot remains the reliable fallback.

## Limitations

- EPL-only congestion excludes cups and European competitions because no reliable bundled schedule is available.
- Real xG coverage is sparse and source-dependent.
- Injuries, line-ups, transfers and tactical changes are not modelled.
- The future-fixtures section stays unavailable when the source has no reliable dated fixtures; dates are never fabricated.
- Temperature calibration is deliberately simple and may drift.
- An ML model is intentionally not bundled in V2: the explainable statistical models keep Render startup fast and dependency size small. It is a candidate for V3 after a strictly chronological feature-store design is proven.
- Backtest performance does not guarantee future performance.

This is a probabilistic model. A 54% home-win probability means that outcome is estimated to occur about 54 times in 100 comparable situations—not that the home team will definitely win.
