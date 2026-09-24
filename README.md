# EPL Predictor V3

A mobile-first Flask application for explainable English Premier League 1X2 probabilities. Application V3 adds a chronological feature store, a lightweight machine-learning challenger, a validation-weighted ensemble, genuine historical xG coverage and reliable future-fixture ingestion without replacing the working V2 live model.

The app returns home/draw/away probabilities, expected goals, the most likely score and the top five scorelines. It does **not** provide betting picks, staking advice or guarantees.

## Promotion decision

V3 initially runs alongside V2. The live default remains **Enhanced Dixon–Coles V2, model 2.0**. V3 is available through `?model=v3` and is recorded in prospective shadow snapshots.

The fixed promotion rule requires the V3 ensemble to beat V2 on both final-test Log Loss and Brier score without a serious calibration regression. The generated report does not pass that rule:

| Model | Log Loss ↓ | Brier ↓ | RPS ↓ | Accuracy |
|---|---:|---:|---:|---:|
| V1 Elo + Poisson | 1.0232 | 0.2044 | 0.2088 | 50.25% |
| **V2 Dixon–Coles — live** | **1.0082** | **0.2014** | 0.2056 | 49.75% |
| V3 HistGradientBoosting ML | 1.0396 | 0.2080 | 0.2123 | 48.15% |
| V3 Ensemble — challenger | 1.0086 | 0.2014 | **0.2055** | 50.37% |
| Bookmaker benchmark | 0.9982 | 0.1992 | 0.2018 | 50.99% |

The validation-selected ensemble is 90% V2 and 10% calibrated ML. On the 810-match final evaluation, V3 was worse by `+0.00040` Log Loss and `+0.00004` Brier, despite a tiny RPS improvement. It therefore was **not promoted**.

## Versions and models

| Component | Version | Role |
|---|---:|---|
| Application | 3.0 | providers, ML challenger, ensemble, shadow evaluation |
| V1 | 1.0 | retained Elo + independent Poisson baseline |
| V2 | 2.0 | retained live Elo/form/Dixon–Coles model |
| V3 ML | 3.0-ml | HistGradientBoosting classifier |
| V3 Ensemble | 3.0 | validation-weighted V2 + calibrated ML challenger |

V2 expected-goals, Elo, form and Dixon–Coles logic is unchanged.
`config/model_registry.json` records the live/challenger assignment and the non-negotiable promotion rule.

## Data provenance

### Historical results

[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php) remains the historical EPL source. Optional columns are detected safely. Available fields include goals/results, shots, shots on target, genuine `HxG/AxG`, and decimal 1X2 odds.

Odds are used only as a margin-normalized benchmark, never as ordinary ML features.

### Upcoming fixtures

`OpenFootballFixtureProvider` reads the current EPL JSON from [openfootball/football.json](https://github.com/openfootball/football.json), whose data is public domain. It normalizes names to Football-Data names and accepts only feed-supplied dates and kickoff times. A committed cache supports temporary upstream failures.

Snapshot automation considers genuine fixtures within the next 14 days. If the provider and cache are unavailable, the command exits safely without inventing fixtures.

### Historical advanced statistics

`CompositeXGProvider` uses:

1. genuine `HxG/AxG` rows supplied by Football-Data; then
2. the final public [FiveThirtyEight Soccer SPI archive](https://github.com/fivethirtyeight/data/tree/master/soccer-spi), licensed CC BY 4.0 and cached from its 2023 Internet Archive capture.

The committed cache contains all 760 EPL matches from 2021/22 and 2022/23. Together with Football-Data rows, the current feature store has 810 genuine xG matches. Missing xG stays missing; expected goals produced by the model are never relabelled as source xG.

### Optional providers

Interfaces exist for all-competition schedules and player availability. They are deliberately disabled because no sufficiently complete, stable, structured and appropriately licensed free source is currently configured.

- congestion scope is reported as `EPL only`;
- cup and European fixtures are not fabricated;
- injury rumours are not treated as facts;
- no subjective player-quality scores are assigned.

Provenance and limitations are exposed at `GET /api/data`.

## Chronological feature store

`ChronologicalFeatureStore` walks matches in date order. It creates every same-day feature row before revealing any result from that date. Features include:

- home/away Elo and Elo difference;
- recent 5/10 goals for and against;
- venue attack and defence strengths with league-average shrinkage;
- shots and shots on target;
- genuine rolling xG/xGA, xG difference/trend and goals-minus-xG finishing difference;
- rest days and EPL matches in the previous 7/14 days;
- team/season sample counts and an early-season indicator.

It never uses final standings, future Elo, later injuries, target goals/result/xG, target-match post-match statistics or bookmaker closing odds as ML features. Team names are not encoded as arbitrary IDs.

## V3 ML and ensemble

The challenger uses scikit-learn `HistGradientBoostingClassifier` with a small regularized tree configuration suitable for Render free/basic resources. It outputs three-class probabilities. Temperature scaling is selected on the chronological validation season only; isotonic is skipped because the validation set is too small for a stable multiclass fit.

The ensemble searches V2 weights `0.0, 0.1, …, 1.0` using validation Log Loss. The selected weight is frozen before the 2024/25–2026/27 final evaluation. The final test is never used for calibration or weight selection.

Feature importance is permutation importance measured by Log Loss on a chronological holdout. It describes model use, not causation.

## Backtests

The existing V2 reports remain unchanged:

- `reports/backtest_v2.json`
- `reports/backtest_v2.csv`

V3 reports are generated by:

```bash
python manage.py backtest_v3
```

Outputs:

- `reports/backtest_v3.json`
- `reports/backtest_v3.csv`

Design:

1. train on 2021/22–2022/23;
2. use 2023/24 only for ML temperature and ensemble-weight selection;
3. freeze those choices;
4. evaluate 2024/25, 2025/26 and available 2026/27 matches;
5. at every matchday, train only on earlier rows, predict all same-day fixtures, then reveal that day's results.

The report contains Log Loss, Brier, RPS, accuracy, calibration error, calibration bins, per-season results and descriptive robustness slices for favourites, balanced matches, promoted-team proxy rows and early-season games. Small groups are flagged; no statistical-significance claim is made.

## Live prospective shadow evaluation

Historical backtests and live performance are never combined.

```bash
python manage.py snapshot_predictions
python manage.py score_predictions
```

Each scheduled snapshot fetches the fixture provider and records both V2 and V3 before kickoff. The fixture/model version is part of the immutable key. A 24-hour window prevents duplicate snapshots. Original probabilities are never changed when results arrive.

`reports/live_performance.json` reports V2 and V3 separately, including overall, last-20, last-50 and season-to-date Log Loss, Brier, RPS and accuracy. Fewer than 50 settled predictions shows a small-sample warning.

Render's runtime filesystem is ephemeral. Durable free-tier history is provided by `.github/workflows/live_predictions.yml`, which commits only changed cache/prediction/report files with the built-in `GITHUB_TOKEN`. If branch protection disallows workflow pushes, scheduled results will not persist; configure the repository to allow the workflow token or adopt a reviewed pull-request workflow.

## Automatic freshness

Football-Data history retains V2.1's stale-while-revalidate behavior:

- load the committed snapshot first;
- consider data stale after 12 hours;
- check at most once every five minutes on requests;
- serve the current immutable model immediately;
- run one guarded background refresh;
- build V1, V2, ML and ensemble away from live traffic;
- atomically swap state only after the full rebuild succeeds;
- keep the prior state and expose the error if refreshing fails.

There is no public `POST /api/refresh` endpoint and no refresh token. A local CLI refresh remains available for maintenance.

## API

```text
GET /api/predict?home=Arsenal&away=Chelsea          # live V2
GET /api/predict?home=Arsenal&away=Chelsea&model=v1
GET /api/predict?home=Arsenal&away=Chelsea&model=ml
GET /api/predict?home=Arsenal&away=Chelsea&model=v3 # challenger
GET /api/fixtures?limit=20
GET /api/model
GET /api/data
GET /health
```

Prediction metadata identifies `app_version`, `model_version`, freshness, xG status and whether the requested model is live or shadow.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py run
```

Open <http://127.0.0.1:5000>.

Validation:

```bash
pytest -q
python manage.py smoke_test
python -c "from wsgi import app; assert app is not None"
```

The full V3 walk-forward takes longer and is intentionally excluded from ordinary CI.

## GitHub Actions, Docker and Render

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests. It installs dependencies, runs pytest, performs V1/V2/V3 normalization and chronological-leakage smoke checks, and imports the production WSGI app.

```bash
docker build -t epl-predictor .
docker run --rm -p 8000:8000 -e PORT=8000 epl-predictor
```

`render.yaml` preserves one Gunicorn worker with four threads, Docker deployment and `/health`. V3 trains only once per complete runtime rebuild, never per page request. No GPU, API key, paid database or paid infrastructure is required.

## Limitations

- V3 did not beat V2 on the two primary probability metrics and remains a challenger.
- FiveThirtyEight xG ends in 2022/23; later genuine coverage is sparse.
- Congestion currently covers EPL matches only.
- Player availability and confirmed lineups are unavailable.
- openfootball is a community dataset; cached fixtures can become stale after schedule changes.
- Live conclusions remain unreliable at small sample sizes.
- Backtests and historical improvement do not guarantee future performance.

This model is probabilistic. A 54% home-win probability means roughly 54 wins in 100 comparable situations—not that the home team will definitely win.
