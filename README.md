# EPL Predictor

A small, mobile-first English Premier League predictor. The V1 model combines team Elo ratings with recency-weighted Poisson goal rates to estimate:

- home win / draw / away win probabilities (1X2)
- expected goals for each team
- the most likely scoreline
- recent completed EPL fixtures

The app does **not** provide betting recommendations.

## Data

Historical and current-season results come from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). A six-season snapshot is committed so the app starts without a network request. Only the match fields used by V1 are retained. The **更新資料** button or CLI refresh command downloads the latest `E0.csv` files and rebuilds the model in memory.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py run
```

Open <http://127.0.0.1:5000>.

Refresh the dataset from Football-Data.co.uk:

```bash
python manage.py refresh
```

Run tests:

```bash
pytest -q
```

## API

```text
GET  /api/predict?home=Arsenal&away=Chelsea
GET  /api/fixtures?limit=20
POST /api/refresh
GET  /health
```

The three values in `probabilities` are normalized to sum to `1.0` (subject only to floating-point precision).

## Docker

```bash
docker build -t epl-predictor .
docker run --rm -p 8000:8000 -e PORT=8000 epl-predictor
```

Open <http://127.0.0.1:8000>.

## Deploy to Render

This repository includes `render.yaml`, so the easiest deployment is a Render Blueprint:

1. Push or merge this repository to GitHub.
2. In Render, choose **New → Blueprint**.
3. Connect `nelson10tech/epl-predictor` and approve the `render.yaml` plan.
4. Wait for the Docker build and `/health` check to pass.

Render runs the included Docker image with Gunicorn. The bundled data snapshot makes cold starts reliable; clicking **更新資料** refreshes the running instance. Render's free filesystem is ephemeral, so refreshed CSV files can be lost after a restart while the committed snapshot remains available.

## Model notes

V1 is intentionally simple and explainable:

1. Elo ratings update after every completed match, including a fixed home advantage.
2. Home and away scoring/defending rates use exponential time decay and shrink toward league averages when samples are small.
3. The Elo gap adjusts the two teams' expected goals.
4. Independent Poisson score distributions produce the 1X2 probabilities and most likely score.

This baseline omits injuries, line-ups, rest, transfers and tactical context. Predictions are informational estimates, not guarantees.
