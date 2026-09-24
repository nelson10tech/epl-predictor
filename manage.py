from __future__ import annotations

import argparse
import json

from app import create_app
from app.backtest import (
    run_smoke_backtest,
    run_walk_forward_backtest,
    write_backtest_report,
)
from app.backtest_v3 import (
    run_v3_backtest,
    run_v3_smoke_backtest,
    write_v3_backtest_report,
)
from app.live import score_predictions, snapshot_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="EPL Predictor maintenance commands")
    parser.add_argument(
        "command",
        choices=[
            "refresh",
            "backtest",
            "backtest_v3",
            "run",
            "snapshot_predictions",
            "score_predictions",
            "smoke_test",
        ],
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()

    app = create_app({"AUTO_REFRESH": args.command == "run"})
    runtime = app.extensions["prediction_runtime"]
    state = runtime.snapshot()
    reports_dir = app.extensions["reports_dir"]
    if args.command == "refresh":
        summary = state.repository.refresh()
        print(
            f"Loaded {summary['matches']} completed matches from "
            f"{len(summary['downloaded_seasons'])} seasons."
        )
    elif args.command == "backtest":
        report = run_walk_forward_backtest(state.repository.matches)
        json_path, csv_path = write_backtest_report(report, reports_dir)
        print(f"Wrote {json_path} and {csv_path} for {report['sample_size']} matches.")
    elif args.command == "backtest_v3":
        report = run_v3_backtest(
            state.repository.matches,
            v2_report=app.extensions.get("backtest_v2_report", {}),
        )
        json_path, csv_path = write_v3_backtest_report(report, reports_dir)
        print(f"Wrote {json_path} and {csv_path} for {report['sample_size']} matches.")
    elif args.command == "snapshot_predictions":
        provider = app.extensions["fixture_provider"]
        fixtures = provider.fetch_upcoming(
            known_teams=set(state.repository.active_teams())
        )
        if fixtures:
            state.repository.fixtures = fixtures
        summary = snapshot_predictions(
            state.repository,
            {
                key: state.predictors[key]
                for key in ("v2", "v3")
                if key in state.predictors
            },
            reports_dir,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "score_predictions":
        summary = score_predictions(
            state.repository,
            reports_dir,
            historical_report=(
                app.extensions.get("backtest_v3_report")
                or app.extensions.get("backtest_report", {})
            ),
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "smoke_test":
        for key, predictor in state.predictors.items():
            prediction = predictor.predict("Arsenal", "Chelsea")
            probability_sum = sum(prediction["probabilities"].values())
            if abs(probability_sum - 1.0) >= 1e-9:
                raise RuntimeError(f"{key} probabilities are not normalized")
            if min(prediction["expected_goals"].values()) < 0:
                raise RuntimeError(f"{key} returned negative expected goals")
        smoke = run_smoke_backtest(state.repository.matches)
        if not smoke["probabilities_normalized"] or not smoke["leakage_check_passed"]:
            raise RuntimeError("Chronological smoke backtest failed")
        v3_smoke = run_v3_smoke_backtest(state.repository.matches)
        if (
            not v3_smoke["probabilities_normalized"]
            or not v3_smoke["leakage_check_passed"]
        ):
            raise RuntimeError("V3 chronological smoke backtest failed")
        smoke["v3"] = v3_smoke
        print(json.dumps(smoke, indent=2))
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
