from __future__ import annotations

import argparse
from pathlib import Path

from app import create_app
from app.backtest import run_walk_forward_backtest, write_backtest_report


def main() -> None:
    parser = argparse.ArgumentParser(description="EPL Predictor maintenance commands")
    parser.add_argument("command", choices=["refresh", "backtest", "run"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()

    app = create_app({"AUTO_REFRESH": args.command == "run"})
    if args.command == "refresh":
        summary = app.extensions["match_repository"].refresh()
        print(
            f"Loaded {summary['matches']} completed matches from "
            f"{len(summary['downloaded_seasons'])} seasons."
        )
    elif args.command == "backtest":
        report = run_walk_forward_backtest(app.extensions["match_repository"].matches)
        json_path, csv_path = write_backtest_report(report, Path("reports"))
        print(f"Wrote {json_path} and {csv_path} for {report['sample_size']} matches.")
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
