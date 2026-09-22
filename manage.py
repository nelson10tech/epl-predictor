from __future__ import annotations

import argparse

from app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="EPL Predictor maintenance commands")
    parser.add_argument("command", choices=["refresh", "run"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()

    app = create_app()
    if args.command == "refresh":
        summary = app.extensions["match_repository"].refresh()
        print(
            f"Loaded {summary['matches']} completed matches from "
            f"{len(summary['downloaded_seasons'])} seasons."
        )
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

