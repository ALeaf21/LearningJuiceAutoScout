import argparse

from autoscout.dashboard_server import run_dashboard_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the local AutoScout dashboard server.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765,
                        help="Bind port. Default: 8765")
    args = parser.parse_args()
    run_dashboard_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
