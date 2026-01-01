"""CLI commands for Pixie SDK."""

import dotenv


def main():
    """Start the Pixie server."""
    dotenv.load_dotenv()

    from pixie.server import start_server

    start_server(reload=True)


if __name__ == "__main__":
    main()
