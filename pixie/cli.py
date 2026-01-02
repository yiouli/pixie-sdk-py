"""CLI commands for Pixie SDK."""

import os
import dotenv


def main():
    """Start the Pixie server."""
    dotenv.load_dotenv(os.getcwd() + "/.env")

    from pixie.server import start_server

    start_server(reload=True)


if __name__ == "__main__":
    main()
