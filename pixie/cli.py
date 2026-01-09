"""CLI commands for Pixie SDK."""

import os
import dotenv

from pixie.server import start_server


def main():
    """Start the Pixie server."""
    dotenv.load_dotenv(os.getcwd() + "/.env")
    port = int(os.getenv("PIXIE_SDK_PORT", "8000"))

    start_server(port=port, reload=True)


if __name__ == "__main__":
    main()
