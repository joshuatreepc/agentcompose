"""Shared pytest configuration.

Loads the project's ``.env`` file once at test collection time so integration
tests (which need real API keys) can read them via ``os.environ`` without the
user having to export vars manually or pass ``--env-file`` to pytest.
"""

from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)


class MockAgent:
    def __init__(self):
        pass