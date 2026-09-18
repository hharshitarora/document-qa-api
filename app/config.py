"""Configuration. Reads .env, keeps every tunable value in one place.

Standard library only on purpose: nothing is installed yet, and config should
work before the rest of the stack exists.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overwriting real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value.isdigit() else default


class Settings:
    def __init__(self) -> None:
        _load_env_file(ENV_FILE)

        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        # Zania's brief restricts the chat model to gpt-4o-mini.
        self.chat_model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

        # Limits. Modest on purpose: every one of them bounds either memory, spend or
        # latency, and all are overridable by environment variable.
        self.max_upload_bytes = _int_env("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
        self.max_questions = _int_env("MAX_QUESTIONS", 50)
        self.max_concurrent_questions = _int_env("MAX_CONCURRENT_QUESTIONS", 5)
        self.request_timeout_seconds = _int_env("REQUEST_TIMEOUT_SECONDS", 30)
        self.cache_documents = _int_env("CACHE_DOCUMENTS", 8)

    def __repr__(self) -> str:
        # Never let the key reach a log line or a traceback.
        return f"Settings(chat_model={self.chat_model!r}, embedding_model={self.embedding_model!r})"


settings = Settings()
