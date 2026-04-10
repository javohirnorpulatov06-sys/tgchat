"""Application configuration helpers.

This module reads all required environment variables for the bot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    telegram_token: str
    gemini_api_key: str
    database_url: str
    twin_name: str
    user_nickname: str
    gemini_model: str


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Environment variable is required: {name}")
    return value


def load_settings() -> Settings:
    """Load and validate all settings."""
    return Settings(
        telegram_token=_require_env("TELEGRAM_TOKEN"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        database_url=_require_env("DATABASE_URL"),
        twin_name=os.getenv("BOT_TWIN_NAME", "Hamroh").strip() or "Hamroh",
        user_nickname=os.getenv("USER_NICKNAME", "aka").strip() or "aka",
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro").strip()
        or "gemini-1.5-pro",
    )
