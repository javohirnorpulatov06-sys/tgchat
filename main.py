"""Telegram twin chatbot entrypoint."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import Settings, load_settings
from db import Database
from gemini import GeminiService


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    welcome_text = (
        f"Assalomu alaykum! Men {settings.twin_name}.\n"
        "Yurakdan gaplashamizmi? Menga oddiy yozing, men doim yoningizdaman."
    )
    if update.message:
        await update.message.reply_text(welcome_text)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return

    db: Database = context.application.bot_data["db"]
    gemini_service: GeminiService = context.application.bot_data["gemini"]

    user_id = int(update.effective_user.id)
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    try:
        await db.add_message(user_id=user_id, role="user", content=user_text)
        history = await db.get_recent_messages(user_id=user_id, limit=20)
        reply = await gemini_service.generate_reply(history=history, user_message=user_text)
        await db.add_message(user_id=user_id, role="assistant", content=reply)
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Failed while processing user message")
        await update.message.reply_text("Kechirasiz, hozir javob bera olmadim")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram error", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text("Kechirasiz, hozir javob bera olmadim")
        except Exception:
            logger.exception("Failed to send fallback error message")


async def post_init(application: Application) -> None:
    """Open shared resources once bot starts."""
    settings: Settings = application.bot_data["settings"]
    db = Database(settings.database_url)
    await db.connect()
    application.bot_data["db"] = db
    application.bot_data["gemini"] = GeminiService(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        twin_name=settings.twin_name,
        user_nickname=settings.user_nickname,
    )
    logger.info("Bot resources initialized")


async def post_shutdown(application: Application) -> None:
    """Close shared resources on shutdown."""
    db: Database | None = application.bot_data.get("db")
    if db:
        await db.close()
    logger.info("Bot resources closed")


def build_application(settings: Settings) -> Application:
    app = (
        Application.builder()
        .token(settings.telegram_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.bot_data["settings"] = settings
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)
    return app


def main() -> None:
    settings = load_settings()
    app = build_application(settings)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # PTB internally manages its own event loop inside run_polling.
    main()
