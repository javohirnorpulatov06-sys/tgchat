"""Telegram twin chatbot entrypoint."""

from __future__ import annotations

import io
import logging

from telegram import Update
from telegram.constants import ChatAction
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
        "Yurakdan gaplashamizmi? Menga matn yoki ovoz yuboring, men doim yoningizdaman."
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

    await _process_and_reply(update=update, context=context, user_id=user_id, user_text=user_text)


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user or not update.message.voice:
        return

    gemini_service: GeminiService = context.application.bot_data["gemini"]
    voice = update.message.voice
    user_id = int(update.effective_user.id)

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        telegram_file = await context.bot.get_file(voice.file_id)
        voice_bytes = bytes(await telegram_file.download_as_bytearray())
        transcription = (await gemini_service.transcribe_voice(voice_bytes=voice_bytes, mime_type="audio/ogg")).strip()
        if not transcription:
            await update.message.reply_text("Ovozni to'liq tushunolmadim, iltimos qayta yuboring.")
            return

        await _process_and_reply(
            update=update,
            context=context,
            user_id=user_id,
            user_text=transcription,
            include_voice=True,
        )
    except Exception:
        logger.exception("Failed while processing voice message")
        await update.message.reply_text("Kechirasiz, hozir javob bera olmadim")


async def _process_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    user_text: str,
    include_voice: bool = False,
) -> None:
    db: Database = context.application.bot_data["db"]
    gemini_service: GeminiService = context.application.bot_data["gemini"]

    if not update.message:
        return

    try:
        await db.add_message(user_id=user_id, role="user", content=user_text)
        history = await db.get_recent_messages(user_id=user_id, limit=20)
        reply = await gemini_service.generate_reply(history=history, user_message=user_text)
        await db.add_message(user_id=user_id, role="assistant", content=reply)

        await update.message.reply_text(reply)

        if include_voice:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
            audio_bytes = await gemini_service.synthesize_speech(reply)
            if audio_bytes:
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "hamroh-voice.mp3"
                await update.message.reply_audio(audio=audio_file, title="Hamroh ovozi")
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
        tts_voice=settings.tts_voice,
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
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
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
