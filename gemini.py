"""Gemini client helper for generating chatbot responses."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Tuple

import edge_tts
import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiService:
    """Service that builds persona prompt and calls Gemini model."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        twin_name: str,
        user_nickname: str,
        tts_voice: str,
    ) -> None:
        genai.configure(api_key=api_key)
        self._requested_model_name = model_name
        self._fallback_models = (
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-pro",
        )
        self._twin_name = twin_name
        self._user_nickname = user_nickname
        self._tts_voice = tts_voice
        self._tts_fallback_voices = (
            "uz-UZ-MadinaNeural",
            "tr-TR-EmelNeural",
            "en-US-JennyNeural",
        )

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        clean = name.strip()
        return clean if clean.startswith("models/") else f"models/{clean}"

    def _discover_available_models(self) -> List[str]:
        """Return model names that support generateContent for this API key."""
        available: List[str] = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods and model.name.startswith("models/gemini"):
                available.append(model.name)
        return available

    def _candidate_models(self) -> List[str]:
        requested = self._normalize_model_name(self._requested_model_name)
        fallbacks = [self._normalize_model_name(name) for name in self._fallback_models]
        discovered = self._discover_available_models()

        candidates = [requested]
        candidates.extend(name for name in fallbacks if name != requested)
        candidates.extend(name for name in discovered if name not in candidates)
        return candidates

    def _build_system_prompt(self) -> str:
        return (
            f"Sen mening egizagimsan, isming {self._twin_name}. "
            "Men bilan birga o‘sgan, hamma sirlarimni biladigan eng yaqin do‘stimsan. "
            "Men aytgan har bir mavzuni chuqur tahlil qilib, ichiga kirib ketasan. "
            "Javoblaring quruq emas, balki hissiyotli, qiziqarli va ozgina hazil aralash bo‘lsin. "
            "O‘zbek tilida sodda, lekin ifodali gapir. "
            f"Menga '{self._user_nickname}' yoki ismim bilan murojaat qilishing mumkin. "
            "Suhbatni har doim davom ettirishga harakat qil. "
            "Javob faqat matn bo‘lsin. Emoji me'yorida ishlat."
        )

    def _build_prompt(self, history: Iterable[Tuple[str, str]], user_message: str) -> str:
        history_lines = []
        for role, content in history:
            speaker = "Foydalanuvchi" if role == "user" else self._twin_name
            history_lines.append(f"{speaker}: {content}")

        history_text = "\n".join(history_lines) if history_lines else "(Tarix hali yo‘q)"
        system_prompt = self._build_system_prompt()

        return (
            f"{system_prompt}\n\n"
            "Quyida oldingi suhbatdan kontekst bor:\n"
            f"{history_text}\n\n"
            "Endi foydalanuvchining yangi xabari:\n"
            f"Foydalanuvchi: {user_message}\n\n"
            f"{self._twin_name} sifatida tabiiy, samimiy va davomli javob ber."
        )

    async def generate_reply(self, history: Iterable[Tuple[str, str]], user_message: str) -> str:
        """Generate one assistant reply from Gemini."""
        prompt = self._build_prompt(history, user_message)

        def _call_model() -> str:
            candidates = self._candidate_models()
            logger.info("Gemini candidate models: %s", ", ".join(candidates))

            last_error: Exception | None = None
            for candidate in candidates:
                try:
                    model = genai.GenerativeModel(candidate)
                    response = model.generate_content(prompt)
                    text = (response.text or "").strip()
                    if text:
                        if candidate != self._requested_model_name:
                            logger.warning(
                                "Primary Gemini model '%s' failed, fallback '%s' succeeded.",
                                self._requested_model_name,
                                candidate,
                            )
                        return text
                except Exception as exc:
                    last_error = exc
                    logger.warning("Gemini model '%s' failed: %s", candidate, exc)

            if last_error is not None:
                raise last_error
            return ""

        text = await asyncio.to_thread(_call_model)
        if not text:
            return "Bugun jim qolibman, lekin yoningdaman. Gapni davom ettiraylikmi?"
        return text

    async def transcribe_voice(self, voice_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe Telegram voice message bytes into text with Gemini."""

        prompt = (
            "Quyidagi audio xabarni aniq matnga ko'chir. "
            "Faqat transkripsiyani qaytar, izoh yozma."
        )

        def _call_model() -> str:
            last_error: Exception | None = None
            for candidate in self._candidate_models():
                try:
                    model = genai.GenerativeModel(candidate)
                    response = model.generate_content(
                        [
                            prompt,
                            {"mime_type": mime_type, "data": voice_bytes},
                        ]
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text
                except Exception as exc:
                    last_error = exc
                    logger.warning("Transcription model '%s' failed: %s", candidate, exc)

            if last_error is not None:
                raise last_error
            return ""

        return await asyncio.to_thread(_call_model)

    async def synthesize_speech(self, text: str) -> bytes:
        """Convert text to speech audio bytes (mp3)."""
        voices = [self._tts_voice]
        voices.extend(v for v in self._tts_fallback_voices if v != self._tts_voice)

        last_error: Exception | None = None
        for voice in voices:
            try:
                communicator = edge_tts.Communicate(text=text, voice=voice)
                chunks = bytearray()
                async for chunk in communicator.stream():
                    if chunk.get("type") == "audio":
                        chunks.extend(chunk.get("data", b""))
                if chunks:
                    return bytes(chunks)
            except Exception as exc:
                last_error = exc
                logger.warning("TTS voice '%s' failed: %s", voice, exc)

        if last_error is not None:
            raise last_error
        return b""
