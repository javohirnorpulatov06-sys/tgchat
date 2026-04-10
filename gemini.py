"""Gemini client helper for generating chatbot responses."""

from __future__ import annotations

import asyncio
from typing import Iterable, Tuple

import google.generativeai as genai


class GeminiService:
    """Service that builds persona prompt and calls Gemini model."""

    def __init__(self, api_key: str, model_name: str, twin_name: str, user_nickname: str) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
        self._twin_name = twin_name
        self._user_nickname = user_nickname

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
            response = self._model.generate_content(prompt)
            text = (response.text or "").strip()
            return text

        text = await asyncio.to_thread(_call_model)
        if not text:
            return "Bugun jim qolibman, lekin yoningdaman. Gapni davom ettiraylikmi?"
        return text
