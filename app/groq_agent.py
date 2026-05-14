import json
from collections.abc import AsyncIterator

import httpx


class GroqStreamingAgent:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        persona: str,
        temperature: float = 0.7,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.persona = persona
        self.temperature = temperature

    async def stream_response(self, transcript: list[dict[str, str]]) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": self.persona}, *transcript]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = parse_groq_stream_line(line)
                    if delta:
                        yield delta


def parse_groq_stream_line(line: str) -> str | None:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None

    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return None

    parsed = json.loads(data)
    choices = parsed.get("choices") or []
    if not choices:
        return None

    return choices[0].get("delta", {}).get("content") or None


def pop_speakable_chunks(buffer: str, *, force: bool = False) -> tuple[list[str], str]:
    chunks: list[str] = []
    cursor = 0
    punctuation = ".!?\n"
    min_punctuated_chars = 18
    max_unpunctuated_chars = 70

    for index, char in enumerate(buffer):
        if char in punctuation and index + 1 - cursor >= min_punctuated_chars:
            chunks.append(buffer[cursor : index + 1].strip() + " ")
            cursor = index + 1

    remainder = buffer[cursor:]
    if force and remainder.strip():
        chunks.append(remainder.strip())
        remainder = ""
    elif len(remainder) >= max_unpunctuated_chars:
        split_at = remainder.rfind(" ", 0, max_unpunctuated_chars)
        if split_at > 32:
            chunks.append(remainder[:split_at].strip() + " ")
            remainder = remainder[split_at:]

    return chunks, remainder
