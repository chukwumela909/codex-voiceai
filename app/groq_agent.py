import json
from collections.abc import AsyncIterator

import httpx

from app.speech_tags import ALLOWED_TAGS, _TAG_PATTERN


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

    in_tag, paired = _scan_tag_depth(buffer)

    for index, char in enumerate(buffer):
        if char in punctuation and index + 1 - cursor >= min_punctuated_chars:
            if not _safe_split_after(in_tag, paired, index):
                continue
            chunks.append(buffer[cursor : index + 1].strip() + " ")
            cursor = index + 1

    remainder = buffer[cursor:]
    if force and remainder.strip():
        chunks.append(remainder.strip())
        remainder = ""
    elif len(remainder) >= max_unpunctuated_chars:
        window = min(max_unpunctuated_chars, len(remainder))
        split_at = -1
        for offset in range(window):
            if remainder[offset] == " " and _safe_split_after(in_tag, paired, cursor + offset):
                split_at = offset
        if split_at > 32:
            chunks.append(remainder[:split_at].strip() + " ")
            remainder = remainder[split_at:]

    return chunks, remainder


def _scan_tag_depth(text: str) -> tuple[list[bool], list[int]]:
    """For each char index i, capture (in_tag_after_i, paired_depth_after_i).

    A position is safe to split after iff in_tag_after_i is False and
    paired_depth_after_i is 0 — i.e., we are not inside a tag's angle
    brackets and not between an open paired tag and its close.
    """
    n = len(text)
    in_tag_arr = [False] * n
    paired_arr = [0] * n

    cur_in_tag = False
    cur_paired = 0
    tag_buf: list[str] = []

    for i, ch in enumerate(text):
        if not cur_in_tag and ch == "<":
            cur_in_tag = True
            tag_buf = ["<"]
        elif cur_in_tag:
            tag_buf.append(ch)
            if ch == ">":
                cur_in_tag = False
                match = _TAG_PATTERN.fullmatch("".join(tag_buf))
                tag_buf = []
                if match:
                    name = match.group("name").lower()
                    spec = ALLOWED_TAGS.get(name)
                    if spec and not spec.void and not match.group("self"):
                        if match.group("slash"):
                            cur_paired = max(0, cur_paired - 1)
                        else:
                            cur_paired += 1
        in_tag_arr[i] = cur_in_tag
        paired_arr[i] = cur_paired

    return in_tag_arr, paired_arr


def _safe_split_after(in_tag: list[bool], paired: list[int], index: int) -> bool:
    return not in_tag[index] and paired[index] == 0
