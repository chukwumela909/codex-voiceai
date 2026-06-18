"""Inline control-tag registry and sanitizer.

``sanitize`` strips any tag outside the caller-supplied allow-list (unknown tag,
unknown attribute, malformed structure, unbalanced paired tag) while preserving
the inner text so speech keeps playing. ElevenLabs has no tag support, so the
TTS path passes an empty allow-list to strip everything; the ``ALLOWED_TAGS``
registry below is kept as a reusable default for any caller that wants it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


EMOTION_LABELS: tuple[str, ...] = (
    "excited",
    "curious",
    "calm",
    "sad",
    "playful",
    "warm",
    "surprised",
    "thoughtful",
    "amused",
    "gentle",
    "serious",
    "reassuring",
)


@dataclass(frozen=True)
class TagSpec:
    void: bool
    attrs: frozenset[str] = field(default_factory=frozenset)
    enum_attrs: dict[str, frozenset[str]] = field(default_factory=dict)


ALLOWED_TAGS: dict[str, TagSpec] = {
    "break": TagSpec(void=True, attrs=frozenset({"time"})),
    "emotion": TagSpec(
        void=True,
        attrs=frozenset({"value"}),
        enum_attrs={"value": frozenset(EMOTION_LABELS)},
    ),
    "spell": TagSpec(void=False),
}


_TAG_PATTERN = re.compile(
    r"""<
        (?P<slash>/?)
        (?P<name>[A-Za-z][A-Za-z0-9_-]*)
        (?P<attrs>(?:\s+[A-Za-z_][A-Za-z0-9_-]*\s*=\s*"[^"<>]*")*)
        \s*
        (?P<self>/?)
        >""",
    re.VERBOSE,
)
_ATTR_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*"([^"<>]*)"')
_BREAK_TIME_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:ms|s)$")


@dataclass
class SanitizeResult:
    text: str
    stripped: int = 0


def sanitize(text: str, *, allowed_tags: dict[str, TagSpec] | None = None) -> SanitizeResult:
    """Strip any tag that is not in the allowed set or is malformed.

    Inner text of stripped paired tags is preserved. Stray <, >, & in plain
    text are escaped so the downstream renderer never sees a half-tag.
    """
    if not text:
        return SanitizeResult(text="", stripped=0)

    allowed = allowed_tags if allowed_tags is not None else ALLOWED_TAGS
    out: list[str] = []
    stripped = 0
    cursor = 0
    open_stack: list[tuple[str, int]] = []

    for match in _TAG_PATTERN.finditer(text):
        out.append(_escape_plain(text[cursor : match.start()]))
        cursor = match.end()

        name = match.group("name").lower()
        is_close = bool(match.group("slash"))
        is_self_closing = bool(match.group("self"))
        spec = allowed.get(name)

        if spec is None:
            stripped += 1
            continue

        if is_close:
            if spec.void:
                stripped += 1
                continue
            if open_stack and open_stack[-1][0] == name:
                open_stack.pop()
                out.append(f"</{name}>")
            else:
                stripped += 1
            continue

        attrs_text = match.group("attrs") or ""
        attrs = dict(_ATTR_PATTERN.findall(attrs_text))
        unknown_attrs = set(attrs) - set(spec.attrs)
        if unknown_attrs:
            stripped += 1
            continue
        if not _attrs_valid(name, attrs, spec):
            stripped += 1
            continue

        if spec.void:
            out.append(_render_tag(name, attrs, void=True))
        else:
            if is_self_closing:
                stripped += 1
                continue
            out.append(_render_tag(name, attrs, void=False))
            open_stack.append((name, len(out) - 1))

    out.append(_escape_plain(text[cursor:]))

    while open_stack:
        name, insert_idx = open_stack.pop()
        out[insert_idx] = ""
        stripped += 1

    return SanitizeResult(text="".join(out), stripped=stripped)


def _attrs_valid(name: str, attrs: dict[str, str], spec: TagSpec) -> bool:
    if name == "break":
        time_value = attrs.get("time", "")
        return bool(_BREAK_TIME_PATTERN.match(time_value))
    for attr, allowed_values in spec.enum_attrs.items():
        value = attrs.get(attr, "").strip().lower()
        if not value or value not in allowed_values:
            return False
    return True


def _render_tag(name: str, attrs: dict[str, str], *, void: bool) -> str:
    attr_text = "".join(f' {key}="{value}"' for key, value in attrs.items())
    suffix = "/>" if void else ">"
    return f"<{name}{attr_text}{suffix}"


def _escape_plain(text: str) -> str:
    # TTS treats input as plain text outside recognized tags, so stray
    # <, >, & must pass through verbatim — XML-escaping them would make
    # the model read out "ampersand l t semicolon".
    return text
