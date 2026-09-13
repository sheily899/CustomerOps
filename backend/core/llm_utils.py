"""LLM response helpers shared by Anthropic-compatible providers."""
import json
from typing import Any, Iterable, List


def extract_text_content(content: Iterable[Any]) -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return "\n".join(t for t in texts if t)


def parse_json_text(raw: str) -> Any:
    """Parse JSON text after removing an optional Markdown code fence.

    JSON Output-capable models can still wrap their JSON in a ``json`` fence.
    We remove only that wrapper and intentionally do not search arbitrary text
    for a JSON-looking substring.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM JSON 响应为空")

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text:
        raise ValueError("LLM JSON 响应去除代码围栏后为空")
    return json.loads(text)
