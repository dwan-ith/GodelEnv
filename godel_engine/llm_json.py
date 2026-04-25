"""
Parse JSON from LLM outputs (often malformed: markdown fences, trailing commas, etc.).
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def _strip_fenced_code(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "```" in t:
        t = t.replace("```json", "").replace("```", "")
    return t.strip()


def parse_llm_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output. Tries strict JSON, then json-repair if installed.
    """
    raw = (text or "").strip()
    if not raw:
        return {}

    candidates: list[str] = []
    for c in (_strip_fenced_code(raw), extract_json_blob(raw) or raw, raw):
        c = c.strip() if c else ""
        if c and c not in candidates:
            candidates.append(c)

    for candidate in candidates:
        try:
            out = json.loads(candidate)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                fixed = repair_json(candidate)
                out = json.loads(fixed)
                if isinstance(out, dict):
                    return out
            except Exception:
                continue

    raise json.JSONDecodeError("Could not parse LLM output as JSON object", raw, 0)
