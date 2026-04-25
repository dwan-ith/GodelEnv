"""
Parse JSON from LLM outputs (often malformed: markdown fences, trailing commas, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("godel_env.llm_json")


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

    last_error: str | None = None
    for i, candidate in enumerate(candidates):
        try:
            out = json.loads(candidate)
            if isinstance(out, dict):
                logger.debug("Parsed JSON successfully on candidate %d", i)
                return out
        except json.JSONDecodeError as e:
            last_error = str(e)
            logger.debug("Strict JSON failed on candidate %d: %s", i, e)
            try:
                from json_repair import repair_json

                fixed = repair_json(candidate)
                out = json.loads(fixed)
                if isinstance(out, dict):
                    logger.debug("json-repair fixed candidate %d", i)
                    return out
            except ImportError:
                logger.warning(
                    "json-repair not installed; install with: pip install json-repair"
                )
            except Exception as repair_exc:
                logger.debug("json-repair failed on candidate %d: %s", i, repair_exc)
                continue

    # Log the raw text for debugging
    logger.error(
        "Failed to parse LLM JSON. Raw text (first 500 chars): %s",
        raw[:500] if len(raw) > 500 else raw,
    )
    raise json.JSONDecodeError(
        f"Could not parse LLM output as JSON object: {last_error or 'unknown'}",
        raw[:200],
        0,
    )
