"""
LLM-first grading with deterministic fallback support.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv(override=False)
logger = logging.getLogger("godel_env.grader")


def _extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


class AgentGrader:
    """API-backed grader that gracefully falls back to local verification."""

    _provider_disabled_global = False

    def __init__(self, max_concurrent: int = 10, timeout: int = 30) -> None:
        self.api_key = (
            os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = os.getenv("API_BASE_URL")
        if self.api_key and not self.base_url and os.getenv("HF_TOKEN"):
            self.base_url = "https://router.huggingface.co/v1"
        default_model = (
            "Qwen/Qwen2.5-7B-Instruct" if self.base_url else "gpt-4o-mini"
        )
        self.model_name = os.getenv("MODEL_NAME", default_model)
        self.grading_mode = os.getenv("GODEL_GRADING_MODE", "auto").strip().lower()
        self._disabled = False

        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
            )
        else:
            self.client = None

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.last_error: str | None = None

    def is_available(self) -> bool:
        return self.client is not None

    async def safe_grade(
        self,
        *,
        task_prompt: str,
        current_solution: str,
        rubrics: Dict[str, str],
    ) -> tuple[float, Dict[str, float], Dict[str, str]] | None:
        """Return an LLM grade when available, otherwise None for fallback."""
        self.last_error = None

        if self.grading_mode == "deterministic":
            self.last_error = "grading mode is deterministic"
            return None

        if self._disabled or AgentGrader._provider_disabled_global:
            self.last_error = "API grader disabled after previous provider failure"
            return None

        if not self.client:
            self.last_error = "no API client configured"
            return None

        rubric_description = "\n".join(
            f"- {name}: {description}" for name, description in rubrics.items()
        )
        system_prompt = (
            "You are grading a candidate solution inside GodelEnv.\n"
            "Score each rubric from 0.0 to 1.0.\n"
            "Return only JSON with keys `scores` and `feedback`.\n\n"
            f"RUBRICS:\n{rubric_description}"
        )
        user_prompt = (
            f"TASK PROMPT:\n{task_prompt}\n\n"
            f"CANDIDATE SOLUTION:\n{current_solution}"
        )

        try:
            async with self.semaphore:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=1200,
                    ),
                    timeout=self.timeout,
                )

            content = response.choices[0].message.content or "{}"
            blob = _extract_json_blob(content)
            payload = json.loads(blob or "{}")

            raw_scores = payload.get("scores", {})
            raw_feedback = payload.get("feedback", {})
            if not isinstance(raw_scores, dict):
                raise ValueError("grader response did not include a scores object")
            if not any(name in raw_scores for name in rubrics):
                raise ValueError("grader response did not include any known rubric scores")
            if not isinstance(raw_feedback, dict):
                raw_feedback = {}

            scores = {
                name: max(0.0, min(1.0, float(raw_scores.get(name, 0.0))))
                for name in rubrics
            }
            feedback = {
                name: str(raw_feedback.get(name, ""))
                for name in rubrics
            }
            total = sum(scores.values()) / len(scores) if scores else 0.0
            return total, scores, feedback
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if "insufficient_quota" in str(exc) or "rate limit" in str(exc).lower():
                # Avoid spamming a failing paid endpoint across every rubric call.
                self._disabled = True
                AgentGrader._provider_disabled_global = True
            logger.warning(
                "LLM grading failed, using deterministic fallback: %s",
                self.last_error,
            )
            return None
