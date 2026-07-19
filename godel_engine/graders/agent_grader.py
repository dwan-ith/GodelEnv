"""
LLM-first grading with deterministic fallback support.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from godel_engine.llm_json import parse_llm_json_object
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    build_provider_client,
    load_provider_configs,
    provider_completion_kwargs,
)


load_dotenv(override=False)
logger = logging.getLogger("godel_env.grader")


class AgentGrader:
    """API-backed grader that gracefully falls back to local verification."""

    def __init__(self, max_concurrent: int = 10, timeout: int = 30) -> None:
        self.providers = load_provider_configs()
        self.grading_mode = (
            os.getenv("GODEL_GRADING_MODE", "auto").strip().lower()
        )

        self.clients: list[tuple[str, str, Any]] = []
        for provider in self.providers:
            if not provider.api_key:
                continue
            self.clients.append(
                (
                    provider.name,
                    provider.model_name,
                    build_provider_client(provider),
                )
            )

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = int(os.getenv("GODEL_GRADING_TIMEOUT", str(timeout)) or timeout)
        self.last_error: str | None = None
        self.last_source = "deterministic"
        self.last_provider: str | None = None

    def is_available(self) -> bool:
        return any(
            not ProviderCircuitBreaker.is_disabled(provider_name, scope="grader")
            for provider_name, _, _ in self.clients
        )

    async def safe_grade(
        self,
        *,
        task_prompt: str,
        current_solution: str,
        rubrics: Dict[str, str],
    ) -> tuple[float, Dict[str, float], Dict[str, str]] | None:
        """Return an LLM grade when available, otherwise None for fallback."""
        self.last_error = None
        self.last_source = "deterministic"
        self.last_provider = None

        if self.grading_mode == "deterministic":
            self.last_error = "grading mode is deterministic"
            return None

        if not self.clients:
            self.last_error = "no API client configured"
            return None

        if all(ProviderCircuitBreaker.is_disabled(provider_name, scope="grader") for provider_name, _, _ in self.clients):
            reason = ProviderCircuitBreaker.reason() or "provider disabled"
            self.last_error = f"API grader disabled after previous provider failure: {reason}"
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

        errors: list[str] = []
        for provider_name, model_name, client in self.clients:
            if ProviderCircuitBreaker.is_disabled(provider_name, scope="grader"):
                errors.append(
                    f"{provider_name}: disabled after previous failure ({ProviderCircuitBreaker.reason(provider_name, scope='grader')})"
                )
                continue

            try:
                async with self.semaphore:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=1200,
                            **(
                                {"response_format": {"type": "json_object"}}
                                if provider_name in {"openai", "ollama"}
                                else {}
                            ),
                            **provider_completion_kwargs(provider_name),
                        ),
                        timeout=self.timeout,
                    )

                choices = getattr(response, "choices", None) or []
                if not choices or getattr(choices[0], "message", None) is None:
                    raise ValueError("grader provider returned no chat choices")
                content = choices[0].message.content or ""
                payload = parse_llm_json_object(content if content.strip() else "{}")

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
                self.last_source = f"llm:{provider_name}"
                self.last_provider = provider_name
                self.last_error = None
                return total, scores, feedback
            except Exception as exc:
                message = ProviderCircuitBreaker.record_failure(
                    provider_name, exc, scope="grader"
                )
                errors.append(f"{provider_name}: {message}")
                logger.warning(
                    "Provider %s failed for grading; trying fallback: %s",
                    provider_name,
                    message,
                )

        self.last_error = "; ".join(errors) if errors else "no enabled providers"
        logger.warning(
            "LLM grading failed, using deterministic fallback: %s",
            self.last_error,
        )
        return None
