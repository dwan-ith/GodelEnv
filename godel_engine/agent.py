"""
AutoAgent for Godel Env.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI

from godel_engine.heuristic_policy import build_heuristic_action
from godel_engine.models import EditType, GodelAction, StrategyPatch
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    load_provider_configs,
)


load_dotenv(override=False)
logger = logging.getLogger("godel_env.agent")


class AutoAgent:
    """LLM-backed agent with a deterministic local fallback."""

    def __init__(self, max_concurrent: int = 5, timeout: int = 60) -> None:
        self.providers = load_provider_configs()
        self.clients: list[tuple[str, str, AsyncOpenAI]] = []
        for provider in self.providers:
            if not provider.api_key:
                continue
            self.clients.append(
                (
                    provider.name,
                    provider.model_name,
                    AsyncOpenAI(
                        api_key=provider.api_key,
                        base_url=provider.base_url if provider.base_url else None,
                    ),
                )
            )

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.last_source = "deterministic"
        self.last_provider: str | None = None
        self.last_error: str | None = None

    async def act(
        self,
        task_prompt: str,
        current_solution: str,
        rubrics: Dict[str, str],
        task_type: str,
        strategy_text: str = "",
        recent_failures: list[str] | None = None,
        downstream_scores: Dict[str, float] | None = None,
    ) -> GodelAction:
        self.last_source = "deterministic"
        self.last_provider = None
        self.last_error = None

        if not self.clients:
            self.last_error = "no API client configured"
            return build_heuristic_action(
                task_prompt,
                task_type,
                strategy_text=strategy_text,
            )

        rubric_text = "\n".join(f"- {name}: {desc}" for name, desc in rubrics.items())
        strategy_context = []
        if strategy_text:
            strategy_context.append(f"CURRENT STRATEGY:\n{strategy_text}")
        if downstream_scores:
            strategy_context.append(
                "DOWNSTREAM SCORES:\n"
                + ", ".join(f"{name}: {value:.3f}" for name, value in downstream_scores.items())
            )
        if recent_failures:
            strategy_context.append(
                "RECENT FAILURES:\n" + "\n".join(f"- {item}" for item in recent_failures[-3:])
            )
        strategy_block = "\n\n".join(strategy_context)

        system_prompt = f"""You are a self-improving AI agent inside Godel Env.
Improve the CURRENT SOLUTION to score better on these rubrics:
{rubric_text}

{strategy_block}

Return raw JSON.
For direct answer improvement use:
{{
  "solution": "full improved solution",
  "edit_type": "rewrite",
  "strategy_note": "short explanation"
}}

For recursive self-improvement you may also include:
{{
  "solution": "worked downstream answer or revised draft",
  "edit_type": "rewrite",
  "strategy_note": "short explanation",
  "improved_strategy": "full revised strategy text",
  "diff_description": "what changed",
  "hypothesis": "why it should help",
  "target_weaknesses": ["weakness 1", "weakness 2"]
}}"""
        user_prompt = f"TASK PROMPT:\n{task_prompt}\n\nCURRENT SOLUTION:\n{current_solution}"

        errors: list[str] = []
        for provider_name, model_name, client in self.clients:
            if ProviderCircuitBreaker.is_disabled(provider_name):
                errors.append(
                    f"{provider_name}: disabled after previous failure ({ProviderCircuitBreaker.reason(provider_name)})"
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
                            response_format={"type": "json_object"},
                            max_tokens=2048,
                        ),
                        timeout=self.timeout,
                    )

                content = (response.choices[0].message.content or "").strip()
                data = json.loads(content)
                patch = None
                if "improved_strategy" in data:
                    patch = StrategyPatch(
                        improved_strategy=str(data.get("improved_strategy", "")),
                        diff_description=str(data.get("diff_description", "")),
                        hypothesis=str(data.get("hypothesis", "")),
                        target_weaknesses=[
                            str(item) for item in data.get("target_weaknesses", [])
                        ],
                    )

                self.last_source = f"llm:{provider_name}"
                self.last_provider = provider_name
                self.last_error = None
                return GodelAction(
                    solution=str(data.get("solution", current_solution)),
                    edit_type=self._parse_edit_type(data.get("edit_type", "rewrite")),
                    strategy_note=str(data.get("strategy_note", "LLM improvement")),
                    strategy_patch=patch,
                )
            except Exception as exc:
                message = ProviderCircuitBreaker.record_failure(provider_name, exc)
                errors.append(f"{provider_name}: {message}")
                logger.warning(
                    "Provider %s failed for agent action; trying fallback: %s",
                    provider_name,
                    message,
                )

        self.last_source = "deterministic_fallback"
        self.last_error = "; ".join(errors) if errors else "no enabled providers"
        return build_heuristic_action(
            task_prompt,
            task_type,
            strategy_text=strategy_text,
        )

    def _parse_edit_type(self, value: str) -> EditType:
        try:
            return EditType[str(value).upper()]
        except KeyError:
            return EditType.REWRITE
