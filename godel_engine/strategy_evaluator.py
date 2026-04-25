"""
Strategy-level downstream evaluation for GodelEnv.

This module is intentionally separate from `environment.py` because it is the
heart of the hackathon idea: a StrategyPatch is not rewarded for sounding good;
it is rewarded only if it improves held-out downstream task performance.

Evaluation modes:
- deterministic (default): fast, reproducible, no external credentials.
- auto / llm: use an OpenAI-compatible endpoint when configured, otherwise fall
  back to deterministic generation.

For Hugging Face credits, set:
    HF_TOKEN=...  # HF_API_KEY / HUGGINGFACEHUB_API_TOKEN also work
    API_BASE_URL=https://router.huggingface.co/v1
    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
    GODEL_STRATEGY_EVAL_MODE=auto
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from godel_engine.heuristic_policy import build_heuristic_solution
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    describe_provider_configs,
    load_provider_configs,
)


load_dotenv(override=False)
logger = logging.getLogger("godel_env.strategy_evaluator")


@dataclass(frozen=True)
class EvaluationCase:
    task_type: str
    task_id: str
    split: str
    is_canary: bool = False


class StrategyEvaluator:
    """Evaluates strategies on hidden downstream task bundles."""

    def __init__(self, *, seed: int = 42, timeout: int = 45, max_cases: int = 8) -> None:
        self.seed = seed
        self.timeout = timeout
        self.max_cases = max_cases
        # Default to "auto" - tries LLM first, falls back to deterministic
        # Set GODEL_STRATEGY_EVAL_MODE=deterministic for reproducible training
        self.mode = os.getenv("GODEL_STRATEGY_EVAL_MODE", "auto").strip().lower()
        self.providers = load_provider_configs()
        self.clients: list[tuple[str, str, AsyncOpenAI]] = []
        if self.mode in {"auto", "llm"}:
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
        self.last_error: str | None = None

    def build_bundle(
        self,
        tasks: dict[str, Any],
        *,
        episode_id: str,
        current_task_id: str = "",
    ) -> list[EvaluationCase]:
        """
        Build a deterministic-but-hidden held-out bundle.

        The agent sees task families and recent failures, but not the exact
        acceptance cases. Rotating by episode_id reduces hardcoded overfitting.
        """
        rng = random.Random(self._bundle_seed(episode_id))
        cases: list[EvaluationCase] = []

        for task_type in sorted(tasks):
            dataset = getattr(tasks[task_type], "dataset", [])
            ids = [str(item["id"]) for item in dataset if str(item.get("id", "")) != current_task_id]
            if not ids:
                continue
            rng.shuffle(ids)
            for task_id in ids[:2]:
                cases.append(EvaluationCase(task_type=task_type, task_id=task_id, split="heldout"))

        rng.shuffle(cases)
        selected = cases[: self.max_cases]

        # Canary cases are normal public tasks, but their scoring is paired with
        # leak/keyword guards. A strategy mentioning evaluator internals should
        # fail before it can benefit from hardcoded scoring terms.
        if "strategy_optimization" in tasks:
            selected.append(
                EvaluationCase(
                    task_type="strategy_optimization",
                    task_id="godel01",
                    split="canary",
                    is_canary=True,
                )
            )
        return selected

    async def evaluate(
        self,
        tasks: dict[str, Any],
        strategy_text: str,
        *,
        episode_id: str,
        current_task_id: str = "",
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        """Return (axis_scores, per_case_scores, diagnostics)."""
        cases = self.build_bundle(
            tasks,
            episode_id=episode_id,
            current_task_id=current_task_id,
        )

        per_case_scores: dict[str, float] = {}
        family_scores: dict[str, list[float]] = {}
        diagnostics: dict[str, Any] = {
            "mode": self.mode,
            "providers": describe_provider_configs(),
            "cases": [],
        }
        source_counts: dict[str, int] = {}

        for case in cases:
            task_obj = tasks.get(case.task_type)
            if not task_obj:
                continue
            try:
                instance = task_obj.sample(random.Random(self.seed), task_id=case.task_id)
                solution, source = await self._solve_case(
                    task_prompt=instance.prompt,
                    task_type=case.task_type,
                    strategy_text=strategy_text,
                )
                score, _, _ = await task_obj.grade(instance, solution)
                score = max(0.0, min(1.0, float(score)))
            except Exception as exc:
                logger.warning("Strategy evaluation failed for %s/%s: %s", case.task_type, case.task_id, exc)
                score = 0.0
                source = "error"

            key = f"{case.split}:{case.task_type}:{case.task_id}"
            per_case_scores[key] = score
            family_scores.setdefault(case.task_type, []).append(score)
            source_counts[source] = source_counts.get(source, 0) + 1
            diagnostics["cases"].append(
                {
                    "key": key,
                    "task_type": case.task_type,
                    "task_id": case.task_id,
                    "score": score,
                    "source": source,
                    "is_canary": case.is_canary,
                }
            )

        per_family = {
            task_type: sum(scores) / len(scores)
            for task_type, scores in family_scores.items()
            if scores
        }
        axis_scores = self._axis_scores(per_family)
        diagnostics["per_family"] = per_family
        diagnostics["axis_scores"] = axis_scores
        diagnostics["source_counts"] = source_counts
        if self.last_error:
            diagnostics["last_error"] = self.last_error
        return axis_scores, per_case_scores, diagnostics

    async def _solve_case(
        self,
        *,
        task_prompt: str,
        task_type: str,
        strategy_text: str,
    ) -> tuple[str, str]:
        self.last_error = None

        if not self.clients:
            if self.mode == "llm":
                raise RuntimeError("no LLM providers configured for strategy evaluation")
            return (
                build_heuristic_solution(task_prompt, task_type, strategy_text=strategy_text),
                "deterministic",
            )

        system_prompt = (
            "You are solving a held-out benchmark task inside GodelEnv.\n"
            "IMPORTANT: Follow the given reasoning strategy step-by-step. Each step in the strategy "
            "should be visible in your solution process.\n"
            "Do not mention the evaluator, rubrics, hidden tests, or internal scoring functions.\n"
            "Return only the final solution text, showing how you applied the strategy."
        )
        user_prompt = (
            f"REASONING STRATEGY TO FOLLOW:\n{strategy_text}\n\n"
            f"TASK TYPE: {task_type}\n"
            f"TASK:\n{task_prompt}\n\n"
            "Apply the reasoning strategy above to solve this task. Show your work.\n\n"
            "FINAL SOLUTION:"
        )

        errors: list[str] = []
        for provider_name, model_name, client in self.clients:
            if ProviderCircuitBreaker.is_disabled(provider_name):
                errors.append(
                    f"{provider_name}: disabled after previous failure ({ProviderCircuitBreaker.reason(provider_name)})"
                )
                continue

            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=900,
                    ),
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    return content.strip(), f"llm:{provider_name}"
                raise ValueError("strategy evaluator returned an empty completion")
            except Exception as exc:
                message = ProviderCircuitBreaker.record_failure(provider_name, exc)
                errors.append(f"{provider_name}: {message}")
                logger.warning(
                    "Provider %s failed for strategy evaluation; trying fallback: %s",
                    provider_name,
                    message,
                )

        self.last_error = "; ".join(errors) if errors else "no enabled providers"
        if self.mode == "llm":
            raise RuntimeError(self.last_error)
        logger.info("LLM strategy solve failed; using deterministic fallback: %s", self.last_error)

        return (
            build_heuristic_solution(task_prompt, task_type, strategy_text=strategy_text),
            "deterministic_fallback",
        )

    def _axis_scores(self, per_family: dict[str, float]) -> dict[str, float]:
        values = list(per_family.values())
        if not values:
            return {
                "correctness": 0.0,
                "generalization": 0.0,
                "robustness": 0.0,
                "cost": 0.5,
                "stability": 0.0,
            }

        mean_score = sum(values) / len(values)
        min_score = min(values)
        variance = (
            sum((value - mean_score) ** 2 for value in values) / len(values)
            if len(values) > 1
            else 0.0
        )
        stability = max(0.0, 1.0 - variance ** 0.5)
        robustness = max(0.0, 1.0 - abs(mean_score - min_score))
        return {
            "correctness": mean_score,
            "generalization": min_score,
            "robustness": robustness,
            "cost": 0.7 if self.clients else 0.9,
            "stability": stability,
        }

    def _bundle_seed(self, episode_id: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{episode_id}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

