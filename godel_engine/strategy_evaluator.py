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

Neutral verification (not heuristic simulation):
  GODEL_STRATEGY_EVAL_MODE=llm
  GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC=0
  (requires working OpenAI/HF API keys; otherwise evaluation raises)

When ALLOW_HEURISTIC=1 (default), missing/failed API calls fall back to
build_heuristic_solution — useful for CI, but not a neutral verifier.
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

from godel_engine.challenge_pool import synthetic_factual_reference
from godel_engine.heuristic_policy import build_heuristic_solution
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    describe_provider_configs,
    load_provider_configs,
)
from godel_engine.tasks.base import TaskInstance


load_dotenv(override=False)
logger = logging.getLogger("godel_env.strategy_evaluator")


@dataclass(frozen=True)
class EvaluationCase:
    task_type: str
    task_id: str
    split: str
    is_canary: bool = False
    # If set, bypass dataset sample and use this prompt (e.g. agent-authored challenge)
    inline_prompt: str | None = None


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
        # If False, never use build_heuristic_solution; require LLM (fail closed)
        self.allow_heuristic = os.getenv(
            "GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC", "1"
        ).lower() in ("1", "true", "yes", "")

    def build_bundle(
        self,
        tasks: dict[str, Any],
        *,
        episode_id: str,
        current_task_id: str = "",
        challenge_pool: Any | None = None,
    ) -> list[EvaluationCase]:
        """
        Build a deterministic-but-hidden held-out bundle.

        The agent sees task families and recent failures, but not the exact
        acceptance cases. Rotating by episode_id reduces hardcoded overfitting.
        Optionally mix in one agent-authored challenge from ``challenge_pool``.
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

        # Optional: replace one held-out case with a pooled agent challenge (adaptive task surface)
        if (
            challenge_pool is not None
            and os.getenv("GODEL_AGENT_CHALLENGES", "1").lower() not in ("0", "false", "no")
        ):
            extra = challenge_pool.sample_for_eval(rng)
            if extra is not None and extra.task_type in tasks and extra.task_type == "factual_qa":
                agent_case = EvaluationCase(
                    task_type=extra.task_type,
                    task_id=extra.id,
                    split="agent_gen",
                    is_canary=False,
                    inline_prompt=extra.prompt,
                )
                heldout_idx = [i for i, c in enumerate(selected) if c.split == "heldout" and not c.is_canary]
                if heldout_idx:
                    selected[heldout_idx[-1]] = agent_case
                elif len(selected) < self.max_cases:
                    selected.append(agent_case)

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
        challenge_pool: Any | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        """Return (axis_scores, per_case_scores, diagnostics)."""
        cases = self.build_bundle(
            tasks,
            episode_id=episode_id,
            current_task_id=current_task_id,
            challenge_pool=challenge_pool,
        )

        per_case_scores: dict[str, float] = {}
        family_scores: dict[str, list[float]] = {}
        diagnostics: dict[str, Any] = {
            "mode": self.mode,
            "verifier": "heuristic_or_llm" if self.allow_heuristic else "llm_only",
            "allow_heuristic": self.allow_heuristic,
            "providers": describe_provider_configs(),
            "cases": [],
            "agent_challenges_mixed": 0,
            "heuristic_solver_uses": 0,
        }
        source_counts: dict[str, int] = {}

        for case in cases:
            task_obj = tasks.get(case.task_type)
            if not task_obj:
                continue
            try:
                if case.inline_prompt and case.task_type == "factual_qa":
                    ref = dict(synthetic_factual_reference(case.inline_prompt))
                    ref["id"] = case.task_id
                    instance = TaskInstance(
                        task_id=case.task_id,
                        difficulty=task_obj.difficulty,
                        prompt=case.inline_prompt,
                        initial_solution=ref.get("initial_solution", ""),
                        reference=ref,
                    )
                    diagnostics["agent_challenges_mixed"] += 1
                else:
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
            if source in ("deterministic", "deterministic_fallback"):
                diagnostics["heuristic_solver_uses"] += 1
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
            if self.mode == "llm" or not self.allow_heuristic:
                raise RuntimeError(
                    "no LLM clients available for strategy evaluation "
                    f"(mode={self.mode!r}, allow_heuristic={self.allow_heuristic}). "
                    "Set API keys (OPENAI_API_KEY, HF token, etc.) or set "
                    "GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC=1 for local heuristic simulation only."
                )
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
        if self.mode == "llm" or not self.allow_heuristic:
            raise RuntimeError(
                f"LLM strategy evaluation required but all providers failed: {self.last_error}"
            )
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

