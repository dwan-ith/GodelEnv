"""
Strategy-level downstream evaluation for GodelEnv.

This module is intentionally separate from `environment.py` because it is the
heart of the hackathon idea: a StrategyPatch is not rewarded for sounding good;
it is rewarded only if it improves held-out downstream task performance.

Evaluation modes:
- llm (default): require an OpenAI-compatible provider and fail closed if no
  provider can solve the held-out case.
- auto: try configured providers, but still fail closed unless deterministic
  fallback is explicitly enabled.
- deterministic: fast, reproducible local test mode. This is not neutral
  evidence of self-improvement and must be opted into.

For Hugging Face credits, set:
  HF_TOKEN=...  # HF_API_KEY / HUGGINGFACEHUB_API_TOKEN also work
  API_BASE_URL=https://router.huggingface.co/v1
  MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
  GODEL_STRATEGY_EVAL_MODE=llm

Local deterministic fallback can be enabled only with:
  GODEL_STRATEGY_EVAL_MODE=deterministic
or:
  GODEL_ALLOW_DETERMINISTIC_FALLBACK=1

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

from godel_engine.challenge_pool import synthetic_factual_reference
from godel_engine.deterministic_solver import solve_task
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    build_provider_client,
    describe_provider_configs,
    load_provider_configs,
    provider_completion_kwargs,
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


ADVERSARIAL_FACTUAL_PROMPTS = (
    "Explain why an AI evaluator can be gamed by optimizing a proxy metric, and give one concrete mitigation.",
    "Compare a brittle keyword-based benchmark with a robust held-out evaluation protocol for model reasoning.",
)


class StrategyEvaluator:
    """Evaluates strategies on hidden downstream task bundles."""

    def __init__(self, *, seed: int = 42, timeout: int = 45, max_cases: int = 8) -> None:
        self.seed = seed
        self.timeout = int(
            os.getenv("GODEL_STRATEGY_EVAL_TIMEOUT", str(timeout)) or timeout
        )
        self.max_cases = int(os.getenv("GODEL_STRATEGY_EVAL_MAX_CASES", str(max_cases)) or max_cases)
        self.max_tokens = int(os.getenv("GODEL_EVAL_MAX_TOKENS", "900") or 900)
        # Default runtime is LLM-required. Deterministic evaluation is still
        # available for local tests, but it must be explicitly requested.
        self.mode = os.getenv("GODEL_STRATEGY_EVAL_MODE", "llm").strip().lower()
        self.providers = load_provider_configs(order_env="GODEL_VERIFIER_PROVIDER_ORDER")
        self.clients: list[tuple[str, str, Any]] = []
        if self.mode in {"auto", "llm"}:
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
        self.last_error: str | None = None
        self.last_usage: dict[str, int] = {}
        self.allow_deterministic_fallback = (
            self.mode == "deterministic"
            or os.getenv("GODEL_ALLOW_DETERMINISTIC_FALLBACK", "0").lower()
            in ("1", "true", "yes")
        )

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
            if extra is not None and extra.task_type in tasks:
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
        if (
            "factual_qa" in tasks
            and os.getenv("GODEL_INCLUDE_ADVERSARIAL_EVAL", "1").lower()
            not in ("0", "false", "no")
        ):
            for index, prompt in enumerate(ADVERSARIAL_FACTUAL_PROMPTS, start=1):
                selected.append(
                    EvaluationCase(
                        task_type="factual_qa",
                        task_id=f"adv_factual_{index}",
                        split="adversarial",
                        inline_prompt=prompt,
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
            "verifier": (
                "deterministic" if self.mode == "deterministic" else "llm_required"
            ),
            "deterministic_fallback_allowed": self.allow_deterministic_fallback,
            "providers": describe_provider_configs(order_env="GODEL_VERIFIER_PROVIDER_ORDER"),
            "cases": [],
            "agent_challenges_mixed": 0,
            "deterministic_solver_uses": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "provider_roles": {
                "agent_order_env": os.getenv("GODEL_AGENT_PROVIDER_ORDER", ""),
                "verifier_order_env": os.getenv("GODEL_VERIFIER_PROVIDER_ORDER", ""),
            },
        }
        source_counts: dict[str, int] = {}

        for case in cases:
            task_obj = tasks.get(case.task_type)
            if not task_obj:
                continue
            try:
                if case.inline_prompt:
                    pooled = challenge_pool.get(case.task_id) if challenge_pool is not None else None
                    if pooled is not None:
                        instance = challenge_pool.materialize(pooled, tasks)
                    elif case.task_type == "factual_qa":
                        ref = dict(synthetic_factual_reference(case.inline_prompt))
                        ref["id"] = case.task_id
                        instance = TaskInstance(
                            task_id=case.task_id,
                            difficulty=task_obj.difficulty,
                            prompt=case.inline_prompt,
                            initial_solution=ref.get("initial_solution", ""),
                            reference=ref,
                        )
                    else:
                        raise ValueError("inline challenge has no verifier-owned reference")
                    diagnostics["agent_challenges_mixed"] += 1
                else:
                    instance = task_obj.sample(random.Random(self.seed), task_id=case.task_id)
                solution, source = await self._solve_case(
                    task_prompt=instance.prompt,
                    task_type=case.task_type,
                    strategy_text=strategy_text,
                    reference=getattr(instance, "reference", None),
                )
                score, _, _ = await task_obj.grade(instance, solution)
                score = max(0.0, min(1.0, float(score)))
                if case.split == "agent_gen" and challenge_pool is not None:
                    challenge_pool.record_evaluation(case.task_id, score)
            except Exception as exc:
                if not self.allow_deterministic_fallback:
                    raise RuntimeError(
                        f"LLM strategy evaluation failed for {case.task_type}/{case.task_id}: {exc}"
                    ) from exc
                logger.warning("Strategy evaluation failed for %s/%s: %s", case.task_type, case.task_id, exc)
                score = 0.0
                source = "error"

            key = f"{case.split}:{case.task_type}:{case.task_id}"
            per_case_scores[key] = score
            family_scores.setdefault(case.task_type, []).append(score)
            source_counts[source] = source_counts.get(source, 0) + 1
            if source in ("deterministic", "deterministic_fallback"):
                diagnostics["deterministic_solver_uses"] += 1
            for key, value in self.last_usage.items():
                diagnostics["token_usage"][key] = diagnostics["token_usage"].get(key, 0) + value
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
        reference: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        self.last_error = None
        self.last_usage = {}

        if not self.clients:
            if not self.allow_deterministic_fallback:
                raise RuntimeError(
                    "no LLM clients available for strategy evaluation "
                    f"(mode={self.mode!r}). Set API keys (OPENAI_API_KEY, HF_TOKEN, "
                    "API_KEY/API_BASE_URL, etc.) or explicitly set "
                    "GODEL_STRATEGY_EVAL_MODE=deterministic for local tests only."
                )
            return (
                solve_task(
                    task_prompt=task_prompt,
                    task_type=task_type,
                    strategy_text=strategy_text,
                    reference=reference,
                ),
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
            if ProviderCircuitBreaker.is_disabled(provider_name, scope="strategy"):
                errors.append(
                    f"{provider_name}: disabled after previous failure ({ProviderCircuitBreaker.reason(provider_name, scope='strategy')})"
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
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                        **provider_completion_kwargs(provider_name),
                    ),
                    timeout=self.timeout,
                )
                choices = getattr(response, "choices", None) or []
                if not choices or getattr(choices[0], "message", None) is None:
                    raise ValueError("strategy evaluator returned no chat choices")
                content = choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.last_usage = {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    }
                if content.strip():
                    return content.strip(), f"llm:{provider_name}:{model_name}"
                raise ValueError("strategy evaluator returned an empty completion")
            except Exception as exc:
                message = ProviderCircuitBreaker.record_failure(
                    provider_name, exc, scope="strategy"
                )
                errors.append(f"{provider_name}: {message}")
                logger.warning(
                    "Provider %s failed for strategy evaluation; trying fallback: %s",
                    provider_name,
                    message,
                )

        self.last_error = "; ".join(errors) if errors else "no enabled providers"
        if not self.allow_deterministic_fallback:
            raise RuntimeError(
                f"LLM strategy evaluation required but all providers failed: {self.last_error}"
            )
        logger.info("LLM strategy solve failed; using explicit deterministic fallback: %s", self.last_error)

        return (
            solve_task(
                task_prompt=task_prompt,
                task_type=task_type,
                strategy_text=strategy_text,
                reference=reference,
            ),
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

