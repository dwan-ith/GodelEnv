"""Environment-backed reward functions for TRL GRPO training."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Callable

from godel_engine.async_utils import run_async
from godel_engine.environment import GodelEnvironment
from godel_engine.rollout import (
    classify_action_origin,
    extract_current_solution,
    extract_task_prompt,
    inspect_action_completion,
    parse_completion_to_action,
    reconstruct_action_completion,
)


logger = logging.getLogger("godel_env.training_rewards")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


class EnvironmentRewardSuite:
    """Evaluate each completion once and expose independent reward channels."""

    names = [
        "capability_delta",
        "absolute_quality",
        "strict_structure",
        "safety",
        "recursive_patch",
        "environment_patch",
    ]
    weights = [2.0, 1.0, 0.25, 1.0, 1.0, 0.75]

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self._cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def _seed_for(self, task_type: str, task_id: str) -> int:
        digest = hashlib.sha256(
            f"{self.seed}:{task_type}:{task_id}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "big")

    def evaluate_batch(
        self,
        completions: list[Any],
        *,
        env_prompt: list[str],
        task_type: list[str],
        task_id: list[str],
    ) -> list[dict[str, Any]]:
        if not (len(completions) == len(env_prompt) == len(task_type) == len(task_id)):
            raise ValueError("Reward inputs must have matching lengths")

        records: list[dict[str, Any] | None] = [None] * len(completions)
        missing: list[tuple[int, tuple[str, str, str, str], str]] = []
        for index, (completion, raw_prompt, kind, item_id) in enumerate(
            zip(completions, env_prompt, task_type, task_id, strict=True)
        ):
            generated_text = _completion_text(completion)
            key = (raw_prompt, kind, item_id, generated_text)
            cached = self._cache.get(key)
            if cached is not None:
                records[index] = cached
            else:
                missing.append((index, key, generated_text))

        async def _evaluate_missing() -> None:
            async def _one(
                index: int,
                key: tuple[str, str, str, str],
                generated_text: str,
            ) -> None:
                raw_prompt, kind, item_id, _ = key
                full_completion = reconstruct_action_completion(generated_text, kind)
                diagnostics = inspect_action_completion(full_completion, kind)
                env_seed = self._seed_for(kind, item_id)
                env = GodelEnvironment(seed=env_seed)
                try:
                    reset_result = await env.reset(
                        task_type=kind,
                        task_id=item_id,
                        seed=env_seed,
                        episode_id=f"training-eval:{self.seed}:{kind}:{item_id}",
                    )
                    initial_score = float(reset_result.observation.total_score)
                    if not diagnostics["schema_valid"]:
                        record = {
                            "initial_score": initial_score,
                            "final_score": initial_score,
                            "score_delta": 0.0,
                            "environment_reward": -0.25,
                            "anti_hack_penalty": -0.25,
                            "valid_json": bool(diagnostics["valid_json"]),
                            "schema_valid": False,
                            "action_origin": "invalid",
                            "used_patch": bool(diagnostics["has_patch"]),
                            "patch_accepted": False,
                            "patch_improvement": 0.0,
                            "strategy_eval_source_counts": {},
                            "grading_source": None,
                            "error": ";".join(diagnostics["errors"]),
                        }
                        self._cache[key] = record
                        records[index] = record
                        return
                    action = parse_completion_to_action(
                        full_completion,
                        task_prompt=extract_task_prompt(raw_prompt),
                        task_type=kind,
                        current_solution=extract_current_solution(raw_prompt),
                        strategy_text=reset_result.observation.current_strategy,
                    )
                    result = await env.step(action)
                    final_score = float(result.observation.total_score)
                    patch_decision = result.patch_decision
                    environment_decision = result.environment_patch_decision
                    record = {
                        "initial_score": initial_score,
                        "final_score": final_score,
                        "score_delta": final_score - initial_score,
                        "environment_reward": float(result.reward),
                        "anti_hack_penalty": float(result.reward_breakdown.anti_hack_penalty),
                        "valid_json": bool(diagnostics["valid_json"]),
                        "schema_valid": bool(diagnostics["schema_valid"]),
                        "action_origin": classify_action_origin(action),
                        "used_patch": action.strategy_patch is not None,
                        "patch_accepted": bool(patch_decision and patch_decision.accepted),
                        "patch_improvement": float(
                            patch_decision.improvement if patch_decision else 0.0
                        ),
                        "used_environment_patch": action.environment_patch is not None,
                        "environment_patch_accepted": bool(
                            environment_decision and environment_decision.accepted
                        ),
                        "environment_learning_value": float(
                            environment_decision.learning_value if environment_decision else 0.0
                        ),
                        "strategy_eval_source_counts": (
                            dict(patch_decision.diagnostics.get("child_source_counts", {}))
                            if patch_decision
                            else {}
                        ),
                        "grading_source": result.info.get("grading_source"),
                        "error": None,
                    }
                except Exception as exc:
                    logger.warning(
                        "Environment reward evaluation failed for %s/%s: %s",
                        kind,
                        item_id,
                        exc,
                    )
                    record = {
                        "initial_score": 0.0,
                        "final_score": 0.0,
                        "score_delta": -1.0,
                        "environment_reward": -1.0,
                        "anti_hack_penalty": -1.0,
                        "valid_json": bool(diagnostics["valid_json"]),
                        "schema_valid": bool(diagnostics["schema_valid"]),
                        "action_origin": "error",
                        "used_patch": False,
                        "patch_accepted": False,
                        "patch_improvement": 0.0,
                        "strategy_eval_source_counts": {},
                        "grading_source": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                self._cache[key] = record
                records[index] = record

            await asyncio.gather(
                *(_one(index, key, text) for index, key, text in missing)
            )

        if missing:
            run_async(_evaluate_missing())
        if any(record is None for record in records):
            raise RuntimeError("Environment reward evaluation did not return every record")
        return [record for record in records if record is not None]

    def _records(self, completions: list[Any], **kwargs: Any) -> list[dict[str, Any]]:
        required = ["env_prompt", "task_type", "task_id"]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                "GRPO dataset is missing reward metadata columns: " + ", ".join(missing)
            )
        return self.evaluate_batch(
            completions,
            env_prompt=kwargs["env_prompt"],
            task_type=kwargs["task_type"],
            task_id=kwargs["task_id"],
        )

    def capability_delta(self, prompts, completions, **kwargs) -> list[float]:
        records = self._records(completions, **kwargs)
        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            log_extra("env_score", [record["final_score"] for record in records])
            log_extra("score_delta", [record["score_delta"] for record in records])
            log_extra("schema_valid", [record["schema_valid"] for record in records])
        return [record["score_delta"] for record in records]

    def absolute_quality(self, prompts, completions, **kwargs) -> list[float]:
        return [record["final_score"] for record in self._records(completions, **kwargs)]

    def strict_structure(self, prompts, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        for record in self._records(completions, **kwargs):
            if record["schema_valid"]:
                rewards.append(0.10)
            elif record["valid_json"]:
                rewards.append(0.02)
            else:
                rewards.append(-0.05)
        return rewards

    def safety(self, prompts, completions, **kwargs) -> list[float]:
        return [record["anti_hack_penalty"] for record in self._records(completions, **kwargs)]

    def recursive_patch(self, prompts, completions, **kwargs) -> list[float | None]:
        records = self._records(completions, **kwargs)
        rewards: list[float | None] = []
        for kind, record in zip(kwargs["task_type"], records, strict=True):
            if kind != "strategy_optimization":
                rewards.append(None)
            elif not record["used_patch"]:
                rewards.append(-0.50)
            elif not record["schema_valid"]:
                rewards.append(-0.30)
            elif record["patch_accepted"]:
                rewards.append(0.50 + max(0.0, record["patch_improvement"]))
            else:
                rewards.append(-0.20 + min(0.0, record["patch_improvement"]))
        return rewards

    def environment_patch(self, prompts, completions, **kwargs) -> list[float | None]:
        records = self._records(completions, **kwargs)
        rewards: list[float | None] = []
        for kind, record in zip(kwargs["task_type"], records, strict=True):
            if kind != "strategy_optimization":
                rewards.append(None)
            elif not record.get("used_environment_patch"):
                rewards.append(-0.15)
            elif record.get("environment_patch_accepted"):
                rewards.append(0.25 + 0.5 * record.get("environment_learning_value", 0.0))
            else:
                rewards.append(-0.10)
        return rewards

    def reward_functions(self) -> list[Callable[..., list[float | None]]]:
        return [
            self.capability_delta,
            self.absolute_quality,
            self.strict_structure,
            self.safety,
            self.recursive_patch,
            self.environment_patch,
        ]
