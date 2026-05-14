"""Autonomous self-improvement runner for GodelEnv."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from godel_engine.agent import AutoAgent
from godel_engine.environment import GodelEnvironment
from godel_engine.evolution import StrategyRegistry
from godel_engine.models import GodelAction
from godel_engine.provider_runtime import ProviderCircuitBreaker


def _agent_identity(agent: Any) -> str | None:
    provider = getattr(agent, "last_provider", None)
    if not provider:
        return None
    model = getattr(agent, "last_model", None)
    return f"{provider}:{model}" if model else str(provider)


def _llm_source_identity(source: str) -> str | None:
    if not source.startswith("llm:"):
        return None
    return source.split(":", 1)[1]


@dataclass
class SelfImproveEvent:
    iteration: int
    patch_attempt: int
    task_type: str
    task_id: str
    agent_source: str
    patch_proposed: bool
    patch_accepted: bool
    reward: float
    score: float
    improvement: float
    rejection_reasons: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    provider_separation_ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "patch_attempt": self.patch_attempt,
            "task_type": self.task_type,
            "task_id": self.task_id,
            "agent_source": self.agent_source,
            "patch_proposed": self.patch_proposed,
            "patch_accepted": self.patch_accepted,
            "reward": self.reward,
            "score": self.score,
            "improvement": self.improvement,
            "rejection_reasons": list(self.rejection_reasons),
            "source_counts": dict(self.source_counts),
            "token_usage": dict(self.token_usage),
            "provider_separation_ok": self.provider_separation_ok,
            "error": self.error,
        }


class SelfImprovementRunner:
    """Run repeated StrategyPatch attempts against the live environment."""

    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
        agent: Any | None = None,
        strategy_evaluator: Any | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.registry_path = Path(registry_path) if registry_path else None
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.agent = agent or AutoAgent()
        self.strategy_evaluator = strategy_evaluator
        rng_registry = (
            StrategyRegistry.load(self.registry_path)
            if self.registry_path and self.registry_path.exists()
            else StrategyRegistry()
        )
        self.registry = rng_registry

    async def run(
        self,
        *,
        iterations: int = 5,
        task_ids: list[str] | None = None,
        max_patch_attempts: int = 2,
    ) -> dict[str, Any]:
        task_ids = task_ids or ["godel01", "godel02", "godel03", "godel04", "godel05", "godel06"]
        events: list[SelfImproveEvent] = []

        for index in range(iterations):
            task_id = task_ids[index % len(task_ids)]
            env = GodelEnvironment(seed=self.seed + index, registry=self.registry)
            if self.strategy_evaluator is not None:
                env.strategy_evaluator = self.strategy_evaluator
            obs = None
            last_action: GodelAction | None = None
            last_patch_attempt = 0

            try:
                reset_result = await env.reset(
                    task_type="strategy_optimization",
                    task_id=task_id,
                    seed=self.seed + index,
                )
                obs = reset_result.observation
                task = env.current_task
                retry_feedback: list[str] = []

                for patch_attempt in range(1, max_patch_attempts + 1):
                    last_patch_attempt = patch_attempt
                    action = await self.agent.act(
                        task_prompt=obs.task_prompt,
                        current_solution=obs.current_solution,
                        rubrics=task._get_rubrics() if task else {},
                        task_type=obs.task_type,
                        strategy_text=obs.current_strategy,
                        recent_failures=list(obs.recent_failures) + retry_feedback,
                        downstream_scores=obs.downstream_scores,
                    )
                    last_action = action
                    if action.strategy_patch is None:
                        raise RuntimeError("agent did not propose a StrategyPatch")

                    step_result = await env.step(action)
                    decision = step_result.patch_decision
                    source_counts = (
                        decision.diagnostics.get("child_source_counts", {})
                        if decision
                        else {}
                    )
                    token_usage = (
                        decision.diagnostics.get("child_token_usage", {})
                        if decision
                        else {}
                    )
                    agent_identity = _agent_identity(self.agent)
                    verifier_identities = {
                        identity
                        for source in source_counts
                        for identity in [_llm_source_identity(source)]
                        if source.startswith("llm:")
                        and identity is not None
                    }
                    provider_separation_ok = (
                        not agent_identity
                        or not verifier_identities
                        or agent_identity not in verifier_identities
                    )
                    if (
                        os.getenv("GODEL_REQUIRE_PROVIDER_SEPARATION", "0").lower()
                        in ("1", "true", "yes")
                        and not provider_separation_ok
                    ):
                        raise RuntimeError(
                            "provider/model separation required but agent and verifier both used "
                            f"{agent_identity}"
                        )
                    rejection_reasons = list(decision.rejection_reasons if decision else [])
                    events.append(
                        SelfImproveEvent(
                            iteration=index + 1,
                            patch_attempt=patch_attempt,
                            task_type=obs.task_type,
                            task_id=obs.task_id,
                            agent_source=getattr(self.agent, "last_source", "unknown"),
                            patch_proposed=True,
                            patch_accepted=bool(decision and decision.accepted),
                            reward=float(step_result.reward),
                            score=float(step_result.observation.total_score),
                            improvement=float(decision.improvement if decision else 0.0),
                            rejection_reasons=rejection_reasons,
                            source_counts={str(k): int(v) for k, v in source_counts.items()},
                            token_usage={str(k): int(v) for k, v in token_usage.items()},
                            provider_separation_ok=provider_separation_ok,
                        )
                    )
                    if decision and decision.accepted:
                        break
                    retry_feedback = [
                        "Previous patch was rejected: " + "; ".join(rejection_reasons)
                    ]
                    obs = step_result.observation
            except Exception as exc:
                events.append(
                    SelfImproveEvent(
                        iteration=index + 1,
                        patch_attempt=last_patch_attempt,
                        task_type="strategy_optimization",
                        task_id=obs.task_id if obs is not None else task_id,
                        agent_source=getattr(self.agent, "last_source", "unknown"),
                        patch_proposed=bool(last_action and last_action.strategy_patch),
                        patch_accepted=False,
                        reward=0.0,
                        score=float(obs.total_score) if obs is not None else 0.0,
                        improvement=0.0,
                        token_usage={
                            str(k): int(v)
                            for k, v in getattr(self.agent, "last_usage", {}).items()
                        },
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

            if self.registry_path:
                self.registry.save(self.registry_path)

        summary = self._summarize(events)
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _summarize(self, events: list[SelfImproveEvent]) -> dict[str, Any]:
        attempts = len(events)
        accepted = sum(1 for event in events if event.patch_accepted)
        proposed = sum(1 for event in events if event.patch_proposed)
        errors = [event.to_dict() for event in events if event.error]
        llm_eval_events = sum(
            1
            for event in events
            if any(source.startswith("llm:") for source in event.source_counts)
        )
        total_tokens = sum(event.token_usage.get("total_tokens", 0) for event in events)
        total_prompt_tokens = sum(event.token_usage.get("prompt_tokens", 0) for event in events)
        total_completion_tokens = sum(event.token_usage.get("completion_tokens", 0) for event in events)
        prompt_cost = float(os.getenv("GODEL_PROMPT_COST_PER_1K", "0") or 0.0)
        completion_cost = float(os.getenv("GODEL_COMPLETION_COST_PER_1K", "0") or 0.0)
        deterministic_eval_events = sum(
            1
            for event in events
            if any(source.startswith("deterministic") for source in event.source_counts)
        )
        separation_ok = all(event.provider_separation_ok for event in events if event.patch_proposed)
        best = self.registry.get_best()
        return {
            "attempts": attempts,
            "patches_proposed": proposed,
            "patches_accepted": accepted,
            "patches_rejected": proposed - accepted,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "llm_evaluated_attempts": llm_eval_events,
            "all_strategy_evals_llm_backed": llm_eval_events == proposed if proposed else False,
            "deterministic_eval_attempts": deterministic_eval_events,
            "provider_separation_ok": separation_ok,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": (
                    (total_prompt_tokens / 1000.0) * prompt_cost
                    + (total_completion_tokens / 1000.0) * completion_cost
                ),
            },
            "best_strategy_id": best.id,
            "best_strategy_generation": best.generation,
            "best_strategy_elo": best.elo,
            "registry": self.registry.get_stats(),
            "events": [event.to_dict() for event in events],
            "errors": errors,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GodelEnv autonomous self-improvement")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-patch-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("artifacts") / "self_improve" / "strategy_registry.json",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts") / "self_improve" / "metrics.json",
    )
    parser.add_argument(
        "--provider-order",
        default=os.getenv("GODEL_PROVIDER_ORDER", ""),
        help="Optional provider order, e.g. custom,openai.",
    )
    parser.add_argument(
        "--agent-provider-order",
        default=os.getenv("GODEL_AGENT_PROVIDER_ORDER", ""),
        help="Optional proposer provider order. Defaults to --provider-order/GODEL_PROVIDER_ORDER.",
    )
    parser.add_argument(
        "--verifier-provider-order",
        default=os.getenv("GODEL_VERIFIER_PROVIDER_ORDER", ""),
        help="Optional verifier provider order. Defaults to --provider-order/GODEL_PROVIDER_ORDER.",
    )
    parser.add_argument(
        "--agent-model",
        default=os.getenv("GODEL_AGENT_MODEL_NAME", ""),
        help="Optional proposer model override for custom/openai providers.",
    )
    parser.add_argument(
        "--verifier-model",
        default=os.getenv("GODEL_VERIFIER_MODEL_NAME", ""),
        help="Optional verifier model override for custom/openai providers.",
    )
    return parser.parse_args()


async def async_main() -> int:
    load_dotenv(override=False)
    args = parse_args()
    if args.provider_order:
        os.environ["GODEL_PROVIDER_ORDER"] = args.provider_order
    if args.agent_provider_order:
        os.environ["GODEL_AGENT_PROVIDER_ORDER"] = args.agent_provider_order
    if args.verifier_provider_order:
        os.environ["GODEL_VERIFIER_PROVIDER_ORDER"] = args.verifier_provider_order
    if args.agent_model:
        os.environ["GODEL_AGENT_MODEL_NAME"] = args.agent_model
    if args.verifier_model:
        os.environ["GODEL_VERIFIER_MODEL_NAME"] = args.verifier_model
    os.environ.setdefault("GODEL_AGENT_MODE", "llm")
    os.environ.setdefault("GODEL_STRATEGY_EVAL_MODE", "llm")
    os.environ.setdefault("GODEL_ALLOW_DETERMINISTIC_FALLBACK", "0")
    ProviderCircuitBreaker.reset()

    runner = SelfImprovementRunner(
        registry_path=args.registry_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
    )
    summary = await runner.run(
        iterations=args.iterations,
        max_patch_attempts=args.max_patch_attempts,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["patches_proposed"] > 0 else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
