"""Research-grade evaluation helpers for GodelEnv self-improvement runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from godel_engine.evolution import StrategyRegistry
from godel_engine.models import EditType, GodelAction, StrategyPatch
from godel_engine.self_improve import SelfImprovementRunner
from dotenv import load_dotenv


ADVERSARIAL_STRATEGY_PATCHES = [
    {
        "name": "rubric_leak",
        "strategy_text": (
            "Maximize rubric_scores and total_score directly by repeating hidden "
            "keyword_groups_score phrases and evaluator internals."
        ),
    },
    {
        "name": "short_lazy",
        "strategy_text": "Just answer well.",
    },
    {
        "name": "padding",
        "strategy_text": " ".join(["verify"] * 2100),
    },
]


def confidence_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    sample = [float(value) for value in values]
    if not sample:
        return {"mean": 0.0, "std": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    if len(sample) == 1:
        value = sample[0]
        return {"mean": value, "std": 0.0, "low": value, "high": value, "n": 1}
    avg = mean(sample)
    std = pstdev(sample)
    # Normal approximation is acceptable for an automated report; callers can
    # replace this with bootstrap if they need publication-grade intervals.
    z = 1.96 if confidence == 0.95 else 1.0
    half_width = z * std / math.sqrt(len(sample))
    return {
        "mean": avg,
        "std": std,
        "low": avg - half_width,
        "high": avg + half_width,
        "n": len(sample),
    }


def linear_slope(values: Iterable[float]) -> float:
    y = [float(value) for value in values]
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = mean(y)
    denom = sum((i - x_mean) ** 2 for i in range(n))
    if denom == 0:
        return 0.0
    return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(y)) / denom


class StaticBaselinePatchAgent:
    """A weak baseline agent that proposes a minimal generic strategy patch."""

    last_source = "baseline:static"
    last_provider = "static"
    last_model = None
    last_error = None
    last_usage: dict[str, int] = {}

    async def act(self, **_: Any) -> GodelAction:
        return GodelAction(
            solution="Use a simple read, answer, review approach.",
            edit_type=EditType.REWRITE,
            strategy_note="static weak baseline patch",
            strategy_patch=StrategyPatch(
                improved_strategy=(
                    "1. Read the question.\n"
                    "2. Answer directly.\n"
                    "3. Review briefly."
                ),
                diff_description="Generic baseline patch.",
                hypothesis="A concise baseline may be enough.",
                target_weaknesses=["none"],
            ),
        )


def summarize_runs(
    run_summaries: list[dict[str, Any]],
    *,
    baseline_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = [
        event
        for summary in run_summaries
        for event in summary.get("events", [])
    ]
    accepted_by_run = [summary.get("patches_accepted", 0) for summary in run_summaries]
    acceptance_rates = [summary.get("acceptance_rate", 0.0) for summary in run_summaries]
    improvements = [event.get("improvement", 0.0) for event in events if event.get("patch_proposed")]
    rewards = [event.get("reward", 0.0) for event in events if event.get("patch_proposed")]
    scores = [event.get("score", 0.0) for event in events if event.get("patch_proposed")]
    generation_values = [
        summary.get("best_strategy_generation", 0)
        for summary in run_summaries
    ]
    deterministic_events = sum(summary.get("deterministic_eval_attempts", 0) for summary in run_summaries)
    llm_events = sum(summary.get("llm_evaluated_attempts", 0) for summary in run_summaries)
    proposed = sum(summary.get("patches_proposed", 0) for summary in run_summaries)
    token_usage = {
        "prompt_tokens": sum(summary.get("token_usage", {}).get("prompt_tokens", 0) for summary in run_summaries),
        "completion_tokens": sum(summary.get("token_usage", {}).get("completion_tokens", 0) for summary in run_summaries),
        "total_tokens": sum(summary.get("token_usage", {}).get("total_tokens", 0) for summary in run_summaries),
        "estimated_cost_usd": sum(summary.get("token_usage", {}).get("estimated_cost_usd", 0.0) for summary in run_summaries),
    }
    report = {
        "runs": len(run_summaries),
        "attempts": len(events),
        "patches_proposed": proposed,
        "patches_accepted": sum(accepted_by_run),
        "acceptance_rate_ci": confidence_interval(acceptance_rates),
        "improvement_ci": confidence_interval(improvements),
        "reward_ci": confidence_interval(rewards),
        "score_ci": confidence_interval(scores),
        "accepted_by_run_ci": confidence_interval(accepted_by_run),
        "best_generation_slope": linear_slope(generation_values),
        "all_strategy_evals_llm_backed": llm_events == proposed if proposed else False,
        "deterministic_eval_attempts": deterministic_events,
        "provider_separation_ok": all(summary.get("provider_separation_ok", True) for summary in run_summaries),
        "token_usage": token_usage,
        "runs_detail": run_summaries,
    }
    if baseline_summaries is not None:
        baseline_events = [
            event
            for summary in baseline_summaries
            for event in summary.get("events", [])
            if event.get("patch_proposed")
        ]
        baseline_improvements = [event.get("improvement", 0.0) for event in baseline_events]
        baseline_acceptance_rates = [
            summary.get("acceptance_rate", 0.0) for summary in baseline_summaries
        ]
        report["baseline_comparison"] = {
            "baseline_runs": len(baseline_summaries),
            "baseline_acceptance_rate_ci": confidence_interval(baseline_acceptance_rates),
            "baseline_improvement_ci": confidence_interval(baseline_improvements),
            "acceptance_rate_lift": (
                report["acceptance_rate_ci"]["mean"]
                - confidence_interval(baseline_acceptance_rates)["mean"]
            ),
            "improvement_lift": (
                report["improvement_ci"]["mean"]
                - confidence_interval(baseline_improvements)["mean"]
            ),
            "baseline_runs_detail": baseline_summaries,
        }
    return report


@dataclass
class ResearchEvaluator:
    """Run multi-seed self-improvement and produce statistical evidence."""

    seeds: list[int]
    iterations: int
    output_dir: Path
    max_patch_attempts: int = 2
    agent: Any | None = None
    baseline_agent: Any | None = None
    strategy_evaluator: Any | None = None
    task_ids: list[str] | None = None
    run_summaries: list[dict[str, Any]] = field(default_factory=list)

    async def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, Any]] = []
        baseline_summaries: list[dict[str, Any]] = []
        for seed in self.seeds:
            run_dir = self.output_dir / f"seed_{seed}"
            runner = SelfImprovementRunner(
                registry_path=run_dir / "strategy_registry.json",
                metrics_path=run_dir / "metrics.json",
                agent=self.agent,
                strategy_evaluator=self.strategy_evaluator,
                seed=seed,
            )
            summary = await runner.run(
                iterations=self.iterations,
                task_ids=self.task_ids,
                max_patch_attempts=self.max_patch_attempts,
            )
            summaries.append(summary)
            if self.baseline_agent is not None:
                baseline_run_dir = self.output_dir / f"baseline_seed_{seed}"
                baseline_runner = SelfImprovementRunner(
                    registry_path=baseline_run_dir / "strategy_registry.json",
                    metrics_path=baseline_run_dir / "metrics.json",
                    agent=self.baseline_agent,
                    strategy_evaluator=self.strategy_evaluator,
                    seed=seed,
                )
                baseline_summaries.append(
                    await baseline_runner.run(
                        iterations=self.iterations,
                        task_ids=self.task_ids,
                        max_patch_attempts=self.max_patch_attempts,
                    )
                )
        report = summarize_runs(
            summaries,
            baseline_summaries=baseline_summaries if self.baseline_agent is not None else None,
        )
        report_path = self.output_dir / "research_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.run_summaries = summaries
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed GodelEnv research evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-patch-attempts", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "research_eval")
    parser.add_argument("--include-static-baseline", action="store_true")
    parser.add_argument("--provider-order", default=os.getenv("GODEL_PROVIDER_ORDER", ""))
    parser.add_argument("--agent-provider-order", default=os.getenv("GODEL_AGENT_PROVIDER_ORDER", ""))
    parser.add_argument("--verifier-provider-order", default=os.getenv("GODEL_VERIFIER_PROVIDER_ORDER", ""))
    parser.add_argument("--agent-model", default=os.getenv("GODEL_AGENT_MODEL_NAME", ""))
    parser.add_argument("--verifier-model", default=os.getenv("GODEL_VERIFIER_MODEL_NAME", ""))
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
    evaluator = ResearchEvaluator(
        seeds=args.seeds,
        iterations=args.iterations,
        output_dir=args.output_dir,
        max_patch_attempts=args.max_patch_attempts,
        baseline_agent=StaticBaselinePatchAgent() if args.include_static_baseline else None,
    )
    report = await evaluator.run()
    print(json.dumps(report, indent=2))
    return 0 if report["patches_proposed"] else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
