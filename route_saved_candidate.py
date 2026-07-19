"""Build and evaluate a safe task-conditional route from a saved LoRA candidate."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", default="grpo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    from godel_engine.rollout import collect_train_eval_prompt_datasets
    from godel_engine.training_support import (
        compare_paired_evaluations,
        evaluate_model_freeform,
        load_tokenizer,
        plot_before_after,
        select_and_gate_candidate,
    )
    from reevaluate import _improvement, _load_adapter, _release

    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stage = args.stage
    baseline = metrics["baseline"]
    stage_metrics = metrics[stage]
    comparison = compare_paired_evaluations(baseline, stage_metrics)
    fallback_tasks = sorted(
        task_type
        for task_type, delta in comparison["per_task_score_delta"].items()
        if float(delta) < -0.02
    )
    if not fallback_tasks:
        raise RuntimeError(f"{stage} has no regressed task family to route")

    model_name = str(metrics["base_model"])
    os.environ["GODEL_BASE_MODEL"] = model_name
    os.environ["GODEL_GRADING_MODE"] = str(metrics["grading_mode"])
    os.environ["GODEL_STRATEGY_EVAL_MODE"] = str(metrics["strategy_eval_mode"])
    _, eval_data, split = collect_train_eval_prompt_datasets(
        num_train_prompts=int(metrics["prompt_count"]),
        tasks=list(metrics["tasks"]),
        eval_fraction=0.25,
        seed=42,
    )
    if split != metrics["data_split"]:
        raise RuntimeError("Saved split does not match current deterministic split")

    tokenizer = load_tokenizer()
    source_dir = output_dir / f"candidate_{stage}"
    model = _load_adapter(model_name, tokenizer, source_dir, args.device)
    routed = evaluate_model_freeform(
        model,
        tokenizer,
        eval_data,
        max_new_tokens=int(metrics.get("max_new_tokens", 320)),
        max_input_length=int(metrics.get("max_input_length", 768)),
        policy_mode="model",
        seed=42,
        base_fallback_tasks=set(fallback_tasks),
    )
    _release(model)

    routed_stage = f"{stage}_routed"
    routed_dir = output_dir / f"candidate_{routed_stage}"
    if routed_dir.exists():
        shutil.rmtree(routed_dir)
    shutil.copytree(source_dir, routed_dir)
    (routed_dir / "routing.json").write_text(
        json.dumps(
            {
                "type": "task_conditional_lora",
                "base_fallback_tasks": fallback_tasks,
                "task_regression_tolerance": 0.02,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    candidates = dict(metrics.get("candidate_metrics", {}))
    if not candidates:
        candidates = {
            name: metrics[name]
            for name in ("sft", "grpo")
            if name in metrics
        }
    candidates[routed_stage] = routed
    promotion = select_and_gate_candidate(baseline, candidates)
    selected = candidates[promotion["selected_stage"]]
    final_model_dir = output_dir / "final_model"
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)
    if promotion["promoted"]:
        shutil.copytree(
            output_dir / f"candidate_{promotion['selected_stage']}", final_model_dir
        )

    comparisons = {
        name: compare_paired_evaluations(baseline, candidate)
        for name, candidate in candidates.items()
    }
    metrics[routed_stage] = routed
    metrics["candidate_metrics"] = candidates
    metrics["stage_comparisons"] = comparisons
    metrics["promotion"] = promotion
    metrics["trained"] = selected
    metrics["improvement"] = _improvement(baseline, selected, promotion["comparison"])
    metrics["evidence_quality"] = promotion["evidence_label"]
    metrics["evidence_note"] = (
        "A task-conditional LoRA route passed the no-regression gate; regressed families "
        "use untouched base weights."
        if promotion["promoted"]
        else "No candidate or routed candidate passed the no-regression gate."
    )
    if promotion["promoted"] and promotion["coevolution_evidence"]:
        metrics["evidence_note"] += (
            " The routed policy also passed both recursive mutation gates."
        )
    metrics["artifacts"]["model_dir"] = (
        str(final_model_dir) if promotion["promoted"] else None
    )
    metrics["artifacts"]["routed_candidate_dir"] = str(routed_dir)
    metrics["artifacts"]["before_after"] = str(
        plot_before_after(baseline, selected, output_dir)
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
