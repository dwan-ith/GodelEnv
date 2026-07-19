"""Re-evaluate saved GodelEnv adapters under the current evidence protocol."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate base, SFT, and GRPO checkpoints without retraining."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training_run"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    return parser.parse_args()


def _load_base(model_name: str, tokenizer, device: str):
    from transformers import AutoModelForCausalLM
    from godel_engine.training_support import (
        _cached_model_source,
        _ensure_tokenizer_capacity,
    )

    local_source = _cached_model_source(model_name)
    if local_source is not None:
        model = AutoModelForCausalLM.from_pretrained(
            local_source,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    _ensure_tokenizer_capacity(model, tokenizer)
    return model.to(device)


def _load_adapter(model_name: str, tokenizer, adapter_dir: Path, device: str):
    from peft import PeftModel

    base = _load_base(model_name, tokenizer, device)
    return PeftModel.from_pretrained(base, adapter_dir).to(device)


def _release(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _improvement(baseline: dict[str, Any], trained: dict[str, Any], comparison: dict[str, Any]):
    return {
        **comparison,
        "structured_action_rate_delta": trained.get("structured_action_rate", 0.0)
        - baseline.get("structured_action_rate", 0.0),
        "json_action_rate_delta": trained.get("json_action_rate", 0.0)
        - baseline.get("json_action_rate", 0.0),
        "patch_rate_delta": trained.get("strategy_patch_rate", 0.0)
        - baseline.get("strategy_patch_rate", 0.0),
        "patch_acceptance_delta": trained.get("patch_acceptance_rate", 0.0)
        - baseline.get("patch_acceptance_rate", 0.0),
        "mean_patch_improvement_delta": trained.get("mean_patch_improvement", 0.0)
        - baseline.get("mean_patch_improvement", 0.0),
        "environment_patch_rate_delta": trained.get("environment_patch_rate", 0.0)
        - baseline.get("environment_patch_rate", 0.0),
        "environment_patch_acceptance_delta": trained.get(
            "environment_patch_acceptance_rate", 0.0
        )
        - baseline.get("environment_patch_acceptance_rate", 0.0),
        "environment_learning_value_delta": trained.get(
            "mean_environment_learning_value", 0.0
        )
        - baseline.get("mean_environment_learning_value", 0.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from godel_engine.rollout import collect_train_eval_prompt_datasets
    from godel_engine.training_support import (
        compare_paired_evaluations,
        evaluate_model_freeform,
        load_tokenizer,
        plot_before_after,
        select_and_gate_candidate,
    )

    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_name = str(metrics["base_model"])
    tasks = [str(item) for item in metrics["tasks"]]
    prompt_count = int(metrics["prompt_count"])
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    os.environ["GODEL_BASE_MODEL"] = model_name
    os.environ["GODEL_GRADING_MODE"] = str(metrics.get("grading_mode", "deterministic"))
    os.environ["GODEL_STRATEGY_EVAL_MODE"] = str(
        metrics.get("strategy_eval_mode", "deterministic")
    )
    os.environ.setdefault("GODEL_ALLOW_DETERMINISTIC_FALLBACK", "1")

    _, eval_data, split = collect_train_eval_prompt_datasets(
        num_train_prompts=prompt_count,
        tasks=tasks,
        eval_fraction=0.25,
        seed=42,
    )
    if split != metrics["data_split"]:
        raise RuntimeError("Saved data split does not match the reproducible split")

    tokenizer = load_tokenizer()

    def evaluate(model) -> dict[str, Any]:
        return evaluate_model_freeform(
            model,
            tokenizer,
            eval_data,
            max_new_tokens=args.max_new_tokens,
            max_input_length=args.max_input_length,
            policy_mode="model",
            seed=42,
        )

    base_model = _load_base(model_name, tokenizer, device)
    baseline = evaluate(base_model)
    _release(base_model)

    candidates: dict[str, dict[str, Any]] = {}
    for stage in ("sft", "grpo"):
        adapter_dir = output_dir / f"candidate_{stage}"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Missing saved adapter: {adapter_dir}")
        model = _load_adapter(model_name, tokenizer, adapter_dir, device)
        candidates[stage] = evaluate(model)
        _release(model)

    promotion = select_and_gate_candidate(baseline, candidates)
    trained = candidates[promotion["selected_stage"]]
    comparisons = {
        name: compare_paired_evaluations(baseline, candidate)
        for name, candidate in candidates.items()
    }

    final_model_dir = output_dir / "final_model"
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)
    if promotion["promoted"]:
        shutil.copytree(
            output_dir / f"candidate_{promotion['selected_stage']}", final_model_dir
        )

    before_after = plot_before_after(baseline, trained, output_dir)
    evidence_note = (
        "Held-out capability improved without material task-family regression; "
        "the selected checkpoint was promoted."
        if promotion["promoted"]
        else "No checkpoint was promoted because every candidate failed the no-regression gate."
    )
    if promotion["promoted"] and not promotion["recursive_evidence"]:
        evidence_note += (
            " This run demonstrates task-policy improvement but not recursive strategy-patch "
            "improvement; no accepted positive patch was observed."
        )
    if promotion["promoted"] and not promotion["environment_recursive_evidence"]:
        evidence_note += (
            " It also does not demonstrate learned environment evolution; no accepted "
            "positive environment patch was observed from the trained policy."
        )
    if promotion["promoted"] and promotion["coevolution_evidence"]:
        evidence_note += (
            " The promoted policy produced accepted positive strategy and environment "
            "mutations, so this run meets the verified coevolution evidence gate."
        )

    metrics.update(
        {
            "evidence_quality": promotion["evidence_label"],
            "evidence_note": evidence_note,
            "evaluation_protocol": {
                "paired_examples": True,
                "fixed_strategy_bundle_key": "training-eval:42:{task_type}:{task_id}",
                "task_regression_tolerance": 0.02,
                "max_regressed_fraction": 0.25,
            },
            "baseline": baseline,
            "sft": candidates["sft"],
            "grpo": candidates["grpo"],
            "trained": trained,
            "stage_comparisons": comparisons,
            "candidate_metrics": candidates,
            "promotion": promotion,
            "improvement": _improvement(baseline, trained, promotion["comparison"]),
        }
    )
    metrics["artifacts"]["before_after"] = str(before_after)
    metrics["artifacts"]["model_dir"] = str(final_model_dir) if promotion["promoted"] else None
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
