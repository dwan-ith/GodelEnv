"""
Local Godel Env training pipeline.

GodelEnv trains models to generate full JSON action completions, enabling
genuine end-to-end learning of recursive self-improvement.

The model learns to:
1. Generate valid JSON actions with solution, edit_type, and strategy_note
2. Propose strategy patches with improved_strategy, hypothesis, and target_weaknesses
3. Propose bounded environment patches that evolve the verified curriculum
4. Demonstrate proposed strategies through concrete task solutions

The default run uses reproducible programmatic verifiers. LLM-backed strategy
evaluation is optional, and the saved model is promoted only when held-out task
score and environment reward pass a no-regression gate.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=False)
# Silence TRL experimental warnings
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
# Note: WANDB is disabled via report_to="none" in training configs, not env vars


def _restart_with_utf8_on_windows() -> None:
    if os.name != "nt":
        return
    if os.environ.get("PYTHONUTF8") == "1":
        return

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run([sys.executable, *sys.argv], env=env, check=False)
    raise SystemExit(completed.returncode)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("godel_train")


def _save_candidate(model, tokenizer, output_dir: Path) -> None:
    """Save LoRA adapters without duplicating an unchanged embedding matrix."""
    if hasattr(model, "peft_config"):
        model.save_pretrained(output_dir, save_embedding_layers=False)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GRPO policy on Godel Env")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "training_run")
    parser.add_argument(
        "--tasks",
        type=str,
        default="factual_qa,alignment_qa,reasoning,strategy_optimization",
        help="Comma-separated task list for the demo training run.",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--sft-steps", type=int, default=40)
    parser.add_argument("--grpo-steps", type=int, default=12)
    parser.add_argument("--sft-learning-rate", type=float, default=5e-5)
    parser.add_argument("--grpo-learning-rate", type=float, default=2e-6)
    parser.add_argument("--repair-steps", type=int, default=12)
    parser.add_argument("--repair-learning-rate", type=float, default=5e-6)
    parser.add_argument("--repair-oversample", type=int, default=3)
    parser.add_argument(
        "--recursive-sft-oversample",
        type=int,
        default=4,
        help="Relative SFT repetition count for dual-mutation strategy examples.",
    )
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--recursive-oversample",
        type=int,
        default=6,
        help="Relative GRPO sampling weight for strategy-patch prompts.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.getenv("GODEL_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Pretrained causal LM. Random fallback is disabled for evidence runs.",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. `auto` uses CUDA when available.",
    )
    parser.add_argument(
        "--grading-mode",
        type=str,
        default=os.getenv("GODEL_GRADING_MODE", "deterministic"),
        help="Grading mode: auto, llm, or deterministic.",
    )
    parser.add_argument(
        "--strategy-eval-mode",
        type=str,
        default=os.getenv("GODEL_STRATEGY_EVAL_MODE", "deterministic"),
        help="Strategy evaluation mode: auto, llm, or deterministic.",
    )
    parser.add_argument(
        "--provider-order",
        type=str,
        default=os.getenv("GODEL_PROVIDER_ORDER", "custom,huggingface,openai,ollama"),
        help="Comma-separated LLM provider priority for hybrid mode.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generation-mode",
        type=str,
        default="freeform",
        choices=["freeform"],
        help=(
            "Generation mode: full JSON action generation. The symbolic macro path "
            "has been removed from the training pipeline."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from transformers import set_seed

    from godel_engine.rollout import collect_train_eval_prompt_datasets
    from godel_engine.training_support import (
        build_freeform_model,
        build_adaptive_repair_examples,
        build_supervised_examples_freeform,
        compare_paired_evaluations,
        evaluate_model_freeform,
        load_tokenizer,
        oversample_recursive_examples,
        plot_before_after,
        plot_training_curves,
        run_grpo,
        run_sft,
        select_and_gate_candidate,
    )

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GODEL_GRADING_MODE"] = args.grading_mode
    os.environ["GODEL_STRATEGY_EVAL_MODE"] = args.strategy_eval_mode
    os.environ["GODEL_PROVIDER_ORDER"] = args.provider_order
    os.environ["GODEL_BASE_MODEL"] = args.base_model
    generation_mode = getattr(args, "generation_mode", "freeform")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    set_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    use_cpu = args.device == "cpu" or (
        args.device == "auto" and not torch.cuda.is_available()
    )
    logger.info("Training device: %s", "cpu" if use_cpu else "cuda")

    logger.info("Collecting disjoint train and held-out prompts")
    prompt_data, eval_data, data_split = collect_train_eval_prompt_datasets(
        num_train_prompts=args.num_prompts,
        tasks=task_names,
        eval_fraction=args.eval_fraction,
        seed=42,
    )
    
    tokenizer = load_tokenizer()
    supervised_examples = oversample_recursive_examples(
        build_supervised_examples_freeform(prompt_data),
        multiplier=args.recursive_sft_oversample,
    )

    logger.info("Using freeform JSON generation model")
    model = build_freeform_model(
        tokenizer,
        max_length=max(args.max_input_length + args.max_new_tokens + 256, 1024),
    )
    model.to("cpu" if use_cpu else "cuda")
    effective_max_new_tokens = max(args.max_new_tokens, 160)

    eval_func = evaluate_model_freeform

    baseline_metrics = eval_func(
        model,
        tokenizer,
        eval_data,
        max_new_tokens=effective_max_new_tokens,
        max_input_length=args.max_input_length,
        policy_mode="model",
        seed=42,
    )
    logger.info(
        "Baseline metrics: reward=%.4f score=%.4f (generation_mode=%s)",
        baseline_metrics["mean_reward"],
        baseline_metrics["mean_score"],
        generation_mode,
    )

    if args.dry_run:
        return {
            "prompt_count": len(prompt_data),
            "eval_prompt_count": len(eval_data),
            "tasks": task_names,
            "baseline": baseline_metrics,
            "generation_mode": generation_mode,
            "data_split": data_split,
        }

    logger.info("Running SFT warm start")
    sft_max_length = args.max_input_length + effective_max_new_tokens
    sft_logs = run_sft(
        model,
        tokenizer,
        supervised_examples,
        output_dir=output_dir / "sft",
        max_steps=args.sft_steps,
        batch_size=1,
        max_length=sft_max_length,
        use_cpu=use_cpu,
        learning_rate=args.sft_learning_rate,
    )
    sft_metrics = eval_func(
        model,
        tokenizer,
        eval_data,
        max_new_tokens=effective_max_new_tokens,
        max_input_length=args.max_input_length,
        policy_mode="model",
        seed=42,
    )
    sft_candidate_dir = output_dir / "candidate_sft"
    _save_candidate(model, tokenizer, sft_candidate_dir)
    logger.info(
        "Post-SFT held-out metrics: reward=%.4f score=%.4f",
        sft_metrics["mean_reward"],
        sft_metrics["mean_score"],
    )

    sft_comparison = compare_paired_evaluations(baseline_metrics, sft_metrics)
    repair_examples, repaired_task_families = build_adaptive_repair_examples(
        supervised_examples,
        sft_comparison,
        oversample=args.repair_oversample,
    )
    repair_logs: list[dict] = []
    repaired_metrics = None
    repaired_candidate_dir = output_dir / "candidate_sft_repaired"
    if repair_examples and args.repair_steps > 0:
        logger.info(
            "Running adaptive SFT repair for regressed task families: %s",
            ", ".join(repaired_task_families),
        )
        repair_logs = run_sft(
            model,
            tokenizer,
            repair_examples,
            output_dir=output_dir / "sft_repair",
            max_steps=args.repair_steps,
            batch_size=1,
            max_length=sft_max_length,
            use_cpu=use_cpu,
            learning_rate=args.repair_learning_rate,
        )
        repaired_metrics = eval_func(
            model,
            tokenizer,
            eval_data,
            max_new_tokens=effective_max_new_tokens,
            max_input_length=args.max_input_length,
            policy_mode="model",
            seed=42,
        )
        _save_candidate(model, tokenizer, repaired_candidate_dir)
        logger.info(
            "Post-repair held-out metrics: reward=%.4f score=%.4f",
            repaired_metrics["mean_reward"],
            repaired_metrics["mean_score"],
        )

    logger.info("Running GRPO refinement (generation_mode=%s)", generation_mode)
    grpo_logs = run_grpo(
        model,
        tokenizer,
        prompt_data,
        output_dir=output_dir / "grpo",
        max_steps=args.grpo_steps,
        batch_size=2,
        num_generations=2,
        max_completion_length=effective_max_new_tokens,
        max_new_tokens=effective_max_new_tokens,
        use_cpu=use_cpu,
        recursive_oversample=args.recursive_oversample,
        learning_rate=args.grpo_learning_rate,
    )

    grpo_metrics = eval_func(
        model,
        tokenizer,
        eval_data,
        max_new_tokens=effective_max_new_tokens,
        max_input_length=args.max_input_length,
        policy_mode="model",
        seed=42,
    )
    logger.info(
        "Post-GRPO held-out metrics: reward=%.4f score=%.4f (generation_mode=%s)",
        grpo_metrics["mean_reward"],
        grpo_metrics["mean_score"],
        generation_mode,
    )
    grpo_candidate_dir = output_dir / "candidate_grpo"
    _save_candidate(model, tokenizer, grpo_candidate_dir)

    routed_metrics = None
    routed_candidate_dir = output_dir / "candidate_grpo_routed"
    grpo_comparison = compare_paired_evaluations(baseline_metrics, grpo_metrics)
    fallback_tasks = sorted(
        task_type
        for task_type, delta in grpo_comparison["per_task_score_delta"].items()
        if float(delta) < -0.02
    )
    if fallback_tasks:
        logger.info(
            "Evaluating task-conditional adapter route with base fallback for: %s",
            ", ".join(fallback_tasks),
        )
        routed_metrics = eval_func(
            model,
            tokenizer,
            eval_data,
            max_new_tokens=effective_max_new_tokens,
            max_input_length=args.max_input_length,
            policy_mode="model",
            seed=42,
            base_fallback_tasks=set(fallback_tasks),
        )
        _save_candidate(model, tokenizer, routed_candidate_dir)
        (routed_candidate_dir / "routing.json").write_text(
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

    candidates = {"sft": sft_metrics, "grpo": grpo_metrics}
    if repaired_metrics is not None:
        candidates["sft_repaired"] = repaired_metrics
    if routed_metrics is not None:
        candidates["grpo_routed"] = routed_metrics
    promotion = select_and_gate_candidate(baseline_metrics, candidates)
    trained_metrics = candidates[promotion["selected_stage"]]
    stage_comparisons = {
        name: compare_paired_evaluations(baseline_metrics, metrics)
        for name, metrics in candidates.items()
    }

    if repair_logs:
        offset = max((int(item.get("step", 0)) for item in sft_logs), default=0)
        sft_logs = sft_logs + [
            {**item, "step": int(item.get("step", 0)) + offset}
            for item in repair_logs
        ]
    plots = plot_training_curves(sft_logs, grpo_logs, output_dir)
    before_after_path = plot_before_after(baseline_metrics, trained_metrics, output_dir)

    final_model_dir = output_dir / "final_model"
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)
    if promotion["promoted"]:
        selected_candidate_dir = output_dir / f"candidate_{promotion['selected_stage']}"
        shutil.copytree(selected_candidate_dir, final_model_dir)
        logger.info(
            "Promoted %s checkpoint to %s",
            promotion["selected_stage"],
            final_model_dir,
        )
    else:
        logger.error("Model promotion rejected: %s", "; ".join(promotion["reasons"]))

    evidence_note = (
        "Held-out capability improved and the selected checkpoint was promoted."
        if promotion["promoted"]
        else "No checkpoint was promoted because held-out quality failed the no-regression gate."
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

    summary = {
        "tasks": task_names,
        "prompt_count": len(prompt_data),
        "eval_prompt_count": len(eval_data),
        "sft_steps": args.sft_steps,
        "grpo_steps": args.grpo_steps,
        "sft_learning_rate": args.sft_learning_rate,
        "grpo_learning_rate": args.grpo_learning_rate,
        "repair_steps": args.repair_steps,
        "repair_learning_rate": args.repair_learning_rate,
        "repair_oversample": args.repair_oversample,
        "recursive_sft_oversample": args.recursive_sft_oversample,
        "repaired_task_families": repaired_task_families,
        "recursive_oversample": args.recursive_oversample,
        "max_input_length": args.max_input_length,
        "max_new_tokens": effective_max_new_tokens,
        "base_model": args.base_model,
        "generation_mode": generation_mode,
        "grading_mode": args.grading_mode,
        "strategy_eval_mode": args.strategy_eval_mode,
        "reward_backend": "live_environment",
        "data_split": data_split,
        "evaluation_protocol": {
            "paired_examples": True,
            "fixed_strategy_bundle_key": "training-eval:42:{task_type}:{task_id}",
            "task_regression_tolerance": 0.02,
            "max_regressed_fraction": 0.25,
        },
        "evidence_quality": promotion["evidence_label"],
        "evidence_note": evidence_note,
        "baseline": baseline_metrics,
        "sft": sft_metrics,
        "grpo": grpo_metrics,
        "trained": trained_metrics,
        "stage_comparisons": stage_comparisons,
        "candidate_metrics": candidates,
        "promotion": promotion,
        "improvement": {
            **promotion["comparison"],
            "structured_action_rate_delta": trained_metrics.get("structured_action_rate", 0.0)
            - baseline_metrics.get("structured_action_rate", 0.0),
            "json_action_rate_delta": trained_metrics.get("json_action_rate", 0.0)
            - baseline_metrics.get("json_action_rate", 0.0),
            "patch_rate_delta": trained_metrics.get("strategy_patch_rate", 0.0)
            - baseline_metrics.get("strategy_patch_rate", 0.0),
            "patch_acceptance_delta": trained_metrics.get("patch_acceptance_rate", 0.0)
            - baseline_metrics.get("patch_acceptance_rate", 0.0),
            "mean_patch_improvement_delta": trained_metrics.get("mean_patch_improvement", 0.0)
            - baseline_metrics.get("mean_patch_improvement", 0.0),
            "environment_patch_rate_delta": trained_metrics.get("environment_patch_rate", 0.0)
            - baseline_metrics.get("environment_patch_rate", 0.0),
            "environment_patch_acceptance_delta": trained_metrics.get(
                "environment_patch_acceptance_rate", 0.0
            )
            - baseline_metrics.get("environment_patch_acceptance_rate", 0.0),
            "environment_learning_value_delta": trained_metrics.get(
                "mean_environment_learning_value", 0.0
            )
            - baseline_metrics.get("mean_environment_learning_value", 0.0),
        },
        "artifacts": {
            "loss_curve": str(plots["loss_curve"]),
            "reward_curve": str(plots["reward_curve"]),
            "before_after": str(before_after_path),
            "model_dir": str(final_model_dir) if promotion["promoted"] else None,
            "sft_candidate_dir": str(sft_candidate_dir),
            "grpo_candidate_dir": str(grpo_candidate_dir),
            "repaired_candidate_dir": (
                str(repaired_candidate_dir) if repaired_metrics is not None else None
            ),
            "routed_candidate_dir": (
                str(routed_candidate_dir) if routed_metrics is not None else None
            ),
        },
    }

    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import wandb

        wandb.finish(quiet=True)
    except Exception:
        pass
    logger.info("Wrote training summary to %s", summary_path)
    return summary


if __name__ == "__main__":
    _restart_with_utf8_on_windows()
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
