"""
Local Godel Env training pipeline.

GodelEnv trains models to generate full JSON action completions, enabling
genuine end-to-end learning of recursive self-improvement.

The model learns to:
1. Generate valid JSON actions with solution, edit_type, and strategy_note
2. Propose strategy patches with improved_strategy, hypothesis, and target_weaknesses
3. Demonstrate proposed strategies through concrete task solutions

Evidence quality depends on:
- grading_mode: 'auto' (uses LLM) vs 'deterministic' (uses heuristic graders)
- strategy_eval_mode: 'auto' (uses LLM) vs 'deterministic' (uses heuristic solvers)

The training path defaults to deterministic grading so the committed evidence is
reproducible even if stale API credentials are present in the shell. Set
`--grading-mode auto --strategy-eval-mode auto` to use a configured LLM provider.

For the strongest evidence (genuine recursive self-improvement):
  python train.py --grading-mode auto --strategy-eval-mode auto
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GRPO policy on Godel Env")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "training_run")
    parser.add_argument(
        "--tasks",
        type=str,
        default="factual_qa,alignment_qa,reasoning,strategy_optimization",
        help="Comma-separated task list for the demo training run.",
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--sft-steps", type=int, default=20)
    parser.add_argument("--grpo-steps", type=int, default=10)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=224)
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

    from godel_engine.rollout import collect_local_prompt_dataset
    from godel_engine.training_support import (
        build_freeform_model,
        build_supervised_examples_freeform,
        evaluate_model_freeform,
        load_tokenizer,
        plot_before_after,
        plot_training_curves,
        run_grpo,
        run_sft,
        warn_evidence_quality,
    )

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GODEL_GRADING_MODE"] = args.grading_mode
    os.environ["GODEL_STRATEGY_EVAL_MODE"] = args.strategy_eval_mode

    # Warn about evidence quality based on configuration
    generation_mode = getattr(args, "generation_mode", "freeform")
    evidence_quality = warn_evidence_quality(
        generation_mode=generation_mode,
        grading_mode=args.grading_mode,
        strategy_eval_mode=args.strategy_eval_mode,
    )
    logger.info("Evidence quality: %s (generation=%s, grading=%s, strategy_eval=%s)",
                evidence_quality, generation_mode, args.grading_mode, args.strategy_eval_mode)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    set_seed(42)

    logger.info("Collecting prompts from the local environment")
    prompt_data = collect_local_prompt_dataset(
        num_prompts=args.num_prompts,
        tasks=task_names,
        seed=42,
    )
    
    is_freeform = True

    tokenizer = load_tokenizer()
    supervised_examples = build_supervised_examples_freeform(prompt_data)

    logger.info("Using freeform JSON generation model")
    model = build_freeform_model(
        tokenizer,
        max_length=max(args.max_input_length + args.max_new_tokens + 256, 1024),
    )
    effective_max_new_tokens = max(args.max_new_tokens, 160)

    eval_func = evaluate_model_freeform

    baseline_metrics = eval_func(
        model,
        tokenizer,
        prompt_data,
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
            "tasks": task_names,
            "baseline": baseline_metrics,
            "generation_mode": generation_mode,
            "evidence_quality": evidence_quality,
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
        use_cpu=True,
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
        use_cpu=True,
    )

    trained_metrics = eval_func(
        model,
        tokenizer,
        prompt_data,
        max_new_tokens=effective_max_new_tokens,
        max_input_length=args.max_input_length,
        policy_mode="model",
        seed=42,
    )
    logger.info(
        "Trained metrics: reward=%.4f score=%.4f (generation_mode=%s)",
        trained_metrics["mean_reward"],
        trained_metrics["mean_score"],
        generation_mode,
    )

    plots = plot_training_curves(sft_logs, grpo_logs, output_dir)
    before_after_path = plot_before_after(baseline_metrics, trained_metrics, output_dir)

    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Build evidence quality note for the summary
    evidence_note = None
    if evidence_quality != "genuine":
        evidence_note = (
            f"EVIDENCE QUALITY: {evidence_quality.upper()}. "
            f"generation_mode={generation_mode}, grading_mode={args.grading_mode}, "
            f"strategy_eval_mode={args.strategy_eval_mode}. "
            "For genuine evidence, use: --generation-mode freeform --grading-mode auto --strategy-eval-mode auto"
        )

    summary = {
        "tasks": task_names,
        "prompt_count": len(prompt_data),
        "generation_mode": generation_mode,
        "grading_mode": args.grading_mode,
        "strategy_eval_mode": args.strategy_eval_mode,
        "evidence_quality": evidence_quality,
        "evidence_note": evidence_note,
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "improvement": {
            "reward_delta": trained_metrics["mean_reward"] - baseline_metrics["mean_reward"],
            "score_delta": trained_metrics["mean_score"] - baseline_metrics["mean_score"],
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
        },
        "artifacts": {
            "loss_curve": str(plots["loss_curve"]),
            "reward_curve": str(plots["reward_curve"]),
            "before_after": str(before_after_path),
            "model_dir": str(final_model_dir),
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
