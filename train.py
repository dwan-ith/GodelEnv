"""
Local Godel Env training pipeline.

This script supports two training modes:

1. SYMBOLIC mode (default, fast demo):
   - Model learns to emit one of ~3 compact action tokens
   - Tokens expand to handcrafted solutions via heuristic_policy
   - Evidence quality: PARTIAL - measures policy selection over hardcoded macros
   - Use for: quick smoke tests, CI, demo runs

2. FREEFORM mode (slower, genuine evidence):
   - Model learns to generate full JSON action completions
   - No expansion through hardcoded heuristics
   - Evidence quality: GENUINE - measures actual learned generation
   - Use for: real training runs, publishable results

The training path defaults to deterministic grading so the committed evidence is
reproducible even if stale API credentials are present in the shell. Set
`--grading-mode auto --strategy-eval-mode auto` to use a configured LLM provider.

For the strongest evidence (genuine recursive self-improvement):
  python train.py --generation-mode freeform --grading-mode auto --strategy-eval-mode auto
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
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


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
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1)
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
        default="symbolic",
        choices=["symbolic", "freeform"],
        help=(
            "Generation mode: 'symbolic' trains token classification over ~3 action macros "
            "(fast, partial evidence); 'freeform' trains actual JSON generation (slower, "
            "genuine evidence). Default: symbolic."
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
        build_supervised_examples,
        build_supervised_examples_freeform,
        build_tiny_model,
        evaluate_model,
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
    generation_mode = getattr(args, "generation_mode", "symbolic")
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
    
    # Build examples and model based on generation mode
    is_freeform = generation_mode == "freeform"
    
    if is_freeform:
        logger.info("Using FREEFORM mode: training actual JSON generation")
        supervised_examples = build_supervised_examples_freeform(prompt_data)
        tokenizer = load_tokenizer()
        model = build_freeform_model(
            tokenizer,
            max_length=max(args.max_input_length + args.max_new_tokens + 256, 1024),
        )
        eval_func = evaluate_model_freeform
        # Freeform needs more tokens to generate JSON
        effective_max_new_tokens = max(args.max_new_tokens, 256)
    else:
        logger.info("Using SYMBOLIC mode: training compact token classification")
        supervised_examples = build_supervised_examples(prompt_data)
        tokenizer = load_tokenizer()
        model = build_tiny_model(
            tokenizer,
            max_length=max(args.max_input_length + args.max_new_tokens + 128, 768),
        )
        eval_func = evaluate_model
        effective_max_new_tokens = args.max_new_tokens

    baseline_metrics = eval_func(
        model,
        tokenizer,
        prompt_data,
        max_new_tokens=effective_max_new_tokens,
        max_input_length=args.max_input_length,
        policy_mode="random",
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
    # For freeform, we need longer max_length to fit JSON completions
    sft_max_length = args.max_input_length + effective_max_new_tokens if is_freeform else args.max_input_length
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
        generation_mode=generation_mode,
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
            - baseline_metrics.get("json_action_rate", 0.0) if is_freeform else None,
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
