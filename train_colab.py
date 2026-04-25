# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GodelEnv 2.0 — Training Evidence Notebook
#
# This notebook demonstrates the full training pipeline for GodelEnv,
# a recursive self-improvement environment where an LLM proposes
# StrategyPatch mutations to its own reasoning policy.
#
# **Pipeline:**
# 1. Collect prompts from the deterministic environment
# 2. Warm-start a tiny language model with heuristic traces (SFT)
# 3. Refine with Group Relative Policy Optimization (GRPO)
# 4. Export loss / reward curves and before/after metrics
#
# **Requirements:** run this notebook from the repo root, or let the first cell
# clone and install the repo automatically when opened in Google Colab.

# %%
from pathlib import Path
import json
import os
import subprocess
import sys

from dotenv import load_dotenv


load_dotenv(override=False)


def _in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


if _in_colab() and not (Path.cwd() / "pyproject.toml").exists():
    repo_dir = Path("/content/GodelEnv")
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/dwan-ith/GodelEnv.git", str(repo_dir)],
            check=True,
        )
    os.chdir(repo_dir)

if _in_colab():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"],
        check=True,
    )


def _restart_with_utf8_on_windows() -> None:
    # TRL reads some packaged templates via Path.read_text() without an explicit
    # encoding; on Windows this can default to cp1252 and crash with UnicodeDecodeError.
    # The simplest robust fix is to restart with UTF-8 mode enabled.
    if os.name != "nt":
        return
    if os.environ.get("PYTHONUTF8") == "1":
        return

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run([sys.executable, *sys.argv], env=env, check=False)
    raise SystemExit(completed.returncode)

_restart_with_utf8_on_windows()

from godel_engine.rollout import collect_local_prompt_dataset
from godel_engine.training_support import (
    build_supervised_examples,
    build_tiny_model,
    evaluate_model,
    load_tokenizer,
    plot_before_after,
    plot_training_curves,
    run_grpo,
    run_sft,
)


# %% [markdown]
# ## 1. Configure the Run

# %%
TASKS = ["factual_qa", "alignment_qa", "reasoning", "strategy_optimization"]
NUM_PROMPTS = 16
SFT_STEPS = 20
GRPO_STEPS = 10
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128
OUTPUT_DIR = Path("artifacts/training_run")
GRADING_MODE = os.getenv("GODEL_GRADING_MODE", "deterministic")
STRATEGY_EVAL_MODE = os.getenv("GODEL_STRATEGY_EVAL_MODE", "deterministic")
os.environ["GODEL_GRADING_MODE"] = GRADING_MODE
os.environ["GODEL_STRATEGY_EVAL_MODE"] = STRATEGY_EVAL_MODE
print(f"Training config: {NUM_PROMPTS} prompts, {SFT_STEPS} SFT steps, {GRPO_STEPS} GRPO steps")
print(f"Modes: grading={GRADING_MODE}, strategy_eval={STRATEGY_EVAL_MODE}")


# %% [markdown]
# ## 2. Collect Prompts and Build the Warm-Start Dataset

# %%
prompt_data = collect_local_prompt_dataset(
    num_prompts=NUM_PROMPTS,
    tasks=TASKS,
    seed=42,
)
supervised_examples = build_supervised_examples(prompt_data)

print(f"Collected {len(prompt_data)} prompts across {len(TASKS)} task families")
for task in TASKS:
    count = sum(1 for p in prompt_data if p["task_type"] == task)
    print(f"  {task}: {count} prompts")


# %% [markdown]
# ## 3. Build the Tiny Local Model
#
# We use a 2-layer GPT-2 (128-dim embeddings) as a proof-of-concept.
# For production training, replace with Unsloth + Qwen2.5-7B on GPU.

# %%
tokenizer = load_tokenizer()
model = build_tiny_model(
    tokenizer,
    max_length=max(MAX_INPUT_LENGTH + MAX_NEW_TOKENS + 64, 384),
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: GPT-2 tiny ({total_params:,} parameters)")


# %% [markdown]
# ## 4. Baseline Evaluation (Before Training)

# %%
baseline_metrics = evaluate_model(
    model,
    tokenizer,
    prompt_data,
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_length=MAX_INPUT_LENGTH,
)
print(f"Baseline: mean_reward={baseline_metrics['mean_reward']:.4f}, mean_score={baseline_metrics['mean_score']:.4f}")
print(
    "Patch behavior: "
    f"proposal_rate={baseline_metrics.get('strategy_patch_rate', 0.0):.2%}, "
    f"acceptance_rate={baseline_metrics.get('patch_acceptance_rate', 0.0):.2%}, "
    f"mean_patch_delta={baseline_metrics.get('mean_patch_improvement', 0.0):+.4f}"
)
for ep in baseline_metrics["episodes"][:4]:
    print(f"  {ep['task_type']:25s} score={ep['score']:.3f}  reward={ep['reward']:+.3f}")


# %% [markdown]
# ## 5. Warm-Start with Supervised Traces (SFT)
#
# We train the tiny model on heuristic-generated solutions to teach it the
# output format and basic task structure before reinforcement learning.

# %%
sft_logs = run_sft(
    model,
    tokenizer,
    supervised_examples,
    output_dir=OUTPUT_DIR / "sft",
    max_steps=SFT_STEPS,
    batch_size=1,
    max_length=MAX_INPUT_LENGTH,
    use_cpu=True,
)
sft_loss_entries = [item for item in sft_logs if "loss" in item]
print(f"SFT complete: {len(sft_loss_entries)} steps logged")
if sft_loss_entries:
    print(f"  Initial loss: {sft_loss_entries[0]['loss']:.4f}")
    print(f"  Final loss:   {sft_loss_entries[-1]['loss']:.4f}")


# %% [markdown]
# ## 6. Refine with Group Relative Policy Optimization (GRPO)
#
# GRPO generates multiple completions per prompt, scores each via the
# environment's multi-channel reward (task_score_delta, format_compliance,
# anti_hack_penalty, patch_quality), and updates the policy to favor
# higher-reward completions.

# %%
grpo_logs = run_grpo(
    model,
    tokenizer,
    prompt_data,
    output_dir=OUTPUT_DIR / "grpo",
    max_steps=GRPO_STEPS,
    batch_size=2,
    num_generations=2,
    max_completion_length=MAX_NEW_TOKENS,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cpu=True,
)
grpo_reward_entries = [item for item in grpo_logs if "reward" in item]
print(f"GRPO complete: {len(grpo_reward_entries)} steps logged")
if grpo_reward_entries:
    print(f"  Initial reward: {grpo_reward_entries[0]['reward']:.4f}")
    print(f"  Final reward:   {grpo_reward_entries[-1]['reward']:.4f}")


# %% [markdown]
# ## 7. Evaluate the Trained Model

# %%
trained_metrics = evaluate_model(
    model,
    tokenizer,
    prompt_data,
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_length=MAX_INPUT_LENGTH,
)
print(f"Trained: mean_reward={trained_metrics['mean_reward']:.4f}, mean_score={trained_metrics['mean_score']:.4f}")
print(f"\nImprovement:")
print(f"  Reward: {baseline_metrics['mean_reward']:.4f} → {trained_metrics['mean_reward']:.4f} "
      f"(Δ={trained_metrics['mean_reward'] - baseline_metrics['mean_reward']:+.4f})")
print(f"  Score:  {baseline_metrics['mean_score']:.4f} → {trained_metrics['mean_score']:.4f} "
      f"(Δ={trained_metrics['mean_score'] - baseline_metrics['mean_score']:+.4f})")
print(
    "  Patch proposal rate: "
    f"{baseline_metrics.get('strategy_patch_rate', 0.0):.2%} → {trained_metrics.get('strategy_patch_rate', 0.0):.2%}"
)
print(
    "  Patch acceptance rate: "
    f"{baseline_metrics.get('patch_acceptance_rate', 0.0):.2%} → {trained_metrics.get('patch_acceptance_rate', 0.0):.2%}"
)
print(
    "  Mean patch delta: "
    f"{baseline_metrics.get('mean_patch_improvement', 0.0):+.4f} → {trained_metrics.get('mean_patch_improvement', 0.0):+.4f}"
)


# %% [markdown]
# ## 8. Save Plots and Model Artifacts

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plots = plot_training_curves(sft_logs, grpo_logs, OUTPUT_DIR)
before_after_path = plot_before_after(baseline_metrics, trained_metrics, OUTPUT_DIR)

model_dir = OUTPUT_DIR / "final_model"
model_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

summary = {
    "tasks": TASKS,
    "prompt_count": len(prompt_data),
    "sft_steps": SFT_STEPS,
    "grpo_steps": GRPO_STEPS,
    "baseline": baseline_metrics,
    "trained": trained_metrics,
    "improvement": {
        "reward_delta": trained_metrics["mean_reward"] - baseline_metrics["mean_reward"],
        "score_delta": trained_metrics["mean_score"] - baseline_metrics["mean_score"],
        "patch_rate_delta": trained_metrics.get("strategy_patch_rate", 0.0) - baseline_metrics.get("strategy_patch_rate", 0.0),
        "patch_acceptance_delta": trained_metrics.get("patch_acceptance_rate", 0.0) - baseline_metrics.get("patch_acceptance_rate", 0.0),
        "mean_patch_improvement_delta": trained_metrics.get("mean_patch_improvement", 0.0) - baseline_metrics.get("mean_patch_improvement", 0.0),
    },
    "artifacts": {
        "loss_curve": str(plots["loss_curve"]),
        "reward_curve": str(plots["reward_curve"]),
        "before_after": str(before_after_path),
        "model_dir": str(model_dir),
    },
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Artifacts saved:")
for name, path in summary["artifacts"].items():
    print(f"  {name}: {path}")


# %% [markdown]
# ## 9. Training Evidence — Inline Plots

# %%
from IPython.display import Image, display

print("=== SFT Loss Curve ===")
display(Image(filename=str(plots["loss_curve"])))
print("\n=== GRPO Reward Curve ===")
display(Image(filename=str(plots["reward_curve"])))
print("\n=== Before / After Training ===")
display(Image(filename=str(before_after_path)))
