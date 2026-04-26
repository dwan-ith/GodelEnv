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
# # GodelEnv
#
# This notebook runs one end-to-end training pass and writes the results to
# `artifacts/training_run`.
#
# The default mode is hybrid:
# - use an LLM provider for grading and strategy evaluation when credentials exist
# - fall back to deterministic scoring only when provider calls fail
#
# The run is sized for Colab T4 use. The notebook below is meant to read like a
# normal experiment log: configuration, run, evaluation, and saved artifacts.

# %%
from pathlib import Path
import json
import os
import subprocess
import sys

from dotenv import load_dotenv
import torch


load_dotenv(override=False)
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


def _in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def _load_colab_secret(env_name: str, *secret_names: str) -> None:
    if not _in_colab() or os.getenv(env_name):
        return
    try:
        from google.colab import userdata
    except Exception:
        return

    for secret_name in secret_names:
        try:
            value = userdata.get(secret_name)
        except Exception:
            value = None
        if value:
            os.environ[env_name] = value
            break


for env_name, secret_names in (
    ("OPENROUTER_API_KEY", ("OPENROUTER_API_KEY", "secretName")),
    ("OPENROUTER_MODEL_NAME", ("OPENROUTER_MODEL_NAME",)),
    ("OPENAI_API_KEY", ("OPENAI_API_KEY",)),
    ("OPENAI_MODEL_NAME", ("OPENAI_MODEL_NAME",)),
    ("HF_TOKEN", ("HF_TOKEN", "HF_API_KEY")),
    ("HF_MODEL_NAME", ("HF_MODEL_NAME",)),
    ("GODEL_PROVIDER_ORDER", ("GODEL_PROVIDER_ORDER",)),
    ("GODEL_BASE_MODEL", ("GODEL_BASE_MODEL",)),
):
    _load_colab_secret(env_name, *secret_names)


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

from godel_engine.rollout import collect_local_prompt_dataset
from godel_engine.training_support import (
    build_supervised_examples,
    build_freeform_model,
    evaluate_model,
    load_tokenizer,
    plot_before_after,
    plot_training_curves,
    run_grpo,
    run_sft,
)


# %% [markdown]
# ## Run Configuration
#
# These defaults are tuned for a quick Colab T4 evidence pass. Increase them if
# you want a longer run after the first successful notebook execution.

# %%
TASKS = ["factual_qa", "alignment_qa", "reasoning", "strategy_optimization"]
NUM_PROMPTS = 8
SFT_STEPS = 12
GRPO_STEPS = 4
MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 160
EFFECTIVE_MAX_NEW_TOKENS = max(MAX_NEW_TOKENS, 160)
OUTPUT_DIR = Path("artifacts/training_run")
BASE_MODEL = os.getenv("GODEL_BASE_MODEL", "gpt2")
PROVIDER_ORDER = os.getenv("GODEL_PROVIDER_ORDER", "custom,huggingface,openai,ollama")
GRADING_MODE = os.getenv("GODEL_GRADING_MODE", "auto")
STRATEGY_EVAL_MODE = os.getenv("GODEL_STRATEGY_EVAL_MODE", "auto")
USE_CPU = not torch.cuda.is_available()
os.environ["GODEL_GRADING_MODE"] = GRADING_MODE
os.environ["GODEL_STRATEGY_EVAL_MODE"] = STRATEGY_EVAL_MODE
os.environ["GODEL_BASE_MODEL"] = BASE_MODEL
os.environ["GODEL_PROVIDER_ORDER"] = PROVIDER_ORDER
print(f"Training config: {NUM_PROMPTS} prompts, {SFT_STEPS} SFT steps, {GRPO_STEPS} GRPO steps")
print(f"Effective completion length for GRPO/eval: {EFFECTIVE_MAX_NEW_TOKENS} tokens")
print(f"Base model: {BASE_MODEL}")
print(f"Provider order: {PROVIDER_ORDER}")
print(f"Modes: grading={GRADING_MODE}, strategy_eval={STRATEGY_EVAL_MODE}")
print(f"CUDA available: {torch.cuda.is_available()} | using_cpu={USE_CPU}")


# %% [markdown]
# ## Prompt Collection

# %%
prompt_data = collect_local_prompt_dataset(
    num_prompts=NUM_PROMPTS,
    tasks=TASKS,
    seed=42,
)
supervised_examples = build_supervised_examples(prompt_data)

print(f"Collected {len(prompt_data)} prompts across {len(TASKS)} task families")
for task in TASKS:
    count = sum(1 for prompt in prompt_data if prompt["task_type"] == task)
    print(f"  {task}: {count} prompts")


# %% [markdown]
# ## Model Setup

# %%
tokenizer = load_tokenizer()
model = build_freeform_model(
    tokenizer,
    max_length=max(MAX_INPUT_LENGTH + EFFECTIVE_MAX_NEW_TOKENS + 64, 1024),
)
total_params = sum(parameter.numel() for parameter in model.parameters())
print(f"Model: {BASE_MODEL} freeform ({total_params:,} parameters)")


# %% [markdown]
# ## Baseline Evaluation

# %%
baseline_metrics = evaluate_model(
    model,
    tokenizer,
    prompt_data,
    max_new_tokens=EFFECTIVE_MAX_NEW_TOKENS,
    max_input_length=MAX_INPUT_LENGTH,
    policy_mode="model",
    seed=42,
)
print(
    f"Baseline: mean_reward={baseline_metrics['mean_reward']:.4f}, "
    f"mean_score={baseline_metrics['mean_score']:.4f}"
)
print(
    "Structured policy: "
    f"structured_action_rate={baseline_metrics.get('structured_action_rate', 0.0):.2%}, "
    "Patch behavior: "
    f"proposal_rate={baseline_metrics.get('strategy_patch_rate', 0.0):.2%}, "
    f"acceptance_rate={baseline_metrics.get('patch_acceptance_rate', 0.0):.2%}, "
    f"mean_patch_delta={baseline_metrics.get('mean_patch_improvement', 0.0):+.4f}"
)
for episode in baseline_metrics["episodes"][:4]:
    print(
        f"  {episode['task_type']:25s} "
        f"score={episode['score']:.3f}  reward={episode['reward']:+.3f}"
    )


# %% [markdown]
# ## Supervised Warm Start

# %%
sft_logs = run_sft(
    model,
    tokenizer,
    supervised_examples,
    output_dir=OUTPUT_DIR / "sft",
    max_steps=SFT_STEPS,
    batch_size=1,
    max_length=MAX_INPUT_LENGTH + EFFECTIVE_MAX_NEW_TOKENS,
    use_cpu=USE_CPU,
)
sft_loss_entries = [item for item in sft_logs if "loss" in item]
print(f"SFT complete: {len(sft_loss_entries)} steps logged")
if sft_loss_entries:
    print(f"  Initial loss: {sft_loss_entries[0]['loss']:.4f}")
    print(f"  Final loss:   {sft_loss_entries[-1]['loss']:.4f}")


# %% [markdown]
# ## GRPO Refinement

# %%
grpo_logs = run_grpo(
    model,
    tokenizer,
    prompt_data,
    output_dir=OUTPUT_DIR / "grpo",
    max_steps=GRPO_STEPS,
    batch_size=2,
    num_generations=2,
    max_completion_length=EFFECTIVE_MAX_NEW_TOKENS,
    max_new_tokens=EFFECTIVE_MAX_NEW_TOKENS,
    use_cpu=USE_CPU,
)
grpo_reward_entries = [item for item in grpo_logs if "reward" in item]
print(f"GRPO complete: {len(grpo_reward_entries)} steps logged")
if grpo_reward_entries:
    print(f"  Initial reward: {grpo_reward_entries[0]['reward']:.4f}")
    print(f"  Final reward:   {grpo_reward_entries[-1]['reward']:.4f}")


# %% [markdown]
# ## Post-Training Evaluation

# %%
trained_metrics = evaluate_model(
    model,
    tokenizer,
    prompt_data,
    max_new_tokens=EFFECTIVE_MAX_NEW_TOKENS,
    max_input_length=MAX_INPUT_LENGTH,
    policy_mode="model",
    seed=42,
)
print(
    f"Trained: mean_reward={trained_metrics['mean_reward']:.4f}, "
    f"mean_score={trained_metrics['mean_score']:.4f}"
)
print("\nImprovement:")
print(
    f"  Reward: {baseline_metrics['mean_reward']:.4f} -> {trained_metrics['mean_reward']:.4f} "
    f"(delta={trained_metrics['mean_reward'] - baseline_metrics['mean_reward']:+.4f})"
)
print(
    f"  Score:  {baseline_metrics['mean_score']:.4f} -> {trained_metrics['mean_score']:.4f} "
    f"(delta={trained_metrics['mean_score'] - baseline_metrics['mean_score']:+.4f})"
)
print(
    "  Structured action rate: "
    f"{baseline_metrics.get('structured_action_rate', 0.0):.2%} -> "
    f"{trained_metrics.get('structured_action_rate', 0.0):.2%}"
)
print(
    "  Patch proposal rate: "
    f"{baseline_metrics.get('strategy_patch_rate', 0.0):.2%} -> "
    f"{trained_metrics.get('strategy_patch_rate', 0.0):.2%}"
)
print(
    "  Patch acceptance rate: "
    f"{baseline_metrics.get('patch_acceptance_rate', 0.0):.2%} -> "
    f"{trained_metrics.get('patch_acceptance_rate', 0.0):.2%}"
)
print(
    "  Mean patch delta: "
    f"{baseline_metrics.get('mean_patch_improvement', 0.0):+.4f} -> "
    f"{trained_metrics.get('mean_patch_improvement', 0.0):+.4f}"
)


# %% [markdown]
# ## Save Artifacts

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
        "structured_action_rate_delta": trained_metrics.get("structured_action_rate", 0.0)
        - baseline_metrics.get("structured_action_rate", 0.0),
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
        "model_dir": str(model_dir),
    },
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
try:
    import wandb

    wandb.finish(quiet=True)
except Exception:
    pass
print("Artifacts saved:")
for name, path in summary["artifacts"].items():
    print(f"  {name}: {path}")


# %% [markdown]
# ## Plots

# %%
from IPython.display import Image, display

print("=== SFT Loss Curve ===")
display(Image(filename=str(plots["loss_curve"])))
print("\n=== GRPO Reward Curve ===")
display(Image(filename=str(plots["reward_curve"])))
print("\n=== Before / After Training ===")
display(Image(filename=str(before_after_path)))
