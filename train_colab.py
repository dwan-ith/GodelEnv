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
# # GodelEnv - Reproducible Coevolution SFT + GRPO Training
#
# This notebook uses the exact same pipeline as `train.py`:
#
# 1. Create task-stratified, disjoint train and held-out task-ID splits.
# 2. Measure the untouched base model on the held-out split.
# 3. Run completion-only SFT on reference-grounded dual-mutation JSON actions.
# 4. Repair regressed task families while replaying all anchor families.
# 5. Run GRPO with rewards computed by executing every completion in GodelEnv.
# 6. Evaluate every checkpoint on identical held-out examples.
# 7. Measure model-patch and environment-patch behavior independently.
# 8. Promote only a policy that passes mean, example, schema, and task-family gates.
# 9. Route regressed families to untouched base weights when that policy passes.
#
# Deterministic verifiers are the default because they are reproducible. The
# optional hybrid cell below enables fail-closed API-backed strategy evaluation.

# %%
from argparse import Namespace
from pathlib import Path
import os
import subprocess
import sys

from dotenv import load_dotenv


load_dotenv(override=False)
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


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


def _force_utf8_path_reads_on_windows() -> None:
    # Some TRL templates are read without an explicit encoding. Notebook kernels
    # cannot safely restart themselves with PYTHONUTF8, so make that default local.
    if os.name != "nt" or os.environ.get("PYTHONUTF8") == "1":
        return
    if getattr(Path, "_godel_utf8_patch", False):
        return
    original_read_text = Path.read_text

    def _read_text(self, *args, **kwargs):
        if not args and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original_read_text(self, *args, **kwargs)

    Path.read_text = _read_text
    Path._godel_utf8_patch = True


_force_utf8_path_reads_on_windows()

# %% [markdown]
# ## 1. Configuration

# %%
TASKS = "factual_qa,alignment_qa,reasoning,strategy_optimization"
BASE_MODEL = os.getenv("GODEL_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
NUM_TRAIN_PROMPTS = 32
SFT_STEPS = 80
GRPO_STEPS = 6
SFT_LEARNING_RATE = 5e-5
GRPO_LEARNING_RATE = 2e-6
REPAIR_STEPS = 12
REPAIR_LEARNING_RATE = 5e-6
REPAIR_OVERSAMPLE = 3
RECURSIVE_SFT_OVERSAMPLE = 4
MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 384
RECURSIVE_OVERSAMPLE = 6
EVAL_FRACTION = 0.25
OUTPUT_DIR = Path("artifacts/training_run")

# Set True only after adding OPENAI_API_KEY (or a compatible provider secret)
# to the Colab secret store. Hybrid mode never silently falls back here.
USE_HYBRID_LLM = False
if USE_HYBRID_LLM:
    from google.colab import userdata

    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    os.environ["GODEL_ALLOW_DETERMINISTIC_FALLBACK"] = "0"
    os.environ["GODEL_REQUIRE_PROVIDER_SEPARATION"] = "1"
    os.environ.setdefault("GODEL_AGENT_MODEL_NAME", "gpt-4o-mini")
    os.environ.setdefault("GODEL_VERIFIER_MODEL_NAME", "gpt-4.1-mini")
    os.environ["GODEL_AGENT_PROVIDER_ORDER"] = "openai"
    os.environ["GODEL_VERIFIER_PROVIDER_ORDER"] = "openai"

GRADING_MODE = "llm" if USE_HYBRID_LLM else os.getenv("GODEL_GRADING_MODE", "deterministic")
STRATEGY_EVAL_MODE = "llm" if USE_HYBRID_LLM else os.getenv("GODEL_STRATEGY_EVAL_MODE", "deterministic")
PROVIDER_ORDER = os.getenv(
    "GODEL_PROVIDER_ORDER", "custom,huggingface,openai,ollama"
)

print(f"Base model: {BASE_MODEL}")
print(f"Tasks: {TASKS}")
print(f"Verifier modes: grading={GRADING_MODE}, strategy={STRATEGY_EVAL_MODE}")

# %% [markdown]
# ## 2. Run the Shared Training Pipeline

# %%
from train import run


args = Namespace(
    output_dir=OUTPUT_DIR,
    tasks=TASKS,
    num_prompts=NUM_TRAIN_PROMPTS,
    sft_steps=SFT_STEPS,
    grpo_steps=GRPO_STEPS,
    sft_learning_rate=SFT_LEARNING_RATE,
    grpo_learning_rate=GRPO_LEARNING_RATE,
    repair_steps=REPAIR_STEPS,
    repair_learning_rate=REPAIR_LEARNING_RATE,
    repair_oversample=REPAIR_OVERSAMPLE,
    recursive_sft_oversample=RECURSIVE_SFT_OVERSAMPLE,
    max_input_length=MAX_INPUT_LENGTH,
    max_new_tokens=MAX_NEW_TOKENS,
    recursive_oversample=RECURSIVE_OVERSAMPLE,
    base_model=BASE_MODEL,
    eval_fraction=EVAL_FRACTION,
    device="auto",
    grading_mode=GRADING_MODE,
    strategy_eval_mode=STRATEGY_EVAL_MODE,
    provider_order=PROVIDER_ORDER,
    dry_run=False,
    generation_mode="freeform",
)
summary = run(args)

# %% [markdown]
# ## 3. Held-Out Results and Promotion Decision

# %%
baseline = summary["baseline"]
selected = summary["trained"]
promotion = summary["promotion"]
comparison = promotion["comparison"]

print(f"Selected stage: {promotion['selected_stage']}")
print(f"Promoted: {promotion['promoted']}")
print(f"Evidence label: {promotion['evidence_label']}")
print(f"Model recursive evidence: {promotion.get('model_recursive_evidence', False)}")
print(
    "Environment recursive evidence: "
    f"{promotion.get('environment_recursive_evidence', False)}"
)
print(f"Verified coevolution: {promotion.get('coevolution_evidence', False)}")
print(
    f"Held-out score: {baseline['mean_score']:.4f} -> {selected['mean_score']:.4f} "
    f"(delta={comparison['score_delta']:+.4f}, CI95={comparison['score_delta_ci95']})"
)
print(
    f"Held-out reward: {baseline['mean_reward']:.4f} -> {selected['mean_reward']:.4f} "
    f"(delta={comparison['reward_delta']:+.4f})"
)
print(
    f"Strict schema rate: {baseline['schema_valid_rate']:.1%} -> "
    f"{selected['schema_valid_rate']:.1%}"
)
print(f"Reasons: {promotion['reasons'] or ['promotion gate passed']}")
print("Candidate gate decisions:")
for stage, decision in promotion["stage_decisions"].items():
    print(f"  {stage}: passed={decision['passed']} reasons={decision['reasons']}")
print(f"Base fallback tasks: {selected.get('base_fallback_tasks', [])}")
print(
    "Recursive action rates: "
    f"model={selected.get('strategy_patch_rate', 0.0):.1%}, "
    f"environment={selected.get('environment_patch_rate', 0.0):.1%}"
)
print(
    "Governor acceptance rates: "
    f"model={selected.get('patch_acceptance_rate', 0.0):.1%}, "
    f"environment={selected.get('environment_patch_acceptance_rate', 0.0):.1%}"
)

# %% [markdown]
# ## 4. Committed Evidence Plots

# %%
from IPython.display import Image, display


for label, key in [
    ("SFT and adaptive-repair loss", "loss_curve"),
    ("GRPO rewards", "reward_curve"),
    ("Held-out before / after", "before_after"),
]:
    print(f"=== {label} ===")
    display(Image(filename=summary["artifacts"][key]))

# %% [markdown]
# ## 5. Evidence Contract
#
# The loss and reward curves above are real training evidence. The promotion gate
# distinguishes three stronger claims and will not infer one from another:
#
# - held-out task-policy improvement,
# - accepted model strategy mutations,
# - accepted environment curriculum mutations.
#
# `verified_coevolution` is true only when the trained candidate demonstrates both
# recursive behaviors. A small model may pass the task-policy gate without passing
# either recursive gate; that is a valid negative result, not evidence to relabel.

# %%
required_artifacts = [
    OUTPUT_DIR / "metrics.json",
    OUTPUT_DIR / "loss_curve.png",
    OUTPUT_DIR / "reward_curve.png",
    OUTPUT_DIR / "before_after.png",
]
missing_artifacts = [str(path) for path in required_artifacts if not path.exists()]
assert not missing_artifacts, f"Missing required evidence artifacts: {missing_artifacts}"
assert promotion["promoted"], (
    "No candidate passed the no-regression promotion gate. Inspect the stage decisions "
    "above; do not publish a regressed checkpoint."
)
assert comparison["reward_delta_ci95"][0] > 0.0, (
    "The held-out reward improvement is not positive at 95% confidence. Treat this as "
    "a training run, not publishable improvement evidence."
)
assert promotion.get("coevolution_evidence", False), (
    "The selected policy did not pass both recursive gates. Do not label this run as "
    "trained model-environment coevolution."
)
print("Evidence bundle is complete and verified coevolution gates passed.")

# %% [markdown]
# ## 6. Optional Strict Hybrid Coevolution Smoke
#
# This is an integration check, not training evidence. When `USE_HYBRID_LLM=True`,
# it requires the API-backed model to emit both mutation types and requires both
# Governors to return explicit decisions. Rejection is allowed and recorded because
# admission without evidence would defeat the environment's purpose.

# %%
if USE_HYBRID_LLM:
    hybrid_output = OUTPUT_DIR / "hybrid_coevolution_smoke.json"
    subprocess.run(
        [
            sys.executable,
            "hybrid_smoke.py",
            "--require-llm",
            "--require-coevolution",
            "--output",
            str(hybrid_output),
        ],
        check=True,
    )
    print(hybrid_output.read_text(encoding="utf-8"))
else:
    print("Skipped API-backed smoke. Set USE_HYBRID_LLM=True and configure Colab secrets.")
