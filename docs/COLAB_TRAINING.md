# Training GodelEngine at scale (Colab / cloud GPU)

The maintainers of this repository cannot run long GPU jobs on your behalf. For serious training, use a machine with a GPU and API keys for neutral verification.

## What to run

**Learned, end-to-end policy (recommended for evidence):**

```bash
export GODEL_GRADING_MODE=auto
export GODEL_STRATEGY_EVAL_MODE=llm
export GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC=0
export OPENAI_API_KEY=...   # or HF router / other provider in .env

python train.py \
  --output-dir /content/drive/MyDrive/godel_artifacts/run1 \
  --generation-mode freeform \
  --grading-mode auto \
  --strategy-eval-mode llm \
  --num-prompts 64 \
  --sft-steps 200 \
  --grpo-steps 100 \
  --max-input-length 1024
```

- `freeform` uses full JSON generation in both evaluation and **GRPO** (not the old 3-token symbolic classifier).
- `GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC=0` forces real LLM calls for held-out strategy evaluation. If the API is down, training fails loudly instead of silently grading against handcrafted templates.

**CI / no API key (heuristic simulation only):**

```bash
export GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC=1
python train.py --generation-mode symbolic --dry-run
```

## Google Colab (step by step)

### 1. Create a notebook

- Go to [colab.research.google.com](https://colab.research.google.com) → **New notebook**.
- **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is enough to start).

### 2. (Optional) Save artifacts to Drive

If the session disconnects, you keep checkpoints under Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 3. (Optional) Store your Hugging Face token in Colab Secrets

- **Secrets** (key icon in the left sidebar) → add `HF_TOKEN` with your token (read access is enough for the router).
- Or skip and paste the token only in the next cell (less secure if you share the notebook).

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # or: os.environ["HF_TOKEN"] = "hf_..."
```

### 4. Clone the repo and install training dependencies

Replace the URL with **your fork** if you use one (this project is often named `GodelEnv` on GitHub).

```python
%cd /content
!rm -rf GodelEnv
!git clone https://github.com/dwan-ith/GodelEnv.git
%cd GodelEnv
!pip install -q -e ".[train]"
```

### 5. Configure providers (Hugging Face credits)

Use the HF OpenAI-compatible router and a Hub model id:

```python
import os
os.environ["GODEL_PROVIDER_ORDER"] = "huggingface"
os.environ["HF_API_BASE_URL"] = "https://router.huggingface.co/v1"
os.environ["HF_MODEL_NAME"] = "Qwen/Qwen2.5-7B-Instruct"  # or another router-supported model
os.environ["GODEL_GRADING_MODE"] = "auto"
os.environ["GODEL_STRATEGY_EVAL_MODE"] = "llm"
# Set to "0" only when HF_TOKEN is valid — training fails fast if the API is down (no fake heuristic eval).
os.environ["GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
```

If you are **not** using API-backed eval yet, use a cheap CPU smoke run instead:

```python
os.environ["GODEL_STRATEGY_EVAL_ALLOW_HEURISTIC"] = "1"
```

### 6. Run training (start small, then scale up)

**Freeform (learned text + GRPO on generated completions):**

```python
!python train.py \
  --output-dir /content/drive/MyDrive/godel_artifacts/run1 \
  --generation-mode freeform \
  --grading-mode auto \
  --strategy-eval-mode llm \
  --num-prompts 16 \
  --sft-steps 30 \
  --grpo-steps 15 \
  --max-input-length 512
```

If you did **not** mount Drive, use e.g. `--output-dir /content/godel_out/run1` (ephemeral; download `metrics.json` and plots before closing the session).

**Faster / cheaper check (symbolic action tokens, good for “does the pipeline run?” only):**

```python
!python train.py --dry-run --generation-mode symbolic
```

### 7. Download results

If outputs are under `/content/godel_out`, zip and download from the file browser, or copy to Drive in another cell.

## Why not train only on CPU here?

- GRPO + freeform generation is memory- and time-heavy; a cloud GPU is usually 5–20× faster.
- Neutral patch acceptance requires LLM-based strategy eval; that needs network access and keys you control.

## `baseline.py` (rollouts with a real `AutoAgent`)

For interactive self-improvement episodes with a configured LLM (not the tiny local GPT-2 trainer):

```bash
python baseline.py --episodes 20 --tasks strategy_optimization
```

This uses the same environment API as training but drives actions through `AutoAgent` and provider APIs.
