---
title: GodelEnv
emoji: 🔁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# GodelEnv: A Verifiable Environment for Recursive Self-Improvement

GodelEnv is a reinforcement learning environment built on the `openenv-core` framework. It is designed to move beyond static, single-turn task optimization. Instead of merely rewarding an agent for producing a correct final answer, GodelEnv provides the infrastructure for **recursive skill amplification**: an agent can propose mutations to its own reasoning strategy, test those mutations against hidden downstream tasks, and retain them only if they genuinely improve multi-objective utility.

**Product Definition:**
> "GodelEnv is an OpenEnv environment for training agents to propose, test, and selectively adopt self-modifications to their own reasoning and coding strategies under verifier-backed meta-evaluation."

## The Core Idea

GodelEnv implements a practical approximation of [Gödel-machine-style](https://people.idsia.ch/~juergen/gmweb4/gmweb4.html) self-improvement:
- **Self-modification is explicit**: The agent proposes `StrategyPatch` mutations to its reasoning policy
- **Improvement proposals are testable**: Patches are evaluated on held-out task bundles
- **Acceptance depends on objective evidence**: The Governor accepts/rejects based on multi-objective utility, not vibes

This follows the pattern validated by [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/): LLM proposals + automated verifiers + evolutionary selection.

## The Recursive Ladder

GodelEnv supports multiple levels of self-improvement:
- **Level 1**: Improve a solution (legacy mode)
- **Level 2**: Improve the strategy that produces solutions (primary mode)
- **Level 3**: Improve the search procedure that improves the strategy (via training)
- **Level 4**: Improve resource allocation under cost, latency, and safety constraints

## Architecture

The environment is centered on strategy improvement, with task families serving as **evaluation substrate**:

```
┌─────────────────────────────────────────────────────────────┐
│                RECURSIVE SELF-IMPROVEMENT                   │
│                                                             │
│   Agent observes: current_strategy + failures + scores      │
│                          ↓                                  │
│   Agent proposes: StrategyPatch mutation                    │
│                          ↓                                  │
│   Evaluator runs: parent vs child on held-out domains       │
│   (factual_qa, code_improvement, reasoning, alignment_qa,   │
│    python_optimized, adr_writing)                           │
│                          ↓                                  │
│   Governor decides: accept/reject with multi-objective      │
│   utility (correctness, generalization, robustness,         │
│   cost, stability, safety)                                  │
│                          ↓                                  │
│   Registry updates: lineage, Elo ratings, failure cases     │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **StrategyPatch**: The fundamental action — a proposed mutation to the reasoning policy
- **Governor**: Multi-gate acceptance filter (improvement, regression, variance, sample size, catastrophic, canary)
- **StrategyEvaluator**: Runs strategies on held-out task bundles
- **StrategyRegistry**: Stores accepted strategies with lineage and Elo ratings
- **HuxleyTracker**: Tracks Clade-Metaproductivity (strategies that produce better descendants)

### Anti-Hacking Guards

This is absolutely central. The Governor implements multiple gates to prevent reward hacking:
1. **Improvement Gate**: Child must have higher multi-objective utility than parent
2. **Regression Gate**: Child must not regress on too many individual tasks
3. **Stability Gate**: Child must have acceptable variance across domains
4. **Sample Size Gate**: Must evaluate on enough tasks for valid decision
5. **Catastrophic Gate**: No single task can regress too severely
6. **Canary Gate**: Special canary tasks must not regress

## Architecture

GodelEnv behaves as a fully compliant OpenEnv service, making it compatible with existing agentic RL pipelines.

- **OpenEnv Interface**: Conforms strictly to `openenv-core>=0.2.3` standards with standard `/reset`, `/step`, and `/state` API surfaces.
- **Hybrid Runtime**: Supports a dual-execution path. It uses an LLM-first behavior (`auto`) for rigorous grading, agent actions, and strategy evaluation, but gracefully fails over to deterministic grading if provider credentials are stale or rate-limited.
- **Multi-Channel Reward**: Discards the single scalar reward in favor of a detailed, multi-axis vector including `task_score_delta`, `format_compliance`, `generalization_score`, `robustness_score`, and `anti_hack_penalty`.
- **Guardrails**: Includes robust checks for empty responses, repetition, forbidden code patterns, canary leakage, and strategy length limits.

### Supported Providers

GodelEnv supports multiple LLM providers with automatic failover. **LLM mode is the default** - deterministic fallback only occurs when all providers fail.

**Provider Priority** (configurable via `GODEL_PROVIDER_ORDER`):
1. `huggingface` - HF Router / Inference API
2. `ollama` - Local Ollama instance (no API key needed)
3. `custom` - Any OpenAI-compatible endpoint (vLLM, etc.)
4. `openai` - OpenAI API

```bash
# Hugging Face Router (recommended for HF Spaces)
set HF_TOKEN=...
set HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
# Also accepts: HF_API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE_TOKEN, etc.

# Local Ollama (no API key required)
set OLLAMA_MODEL_NAME=qwen2.5:7b
# Or with custom host:
set OLLAMA_API_BASE_URL=http://localhost:11434/v1

# Local vLLM or other OpenAI-compatible server
set API_BASE_URL=http://localhost:8000/v1
set API_KEY=dummy
set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# OpenAI API
set OPENAI_API_KEY=...
set OPENAI_MODEL_NAME=gpt-4o-mini
```

**HF Spaces Secrets** - These work automatically without renaming:
```bash
HF_TOKEN=...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

**Quick Local Setup with Ollama**:
```bash
# Install Ollama: https://ollama.ai/
ollama pull qwen2.5:7b
set OLLAMA_MODEL_NAME=qwen2.5:7b
python demo.py  # Uses local model!
```

**Runtime Flags**:
```bash
# These default to "auto" (LLM-first with deterministic fallback)
set GODEL_GRADING_MODE=auto
set GODEL_STRATEGY_EVAL_MODE=auto

# Customize provider priority
set GODEL_PROVIDER_ORDER=ollama,huggingface,openai

# Require LLM (error instead of fallback)
set GODEL_REQUIRE_LLM=1
```

**Verify your setup**:
```bash
python hybrid_smoke.py --require-llm
```

### Hybrid Mode Diagnostics

The environment provides transparent diagnostics about whether LLM or deterministic fallback is being used:

**Dashboard indicators**: The real-time dashboard shows a color-coded mode indicator:
- **Green (LLM)**: Actions are being generated by the configured LLM provider
- **Yellow (DETERMINISTIC FALLBACK)**: Actions are using heuristic fallback

**API diagnostics**: The `/demo/act` endpoint returns:
- `is_llm_generated`: Whether the action came from an LLM
- `agent_source`: Either `llm:provider_name` or `deterministic_fallback`
- `agent_error`: The error message if LLM failed

**Provider status**: The `/demo/provider-status` endpoint shows:
- Which API keys are detected in the environment
- Which providers are configured and their status
- Circuit breaker state (if a provider was disabled due to errors)

**Require LLM mode**: Set `GODEL_REQUIRE_LLM=1` to error instead of falling back:
```bash
set GODEL_REQUIRE_LLM=1
python demo.py  # Will fail with 503 if no LLM available
```

### Heuristic Fallback Behavior

When LLM providers are unavailable, the environment uses an intelligent heuristic fallback:
- **Varied patches**: Unlike static hardcoded responses, heuristic patches analyze the current strategy and target specific missing capabilities
- **Weakness-aware**: Patches prioritize improvements based on recent failures and weak downstream scores
- **Transparent labeling**: All heuristic actions are labeled with `[HEURISTIC]` in their descriptions

## Verifiable Training Pipeline

To prove the environment mechanics are learnable, GodelEnv ships with a reproducible training pipeline. The pipeline uses a compact action space so a tiny, local CPU policy can be trained and verified without massive compute.

1. **Prompt Collection**: Samples initial states from the live environment.
2. **Warm-Start**: Generates heuristic traces to teach the model the base `Action` or `StrategyPatch` schema.
3. **SFT**: Supervised fine-tuning of a tiny local policy.
4. **GRPO**: Group Relative Policy Optimization against the live environment to maximize the multi-channel reward.

### Training Evidence

The committed training evidence (available in `artifacts/training_run`) demonstrates that the environment provides a clear, learnable gradient. The results below compare a random baseline against the trained policy over 16 prompts across factual QA, alignment QA, reasoning, and strategy optimization tasks.

| Metric | Baseline | Trained | Delta |
| --- | ---: | ---: | ---: |
| Mean reward | 0.4074 | 0.4992 | **+0.0918** |
| Mean score | 0.7221 | 0.8152 | **+0.0931** |

**Per-Task Improvement:**
| Task | Baseline | Trained |
| --- | ---: | ---: |
| factual_qa | 0.9554 | 0.9554 |
| alignment_qa | 0.6319 | 0.7178 |
| reasoning | 0.7917 | 1.0000 |
| strategy_optimization | 0.5093 | 0.5877 |

#### Loss Curve (SFT)
![SFT loss curve](artifacts/training_run/loss_curve.png)

#### Reward Curve (GRPO)
![GRPO reward curve](artifacts/training_run/reward_curve.png)

#### Comparison
![Before and after training](artifacts/training_run/before_after.png)

This shows concrete learning on both reward and downstream score. The current miniature policy converges to a conservative "direct-answer" strategy. The environment path is fully verified and ready for scaling with larger models that can explore richer recursive behaviors.

## Real-Time Dashboard

The environment includes a high-fidelity dashboard to monitor the recursive loop visually:

- **Engine Status**: Real-time tracking of score progression, step counts, and delta.
- **Strategy Stats**: Monitors the agent's Strategy ELO, Generation lineage, and remaining computation budget.
- **Reasoning Strategy**: Displays the specific reasoning policy currently driving the agent's behavior.
- **Recent Failures**: A live feedback loop showing the "challengers" (failed tasks) that guide the next strategy mutation.
- **Telemetry**: Verbose streaming of grading sources and Governor acceptance/rejection decisions.

## Quick Start & Validation

### Run the Environment Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Validate the Engine
Ensure all core mechanics, grading paths, and OpenEnv specifications are functioning:
```bash
openenv validate
pytest -q
python -m compileall godel_engine server train.py train_colab.py demo.py
```

### Run the Training Pipeline
Train the local proof-of-concept policy and export training plots:
```bash
python train.py
```
Or use the pre-configured notebook: `train_colab.ipynb`

## Project Links

- **Hosted Demo**: [HF Space](https://huggingface.co/spaces/litterarum/GodelEnv)
- **Source Code**: [GitHub](https://github.com/dwan-ith/GodelEnv)
- **Blog**: [blog_draft.md](blog_draft.md)
