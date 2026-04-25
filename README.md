# GodelEnv

GodelEnv is an OpenEnv environment for recursive self-improvement. The core action is not only "write a better answer" but "propose a better reasoning strategy, test it on held-out tasks, and keep it only if it improves."

This repo now uses a real OpenEnv server shape, a strategy registry with lineage and Elo tracking, held-out downstream evaluation for `StrategyPatch` proposals, and a hybrid LLM-first runtime with deterministic fallback when the provider is unavailable.

## Links

- Hugging Face Space: [litterarum/GodelEnv](https://huggingface.co/spaces/litterarum/GodelEnv)
- Source repo: [dwan-ith/GodelEnv](https://github.com/dwan-ith/GodelEnv)
- Training notebook: [train_colab.ipynb](train_colab.ipynb)
- Training script: [train.py](train.py)
- Writeup: [blog_draft.md](blog_draft.md)

## Problem

Most LLM RL environments optimize fixed tasks. GodelEnv tries to optimize capability growth itself.

The environment exposes downstream task families such as:

- factual QA
- alignment QA
- reasoning
- code improvement
- Python optimization
- ADR writing
- strategy optimization

The recursive part is that `strategy_optimization` is not just another task. It is the meta-loop: the agent can propose a new reasoning policy, and the environment evaluates parent vs child on episode-hidden downstream cases before accepting or rejecting the patch.

## Environment Design

Each episode gives the agent:

- the current task prompt and current draft
- the current reasoning strategy
- recent downstream failures
- downstream task-family scores
- remaining episode budget

The agent can then do one of two things:

1. Submit a direct answer edit.
2. Submit a `StrategyPatch` containing `improved_strategy`, a diff description, a hypothesis, and target weaknesses.

When a strategy patch is proposed, GodelEnv:

1. Builds a child strategy.
2. Evaluates parent and child on a hidden downstream bundle.
3. Applies anti-hacking guards for leakage, regressions, variance, and padding.
4. Uses the Governor to accept or reject the patch.
5. Updates lineage, Elo, and patch history.

## Hybrid LLM Runtime

The runtime is now hybrid in the new project structure:

- LLM-backed grading, agent actions, and strategy evaluation are the primary path.
- Deterministic verifiers are the fallback path.
- If the provider errors, rate-limits, or credentials are absent, the episode still runs and records `grading_source=deterministic`.

Supported credential patterns:

- `OPENAI_API_KEY` for the default OpenAI-compatible path
- `API_KEY` + `API_BASE_URL` for custom OpenAI-compatible providers
- `HF_TOKEN` + `API_BASE_URL=https://router.huggingface.co/v1` for Hugging Face Router

Useful environment variables:

```bash
set GODEL_GRADING_MODE=auto
set GODEL_STRATEGY_EVAL_MODE=auto
set MODEL_NAME=gpt-4o-mini
```

For fully reproducible offline evidence, set both modes to `deterministic`.

## OpenEnv Structure

The canonical environment surface is now:

- [godel_engine/openenv_environment.py](godel_engine/openenv_environment.py)
- [godel_engine/openenv_models.py](godel_engine/openenv_models.py)
- [server/app.py](server/app.py)
- [godel_engine/client.py](godel_engine/client.py)

Server endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /metadata`
- `GET /schema`
- `GET /health`
- `WS /ws`

The dashboard uses:

- `WS /ws` for the actual multi-step environment session
- `POST /demo/act` for server-side agent action generation

## Reward and Anti-Hacking Design

The reward is multi-channel rather than a single scalar:

- `task_score_delta`
- `format_compliance`
- `step_cost`
- `anti_hack_penalty`
- `process_reward`
- `patch_quality`
- `generalization_score`
- `robustness_score`
- `stability_score`

Anti-hacking checks include:

- repetition and length guards
- forbidden code patterns
- regression gates
- strategy canary checks
- variance penalties
- strategy length checks

## Training Pipeline

The shipped training path is a small proof-of-concept that can run locally:

1. Collect prompts from the environment.
2. Build heuristic warm-start traces.
3. Run SFT on the trace format.
4. Run GRPO against the live environment.
5. Save reward, loss, and before/after plots.

Run:

```bash
python train.py
```

For a quick check:

```bash
python train.py --dry-run
```

The notebook version is [train_colab.ipynb](train_colab.ipynb). It clones/installs the repo automatically when opened in Colab, defaults to deterministic evidence mode for reproducible judging, and can use the hybrid LLM path by setting:

```bash
set GODEL_GRADING_MODE=auto
set GODEL_STRATEGY_EVAL_MODE=auto
```

## Training Evidence

Current committed evidence is in [artifacts/training_run](artifacts/training_run).

From the latest local run:

- Mean reward: `-0.4394 -> -0.1178` (`+0.3216`)
- Mean score: `0.1491 -> 0.1424` (`-0.0068`)
- Prompt count: `16`

Interpretation:

- The tiny CPU proof-of-concept clearly improves reward.
- Aggregate task score is roughly flat with a small negative delta.
- This means the current local miniature model is enough to demonstrate the environment and training loop, but not yet enough to show strong recursive patch behavior on its own.
- The environment itself supports the stronger hybrid LLM path, and that is the intended next scaling direction.

### Loss curve

![SFT loss curve](artifacts/training_run/loss_curve.png)

### Reward curve

![GRPO reward curve](artifacts/training_run/reward_curve.png)

### Before / after summary

![Before and after training](artifacts/training_run/before_after.png)

## Dashboard

The dashboard provides a high-fidelity interface to monitor the recursive self-improvement process:

- **Engine Status**: Real-time tracking of Score, Delta, and Step.
- **Strategy Stats**: Monitors the agent's **Strategy ELO**, current **Generation**, and remaining budget.
- **Reasoning Strategy**: Displays the specific reasoning policy the agent is currently following (and attempting to optimize).
- **Recent Failures**: A feedback loop showing the specific task instances the agent recently failed, which serve as the "challengers" for the next `StrategyPatch`.
- **Grading / Rubrics**: Detailed breakdown of multi-axis scores and per-task feedback.
- **Log Stream**: Verbose logging of agent actions, grading sources, and Governor acceptance/rejection decisions.

## Why This Is Interesting

The point of GodelEnv is not just to fine-tune a text model on canned QA. It is to give an agent a way to:

- inspect its own failures
- propose strategy mutations
- test those mutations on held-out tasks
- keep only the ones that improve utility

That is a practical approximation of recursive self-improvement rather than a toy answer-rewrite loop.

## Validation

Local checks used on this repo:

```bash
pytest -q
openenv validate
python -m compileall godel_engine server train.py train_colab.py
```

## Current Limitations

- The tiny local training run is a proof-of-concept, not the final ceiling of the environment.
- In sandboxed offline settings, live provider calls fall back to deterministic grading.
- The current miniature model still struggles to emit valid `StrategyPatch` JSON consistently; the environment path is ready for stronger API-backed or larger-model runs.

That limitation is exactly why the hybrid architecture matters: the environment and evaluator are no longer fake, and stronger models can plug into the same verified loop.
