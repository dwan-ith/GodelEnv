# GodelEnv: Recursive Self-Improvement with a Hybrid LLM Runtime

GodelEnv is an OpenEnv environment built around a simple idea: an agent should not only improve answers, it should improve the strategy that produces answers.

The practical loop is:

1. The agent sees a task, the current draft, its current reasoning strategy, recent failures, and downstream task-family scores.
2. It can either improve the draft directly or propose a `StrategyPatch`.
3. If it proposes a patch, the environment evaluates the parent and child strategies on hidden downstream tasks.
4. A Governor accepts the patch only if it improves multi-objective utility without broad regressions or evaluator leakage.

That makes the environment a practical approximation of recursive self-improvement rather than a single-turn rewrite benchmark.

## What changed in this repo

The repo now has:

- a real OpenEnv server wrapper instead of a custom API pretending to be OpenEnv
- a persistent websocket client/session path
- strategy lineage, Elo, and patch history
- held-out strategy evaluation separated from the task currently shown to the agent
- anti-hacking guards for regression, variance, canary leakage, and padding
- a hybrid runtime where LLM calls are primary and deterministic grading is fallback

## Why the hybrid path matters

In the old shape, the project could easily become "vibecoded slop": the environment talked like a self-improver, but the actual scoring path was shallow and brittle.

The new structure is different:

- `AutoAgent`, `AgentGrader`, and `StrategyEvaluator` all try the configured LLM provider first.
- If the provider is unavailable, they fall back to deterministic behavior instead of crashing the episode.
- The server/runtime can use `GODEL_GRADING_MODE=auto` and `GODEL_STRATEGY_EVAL_MODE=auto`, so `.env` credentials are used when available.
- The training evidence notebook defaults to deterministic mode so judges can reproduce the committed curves even on machines with stale or missing API credentials.

That gives us two useful modes:

- hybrid demo mode for stronger real inference
- deterministic replay mode for reproducible local evidence

## Current evidence

The latest local proof-of-concept run is committed under `artifacts/training_run/`.

Headline numbers:

- Mean reward improved from `-0.4394` to `-0.1178`
- Mean score moved from `0.1491` to `0.1424`

So the tiny local model learns something measurable, but not enough yet to become a strong recursive patch proposer. That is an honest result, and it is exactly why the environment was built to support the stronger hybrid LLM path.

## What still needs scaling

The environment is now much more real than the earlier repo, but the tiny CPU model is still the bottleneck:

- it improves reward faster than it improves actual downstream quality
- it does not consistently emit valid `StrategyPatch` JSON during evaluation
- it is best understood as a smoke-test trainer, not the final agent

The next serious step is to run the same loop with a stronger model through the configured API path and regenerate the same evidence with real patch proposals.

## Why I think this is worth building

What I want from GodelEnv is not another QA benchmark with a philosophical name. I want an environment where an agent can:

- inspect repeated weaknesses
- mutate its own solver policy
- test the mutation on hidden tasks
- keep the mutation only if it generalizes

That is the version of "Godel machine" thinking that feels practical for modern LLM systems.
