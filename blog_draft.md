# GodelEnv: Turning Recursive Self-Improvement Into a Verifiable OpenEnv Environment

GodelEnv started from a good instinct: an LLM should not only improve answers, it should improve the strategy that produces answers. The problem was that the earlier repo shape made that idea look more real than it actually was. The environment talked about recursive self-improvement, but key pieces of the training and verification loop were brittle, under-specified, or easy to misread as progress.

This revision tightens that loop so the project behaves like a real OpenEnv environment that can be trained against and evaluated honestly.

## The Core Idea

Most LLM environments optimize a fixed task: answer a question, write code, pass tests. GodelEnv adds a meta-action:

- improve the current answer directly, or
- propose a `StrategyPatch` that changes the reasoning policy itself

When the agent proposes a patch, the environment does not blindly accept it. Instead, it evaluates the child strategy against the parent on hidden downstream tasks and keeps the patch only if it improves utility without broad regressions.

That makes the environment a practical self-improvement arena rather than a static benchmark.

## What Changed

The repo now has a cleaner separation between environment dynamics, verification, and training:

- a real OpenEnv server/client shape with `reset`, `step`, and `state`
- held-out downstream evaluation for strategy patches
- multi-channel reward instead of a single vague scalar
- explicit anti-hacking checks for regressions, canary leakage, padding, and variance
- a hybrid runtime where LLM calls are primary and deterministic grading is the fallback
- a provider circuit breaker so stale credentials do not repeatedly poison the run

That last point mattered more than it looks. One of the failure modes in the old codebase was that bad provider configuration could make almost every evaluation path noisy or brittle. Now the environment degrades cleanly instead of pretending the LLM path worked.

## Reward Design

The reward function is intentionally decomposed:

- `task_score_delta`
- `format_compliance`
- `step_cost`
- `anti_hack_penalty`
- `process_reward`
- `patch_quality`
- `generalization_score`
- `robustness_score`
- `stability_score`

This matters for two reasons. First, it gives the trainer a richer signal than a single pass/fail bit. Second, it makes reward hacking harder: the agent has to improve actual outcomes while clearing independent checks.

## Making the Training Path Honest

The biggest repo-level upgrade was the training path.

The old local miniature setup looked like a training pipeline, but it was a poor match for the action space: the tiny model had to emit long free-form JSON-like actions, prompts were bloated, the target action token could be truncated, and prompt tokens were not masked correctly during SFT. It was easy to get curves that looked active without learning a useful policy.

The fixed version uses a compact local policy:

1. The tiny GPT-2 model emits one compact action token.
2. The environment expands that token into a full environment action.
3. The verifier still runs on the real action.

This keeps the CPU proof-of-concept small enough to run locally while preserving the actual environment semantics.

Other training fixes included:

- shorter prompts so the target token is visible and learnable
- prompt masking during SFT so the model learns the completion, not the prompt
- evaluation over the allowed action set instead of uncontrolled full-vocabulary generation
- a random compact baseline instead of misleading "baseline" text generation
- synced notebook and script outputs with committed plots and metrics

## What The Latest Run Shows

The latest committed run evaluates 16 prompts across `factual_qa`, `alignment_qa`, `reasoning`, and `strategy_optimization`.

Headline result:

- mean reward improves from `0.4090` to `0.5224`
- mean score improves from `0.7361` to `0.8384`

The most important thing about that result is not that the numbers are huge. It is that the local proof-of-concept now shows honest improvement on both reward and downstream score.

There is also an important limitation: the trained tiny model learns to prefer the stronger direct-answer action and stops proposing strategy patches in the final evaluation set. In other words, the recursive patch path is present and real, but the tiny CPU policy is still conservative.

That is a useful finding, not a failure to hide. It tells us:

- the environment mechanics are working
- the verification loop is working
- the training evidence is real
- richer recursive behavior will likely need a stronger model or hybrid API-backed training

## Why I Think This Is Interesting

I do not think the goal of an environment like this is to cosplay a "Godel machine" with mystical language. The goal is simpler and more practical:

- let the agent see its recurring weaknesses
- let it mutate its own solver policy
- test that mutation on hidden tasks
- keep the mutation only if it generalizes

That is a useful research direction for self-improving LLM systems, and it is much more compelling when the environment, reward logic, and training notebook all actually run.

## What I Would Do Next

The next scaling path is clear:

1. Keep the same environment and verifier stack.
2. Swap the tiny local model for a stronger model through the hybrid runtime.
3. Regenerate the same evidence with richer patch exploration.
4. Compare conservative direct-answer behavior against genuine recursive patch behavior.

That is the point where GodelEnv stops being only a compact local proof-of-concept and starts becoming a stronger self-improvement benchmark.
