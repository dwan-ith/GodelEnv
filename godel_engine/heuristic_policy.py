"""
Deterministic reference policy used for demos, baselines, and SFT warm starts.

NOTE: This is a FALLBACK policy, not the primary path for recursive self-improvement.
When LLM providers are available, AutoAgent generates novel strategy patches. This
heuristic policy is used only when:
1. No LLM providers are configured
2. All configured providers fail
3. Training requires deterministic reproducibility

The heuristic patches are designed to be VARIED based on the current strategy state,
not hardcoded. Each patch targets specific missing capabilities.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from godel_engine.models import EditType, GodelAction, StrategyPatch


def _strategy_flags(strategy_text: str | None) -> dict[str, bool]:
    text = (strategy_text or "").lower()
    return {
        "decompose": any(token in text for token in ["decompose", "break the problem", "atomic claims"]),
        "evidence": any(token in text for token in ["evidence", "support", "grounding"]),
        "counterargument": any(token in text for token in ["counterargument", "alternative hypothesis", "counter-example", "alternative"]),
        "uncertainty": any(token in text for token in ["uncertainty", "confidence", "confidence level"]),
        "verify": any(token in text for token in ["verify", "self-check", "verification", "check"]),
        "reflect": any(token in text for token in ["reflect", "revision", "lessons learned", "postmortem"]),
        "examples": any(token in text for token in ["example", "worked example", "demonstration"]),
        "structure": any(token in text for token in ["stage", "section", "structured", "organize"]),
        "efficiency": any(token in text for token in ["efficient", "optimization", "complexity", "sqrt", "sieve"]),
        "safety": any(token in text for token in ["policy risk", "safety", "harm", "guardrail", "auditing"]),
    }


CAPABILITY_ADDITIONS = {
    "decompose": (
        "DECOMPOSE: Break the problem into atomic claims, assumptions, and decision points before attempting a solution.",
        "Added problem decomposition step",
        "Decomposition helps identify hidden assumptions and ensures systematic coverage",
    ),
    "evidence": (
        "EVIDENCE: For each major claim, cite supporting evidence or note gaps in knowledge.",
        "Added explicit evidence gathering",
        "Evidence grounding reduces hallucination and improves factual accuracy",
    ),
    "counterargument": (
        "COUNTER: Generate at least one counterargument, edge case, or alternative hypothesis for each conclusion.",
        "Added counterargument generation",
        "Considering alternatives reduces overconfidence and catches blind spots",
    ),
    "uncertainty": (
        "UNCERTAINTY: Explicitly mark confidence levels (high/medium/low) for claims with different evidence strength.",
        "Added uncertainty quantification",
        "Explicit uncertainty helps downstream consumers calibrate trust appropriately",
    ),
    "verify": (
        "VERIFY: Run a self-check before finalizing - does the answer address the actual question? Are there logical gaps?",
        "Added self-verification step",
        "Verification catches errors before they propagate to the final output",
    ),
    "reflect": (
        "REFLECT: After solving, note what worked well and what could improve for similar future problems.",
        "Added reflection/meta-learning step",
        "Reflection enables strategy improvement over time",
    ),
    "examples": (
        "EXAMPLES: Include at least one concrete worked example or demonstration when explaining concepts.",
        "Added worked example requirement",
        "Examples ground abstract concepts and verify understanding",
    ),
    "efficiency": (
        "EFFICIENCY: For computational tasks, analyze time/space complexity and prefer asymptotically optimal solutions.",
        "Added efficiency/complexity analysis",
        "Efficiency awareness prevents naive O(n^2) or worse solutions",
    ),
    "safety": (
        "SAFETY: Check for policy risks, potential harms, or safety concerns before finalizing responses.",
        "Added safety/policy check",
        "Safety awareness prevents harmful outputs and policy violations",
    ),
}


def _select_patch_focus(
    flags: dict[str, bool],
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
) -> str:
    """Select which capability to add based on current weaknesses."""
    failures_text = " ".join(recent_failures or []).lower()
    scores = downstream_scores or {}

    missing = [cap for cap, present in flags.items() if not present]
    if not missing:
        missing = list(CAPABILITY_ADDITIONS.keys())

    priority_order = []

    if "factual" in failures_text or scores.get("factual_qa", 1.0) < 0.5:
        priority_order.extend(["evidence", "verify", "uncertainty"])
    if "code" in failures_text or scores.get("code_improvement", 1.0) < 0.5:
        priority_order.extend(["efficiency", "verify", "decompose"])
    if "reasoning" in failures_text or scores.get("reasoning", 1.0) < 0.5:
        priority_order.extend(["decompose", "counterargument", "examples"])
    if "alignment" in failures_text or scores.get("alignment_qa", 1.0) < 0.5:
        priority_order.extend(["safety", "counterargument", "uncertainty"])

    for cap in priority_order:
        if cap in missing:
            return cap

    return missing[0] if missing else "verify"


def build_heuristic_solution(task_prompt: str, task_type: str, strategy_text: str | None = None) -> str:
    prompt = task_prompt.lower()
    flags = _strategy_flags(strategy_text)

    if task_type == "code_improvement":
        if "palindrome" in prompt:
            solution = (
                "def is_palindrome(s):\n"
                '    """Return True when `s` is a palindrome after normalization."""\n'
                "    normalized = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
                "    return normalized == normalized[::-1]\n"
            )
            if flags["verify"]:
                solution += "\n# Verified by normalizing punctuation and case before comparison.\n"
            return solution

        if flags["efficiency"] or flags["verify"]:
            solution = (
                "def fibonacci(n):\n"
                '    """Return the nth Fibonacci number in O(n) time."""\n'
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            )
        else:
            solution = (
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
            )
        if flags["verify"]:
            solution += "\n# Self-check: handles n=0 and n=1 explicitly.\n"
        return solution

    if task_type == "python_optimized":
        if flags["efficiency"]:
            solution = (
                "def get_primes(n: int) -> list[int]:\n"
                '    """Return all prime numbers smaller than ``n`` using trial division up to sqrt(i)."""\n'
                "    if n < 2:\n"
                "        return []\n"
                "    primes: list[int] = []\n"
                "    for candidate in range(2, n):\n"
                "        is_prime = True\n"
                "        limit = int(candidate ** 0.5) + 1\n"
                "        for divisor in range(2, limit):\n"
                "            if candidate % divisor == 0:\n"
                "                is_prime = False\n"
                "                break\n"
                "        if is_prime:\n"
                "            primes.append(candidate)\n"
                "    return primes\n"
            )
        else:
            solution = (
                "def get_primes(n):\n"
                "    primes = []\n"
                "    for i in range(2, n):\n"
                "        is_prime = True\n"
                "        for j in range(2, i):\n"
                "            if i % j == 0:\n"
                "                is_prime = False\n"
                "                break\n"
                "        if is_prime:\n"
                "            primes.append(i)\n"
                "    return primes\n"
            )
        if flags["verify"]:
            solution += "\n# Self-check: returns an empty list for values smaller than 2.\n"
        return solution

    if task_type == "factual_qa":
        if "reinforcement learning" in prompt:
            parts = [
                "Reinforcement learning trains an agent by letting it interact with an environment, "
                "take actions, and learn from rewards over time. Supervised learning instead starts "
                "with a dataset of labeled examples and learns a direct mapping from inputs to labels. "
                "The key difference is the feedback signal: RL receives delayed reward from the environment, "
                "while supervised learning receives immediate correction from ground-truth data."
            ]
            if flags["evidence"]:
                parts.append(
                    "That difference changes exploration, credit assignment, and the role of sequential decisions."
                )
            if flags["counterargument"]:
                parts.append(
                    "In practice the line can blur, because offline RL and imitation learning borrow ideas from both setups."
                )
            if flags["uncertainty"]:
                parts.append("Confidence: high on the core distinction, lower on edge-case hybrids.")
            return " ".join(parts)
        if "attention mechanism" in prompt:
            parts = [
                "Attention is the mechanism that lets a Transformer decide which tokens matter most for the "
                "current prediction. Queries, keys, and values are compared so the model can build context-aware "
                "representations of the whole sequence. This makes it easier to capture long-range dependencies "
                "and also enables parallel computation across the sequence."
            ]
            if flags["examples"]:
                parts.append("For example, a pronoun can attend back to the noun it refers to earlier in the sentence.")
            return " ".join(parts)
        parts = [
            "Quantum entanglement means two particles can share a linked quantum state, so measuring one of them "
            "instantly tells you something about the other even when they are far apart. The particles are not "
            "sending a normal signal across the distance; instead, the correlation comes from the way their states "
            "were created together in quantum mechanics."
        ]
        if flags["examples"]:
            parts.append("A simple intuition is that the pair behaves like one joint system until measurement.")
        return " ".join(parts)

    if task_type == "alignment_qa":
        if "hhh" in prompt:
            parts = [
                "Helpfulness, honesty, and harmlessness are three alignment goals that often pull in different "
                "directions. A helpful model should answer the user's question, an honest model should avoid making "
                "things up, and a harmless model should refuse requests that create risk. These goals can conflict "
                "when a user asks for something dangerous, because the model must decline the unsafe request while "
                "still being honest about why it is refusing."
            ]
            if flags["safety"]:
                parts.append("Safety policy and context determine when harmlessness should override direct helpfulness.")
            if flags["counterargument"]:
                parts.append("The hard case is when refusing too broadly also harms the user by withholding benign help.")
            return " ".join(parts)
        parts = [
            "Reward misspecification in RLHF happens when the reward model is only a proxy for what humans really "
            "want. The policy then learns to exploit shortcuts that score well on the proxy reward without actually "
            "being aligned with the intended objective."
        ]
        if flags["safety"] or flags["verify"]:
            parts.append(
                "Common mitigations include better evaluation, adversarial testing, monitoring, and holdout audits that look for reward hacking."
            )
        return " ".join(parts)

    if task_type == "reasoning":
        if "microservices" in prompt:
            pros = [
                "- Microservices allow independent deployment and better team autonomy.",
                "- They can improve fault isolation and make it easier to scale specific workloads.",
            ]
            cons = [
                "- They add operational complexity, network latency, and harder debugging.",
                "- They also require stronger observability and coordination across services.",
            ]
            if flags["counterargument"]:
                cons.append("- Some teams overestimate these benefits and split too early before domain boundaries are clear.")
            if flags["verify"]:
                rec = (
                    "- Use a gradual migration only if the monolith is blocking team velocity or scaling, and introduce "
                    "service boundaries incrementally instead of rewriting everything at once."
                )
            else:
                rec = "- Migrate to microservices."
            return "Pros\n" + "\n".join(pros) + "\n\nCons\n" + "\n".join(cons) + "\n\nRecommendation\n" + rec
        pros = [
            "- AGI could raise productivity, automate routine work, and accelerate innovation.",
            "- It may create new industries and lower the cost of expertise.",
        ]
        cons = [
            "- It could displace workers, increase inequality, and create major transition shocks.",
            "- Concentrated control of AGI systems could distort markets and policy.",
        ]
        if flags["counterargument"]:
            cons.append("- Forecasts remain uncertain, so both hype and doom narratives can oversimplify the transition.")
        rec = (
            "- Governments and firms should pair deployment with reskilling, social protections, and clear "
            "governance so the gains are broad-based rather than destabilizing."
            if flags["verify"] or flags["safety"]
            else "- Governments should respond."
        )
        return "Pros\n" + "\n".join(pros) + "\n\nCons\n" + "\n".join(cons) + "\n\nRecommendation\n" + rec

    if task_type == "adr_writing":
        solution = (
            "# Title\nMigration from Monolith to Serverless Event-Driven Architecture\n\n"
            "## Status\nAccepted\n\n"
            "## Context\nThe current monolith slows delivery, scales as a single unit, and makes operational "
            "ownership unclear. We need more elastic scaling and clearer service boundaries.\n\n"
            "## Decision\nAdopt a serverless event-driven architecture built around AWS Lambda and SQS. New "
            "domain workflows will publish events to queues so consumers can scale independently.\n\n"
            "## Consequences\nPositive consequences include lower idle cost, better workload isolation, and faster "
            "team autonomy. Negative consequences include harder observability, cold-start latency, and more "
            "distributed debugging. Operations must invest in tracing, queue monitoring, and replay tooling.\n\n"
            "## Alternatives\nKeep the monolith and optimize it further, or move to containerized microservices. "
            "We rejected both because they either preserve scaling bottlenecks or increase operational overhead "
            "without the serverless cost profile we want."
        )
        if flags["counterargument"]:
            solution += (
                "\n\n## Review Notes\nA phased rollout is safer than a big-bang migration because event contracts and failure handling mature over time."
            )
        return solution

    if task_type == "strategy_optimization":
        steps = ["1. Decompose the prompt into claims, assumptions, and decision points."]
        if flags["evidence"]:
            steps.append("2. Draft an answer with explicit evidence for each major claim.")
        else:
            steps.append("2. Draft an answer.")
        if flags["uncertainty"]:
            steps.append("3. Mark uncertainty and confidence for uncertain claims.")
        if flags["verify"]:
            steps.append("4. Run a self-check for factual accuracy, missing counterarguments, and policy risks.")
        if flags["reflect"]:
            steps.append("5. Revise the strategy itself by recording what the self-check found and how the template should improve.")
        steps.append(f"{len(steps) + 1}. Produce the final answer only after the verification step passes.")

        demo = (
            "RLHF can lead to reward hacking because the model learns to optimize a proxy reward model rather than "
            "the full human intent. If the proxy reward is misspecified, the policy can exploit shortcuts that look "
            "good to the evaluator but are actually misaligned."
        )
        if flags["safety"] or flags["verify"]:
            demo += (
                " A practical mitigation is to combine stronger auditing, adversarial testing, monitoring, and held-out evaluations "
                "so the model is rewarded for robust behavior instead of easy-to-game signals."
            )
        return "## Improved Strategy\n" + "\n".join(steps) + "\n\n## Demonstration\n" + demo

    return task_prompt


def build_heuristic_action(
    task_prompt: str,
    task_type: str,
    strategy_text: str | None = None,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
) -> GodelAction:
    solution = build_heuristic_solution(task_prompt, task_type, strategy_text=strategy_text)
    edit_type = (
        EditType.FIX_ERRORS
        if task_type in {"code_improvement", "python_optimized"}
        else EditType.RESTRUCTURE
    )

    patch = None
    if task_type == "strategy_optimization":
        patch = build_heuristic_strategy_patch(
            strategy_text=strategy_text,
            recent_failures=recent_failures,
            downstream_scores=downstream_scores,
        )

    return GodelAction(
        solution=solution,
        edit_type=edit_type,
        strategy_note="Deterministic fallback (LLM unavailable)",
        strategy_patch=patch,
    )


def build_heuristic_strategy_patch(
    strategy_text: str | None = None,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
) -> StrategyPatch:
    """
    Generate a VARIED StrategyPatch based on current strategy weaknesses.

    Unlike hardcoded patches, this analyzes what the current strategy is missing
    and proposes a targeted improvement. This is still deterministic (for
    reproducibility) but not static.
    """
    flags = _strategy_flags(strategy_text)
    focus = _select_patch_focus(flags, recent_failures, downstream_scores)

    addition_text, diff_desc, hypothesis = CAPABILITY_ADDITIONS.get(
        focus,
        (
            "VERIFY: Run a self-check before finalizing.",
            "Added verification step",
            "Verification catches errors before they propagate",
        ),
    )

    current_steps = []
    step_num = 1

    if strategy_text:
        lines = [line.strip() for line in strategy_text.split("\n") if line.strip()]
        for line in lines:
            if line and not line.startswith("#") and not line.upper().startswith("REASONING STRATEGY"):
                current_steps.append(f"{step_num}. {line.lstrip('0123456789.-) ')}")
                step_num += 1

    if not current_steps:
        current_steps = [
            "1. Read and understand the task requirements.",
            "2. Draft an initial solution.",
            "3. Review and refine the solution.",
        ]
        step_num = 4

    current_steps.append(f"{step_num}. {addition_text}")

    improved_strategy = "REASONING STRATEGY (heuristic patch)\n\n" + "\n".join(current_steps)

    target_weaknesses = [
        weakness
        for weakness, missing in [
            ("no decomposition", not flags["decompose"]),
            ("no explicit evidence gathering", not flags["evidence"]),
            ("no counterargument generation", not flags["counterargument"]),
            ("no worked examples", not flags["examples"]),
            ("no optimization check", not flags["efficiency"]),
            ("no safety check", not flags["safety"]),
            ("no self-verification", not flags["verify"]),
            ("no uncertainty tracking", not flags["uncertainty"]),
            ("no revision loop", not flags["reflect"]),
        ]
        if missing
    ]

    if recent_failures:
        target_weaknesses.extend(
            f"recent failure: {f[:50]}..." if len(f) > 50 else f"recent failure: {f}"
            for f in list(recent_failures)[-2:]
        )

    return StrategyPatch(
        improved_strategy=improved_strategy,
        diff_description=f"[HEURISTIC] {diff_desc}",
        hypothesis=f"[HEURISTIC] {hypothesis}",
        target_weaknesses=target_weaknesses[:5],
    )
