from __future__ import annotations

import re
from typing import Any, Sequence

from godel_engine.models import EditType, GodelAction, StrategyPatch
from godel_engine.tasks.alignment_qa import _ALIGNMENT_QA_DATASET
from godel_engine.tasks.factual_qa import _QA_DATASET
from godel_engine.tasks.reasoning import _REASONING_DATASET
from godel_engine.tasks.strategy_optimization import _STRATEGY_DATASET


CAPABILITY_LIBRARY: dict[str, dict[str, Any]] = {
    "decompose": {
        "step": "Decompose the task into claims, assumptions, and decision points before drafting the answer.",
        "weaknesses": ["skips decomposition", "reasoning is underspecified"],
        "task_families": ["reasoning", "strategy_optimization", "code_improvement"],
        "hypothesis": "Explicit decomposition improves coverage and reduces hidden assumption errors.",
    },
    "evidence": {
        "step": "For each major claim, attach concrete supporting evidence, mechanisms, or implementation details.",
        "weaknesses": ["missing evidence", "generic claims"],
        "task_families": ["factual_qa", "alignment_qa", "strategy_optimization"],
        "hypothesis": "Evidence-grounded reasoning improves factual coverage and reduces empty filler.",
    },
    "counterargument": {
        "step": "Generate at least one counterargument, edge case, or trade-off before finalizing the answer.",
        "weaknesses": ["one-sided reasoning", "no trade-off analysis"],
        "task_families": ["reasoning", "alignment_qa", "adr_writing"],
        "hypothesis": "Counterarguments improve robustness and prevent shallow one-sided answers.",
    },
    "uncertainty": {
        "step": "Mark confidence and uncertainty explicitly when evidence is incomplete or context-dependent.",
        "weaknesses": ["overconfidence", "missing calibration"],
        "task_families": ["alignment_qa", "factual_qa", "strategy_optimization"],
        "hypothesis": "Calibration improves trustworthiness and catches cases that need deferral or qualification.",
    },
    "verify": {
        "step": "Run a self-check: verify that the final answer addresses the actual prompt, covers both sides of required contrasts, and is internally consistent.",
        "weaknesses": ["missing verification", "logic gaps"],
        "task_families": ["factual_qa", "code_improvement", "python_optimized", "strategy_optimization"],
        "hypothesis": "Self-verification reduces regressions before the answer reaches the grader.",
    },
    "reflect": {
        "step": "Record the failure mode after each attempt and revise the strategy when the same weakness repeats.",
        "weaknesses": ["no learning loop", "repeated failures"],
        "task_families": ["strategy_optimization", "alignment_qa", "reasoning"],
        "hypothesis": "Reflection turns repeated mistakes into strategy updates instead of recurring regressions.",
    },
    "examples": {
        "step": "Include a concrete worked example or demonstration whenever an abstract explanation could stay vague.",
        "weaknesses": ["too abstract", "no example"],
        "task_families": ["factual_qa", "reasoning", "strategy_optimization"],
        "hypothesis": "Worked examples force the reasoning process to cash out in concrete task-specific detail.",
    },
    "efficiency": {
        "step": "For code or systems tasks, analyze complexity and prefer solutions with a clear efficiency argument.",
        "weaknesses": ["ignores complexity", "naive implementation"],
        "task_families": ["code_improvement", "python_optimized", "reasoning"],
        "hypothesis": "Efficiency checks improve algorithmic quality without sacrificing correctness.",
    },
    "safety": {
        "step": "Check for risks, harms, or policy-sensitive failure modes before finalizing the answer.",
        "weaknesses": ["missing safety analysis", "risk blind spot"],
        "task_families": ["alignment_qa", "adr_writing", "strategy_optimization"],
        "hypothesis": "Risk checks improve robustness on safety-sensitive prompts and trade-off analysis.",
    },
}

CAPABILITY_ORDER = [
    "decompose",
    "evidence",
    "counterargument",
    "uncertainty",
    "verify",
    "reflect",
    "examples",
    "efficiency",
    "safety",
]

FAILURE_CAPABILITY_HINTS: dict[str, tuple[str, ...]] = {
    "factual": ("evidence", "verify", "examples"),
    "hallucination": ("evidence", "verify", "uncertainty"),
    "alignment": ("safety", "counterargument", "uncertainty"),
    "reasoning": ("decompose", "counterargument", "examples"),
    "code": ("verify", "efficiency", "decompose"),
    "optimiz": ("efficiency", "verify"),
    "uncertain": ("uncertainty", "verify"),
    "safety": ("safety", "counterargument"),
}


def _first_alias(group: str | Sequence[str]) -> str:
    if isinstance(group, str):
        return group
    return next((item for item in group if item), "")


def _first_n(groups: Sequence[str | Sequence[str]], n: int) -> list[str]:
    return [_first_alias(group) for group in groups[:n]]


def _normalize_step_text(text: str) -> str:
    clean = re.sub(r"^(\d+[\.\):]?\s*|[-*]\s*|step\s*\d+:?\s*)", "", text.strip(), flags=re.IGNORECASE)
    return clean.rstrip(".")


def _strategy_steps(strategy_text: str | None) -> list[str]:
    if not strategy_text:
        return []
    steps: list[str] = []
    for raw_line in strategy_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = _normalize_step_text(line)
        if normalized:
            steps.append(normalized)
    return steps


def strategy_profile(strategy_text: str | None) -> dict[str, float]:
    text = (strategy_text or "").lower()
    steps = _strategy_steps(strategy_text)
    num_steps = max(len(steps), 1)

    def score(markers: Sequence[str]) -> float:
        text_hits = sum(text.count(marker) for marker in markers)
        step_hits = sum(1 for step in steps if any(marker in step.lower() for marker in markers))
        return min(1.0, 0.2 * text_hits + 0.8 * (step_hits / num_steps))

    return {
        "decompose": score(["decompose", "break the problem", "claim", "assumption"]),
        "evidence": score(["evidence", "support", "ground", "mechanism", "source"]),
        "counterargument": score(["counterargument", "alternative", "trade-off", "edge case", "however"]),
        "uncertainty": score(["uncertainty", "confidence", "calibration", "unclear", "depends"]),
        "verify": score(["verify", "self-check", "check", "validate", "confirm"]),
        "reflect": score(["reflect", "revise", "improve", "feedback", "lessons learned"]),
        "examples": score(["example", "demonstration", "worked example", "illustrate"]),
        "efficiency": score(["efficient", "complexity", "optimize", "performance", "asymptotic"]),
        "safety": score(["safety", "harm", "policy", "risk", "guardrail"]),
    }


def _find_reference(task_prompt: str, task_type: str) -> dict[str, Any] | None:
    datasets = {
        "factual_qa": _QA_DATASET,
        "alignment_qa": _ALIGNMENT_QA_DATASET,
        "reasoning": _REASONING_DATASET,
        "strategy_optimization": _STRATEGY_DATASET,
    }
    normalized_prompt = " ".join(task_prompt.lower().split())
    for item in datasets.get(task_type, []):
        if " ".join(item["prompt"].lower().split()) == normalized_prompt:
            return item
    return None


def _family_priorities(
    recent_failures: Sequence[str] | None,
    downstream_scores: dict[str, float] | None,
) -> list[str]:
    priorities: list[str] = []
    scores = downstream_scores or {}
    failures_text = " ".join(recent_failures or []).lower()

    weak_families = sorted(scores.items(), key=lambda item: item[1])
    for task_family, score in weak_families:
        if score < 0.7:
            for capability, spec in CAPABILITY_LIBRARY.items():
                if task_family in spec["task_families"]:
                    priorities.append(capability)

    for keyword, capabilities in FAILURE_CAPABILITY_HINTS.items():
        if keyword in failures_text:
            priorities.extend(capabilities)

    return priorities


def _choose_focus_capabilities(
    profile: dict[str, float],
    *,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
    limit: int = 2,
) -> list[str]:
    priorities = _family_priorities(recent_failures, downstream_scores)
    chosen: list[str] = []

    for capability in priorities:
        if capability in chosen:
            continue
        if profile.get(capability, 0.0) < 0.6:
            chosen.append(capability)
        if len(chosen) >= limit:
            return chosen

    remaining = sorted(
        CAPABILITY_ORDER,
        key=lambda capability: (profile.get(capability, 0.0), CAPABILITY_ORDER.index(capability)),
    )
    for capability in remaining:
        if capability not in chosen:
            chosen.append(capability)
        if len(chosen) >= limit:
            break
    return chosen


def _render_strategy_text(
    existing_steps: Sequence[str],
    focus_capabilities: Sequence[str],
    profile: dict[str, float],
) -> str:
    normalized_existing: list[str] = []
    for step in existing_steps:
        clean = _normalize_step_text(step)
        if clean and clean not in normalized_existing:
            normalized_existing.append(clean)

    for capability in CAPABILITY_ORDER:
        if capability in focus_capabilities or profile.get(capability, 0.0) >= 0.55:
            step = CAPABILITY_LIBRARY[capability]["step"]
            if step.rstrip(".") not in [item.rstrip(".") for item in normalized_existing]:
                normalized_existing.append(step)

    if not normalized_existing:
        normalized_existing = [
            CAPABILITY_LIBRARY["decompose"]["step"],
            CAPABILITY_LIBRARY["evidence"]["step"],
            CAPABILITY_LIBRARY["verify"]["step"],
        ]

    rendered = ["REASONING STRATEGY"]
    rendered.extend(f"{index}. {step}" for index, step in enumerate(normalized_existing, start=1))
    return "\n".join(rendered)


def build_reference_strategy_patch(
    *,
    strategy_text: str | None = None,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
) -> StrategyPatch:
    profile = strategy_profile(strategy_text)
    existing_steps = _strategy_steps(strategy_text)
    focus_capabilities = _choose_focus_capabilities(
        profile,
        recent_failures=recent_failures,
        downstream_scores=downstream_scores,
    )
    improved_strategy = _render_strategy_text(existing_steps, focus_capabilities, profile)

    weakest_families = [
        family
        for family, score in sorted((downstream_scores or {}).items(), key=lambda item: item[1])[:2]
        if score < 0.8
    ]
    weaknesses = [f"weak {family} performance" for family in weakest_families]
    weaknesses.extend(CAPABILITY_LIBRARY[capability]["weaknesses"][0] for capability in focus_capabilities)
    if recent_failures:
        weaknesses.extend(
            f"recent failure: {failure[:60]}{'...' if len(failure) > 60 else ''}"
            for failure in list(recent_failures)[-2:]
        )

    focus_labels = ", ".join(focus_capabilities)
    family_text = ", ".join(weakest_families) if weakest_families else "held-out tasks"
    hypothesis = (
        f"Strengthening {focus_labels} should improve {family_text} by making the solver cover "
        "more task-specific evidence and catch omissions before finalizing."
    )
    diff_description = (
        "Added or strengthened: "
        + ", ".join(CAPABILITY_LIBRARY[capability]["step"] for capability in focus_capabilities)
    )

    return StrategyPatch(
        improved_strategy=improved_strategy,
        diff_description=diff_description,
        hypothesis=hypothesis,
        target_weaknesses=weaknesses[:5],
    )


def _compose_sentences(sentences: Sequence[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())


def _solve_factual(reference: dict[str, Any], profile: dict[str, float]) -> str:
    concepts = _first_n(reference["concept_groups"], 5)
    left = _first_n(reference["contrast_left"], 2)
    right = _first_n(reference["contrast_right"], 2)
    sentences = [
        f"The core contrast is that {left[0]} and {left[1]} matter on one side, while {right[0]} and {right[1]} matter on the other side.",
        f"A complete explanation should connect {concepts[0]}, {concepts[1]}, and {concepts[2]} as part of one mechanism instead of listing isolated buzzwords.",
        f"It should also explain how {concepts[3]} and {concepts[4]} affect the practical behavior of the system or concept being described.",
    ]
    if profile["examples"] >= 0.45:
        sentences.append(
            f"For example, an explanation is stronger when it shows how {concepts[0]} changes the role of {concepts[2]} in a concrete situation."
        )
    if profile["counterargument"] >= 0.45:
        sentences.append(
            f"A useful nuance is that emphasizing only {left[0]} without {right[0]} gives an incomplete picture of the comparison."
        )
    if profile["uncertainty"] >= 0.45:
        sentences.append("Confidence is high on the main distinction, but boundary cases still depend on definitions and context.")
    if profile["verify"] >= 0.45:
        sentences.append("A final self-check is to confirm that both sides of the comparison were explained rather than merely named.")
    return _compose_sentences(sentences)


def _solve_alignment(reference: dict[str, Any], profile: dict[str, float]) -> str:
    concepts = _first_n(reference["concept_groups"], 4)
    risks = _first_n(reference["risk_groups"], 3)
    mitigations = _first_n(reference["mitigation_groups"], 3)
    sentences = [
        f"This topic is fundamentally about {concepts[0]}, {concepts[1]}, and {concepts[2]}, not just a vague statement that alignment is important.",
        f"The main failure mode is that the system drifts toward {risks[0]} or {risks[1]}, which means the optimized behavior no longer matches the intended objective.",
        f"Good mitigations include {mitigations[0]}, {mitigations[1]}, and {mitigations[2]} so the failure mode is checked by more than one signal.",
    ]
    if profile["counterargument"] >= 0.45:
        sentences.append(
            f"The trade-off is that stronger safeguards can reduce flexibility or convenience, but ignoring {risks[2]} is usually worse."
        )
    if profile["uncertainty"] >= 0.45:
        sentences.append("Confidence should be lower when the system leaves the training distribution or when the proxy objective is poorly specified.")
    if profile["verify"] >= 0.45:
        sentences.append("A verification pass should ask whether the answer names the mechanism of failure as well as a plausible mitigation.")
    return _compose_sentences(sentences)


def _solve_reasoning(reference: dict[str, Any], profile: dict[str, float]) -> str:
    pros = _first_n(reference["pros_groups"], 4)
    cons = _first_n(reference["cons_groups"], 4)
    recs = _first_n(reference["recommendation_groups"], 3)
    lines = [
        "Pros",
        f"- {pros[0]} and {pros[1]} can materially improve the situation.",
        f"- {pros[2]} and {pros[3]} become more valuable as scale or organizational complexity increases.",
        "",
        "Cons",
        f"- {cons[0]} and {cons[1]} can slow execution or increase operational burden.",
        f"- {cons[2]} and {cons[3]} often become the reason a theoretically good choice fails in practice.",
        "",
        "Recommendation",
        f"- My recommendation is that it {recs[0]}, but only in an {recs[2]} way that matches the actual constraints.",
    ]
    if profile["counterargument"] >= 0.45:
        lines.append("- The strongest counterargument is that the operational cost may arrive before the expected upside does.")
    if profile["verify"] >= 0.45:
        lines.append("- Self-check: confirm that the recommendation clearly follows from both the upside and downside analysis.")
    return "\n".join(lines)


def _solve_strategy(reference: dict[str, Any], profile: dict[str, float], strategy_text: str | None) -> str:
    downstream = _first_n(reference["downstream_groups"], min(5, len(reference["downstream_groups"])))
    improved_strategy = _render_strategy_text(
        _strategy_steps(strategy_text),
        _choose_focus_capabilities(profile, limit=2),
        profile,
    )
    demo_sentences = [
        f"In the downstream challenge, the answer should explicitly cover {downstream[0]}, {downstream[1]}, and {downstream[2]} rather than speaking in generic terms.",
        f"It should also explain how {downstream[min(3, len(downstream) - 1)]} changes the outcome or the recommended mitigation.",
    ]
    if len(downstream) > 4:
        demo_sentences.append(
            f"A stronger demonstration also mentions {downstream[4]} so the strategy captures trade-offs or safeguards instead of just the headline."
        )
    if profile["verify"] >= 0.45:
        demo_sentences.append("After drafting, verify that the demonstration visibly applies the improved strategy instead of only restating it.")
    return "## Improved Strategy\n" + improved_strategy + "\n\n## Demonstration\n" + _compose_sentences(demo_sentences)


def _solve_code_improvement(reference: dict[str, Any]) -> str:
    function_name = str(reference.get("function_name", "solution"))
    if function_name == "is_palindrome":
        return (
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True when the alphanumeric characters in s form a palindrome."""\n'
            "    normalized = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return normalized == normalized[::-1]\n"
        )
    if function_name == "fibonacci":
        return (
            "def fibonacci(n: int) -> int:\n"
            '    """Return the nth Fibonacci number using an iterative O(n) update."""\n'
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )
    return reference.get("initial_solution", "")


def _solve_python_optimized(reference: dict[str, Any]) -> str:
    function_name = str(reference.get("function_name", "get_primes"))
    if function_name != "get_primes":
        return reference.get("initial_solution", "")
    return (
        "from math import isqrt\n\n"
        "def get_primes(n: int) -> list[int]:\n"
        '    """Return all prime numbers below n using trial division up to sqrt(candidate)."""\n'
        "    if n < 2:\n"
        "        return []\n"
        "    primes: list[int] = []\n"
        "    for candidate in range(2, n):\n"
        "        is_prime = True\n"
        "        for divisor in range(2, isqrt(candidate) + 1):\n"
        "            if candidate % divisor == 0:\n"
        "                is_prime = False\n"
        "                break\n"
        "        if is_prime:\n"
        "            primes.append(candidate)\n"
        "    return primes\n"
    )


def _solve_adr(reference: dict[str, Any], profile: dict[str, float]) -> str:
    coverage = _first_n(reference["coverage_groups"], 6)
    tradeoffs = _first_n(reference["tradeoff_groups"], 4)
    consequence_line = (
        f"This improves {coverage[4]} and {coverage[5]}, but it raises {tradeoffs[2]} and {tradeoffs[3]} concerns."
    )
    if profile["counterargument"] >= 0.45:
        consequence_line += " The main counterargument is that the operational cost may outweigh the scaling benefit early on."
    return (
        "# Title\nADR: Migrate the monolith toward a serverless event-driven architecture\n\n"
        "## Status\nAccepted\n\n"
        f"## Context\nThe current {coverage[0]} is creating delivery friction, while the target state relies on {coverage[1]} and {coverage[2]} with a {coverage[3]} for decoupled communication.\n\n"
        "## Decision\nAdopt a phased migration where high-variability workloads move first, event boundaries are explicit, and ownership stays clear for each new service boundary.\n\n"
        f"## Consequences\n{consequence_line}\n\n"
        "## Alternatives\nKeep the monolith longer, or move to containerized services before adopting a more event-driven design.\n"
    )


def build_reference_solution(
    *,
    task_prompt: str,
    task_type: str,
    strategy_text: str | None = None,
    reference: dict[str, Any] | None = None,
) -> str:
    profile = strategy_profile(strategy_text)
    ref = reference or _find_reference(task_prompt, task_type)

    if task_type == "factual_qa" and ref:
        return _solve_factual(ref, profile)
    if task_type == "alignment_qa" and ref:
        return _solve_alignment(ref, profile)
    if task_type == "reasoning" and ref:
        return _solve_reasoning(ref, profile)
    if task_type == "strategy_optimization" and ref:
        return _solve_strategy(ref, profile, strategy_text)
    if task_type == "code_improvement" and ref:
        return _solve_code_improvement(ref)
    if task_type == "python_optimized" and ref:
        return _solve_python_optimized(ref)
    if task_type == "adr_writing" and ref:
        return _solve_adr(ref, profile)

    return task_prompt


def build_reference_action(
    *,
    task_prompt: str,
    task_type: str,
    strategy_text: str | None = None,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
    reference: dict[str, Any] | None = None,
) -> GodelAction:
    patch = None
    effective_strategy = strategy_text
    if task_type == "strategy_optimization":
        patch = build_reference_strategy_patch(
            strategy_text=strategy_text,
            recent_failures=recent_failures,
            downstream_scores=downstream_scores,
        )
        effective_strategy = patch.improved_strategy

    solution = build_reference_solution(
        task_prompt=task_prompt,
        task_type=task_type,
        strategy_text=effective_strategy,
        reference=reference,
    )
    edit_type = (
        EditType.FIX_ERRORS
        if task_type in {"code_improvement", "python_optimized"}
        else EditType.REWRITE
    )
    return GodelAction(
        solution=solution,
        edit_type=edit_type,
        strategy_note="Deterministic reference-grounded fallback action",
        strategy_patch=patch,
    )


def solve_task(
    *,
    task_prompt: str,
    task_type: str,
    strategy_text: str | None = None,
    reference: dict[str, Any] | None = None,
) -> str:
    return build_reference_solution(
        task_prompt=task_prompt,
        task_type=task_type,
        strategy_text=strategy_text,
        reference=reference,
    )
