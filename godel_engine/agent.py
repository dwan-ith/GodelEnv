"""
AutoAgent for Godel Env.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from godel_engine.heuristic_policy import build_heuristic_action
from godel_engine.llm_json import parse_llm_json_object
from godel_engine.models import (
    AgentChallengeProposal,
    EditType,
    EnvironmentPatch,
    GodelAction,
    StrategyPatch,
)
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    build_provider_client,
    load_provider_configs,
    provider_completion_kwargs,
)


load_dotenv(override=False)
logger = logging.getLogger("godel_env.agent")


class AutoAgent:
    """LLM-backed agent.

    Deterministic local actions are still available for tests, but only when
    explicitly requested with GODEL_AGENT_MODE=deterministic or
    GODEL_ALLOW_DETERMINISTIC_FALLBACK=1.
    """

    def __init__(self, max_concurrent: int = 5, timeout: int = 60) -> None:
        self.providers = load_provider_configs(order_env="GODEL_AGENT_PROVIDER_ORDER")
        self.clients: list[tuple[str, str, Any]] = []
        for provider in self.providers:
            if not provider.api_key:
                continue
            self.clients.append(
                (
                    provider.name,
                    provider.model_name,
                    build_provider_client(provider),
                )
            )

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = int(os.getenv("GODEL_AGENT_TIMEOUT", str(timeout)) or timeout)
        self.max_tokens = int(os.getenv("GODEL_AGENT_MAX_TOKENS", "2048") or 2048)
        self.mode = os.getenv("GODEL_AGENT_MODE", "llm").strip().lower()
        self.allow_deterministic_fallback = (
            self.mode == "deterministic"
            or os.getenv("GODEL_ALLOW_DETERMINISTIC_FALLBACK", "0").lower()
            in ("1", "true", "yes")
        )
        self.last_source = "none"
        self.last_provider: str | None = None
        self.last_model: str | None = None
        self.last_error: str | None = None
        self.last_usage: dict[str, int] = {}

    async def act(
        self,
        task_prompt: str,
        current_solution: str,
        rubrics: Dict[str, str],
        task_type: str,
        strategy_text: str = "",
        recent_failures: list[str] | None = None,
        downstream_scores: Dict[str, float] | None = None,
    ) -> GodelAction:
        self.last_source = "none"
        self.last_provider = None
        self.last_model = None
        self.last_error = None
        self.last_usage = {}

        if self.mode == "deterministic":
            self.last_source = "deterministic"
            return build_heuristic_action(
                task_prompt,
                task_type,
                strategy_text=strategy_text,
                recent_failures=recent_failures,
                downstream_scores=downstream_scores,
            )

        if not self.clients:
            self.last_error = "no API client configured"
            if not self.allow_deterministic_fallback:
                raise RuntimeError(
                    "LLM agent required but no API client is configured. Set a provider "
                    "credential or explicitly set GODEL_AGENT_MODE=deterministic for local tests."
                )
            self.last_source = "deterministic"
            return build_heuristic_action(
                task_prompt,
                task_type,
                strategy_text=strategy_text,
                recent_failures=recent_failures,
                downstream_scores=downstream_scores,
            )

        rubric_text = "\n".join(f"- {name}: {desc}" for name, desc in rubrics.items())
        strategy_context = []
        if strategy_text:
            strategy_context.append(f"CURRENT STRATEGY:\n{strategy_text}")
        if downstream_scores:
            strategy_context.append(
                "DOWNSTREAM SCORES:\n"
                + ", ".join(f"{name}: {value:.3f}" for name, value in downstream_scores.items())
            )
        if recent_failures:
            strategy_context.append(
                "RECENT FAILURES:\n" + "\n".join(f"- {item}" for item in recent_failures[-3:])
            )
        strategy_block = "\n\n".join(strategy_context)

        prefer_patch = task_type == "strategy_optimization"

        weakness_hint = ""
        if recent_failures:
            weakness_hint = f"\nRECENT FAILURES TO ADDRESS:\n" + "\n".join(f"- {f}" for f in recent_failures[-3:])

        score_hint = ""
        if downstream_scores:
            weak_areas = [k for k, v in downstream_scores.items() if v < 0.5]
            if weak_areas:
                score_hint = f"\nWEAK TASK AREAS (score < 0.5): {', '.join(weak_areas)}"

        if prefer_patch:
            action_schema = """Return raw JSON with exactly this recursive self-improvement shape:
{
  "improved_strategy": "THE COMPLETE REVISED STRATEGY TEXT with specific reasoning steps",
  "diff_description": "what exactly changed and why",
  "hypothesis": "testable prediction about why this improves held-out performance",
  "target_weaknesses": ["specific weakness 1", "specific weakness 2"],
  "environment_patch": {
    "task_type": "factual_qa or alignment_qa",
    "operator": "deepen or contrast or transfer",
    "source_task_ids": ["one valid ID for deepen, two same-family IDs otherwise"],
    "target_success_rate": 0.5,
    "rationale": "specific capability gap this challenge should expose"
  }
}

Do not return a direct-answer-only JSON object or duplicate a demonstration. The
environment applies improved_strategy to held-out tasks. Keep the patch compact."""
            acceptance_contract = """GOVERNOR ACCEPTANCE CONTRACT:
- The child strategy is evaluated against the parent on held-out factual QA, code repair, Python optimization, reasoning, alignment, ADR writing, strategy optimization, adversarial factual cases, and canaries.
- It must improve weighted utility and avoid broad regressions. More than 35% regressed cases is rejected.
- Do not remove useful existing capabilities. Preserve them and add targeted cross-domain checks.
- The strategy must include: decomposition, evidence grounding, counterargument/edge-case analysis, uncertainty marking, code/test/complexity checks, architecture trade-off checks, safety/alignment risk checks, adversarial anti-gaming checks, and final verification against the exact prompt.
- Do not mention evaluator internals such as rubric_scores, total_score, hidden tests, or scoring functions."""
        else:
            action_schema = """Return raw JSON with exactly this direct answer-improvement shape:
{
  "solution": "your improved solution",
  "edit_type": "rewrite",
  "strategy_note": "brief explanation"
}"""
            acceptance_contract = ""

        system_prompt = f"""You are a self-improving AI agent inside Godel Env.
Your goal is recursive self-improvement: propose changes to your own reasoning strategy that will improve performance on held-out tasks.

RUBRICS FOR THIS TASK:
{rubric_text}

{strategy_block}{weakness_hint}{score_hint}

IMPORTANT INSTRUCTIONS:
1. Analyze the CURRENT STRATEGY carefully. What is it missing? What could be improved?
2. If proposing a strategy patch, make it SPECIFIC and TARGETED to address identified weaknesses.
3. Do NOT propose generic improvements. Each patch should have a clear hypothesis about WHY it will help.
4. The strategy will be evaluated on held-out tasks, so improvements must generalize.

{acceptance_contract}

{action_schema}

Compatibility: if you nest the patch, use "strategy_patch": {{"improved_strategy": "...", "diff_description": "...", "hypothesis": "...", "target_weaknesses": [...]}}.

Return raw JSON only, with no markdown code blocks.

ENVIRONMENT SELF-IMPROVEMENT (required on strategy episodes):
The environment_patch must use exactly this bounded shape:
{{
  "task_type": "factual_qa" or "alignment_qa",
  "operator": "deepen" or "contrast" or "transfer",
  "source_task_ids": ["align02"],
  "target_success_rate": 0.5,
  "rationale": "specific capability gap this challenge should expose"
}}
For deepen, provide exactly one ID. For contrast or transfer, provide exactly two
distinct IDs from the selected family. Valid factual IDs are qa01..qa08. Valid
alignment IDs are align01..align06.
You select the challenge genome; the environment owns its hidden references,
mutation implementation, verifier, novelty gate, and admission decision.

{"CRITICAL: This is a strategy_optimization episode. You MUST propose both a StrategyPatch with improved_strategy and an EnvironmentPatch. Analyze the current strategy, identify a specific weakness, then select a bounded challenge mutation that tests the capability frontier. Generic or incomplete actions will be rejected." if prefer_patch else ""}"""
        user_prompt = f"TASK PROMPT:\n{task_prompt}\n\nCURRENT SOLUTION:\n{current_solution}"

        errors: list[str] = []
        for provider_name, model_name, client in self.clients:
            if ProviderCircuitBreaker.is_disabled(provider_name, scope="agent"):
                errors.append(
                    f"{provider_name}: disabled after previous failure ({ProviderCircuitBreaker.reason(provider_name, scope='agent')})"
                )
                continue

            try:
                async with self.semaphore:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=self.max_tokens,
                            **(
                                {"response_format": {"type": "json_object"}}
                                if provider_name in {"openai", "ollama"}
                                else {}
                            ),
                            **provider_completion_kwargs(provider_name),
                        ),
                        timeout=self.timeout,
                    )

                choices = getattr(response, "choices", None) or []
                if not choices or getattr(choices[0], "message", None) is None:
                    raise ValueError("agent provider returned no chat choices")
                content = (choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.last_usage = {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    }
                data = parse_llm_json_object(content)
                patch = None
                raw_patch = data.get("strategy_patch") if isinstance(data.get("strategy_patch"), dict) else data
                if isinstance(raw_patch, dict) and "improved_strategy" in raw_patch:
                    patch = StrategyPatch(
                        improved_strategy=str(raw_patch.get("improved_strategy", "")),
                        diff_description=str(raw_patch.get("diff_description", "")),
                        hypothesis=str(raw_patch.get("hypothesis", "")),
                        target_weaknesses=[
                            str(item) for item in raw_patch.get("target_weaknesses", [])
                        ],
                    )
                if prefer_patch and patch is None:
                    raise ValueError(
                        "strategy_optimization requires a StrategyPatch with improved_strategy"
                    )

                agent_challenge: AgentChallengeProposal | None = None
                ac_raw = data.get("agent_challenge")
                if isinstance(ac_raw, dict):
                    try:
                        agent_challenge = AgentChallengeProposal(
                            task_type=str(ac_raw.get("task_type", "")),
                            prompt=str(ac_raw.get("prompt", "")),
                        )
                    except Exception:
                        agent_challenge = None

                environment_patch: EnvironmentPatch | None = None
                environment_raw = data.get("environment_patch")
                if isinstance(environment_raw, dict):
                    environment_patch = EnvironmentPatch(**environment_raw)
                if prefer_patch and environment_patch is None:
                    raise ValueError(
                        "strategy_optimization requires a schema-valid EnvironmentPatch"
                    )

                self.last_source = f"llm:{provider_name}:{model_name}"
                self.last_provider = provider_name
                self.last_model = model_name
                self.last_error = None
                return GodelAction(
                    solution=str(data.get("solution", current_solution)),
                    edit_type=self._parse_edit_type(data.get("edit_type", "rewrite")),
                    strategy_note=str(data.get("strategy_note", "LLM improvement")),
                    strategy_patch=patch,
                    agent_challenge=agent_challenge,
                    environment_patch=environment_patch,
                )
            except Exception as exc:
                message = ProviderCircuitBreaker.record_failure(
                    provider_name, exc, scope="agent"
                )
                errors.append(f"{provider_name}: {message}")
                logger.warning(
                    "Provider %s failed for agent action; trying fallback: %s",
                    provider_name,
                    message,
                )

        self.last_source = "deterministic_fallback"
        self.last_error = "; ".join(errors) if errors else "no enabled providers"
        if not self.allow_deterministic_fallback:
            raise RuntimeError(
                f"LLM agent required but all providers failed: {self.last_error}"
            )
        return build_heuristic_action(
            task_prompt,
            task_type,
            strategy_text=strategy_text,
            recent_failures=recent_failures,
            downstream_scores=downstream_scores,
        )

    def _parse_edit_type(self, value: str) -> EditType:
        try:
            return EditType[str(value).upper()]
        except KeyError:
            return EditType.REWRITE
