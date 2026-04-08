"""
AutoAgent — LLM-based agent for Gödel Env.
Reads the task prompt + current solution, then proposes an improved solution.
Strictly uses OpenAI client as per hackathon requirements.
"""
from __future__ import annotations
import os
import json
import asyncio
import logging
from typing import Dict

from openai import AsyncOpenAI
from dotenv import load_dotenv

from godel_engine.models import GodelAction, EditType

load_dotenv(override=True)
logger = logging.getLogger("godel_env.agent")


class AutoAgent:
    """LLM-based agent that reads task + solution and proposes an improvement."""

    def __init__(self, max_concurrent: int = 5, timeout: int = 60):
        # Strictly use keys as requested
        self.api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self.base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout

    async def act(
        self,
        task_prompt: str,
        current_solution: str,
        rubrics: Dict[str, str],
        task_type: str,
    ) -> GodelAction:
        if not self.client:
            logger.warning("No API key found. Returning unmodified solution.")
            return GodelAction(
                solution=current_solution,
                edit_type=EditType.REFINE,
                strategy_note="No API key — skipped",
            )

        rubric_text = "\n".join(f"- {k}: {v}" for k, v in rubrics.items())
        system_prompt = f"""You are a self-improving AI agent inside the Gödel Env reinforcement learning environment.
Your job is to rewrite the CURRENT SOLUTION to earn a higher score against these rubrics:
{rubric_text}

Use chain-of-thought reasoning first, then output a raw JSON object (no markdown) with exactly this schema:
{{
    "reasoning_steps": ["step 1...", "step 2..."],
    "solution": "The complete improved solution text...",
    "edit_type": "rewrite",
    "strategy_note": "One-sentence summary of your main improvement."
}}"""
        user_prompt = (
            f"TASK PROMPT:\n{task_prompt}\n\n"
            f"CURRENT SOLUTION:\n{current_solution}"
        )

        try:
            async with self.semaphore:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=4096,
                    ),
                    timeout=self.timeout,
                )

            content = response.choices[0].message.content.strip()
            # Strip accidental markdown fences
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {"solution": content, "edit_type": "rewrite", "strategy_note": "Salvaged from malformed JSON"}

            solution = data.get("solution", current_solution)
            if not isinstance(solution, str):
                solution = json.dumps(solution, indent=2)

            edit_type_str = data.get("edit_type", "rewrite").upper()
            try:
                edit_type = EditType[edit_type_str]
            except KeyError:
                edit_type = EditType.REWRITE

            return GodelAction(
                solution=solution,
                edit_type=edit_type,
                strategy_note=str(data.get("strategy_note", "LLM auto-improvement")),
            )

        except asyncio.TimeoutError:
            logger.error(f"Agent LLM timed out after {self.timeout}s.")
            return GodelAction(
                solution=current_solution,
                edit_type=EditType.REFINE,
                strategy_note=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            err = str(e).replace("\n", " ")[:200]
            logger.error(f"Agent LLM error: {type(e).__name__} — {err}")
            return GodelAction(
                solution=current_solution,
                edit_type=EditType.REFINE,
                strategy_note=f"API error: {err}",
            )
