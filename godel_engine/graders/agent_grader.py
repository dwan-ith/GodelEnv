"""
Production-Grade Agent Grader using Native OpenAI SDK.
"""
import os
import json
import asyncio
import logging
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger("godel_env.grader")

class AgentGrader:
    def __init__(self, max_concurrent: int = 10, timeout: int = 30):
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

    async def grade(
        self, 
        task_prompt: str, 
        current_solution: str, 
        rubrics: Dict[str, str]
    ) -> tuple[float, Dict[str, float], Dict[str, str]]:
        """
        Evaluate a solution using an LLM with CoT reasoning and concurrency safeguards.
        """
        if not self.client:
            logger.warning("No API key found. Falling back to simulated heuristic grading.")
            return self._simulate_grading(rubrics)

        rubric_description = "\n".join([f"- {name}: {desc}" for name, desc in rubrics.items()])
        
        system_prompt = f"""You are an expert AI grader for a production reinforcement learning environment.
Your goal is to meticulously evaluate the quality of a solution based on the following rubrics:
{rubric_description}

You must output a raw JSON object. Use a Chain-of-Thought approach to explain your reasoning FIRST, and provide the final scores SECOND.

JSON SCHEMA:
{{
  "reasoning_steps": [
    "Step 1: Analyze the solution...",
    "Step 2: Compare against rubric X..."
  ],
  "scores": {{ "rubric_name": 0.8, ... }},
  "feedback": {{ "rubric_name": "Specific actionable feedback", ... }}
}}
"""

        user_content = f"TASK PROMPT:\n{task_prompt}\n\nCURRENT SOLUTION TO GRADE:\n{current_solution}"

        try:
            async with self.semaphore:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={ "type": "json_object" },
                        max_tokens=2048
                    ),
                    timeout=self.timeout
                )
            
            data = json.loads(response.choices[0].message.content)
            scores = data.get("scores", {})
            feedback = data.get("feedback", {})
            
            final_scores = {name: float(scores.get(name, 0.0)) for name in rubrics}
            final_feedback = {name: str(feedback.get(name, "No explicit feedback provided.")) for name in rubrics}
            
            total_score = sum(final_scores.values()) / len(final_scores) if final_scores else 0.0
            
            return total_score, final_scores, final_feedback

        except asyncio.TimeoutError:
            logger.error(f"Grading LLM call timed out after {self.timeout}s.")
            return self._error_grading(rubrics, "TimeoutError: LLM took too long to respond.")
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            logger.error(f"Grading LLM error: {type(e).__name__} - {error_msg}")
            return self._error_grading(rubrics, f"{type(e).__name__}: {error_msg[:150]}")

    def _error_grading(self, rubrics: Dict[str, str], error_msg: str) -> tuple[float, Dict[str, float], Dict[str, str]]:
        scores = {name: 0.0 for name in rubrics}
        feedback = {name: f"[API ERROR] {error_msg}" for name in rubrics}
        return 0.0, scores, feedback

    def _simulate_grading(self, rubrics: Dict[str, str]) -> tuple[float, Dict[str, float], Dict[str, str]]:
        scores = {name: 0.5 for name in rubrics}
        feedback = {name: "Simulated feedback (No API Key)" for name in rubrics}
        return 0.5, scores, feedback
