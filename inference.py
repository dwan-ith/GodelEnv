"""
Inference Script Example for GodelEnv
=====================================
Strictly complies with the Hackathon Output formatting.
"""

import os
import sys
import asyncio
import json
from typing import List

from openai import AsyncOpenAI
from dotenv import load_dotenv

from godel_engine.environment import GodelEnvironment
from godel_engine.agent import AutoAgent
from godel_engine.models import GodelAction

load_dotenv(override=False)

# Force standard ASCII output to avoid charmap errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

def format_action(action: GodelAction) -> str:
    # Safely format the action string without newlines
    edit_type = action.edit_type.value if hasattr(action.edit_type, 'value') else str(action.edit_type)
    # create a super-truncated payload representation to avoid breaking STDOUT regex parsers
    safe_note = action.strategy_note.replace('\n', ' ').replace('\r', '')
    return f"{edit_type}('{safe_note[:30]}')"

async def run_inference():
    env = GodelEnvironment(seed=42)
    agent = AutoAgent()
    
    # We will run 1 episode of each task to demonstrate everything
    # The hackathon script often takes over the environment wrapper, but here we run a loop
    # We'll just run a specific challenging task to show capability
    tasks_to_test = ["factual_qa", "code_improvement", "strategy_optimization"]
    
    for task_type in tasks_to_test:
        try:
            result = await env.reset(task_type=task_type, seed=42)
            obs = result.observation
            
            # [START] task=<task_name> env=<benchmark> model=<model_name>
            print(f"[START] task={task_type} env=GodelEnv model={MODEL_NAME}")
            
            terminated = False
            truncated = False
            rewards = []
            
            while not (terminated or truncated):
                # Step the env
                action_result = await agent.act(
                    task_prompt=obs.task_prompt,
                    current_solution=obs.current_solution,
                    rubrics=obs.rubric_scores.scores or env.current_task._get_rubrics(),
                    task_type=task_type,
                )
                
                try:
                    step_result = await env.step(action_result)
                    error_msg = "null"
                except Exception as e:
                    step_result = result # reuse last result
                    step_result.reward = -1.0
                    step_result.terminated = True
                    error_msg = str(e).replace('\n', ' ')
                
                obs = step_result.observation
                reward_val = step_result.reward
                rewards.append(reward_val)
                terminated = step_result.terminated
                truncated = step_result.truncated
                
                is_done = str(terminated or truncated).lower()
                action_str = format_action(action_result)
                
                # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
                print(f"[STEP] step={obs.step} action={action_str} reward={reward_val:.2f} done={is_done} error={error_msg}")
            
            # Episode complete
            is_success = str(obs.total_score >= 0.95).lower()
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
            print(f"[END] success={is_success} steps={obs.step} score={obs.total_score:.2f} rewards={rewards_str}")

        except Exception as ep_error:
            # If the entire episode crashes
            error_val = str(ep_error).replace("\n", " ")
            print(f"[END] success=false steps=0 score=0.00 rewards= error={error_val}")

if __name__ == "__main__":
    asyncio.run(run_inference())
