"""
Inference Script Example for GodelEnv

"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from godel_engine.client import GodelEngineEnv
from godel_engine.agent import AutoAgent
from godel_engine.models import GodelAction

load_dotenv(override=False)

# Force standard ASCII output to avoid charmap errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

def format_action(action: GodelAction) -> str:
    # Safely format the action string without newlines
    edit_type = action.edit_type.value if hasattr(action.edit_type, 'value') else str(action.edit_type)
    # create a super-truncated payload representation to avoid breaking STDOUT regex parsers
    safe_note = action.strategy_note.replace('\n', ' ').replace('\r', '')
    return f"{edit_type}('{safe_note[:30]}')"

async def run_inference():
    env_url = os.getenv("GODEL_URL", "http://localhost:7860")
    agent = AutoAgent()
    
    tasks_to_test = ["factual_qa", "code_improvement", "strategy_optimization"]
    
    async with GodelEngineEnv(env_url) as env:
        for task_type in tasks_to_test:
            try:
                result = await env.reset(task_type=task_type)
                obs = result.observation
                
                # [START] task=<task_name> env=<benchmark> model=<model_name>
                print(f"[START] task={task_type} env=GodelEnv model={MODEL_NAME}")
                
                terminated = False
                truncated = False
                rewards = []
                last_step_result = result
                error_msg = "null"
                
                while not (terminated or truncated):
                    # Infer rubrics from observation scores
                    inferred_rubrics = {k: f"Optimize {k}" for k in obs.rubric_scores.scores.keys()}
                    
                    # Get agent action
                    action_result = await agent.act(
                        task_prompt=obs.task_prompt,
                        current_solution=obs.current_solution,
                        rubrics=inferred_rubrics,
                        task_type=task_type,
                        strategy_text=obs.current_strategy,
                        recent_failures=obs.recent_failures,
                        downstream_scores=obs.downstream_scores,
                    )
                    
                    # Step the env
                    try:
                        step_result = await env.step(action_result)
                        last_step_result = step_result
                        error_msg = "null"
                    except Exception as e:
                        # Keep the last valid observation so the trace remains parseable.
                        step_result = last_step_result
                        error_msg = str(e).replace('\n', ' ')
                        reward_val = -1.0
                        rewards.append(reward_val)
                        print(
                            f"[STEP] step={obs.step} action={format_action(action_result)} "
                            f"reward={reward_val:.2f} done=true error={error_msg}"
                        )
                        break
                    
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
                is_success = str(obs.total_score >= 0.90).lower()
                clamped_final_score = max(0.001, min(0.999, obs.total_score))
                rewards_str = ",".join([f"{r:.2f}" for r in rewards])
                # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
                print(f"[END] success={is_success} steps={obs.step} score={clamped_final_score:.3f} rewards={rewards_str}")

            except Exception as ep_error:
                # If the entire episode crashes
                error_val = str(ep_error).replace("\n", " ")
                print(f"[END] success=false steps=0 score=0.001 rewards= error={error_val}")

if __name__ == "__main__":
    asyncio.run(run_inference())
