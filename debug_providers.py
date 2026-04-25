"""Debug provider configuration."""
import os
os.environ['GODEL_GRADING_MODE'] = 'auto'
os.environ['GODEL_STRATEGY_EVAL_MODE'] = 'auto'

from godel_engine.provider_runtime import (
    load_provider_configs,
    describe_provider_environment,
    ProviderCircuitBreaker,
)

print('=== Environment Variables ===')
env = describe_provider_environment()
for k, v in env.items():
    print(f'  {k}: {v}')

print()
print('=== Provider Configs ===')
configs = load_provider_configs()
if not configs:
    print('  NO PROVIDERS CONFIGURED!')
for c in configs:
    key_status = '***' if c.api_key else 'MISSING'
    print(f'  {c.name}: api_key={key_status}, base_url={c.base_url}, model={c.model_name}')

print()
print('=== Circuit Breaker ===')
print(f'  Disabled providers: {ProviderCircuitBreaker._disabled}')

print()
print('=== Quick LLM Test ===')
import asyncio
from godel_engine.agent import AutoAgent

async def test():
    agent = AutoAgent(timeout=30)
    print(f'  Agent has {len(agent.clients)} client(s)')
    for name, model, client in agent.clients:
        print(f'    - {name}: {model}')
    
    if not agent.clients:
        print('  NO LLM CLIENTS - will use deterministic fallback')
        return
    
    # Try a simple call
    from godel_engine.models import GodelAction
    result = await agent.act(
        task_prompt='What is 2+2?',
        current_solution='4',
        rubrics={'correctness': 'Is the answer correct?'},
        task_type='factual_qa',
    )
    print(f'  Agent source: {agent.last_source}')
    print(f'  Agent provider: {agent.last_provider}')
    print(f'  Agent error: {agent.last_error}')
    print(f'  Result: {result.solution[:100]}...' if len(result.solution) > 100 else f'  Result: {result.solution}')

asyncio.run(test())
