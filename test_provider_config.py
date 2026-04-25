"""Test provider configuration with HF Space-style secrets."""
import os

# Simulate HF Space secrets (without actual values)
os.environ['HF_TOKEN'] = 'hf_test_token_placeholder'
os.environ['API_BASE_URL'] = 'https://router.huggingface.co/v1'
os.environ['MODEL_NAME'] = 'Qwen/Qwen2.5-7B-Instruct'
os.environ['OPENAI_API_KEY'] = 'sk-test-key-placeholder'

from godel_engine.provider_runtime import load_provider_configs, _openai_model_name

print('=== Testing Provider Config ===\n')

configs = load_provider_configs()
print(f'Loaded {len(configs)} provider(s):\n')

for c in configs:
    print(f'  Provider: {c.name}')
    print(f'    api_key: {"***" if c.api_key else "MISSING"}')
    print(f'    base_url: {c.base_url}')
    print(f'    model_name: {c.model_name}')
    print()

print('=== OpenAI Model Name Resolution ===')
print(f'  _openai_model_name() = {_openai_model_name()}')
print()

# Check that HF doesn't use OpenAI's model and vice versa
hf_config = next((c for c in configs if c.name == 'huggingface'), None)
openai_config = next((c for c in configs if c.name == 'openai'), None)

if hf_config and openai_config:
    print('=== Cross-Check ===')
    print(f'  HF model: {hf_config.model_name}')
    print(f'  OpenAI model: {openai_config.model_name}')
    
    if '/' in openai_config.model_name:
        print('  ERROR: OpenAI has a HuggingFace-style model name!')
    else:
        print('  OK: OpenAI model name looks valid')
