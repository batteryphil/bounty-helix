import os
import json

model_name = 'hermes-3'
adapter_name = 'lora'
adapter_size = 256

experience_tuples_path = f'experience_tuples.jsonl'
planning_path = f'planning.jsonl'

if os.path.exists(experience_tuples_path) and len(open(experience_tuples_path).readlines()) >= 1000:
    print(f'Running advanced planning for {model_name} after 1000 experience tuples')
    os.system(f'python train_lora.py --model_name {model_name} --adapter_name {adapter_name} --adapter_size {adapter_size}')
    os.system('python tools/planning/advanced_planning.py')
    with open(experience_tuples_path, 'w') as f:
        f.write