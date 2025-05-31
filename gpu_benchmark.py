import json
import time
from datetime import datetime
from dstack.api import Task, GPU, Client, Resources, LocalRepo
import re
# Load GPU configurations from the offers file
with open('dstack_offers.json', 'r') as f:
    offers = json.load(f)
import os
from dotenv import load_dotenv
load_dotenv()

print("os.getenv('HF_TOKEN')", os.getenv("HF_TOKEN"))
# Models to test
MODELS = [
    {
        "name": "facebook/opt-125m",
        "min_gpu_mem": 2
    }
]

def sanitize_name(name):
    """Convert a string to a valid dstack resource name."""
    # Remove any non-alphanumeric characters and convert to lowercase
    name = re.sub(r'[^a-zA-Z0-9-]', '-', name.lower())
    # Ensure it starts with a letter
    if not name[0].isalpha():
        name = 'test-' + name
    # Truncate to valid length
    return name[:41]

# Get unique GPU configurations
def get_gpus():
    gpus = []
    for offer in offers['offers']:
        gpu = offer['resources']['gpus'][0]
        gpu_info = {
            "name": gpu['name'],
            "memory_gb": gpu['memory_mib'] / 1024,
            "region": offer['region'],
            "price": offer['price']
        }
        if gpu_info not in gpus:
            gpus.append(gpu_info)
    return gpus

def run_model_on_gpu(model, gpu):
    client = Client.from_config()
    
    print(f"\nTesting {model['name']} on {gpu['name']} ({gpu['memory_gb']:.1f}GB) in {gpu['region']}")
    
    # Create a valid task name
    model_name = model['name'].split('/')[-1]
    gpu_name = gpu['name']
    task_name = sanitize_name(f"test-{model_name}-{gpu_name}")
    

    start_time = time.time()
    
    task = Task(
        name=task_name,
        image="ghcr.io/astral-sh/uv:debian",
        env={
             "HF_TOKEN": os.getenv("HF_TOKEN")},
        commands=[
            "uv sync",
            f'uv run "test_dev.py"'
        ],
        resources=Resources(gpu=GPU(memory=f"{int(gpu['memory_gb'])}GB"))
    )
    repo = LocalRepo.from_dir(".")
    client.repos.init(repo)


    run = client.runs.apply_configuration(
        configuration=task,
        repo=repo
    )
    # run = client.runs.apply_configuration(configuration=task)
    run.attach()
    end_time = time.time()
    
    return {
        "model": model['name'],
        "gpu": gpu['name'],
        "memory": f"{gpu['memory_gb']:.1f}GB",
        "region": gpu['region'],
        "price": gpu['price'],
        "duration": end_time - start_time
    }


def main():
    results = []
    gpus = get_gpus()
    gpus = gpus[1:3]
    
    for model in MODELS:
        # Filter GPUs based on model's memory requirement
        compatible_gpus = [
            gpu for gpu in gpus 
            if gpu['memory_gb'] >= model['min_gpu_mem']
        ]
        
        for gpu in compatible_gpus:
            result = run_model_on_gpu(model, gpu)
            results.append(result)
            
            # Save results after each run
            with open('benchmark_results.json', 'w') as f:
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 