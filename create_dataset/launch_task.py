import sys
import json
import random
import re
import string
from dstack.api import Task, GPU, Client, Resources, LocalRepo
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
import os
from dotenv import load_dotenv

client = Client.from_config()
load_dotenv()

models_to_test = [
    # Small to medium models (~1B–7B)
    "mosaicml/mpt-1b-redpajama-200b",
    # "mistralai/Mistral-7B-Instruct-v0.2",
    # "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "Qwen/Qwen1.5-7B-Chat",
    "TheBloke/Llama-2-7B-Chat-GPTQ",

    # Larger models (13B–32B)
    "TheBloke/Llama-2-13B-chat-GPTQ",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat",
    # "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# Dynamically load 10 random GPU resources from avilable_gpus.json
with open('available_gpus.json', 'r') as f:
    gpus_data = json.load(f)

random.seed(42)
offers = gpus_data["offers"]
selected_offers = random.sample(offers, 10)

def mib_to_gb(mib):
    return f"{int(round(mib / 1024))}GB"

resources_to_test = []
for offer in selected_offers:
    gpu_info = offer["resources"]["gpus"][0]
    resources_to_test.append(
        Resources(
            gpu=GPU(
                name=[gpu_info["name"]],  # must be a list
                memory=mib_to_gb(gpu_info["memory_mib"]),
                vendor=str(gpu_info["vendor"])
            ),
        )
    )

def sanitize_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9-]', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    if not name or not name[0].isalpha():
        name = 'a' + name
    return name[:41]

def random_string(length=6):
    random.seed(None)
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Start the actual task

def main():
    for idx, resources in enumerate(resources_to_test):
        # Create a new Task object for each run
        task = Task(
            image="ghcr.io/astral-sh/uv:debian",
            # ports=["8080"],  # Changed from 80 to 8080
            resources=resources,
            env={
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            }
        )
        offer = selected_offers[idx]
        models_str = ",".join(models_to_test)
        rand_prefix = random_string()
        task.name = sanitize_name(f"{rand_prefix}-benchmark-all-models-{resources.gpu.name[0]}")
        gpu_name = resources.gpu.name
        task.working_dir = "."
        print("\n=== Launching run for all models ===")
        print("GPU features:")
        print(f"  Name: {resources.gpu.name}")
        print(f"  Memory: {resources.gpu.memory}")
        print(f"  Vendor: {resources.gpu.vendor}")
        print(f"  Backend: {offer.get('backend', 'N/A')}")
        print(f"  Region: {offer.get('region', 'N/A')}")
        print(f"  Instance type: {offer.get('instance_type', 'N/A')}")
        print(f"  CPUs: {offer['resources'].get('cpus', 'N/A')}")
        print(f"  RAM: {mib_to_gb(offer['resources'].get('memory_mib', 0))}")
        print(f"  Disk size: {mib_to_gb(offer['resources'].get('disk', {}).get('size_mib', 0))}")
        print("====================\n")
        print(f"Launching task: {task.name} with resources: {resources.gpu.name[0]}")
        # Actually launch the task on the remote VM and run the inference script
        task.commands = [
            "uv sync",
            f"uv run create_dataset/run_inferences.py --models {models_str} --num 10"
        ]

        repo = LocalRepo(repo_dir=".", repo_id="berlin-hackathon")
        client.repos.init(repo)
        
        run = client.runs.apply_configuration(
            configuration=task,
            repo=repo
        )

        run.attach()
        try:
            for log in run.logs():
                sys.stdout.buffer.write(log)
                sys.stdout.buffer.flush()
        except KeyboardInterrupt:
            run.stop(abort=True)
        finally:
            run.detach()

if __name__ == "__main__":
    main()