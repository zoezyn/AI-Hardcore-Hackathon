# llm_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
import pandas as pd
import traceback
import subprocess
import os

models_to_test = [
    # Small to medium models (~1B–7B)
    "mosaicml/mpt-1b-redpajama-200b",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "Qwen/Qwen1.5-7B-Chat",
    "TheBloke/Llama-2-7B-Chat-GPTQ",

    # Larger models (13B–32B)
    "TheBloke/Llama-2-13B-chat-GPTQ",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

def get_gpu_info():
    gpu_info = {
        'gpu_name': None,
        'gpu_brand': None,
        'gpu_cores': None,
        'gpu_memory_total_gb': None
    }
    if torch.cuda.is_available():
        try:
            gpu_idx = torch.cuda.current_device()
            gpu_info['gpu_name'] = torch.cuda.get_device_name(gpu_idx)
            gpu_info['gpu_memory_total_gb'] = round(torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9, 2)
            gpu_info['gpu_cores'] = torch.cuda.get_device_properties(gpu_idx).multi_processor_count
            # Try to get brand from nvidia-smi
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    name = result.stdout.split(',')[0].strip()
                    gpu_info['gpu_brand'] = name.split()[0] if name else None
            except Exception:
                pass
        except Exception:
            pass
    return gpu_info

def get_model_info(model):
    try:
        n_params = sum(p.numel() for p in model.parameters())
        tensor_type = next(model.parameters()).dtype if any(p.requires_grad for p in model.parameters()) else None
        return n_params, str(tensor_type)
    except Exception:
        return None, None

def main():
    prompts = [
        "Hi!",
        "Summarize the plot of Hamlet.",
        "What is the capital of France?",
        "Explain the difference between supervised and unsupervised learning in machine learning.",
        "Write a Python function that returns the Fibonacci sequence up to n.",
        "Describe the process of photosynthesis in detail, including the role of chlorophyll and the light-dependent and light-independent reactions.",
        "List three healthy vegetarian sources of protein.",
        "What are black holes and how do they form?",
        "Give me a summary of the French Revolution in less than 50 words.",
        "Write a short story about a robot learning to love, set in a futuristic city, with at least 100 words.",
        "How does quantum computing differ from classical computing?",
        "What's the future of artificial intelligence?",
        "Describe the process of photosynthesis.",
        "What are some creative ways to reduce plastic waste in daily life, at home, and in the workplace?",
        "In a few sentences, explain how a transformer model works in simple terms for a high school student."
    ]

    csv_path = "inference_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} previous results from {csv_path}")
    else:
        df = pd.DataFrame()

    results = []
    gpu_info = get_gpu_info()
    for model_name in models_to_test:
        print(f"\n==============================\nRunning model: {model_name}\n==============================")
        try:
            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            n_params, tensor_type = get_model_info(model)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            traceback.print_exc()
            continue
        start_time = time.time()
        for prompt in prompts:
            # Check if this inference already exists in the CSV
            if not df.empty and ((df['model_name'] == model_name) & (df['gpu_name'] == gpu_info['gpu_name']) & (df['prompt'] == prompt)).any():
                print(f"Skipping already completed inference for model '{model_name}', GPU '{gpu_info['gpu_name']}', prompt '{prompt[:40]}...'")
                continue
            print(f"\n--- Inference for prompt ---")
            try:
                inf_start = time.time()
                result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
                inf_time = time.time() - inf_start
                print(result)
                row = {
                    'model_name': model_name,
                    'prompt': prompt,
                    'generated_text': result,
                    'gpu_name': gpu_info['gpu_name'],
                    'gpu_brand': gpu_info['gpu_brand'],
                    'gpu_cores': gpu_info['gpu_cores'],
                    'gpu_memory_total_gb': gpu_info['gpu_memory_total_gb'],
                    'model_num_parameters': n_params,
                    'model_tensor_type': tensor_type,
                    'inference_time_sec': round(inf_time, 3)
                }
                # Append to DataFrame and write to CSV immediately
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"Appended result to {csv_path}")
            except Exception as e:
                print(f"Error during inference: {e}")
                traceback.print_exc()
        end_time = time.time()
        print(f"\nTime taken for {len(prompts)} inferences: {end_time - start_time:.2f} seconds")
    print(f"\nAll results saved to {csv_path}")

if __name__ == "__main__":
    main()