import time
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Models to test
MODELS = [
    {
        "name": "facebook/opt-125m",
        "min_gpu_mem": 2
    }
]

# Load model and tokenizer
# model_name = "facebook/opt-125m"
# model_name = "${MODEL_ID}"
model_name = MODELS[0]["name"]
print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                           device_map="auto",
                                           trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test prompt
prompt = "What is machine learning? Answer in one sentence."
print(f"\nRunning inference with prompt: {prompt}")

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Warm-up run
print("Performing warm-up run...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=1)

# Actual timed run
print("\nStarting timed inference...")
torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

torch.cuda.synchronize()
end_time = time.time()

# Get the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
inference_time = end_time - start_time

# Save and display results
result = {
    "prompt": prompt,
    "response": response,
    "inference_time_seconds": inference_time,
    "model": model_name,
}

import json
with open("inference_result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResponse: {response}")
print(f"Inference time: {inference_time:.2f} seconds")
print("\nResults saved to inference_result.json")
