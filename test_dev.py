import time
import torch
import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import threading
import queue
import json

def get_gpu_power():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return float(result.strip())
    except:
        return None

class PowerMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.power_readings = []
        self.should_stop = False
        self.queue = queue.Queue()

    def monitor(self):
        while not self.should_stop:
            power = get_gpu_power()
            if power is not None:
                self.power_readings.append(power)
            time.sleep(self.interval)
        
        if self.power_readings:
            self.queue.put({
                'max_power': max(self.power_readings),
                'avg_power': sum(self.power_readings) / len(self.power_readings),
                'min_power': min(self.power_readings)
            })
        else:
            self.queue.put(None)

    def start(self):
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.should_stop = True
        self.thread.join()
        return self.queue.get()

parser = argparse.ArgumentParser(description="Run inference with a HuggingFace LLM")
parser.add_argument("--model", type=str, help="Single model name or path")
# parser.add_argument("--num", type=int, default=5, help="Number of inferences to generate")
args = parser.parse_args()

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
# model_name = MODELS[0]["name"]
# model_name = args.model
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

# Start power monitoring
power_monitor = PowerMonitor()
power_monitor.start()

# Actual timed run
print("\nStarting timed inference...")
torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

torch.cuda.synchronize()
end_time = time.time()

# Stop power monitoring and get results
power_stats = power_monitor.stop()

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

# Add power consumption stats if available
if power_stats:
    result["power_consumption"] = {
        "max_watts": power_stats["max_power"],
        "avg_watts": power_stats["avg_power"],
        "min_watts": power_stats["min_power"]
    }
    print(f"\nPower Consumption:")
    print(f"Max: {power_stats['max_power']:.2f}W")
    print(f"Avg: {power_stats['avg_power']:.2f}W")
    print(f"Min: {power_stats['min_power']:.2f}W")

with open("inference_result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResponse: {response}")
print(f"Inference time: {inference_time:.2f} seconds")
print("\nResults saved to inference_result.json")
