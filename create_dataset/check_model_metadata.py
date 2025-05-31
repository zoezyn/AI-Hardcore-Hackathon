from transformers import AutoConfig
import json
import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# List of models from launch_task.py
MODELS = [
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "Qwen/Qwen1.5-7B-Chat",
    "TheBloke/Llama-2-7B-Chat-GPTQ",
    "TheBloke/Llama-2-13B-chat-GPTQ",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-32B-Chat",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

def get_model_config(model_id: str):
    """
    Fetch model configuration and parameters.
    """
    try:
        print(f"\nFetching config for {model_id}...")
        config = AutoConfig.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
        
        # Get all the important architecture parameters
        model_params = {
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": getattr(config, "intermediate_size", None),
            "hidden_act": config.hidden_act,
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
            "rope_theta": getattr(config, "rope_theta", None),
            "sliding_window": getattr(config, "sliding_window", None),
        }

        # Calculate number of parameters
        num_params = 0
        # Basic transformer parameters calculation
        num_params += config.vocab_size * config.hidden_size  # Embedding layer
        num_params += config.num_hidden_layers * (
            # Self-attention
            4 * config.hidden_size * config.hidden_size +  # Q, K, V, and output projections
            # FFN
            2 * config.hidden_size * getattr(config, "intermediate_size", 4 * config.hidden_size) +
            # Layer norms
            4 * config.hidden_size
        )
        
        # model_params["approximate_parameters"] = f"{num_params / 1e9:.2f}B"
        
        # Get tensor parallel info if available
        if hasattr(config, "quantization_config"):
            quant_config = config.quantization_config
            model_params["quantization"] = {
                "bits": getattr(quant_config, "bits", None),
                "group_size": getattr(quant_config, "group_size", None),
                "method": getattr(quant_config, "quant_method", None),
            }
        
        return model_params
    except Exception as e:
        return {
            "error": str(e)
        }

def main():
    print("Fetching model configurations...")
    all_configs = {}
    
    for model in MODELS:
        config = get_model_config(model)
        all_configs[model] = config
        
        # Print key architectural information
        print(f"\nModel: {model}")
        if "error" in config:
            print(f"Error: {config['error']}")
            continue
            
        print(f"Architecture type: {config['model_type']}")
        # print(f"Parameters: {config['approximate_parameters']}")
        print(f"Hidden size: {config['hidden_size']}")
        print(f"Layers: {config['num_hidden_layers']}")
        print(f"Attention heads: {config['num_attention_heads']}")
        if config['num_key_value_heads'] != config['num_attention_heads']:
            print(f"KV heads: {config['num_key_value_heads']} (GQA)")
        print(f"Vocab size: {config['vocab_size']}")
        print(f"Max sequence length: {config['max_position_embeddings']}")
        if config.get('quantization'):
            print("Quantization:")
            print(f"  Bits: {config['quantization']['bits']}")
            print(f"  Group size: {config['quantization']['group_size']}")
            print(f"  Method: {config['quantization']['method']}")
        if config.get('sliding_window'):
            print(f"Sliding window size: {config['sliding_window']}")
        print("-" * 80)
    
    # Save full configurations to file
    with open("model_configs.json", "w") as f:
        json.dump(all_configs, f, indent=2)
    print("\nFull configurations saved to model_configs.json")

if __name__ == "__main__":
    main() 