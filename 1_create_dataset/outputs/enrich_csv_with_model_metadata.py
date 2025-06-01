import os
import argparse
import pandas as pd
from transformers import AutoConfig
from dotenv import load_dotenv

load_dotenv()

def get_model_config(model_id: str):
    """
    Fetch model configuration and parameters from the HuggingFace API.
    Handles alternative attribute names for different model families.
    Always sets trust_remote_code=True to avoid prompts.
    """
    try:
        config = AutoConfig.from_pretrained(model_id, token=os.getenv("HF_TOKEN"), trust_remote_code=True)
        model_params = {
            "model_type": getattr(config, "model_type", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
            "num_hidden_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layers", None)),
            "num_attention_heads": getattr(config, "num_attention_heads", getattr(config, "n_heads", None)),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "hidden_act": getattr(config, "hidden_act", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", getattr(config, "n_heads", None))),
            "rope_theta": getattr(config, "rope_theta", None),
            "sliding_window": getattr(config, "sliding_window", None),
        }
        if hasattr(config, "quantization_config"):
            quant_config = config.quantization_config
            model_params["quantization"] = {
                "bits": getattr(quant_config, "bits", None),
                "group_size": getattr(quant_config, "group_size", None),
                "method": getattr(quant_config, "quant_method", None),
            }
        return model_params
    except Exception as e:
        return {"error": str(e)}

def enrich_csv(input_csv, output_csv=None, model_col='model'):
    df = pd.read_csv(input_csv)
    # Collect all model names
    model_names = df[model_col].unique()
    # Fetch metadata for all models
    model_configs = {}
    for name in model_names:
        config = get_model_config(name)
        if 'error' in config:
            print(f"Error fetching metadata for model '{name}': {config['error']}")
            config = {}  # Use empty dict for errored models
        model_configs[name] = config
    # Find all metadata keys
    all_keys = set()
    for config in model_configs.values():
        if isinstance(config, dict):
            all_keys.update(config.keys())
            if 'quantization' in config and isinstance(config['quantization'], dict):
                all_keys.update(f"quant_{k}" for k in config['quantization'].keys())
    all_keys.discard('quantization')
    all_keys.discard('error')
    all_keys = sorted(all_keys)
    # Prefix all new columns with 'model_'
    prefixed_keys = [f"model_{k}" for k in all_keys]
    def get_metadata(model_name):
        config = model_configs.get(model_name, {})
        meta = {}
        for k, pk in zip(all_keys, prefixed_keys):
            if k.startswith('quant_'):
                quant_key = k.replace('quant_', '')
                meta[pk] = config.get('quantization', {}).get(quant_key) if isinstance(config.get('quantization'), dict) else None
            else:
                meta[pk] = config.get(k)
        return pd.Series(meta)
    meta_df = df[model_col].apply(get_metadata)
    df = pd.concat([df, meta_df], axis=1)
    out_path = input_csv if output_csv is None else output_csv
    df.to_csv(out_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Enrich a CSV file with model metadata.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", help="Path to the output CSV file (if omitted, input is overwritten)")
    parser.add_argument("--model-col", default="model_name", help="Column name for the model (default: model)")
    args = parser.parse_args()
    enrich_csv(args.input_csv, args.output_csv, args.model_col)
    print(f"Done. Output written to {args.output_csv or args.input_csv}")

if __name__ == '__main__':
    main()
