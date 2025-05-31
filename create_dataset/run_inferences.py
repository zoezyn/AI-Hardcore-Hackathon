# llm_infer.py

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time

def main():
    parser = argparse.ArgumentParser(description="Run text generation with a Hugging Face LLM")
    parser.add_argument("--models", type=str, help="Comma-separated list of Hugging Face model names or paths")
    parser.add_argument("--model", type=str, help="(Deprecated) Single model name or path")
    parser.add_argument("--num", type=int, default=5, help="Number of inferences to generate")
    args = parser.parse_args()

    # Support both --models (preferred) and --model (legacy)
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        model_list = [args.model]
    else:
        raise ValueError("You must provide either --models or --model argument.")

    prompts = [
        "What are the main causes of climate change?",
        "Explain how a transformer model works in simple terms.",
        "Write a short story about a robot learning to love.",
        "List five healthy vegetarian sources of protein.",
        "How does quantum computing differ from classical computing?",
        "What's the future of artificial intelligence?",
        "Describe the process of photosynthesis.",
        "What are black holes and how do they form?",
        "Give me a summary of the French Revolution.",
        "What are some creative ways to reduce plastic waste?"
    ]

    for model_name in model_list:
        print(f"\n==============================\nRunning model: {model_name}\n==============================")
        print(f"Task name: {args.model if args.model else args.models}")
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        start_time = time.time()
        for i in range(min(args.num, len(prompts))):
            prompt = prompts[i]
            print(f"\n--- Inference {i+1} ---")
            result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
            print(result)
        end_time = time.time()

        print(f"\nTime taken for {args.num} inferences: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()