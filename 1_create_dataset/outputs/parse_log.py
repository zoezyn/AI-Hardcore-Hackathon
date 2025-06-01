import re
import csv
import argparse

def parse_log_file(log_path):
    models = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
    print(f"[DEBUG] Log file loaded: {len(lines)} lines")

    current_model = None
    for i, line in enumerate(lines):
        # Print every line that contains 'model' or 'Time taken' for debugging
        if 'model' in line or 'Time taken' in line:
            print(f"[DEBUG] Line {i}: {line.rstrip()}")
        # Detect model name (robust: Running or Loading)
        model_match = re.search(r"(?:Running|Loading) model:\s*(.+)", line)
        if model_match:
            print(f"[DEBUG] Found model on line {i}: {model_match.group(1).strip()}")
            current_model = model_match.group(1).strip()
        # Detect inference time (robust)
        time_match = re.search(r"Time taken for\s*(\d+) inferences:\s*([\d.]+) seconds", line)
        if time_match and current_model:
            print(f"[DEBUG] Found time on line {i}: {time_match.group(0)} for model {current_model}")
            num_inferences = int(time_match.group(1))
            total_time = float(time_match.group(2))
            avg_time = total_time / num_inferences if num_inferences else 0
            models.append({
                'model_name': current_model,
                'num_inferences': num_inferences,
                'total_time': total_time,
                'avg_inference_time': avg_time
            })
    if not models:
        print("No models found. Please check the log format or regex.")
    return models

def write_csv(models, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['model_name', 'num_inferences', 'total_time', 'avg_inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            writer.writerow(model)

def main():
    parser = argparse.ArgumentParser(description="Parse log file and output CSV with model inference times.")
    parser.add_argument("input_log", help="Path to input log file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    args = parser.parse_args()
    log_path = args.input_log
    output_path = args.output_csv
    models = parse_log_file(log_path)
    write_csv(models, output_path)
    print(f"Parsed {len(models)} models. Output written to {output_path}")

if __name__ == "__main__":
    main()
