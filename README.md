# AI-Hardcore-Hackathon

This project aims to predict the inference time of machine learning models on specific GPUs using only metadata — such as model architecture parameters (e.g., hidden size, number of parameters) and GPU specifications (e.g., core count, memory size, vendor) — without requiring the model or hardware to be physically available.

## Approaches

### Approach 1: Creating a Specialized Dataset

For this hackathon, we focused on large language models (LLMs), but the approach is general. The workflow benchmarks preselected models on various preselected GPUs, collects metadata and inference time, and enriches the data with metadata from Hugging Face and TechPowerUp. The goal is to train a regression model to predict inference time from model and GPU characteristics. 

#### Setup
1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Synchronize packages: `uv sync`
3. Set up DStack: `uv tool install dstack -U`
4. create `.env` and add your `HF_TOKEN` and `DSTACK_TOKEN` (you can use `.env.template`)
5. Connect to the DStack server:
```
dstack project add --name SpongeBobTheHacker --url https://sky.dstack.ai --token $DSTACK_TOKEN
```
6. Initialize DStack: `dstack init`

#### Create a Dataset
1. Run `uv run 1_create_dataset/launch_task.py`
2. View your run on the runs page: [`https://sky.dstack.ai/runs`](https://sky.dstack.ai/runs)
3. When the run is complete, extract the logs with `dstack logs {RUN_NAME} > 1_create_dataset/outputs/{RUN_NAME}.txt`
4. Parse the logs to extract inference information: `uv run 1_create_dataset/outputs/parse_log.py 1_create_dataset/outputs/{RUN_NAME}.txt`. This will create a CSV file.
5. Enrich the features using the Hugging Face API and the TechPowerUp website:
  - `uv run 1_create_dataset/outputs/enrich_csv_with_model_metadata.py --csv_in 1_create_dataset/outputs/{RESULTS_CSV_FILE}`
  - `uv run 1_create_dataset/outputs/enrich_csv_with_gpu_metadata.py --csv_in 1_create_dataset/outputs/{RESULTS_CSV_FILE}`. The name of the GPU can be specified, but if left empty, the script will attempt to infer it from the file name.

An example of such a dataset can be found at `1_create_dataset/outputs/a0gib0h-benchmark-all-models-rtx3090.csv`

### Approach 2: Training Regression Model using an Existing Dataset

This approach leverages existing datasets to analyze and benchmark model inference times. The `2_benchmark_model_with_existing_dataset` directory contains workflows and notebooks for this purpose. You can use the provided `2_benchmark_model_with_existing_dataset/model_analysis.ipynb` notebook to:
- Load and explore existing benchmark datasets (e.g., `2_benchmark_model_with_existing_dataset/tpu-gpu.csv`)
- Perform analysis and visualization of model and GPU characteristics
- Train and evaluate regression models to predict inference time

This approach is useful if you already have benchmark data and want to focus on analysis and modeling rather than running new benchmarks.

## Acknowledgments

Huge thanks to [**DSTack**](https://dstack.ai) for providing the infrastructure and tools that made this benchmarking pipeline possible.

We also thank the organizers of the **AI Hardcore Hackathon** and [**{Tech: Europe}**](https://blog.techeurope.io) for making this event happen, and the **challenge organizer** for designing an inspiring and technically rich problem to work on.