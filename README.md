# AI-Hardcore-Hackathon

## Setup
1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Synchronise packages: `uv sync`
3. Setup DStack: `uv tool install dstack -U`
4. Connect to the dstack server: 
```
dstack project add --name SpongeBobTheHacker --url https://sky.dstack.ai --token $DSTACK_TOKEN
```
5. Init Dstack: `dstack init`
6. run `uv run create_dataset/launch_task.py`
8. See your run on the runs page: [`https://sky.dstack.ai/runs`](https://sky.dstack.ai/runs)