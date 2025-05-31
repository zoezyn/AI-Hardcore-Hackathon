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
6. Create a configuration file for the run, where you define which script to run (see example at `.dstack.yml`)
7. start the run: `dstack apply -f .dstack.yml`
8. See your run on the runs page: [`https://sky.dstack.ai/runs`](https://sky.dstack.ai/runs)