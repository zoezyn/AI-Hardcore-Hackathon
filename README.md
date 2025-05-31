# AI-Hardcore-Hackathon

## Setup (WIP)
1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Synchronise packages: `uv sync`
3. Setup DStack: `uv tool install dstack -U`
4. Connect to the dstack server: 
```
dstack project add \
    --name main \
    --url http://127.0.0.1:3000 \                    # edit that when we get the server
    --token bbae0f28-d3dd-4820-bf61-8f4bb40815da     # edit that when we get the server
```
5. Init Dstack: `dstack init`
6. Applu the `.dtack.yml` config (adjust if needed): `dstack apply`
7. ...???? TBD

## How to Make an Inference with DStack
````
dstack run inference.py --input data.csv --output predictions.txt
```