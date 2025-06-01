import requests
from bs4 import BeautifulSoup
import re
import argparse
import os
import pandas as pd

DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"


def search_gpu_page_duckduckgo(gpu_query):
    """
    Uses DuckDuckGo to search for a GPU on TechPowerUp and returns the first matching URL.
    """
    params = {
        "q": f"{gpu_query} site:techpowerup.com/gpu-specs"
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.post(DUCKDUCKGO_SEARCH_URL, data=params, headers=headers)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)

    for link in links:
        href = link["href"]
        if "techpowerup.com/gpu-specs/" in href:
            return href  # Already absolute
    return None


def parse_gpu_specs(url):
    """
    Parses GPU specs from the given TechPowerUp URL.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"error": "Failed to fetch GPU detail page."}

    soup = BeautifulSoup(response.content, "html.parser")
    info = {}

    spec_entries = soup.select("dl.gpudb-specs-large div.gpudb-specs-large__entry")
    for entry in spec_entries:
        key_tag = entry.find("dt", class_="gpudb-specs-large__title")
        value_tag = entry.find("dd", class_="gpudb-specs-large__value")
        if key_tag and value_tag:
            key = key_tag.text.strip()
            value = value_tag.text.strip()
            info[key] = value

    info["source_url"] = url
    return info


def get_gpu_info(gpu_query):
    url = search_gpu_page_duckduckgo(gpu_query)
    if not url:
        return {"error": f"No GPU found for query '{gpu_query}'."}
    return parse_gpu_specs(url)


def infer_gpu_name_from_filename(filename):
    # Try to extract GPU name from filename (e.g., 'benchmark-all-models-rtx3090.csv')
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    # Look for common GPU name patterns (rtx3090, a2000, etc.)
    tokens = name.split('-')
    for token in tokens[::-1]:
        if any(char.isdigit() for char in token):
            return token
    return None


def main():
    parser = argparse.ArgumentParser(description="Add GPU info to CSV.")
    parser.add_argument("--csv_in", required=True, help="Path to the CSV file.")
    parser.add_argument("--gpu", help="GPU name (optional, inferred from filename if not provided).", default=None)
    args = parser.parse_args()

    csv_path = args.csv_in
    gpu_name = args.gpu or infer_gpu_name_from_filename(csv_path)
    if not gpu_name:
        print("Could not infer GPU name from filename and none provided.")
        return

    gpu_info = get_gpu_info(gpu_name)
    if "error" in gpu_info:
        print(f"Error: {gpu_info['error']}")
        return

    df = pd.read_csv(csv_path)
    # Add GPU info as new columns (prefix with 'gpu_')
    for k, v in gpu_info.items():
        df[f"gpu_{k}"] = v
    # Overwrite the input file
    df.to_csv(csv_path, index=False)
    print(f"Overwrote {csv_path} with GPU info.")


if __name__ == "__main__":
    main()