#!/usr/bin/env python3
"""
Upload classified speeches to Cosmos DB.
"""

import json
import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from tqdm import tqdm

# Load .env
load_dotenv(override=False)

def getenv(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if isinstance(value, str):
        cleaned = value.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1]
        return cleaned
    return value

print("=" * 70)
print("Upload BSP Speeches to Cosmos DB")
print("=" * 70)

# Load configuration
cosmos_endpoint = getenv("COSMOS_ENDPOINT").rstrip('/')
cosmos_key = getenv("COSMOS_KEY")
cosmos_database = getenv("COSMOS_DATABASE")
cosmos_container = getenv("COSMOS_SPEECHES_CONTAINER")

print(f"\n1. Configuration:")
print(f"   Endpoint: {cosmos_endpoint}")
print(f"   Database: {cosmos_database}")
print(f"   Container: {cosmos_container}")

# Load classified speeches
print(f"\n2. Loading classified speeches...")
with open("bsp_speeches_classified.json", 'r', encoding='utf-8') as f:
    speeches = json.load(f)
print(f"   ✓ Loaded {len(speeches)} speeches")

# Connect to Cosmos DB
print(f"\n3. Connecting to Cosmos DB...")
try:
    cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
    database = cosmos_client.get_database_client(cosmos_database)
    container = database.get_container_client(cosmos_container)
    print(f"   ✓ Connected successfully")
except Exception as e:
    print(f"   ✗ Failed to connect: {e}")
    exit(1)

# Upload documents
print(f"\n4. Uploading {len(speeches)} documents...")
uploaded = 0
failed = 0
errors = []

for speech in tqdm(speeches, desc="Uploading"):
    try:
        container.upsert_item(speech)
        uploaded += 1
    except Exception as e:
        failed += 1
        error_msg = f"ID {speech['id']}: {str(e)[:100]}"
        if failed <= 10:  # Keep first 10 errors
            errors.append(error_msg)

print(f"\n5. Upload Summary:")
print(f"   ✓ Successfully uploaded: {uploaded}")
if failed > 0:
    print(f"   ✗ Failed: {failed}")
    print(f"\n   First errors:")
    for err in errors:
        print(f"   - {err}")

print("\n" + "=" * 70)
print("UPLOAD COMPLETE!")
print("=" * 70)
