#!/usr/bin/env python3
"""
Classify scraped BSP speeches by audience setting using Azure OpenAI.
Reads from bsp_speeches.xlsx, classifies each speech, and saves results to a new Excel file.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm
from azure.cosmos import CosmosClient

# Enhanced Audience/Setting Classification Labels with BSP-specific cues
LABEL_DEFS = {
    "INTERNAL_BSP": {
        "description": "Internal BSP staff/leadership audience; culture, morale, internal milestones.",
        "cues": [
            "fellow BSPers", "turnover", "anniversary", "our colleagues", "within the BSP", 
            "staff", "employees", "team", "organization", "internal", "BSP family",
            "our mandate", "our mission", "employee", "workforce", "colleagues",
            "fellow central bankers", "oath-taking", "promotion", "appointment"
        ],
    },
    "GOV_OVERSIGHT": {
        "description": "Government oversight/accountability forum (Congress, DBCC, executive briefings).",
        "cues": [
            "Honorable", "House of Representatives", "Senate", "DBCC", "budget", 
            "assumptions", "committee hearing", "legislative", "Congress", "Senators",
            "Representatives", "fiscal policy", "national budget", "appropriations",
            "oversight", "accountability", "governance", "Cabinet", "President",
            "executive branch", "government officials"
        ],
    },
    "INDUSTRY_MARKET": {
        "description": "Regulated industry / market stakeholders (banks, associations, FM participants, infrastructures).",
        "cues": [
            "association", "banks", "bankers", "industry", "convention", "payments", 
            "trade finance", "correspondent banking", "banking community", "financial institutions",
            "universal banks", "commercial banks", "thrift banks", "rural banks",
            "credit unions", "cooperatives", "payment system", "clearing", "settlement",
            "financial market", "capital market", "forex", "treasury", "dealers",
            "market participants", "stakeholders", "regulated entities", "compliance",
            "supervision", "prudential", "risk management", "credit rating"
        ],
    },
    "INTERNATIONAL_OFFICIAL": {
        "description": "International/multilateral policy community (IMF/BIS/World Bank etc.).",
        "cues": [
            "IMF", "BIS", "World Bank", "OECD", "international cooperation", "cross-border", 
            "delegates", "multilateral", "regional cooperation", "ASEAN", "SEACEN",
            "Asian Development Bank", "ADB", "global", "international forum",
            "central bank governors", "foreign central banks", "Basel", "international standards",
            "EMEAP", "G20", "G7", "international financial architecture",
            "correspondent relations", "bilateral", "monetary cooperation"
        ],
    },
    "PUBLIC_REGIONAL": {
        "description": "General public / regional outreach / broad economic briefings.",
        "cues": [
            "economic briefing", "public", "community", "region", "local", "Iloilo", 
            "Cebu", "Davao", "inclusive growth", "financial literacy", "consumer protection",
            "financial education", "citizens", "Filipinos", "countrymen", "people",
            "grassroots", "microfinance", "SME", "small business", "entrepreneurs",
            "youth", "students", "women", "farmers", "fisherfolk", "OFWs",
            "regional", "provincial", "municipal", "barangay", "stakeholders' congress",
            "town hall", "forum", "symposium", "summit"
        ],
    },
}

# Governor periods for date-based determination
GOVERNOR_PERIODS = [
    {
        "name": "Governor Amando M. Tetangco, Jr.",
        "start": datetime(1900, 1, 1),  # Before our date range
        "end": datetime(2017, 7, 2),
    },
    {
        "name": "Governor Nestor A. Espenilla, Jr.",
        "start": datetime(2017, 7, 3),
        "end": datetime(2019, 2, 23),
    },
    {
        "name": "Governor Benjamin E. Diokno",
        "start": datetime(2019, 3, 4),
        "end": datetime(2022, 6, 30),
    },
    {
        "name": "Governor Felipe M. Medalla",
        "start": datetime(2022, 6, 30),
        "end": datetime(2023, 7, 3),
    },
    {
        "name": "Governor Eli M. Remolona, Jr.",
        "start": datetime(2023, 7, 3),
        "end": datetime(2099, 12, 31),  # Present
    },
]


def load_config() -> Dict[str, str]:
    """Load Azure OpenAI configuration from environment variables."""
    # Load .env from workspace root (searches parent directories)
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
    
    config = {
        "openai_endpoint": getenv("AZURE_OPENAI_ENDPOINT"),
        "openai_key": getenv("AZURE_OPENAI_KEY"),
        "openai_chat_deployment": getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "tenant_id": getenv("TENANT_ID") or None,
        "cosmos_endpoint": getenv("COSMOS_ENDPOINT"),
        "cosmos_key": getenv("COSMOS_KEY"),
        "cosmos_connection_string": getenv("COSMOS_CONNECTION_STRING"),
        "cosmos_database": getenv("COSMOS_DATABASE"),
        "cosmos_container": getenv("COSMOS_SPEECHES_CONTAINER"),
    }
    
    # Validate required config
    missing = []
    if not config["openai_endpoint"]:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not config["openai_key"]:
        missing.append("AZURE_OPENAI_KEY")
    if not config["openai_chat_deployment"]:
        missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
    
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")
    
    return config


def normalize_governor_name(speaker: str) -> Optional[str]:
    """Extract and normalize governor name from Speaker field."""
    if not speaker:
        return None
    
    speaker_lower = speaker.lower()
    
    # Check for each governor's name in the speaker field
    if "tetangco" in speaker_lower:
        return "Governor Amando M. Tetangco, Jr."
    elif "espenilla" in speaker_lower:
        return "Governor Nestor A. Espenilla, Jr."
    elif "diokno" in speaker_lower:
        return "Governor Benjamin E. Diokno"
    elif "medalla" in speaker_lower:
        return "Governor Felipe M. Medalla"
    elif "remolona" in speaker_lower:
        return "Governor Eli M. Remolona, Jr."
    
    return None


def determine_governor_by_date(date_str: str) -> Optional[str]:
    """Determine governor based on speech date."""
    if not date_str:
        return None
    
    # Parse date string
    date_formats = [
        '%B %d, %Y', '%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y',
        '%d %B %Y', '%d %b %Y', '%B %d %Y',
    ]
    
    speech_date = None
    for fmt in date_formats:
        try:
            speech_date = datetime.strptime(date_str.strip(), fmt)
            break
        except ValueError:
            continue
    
    if not speech_date:
        return None
    
    # Find matching governor period
    for period in GOVERNOR_PERIODS:
        if period["start"] <= speech_date <= period["end"]:
            return period["name"]
    
    return None


def classify_speech_audience(
    speech_text: str,
    title: str,
    occasion: str,
    place: str,
    speaker: str,
    client: AzureOpenAI,
    deployment_name: str,
    max_chars: int = 20000,
) -> Dict[str, Any]:
    """
    Classify the speech into one of five audience/setting categories using Azure OpenAI.
    
    Returns:
      {
        "label": one of LABEL_DEFS keys,
        "confidence": float 0..1,
        "reasons": [str, ...],
        "evidence": [str, ...]   # short exact snippets from the speech
      }
    """
    # Combine metadata and content for better classification
    text = f"""Title: {title}
Occasion: {occasion}
Place: {place}
Speaker: {speaker}

Content:
{speech_text or ""}""".strip()
    
    if not text:
        return {
            "label": "PUBLIC_REGIONAL",
            "confidence": 0.2,
            "reasons": ["Empty input"],
            "evidence": []
        }
    
    # Cap length to keep calls efficient
    if len(text) > max_chars:
        third = max_chars // 3
        text = text[:third] + "\n...\n" + text[len(text)//2:len(text)//2 + third] + "\n...\n" + text[-third:]
    
    system = (
        "You are a strict classifier. "
        "Choose EXACTLY ONE Audience/Setting label for a BSP speech. "
        "Return ONLY valid JSON. No markdown, no extra keys."
    )
    
    labels_json = json.dumps(
        {k: {"description": v["description"], "cues": v["cues"]} for k, v in LABEL_DEFS.items()},
        ensure_ascii=False,
        indent=2,
    )
    
    user = f"""
LABELS (choose exactly one):
{labels_json}

TASK:
Classify the speech into ONE label based on the intended audience/setting.
Use strongest cues: addressed titles, venue, event type, stakeholders mentioned.

OUTPUT JSON SCHEMA:
{{
  "label": "INTERNAL_BSP|GOV_OVERSIGHT|INDUSTRY_MARKET|INTERNATIONAL_OFFICIAL|PUBLIC_REGIONAL",
  "confidence": 0.0,
  "reasons": ["...", "..."],
  "evidence": ["short exact phrase from speech", "..."]
}}

SPEECH:
{text}
""".strip()
    
    try:
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        
        content = resp.choices[0].message.content or "{}"
        
        # Parse with safety fallback
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Attempt to extract the first JSON object found
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except Exception:
                    pass
            return {
                "label": "PUBLIC_REGIONAL",
                "confidence": 0.3,
                "reasons": ["Model output not parseable as JSON"],
                "evidence": [],
                "raw": content,
            }
    except Exception as e:
        return {
            "label": "PUBLIC_REGIONAL",
            "confidence": 0.1,
            "reasons": [f"API error: {str(e)}"],
            "evidence": []
        }


def create_metadata_item(row: pd.Series, classification: Dict[str, Any], tenant_id: str, governor: str) -> Dict[str, Any]:
    """Create a metadata item following the specified schema."""
    timestamp = int(time.time())
    
    # Generate document ID from ItemId
    doc_id = f"bsp_speech_{row.get('ItemId', timestamp)}"
    
    # Create partition key
    title_clean = str(row.get('Title', 'untitled')).replace(' ', '_')[:50]
    speaker_clean = str(row.get('Speaker', 'unknown')).replace(' ', '_').replace(',', '')[:30]
    partition_key = f"{title_clean}_speech_{speaker_clean}_{timestamp}"
    
    # Parse date
    date_str = str(row.get('Date', ''))
    
    item = {
        "id": doc_id,
        "PartitionKey": partition_key,
        "title": str(row.get('Title', '')),
        "sourceType": "speech",
        "sourcePath": str(row.get('URL', '')),
        "date": date_str,
        "place": str(row.get('Place', '')),
        "Occasion": str(row.get('Occasion', '')),  # Capital O to match schema
        "audience_setting_classification": classification.get("label", "PUBLIC_REGIONAL"),
        "Speaker": str(row.get('Speaker', '')),  # Capital S to match schema
        "Governor": governor or "",  # New Governor field
        "content": str(row.get('Content', ''))[:10000],  # Limit content length
        "language": "en",
        "tags": ["bsp", "speech", "government"],
        "group_ids": ["policy_team"],
        "tenantId": tenant_id,
        "lastModified": datetime.now().isoformat(),
        "summary": "",
        "isHeader": False,
        "scoreBoost": 1.0,
        "_ts": timestamp,
        # Additional classification metadata
        "classification_confidence": classification.get("confidence"),
        "classification_reasons": classification.get("reasons", []),
        "classification_evidence": classification.get("evidence", []),
        "ItemId": str(row.get('ItemId', '')),
    }
    
    return item


def main():
    print("=" * 70)
    print("BSP Speech Audience Classifier")
    print("=" * 70)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    print(f"   ✓ Azure OpenAI endpoint: {config['openai_endpoint'][:50]}...")
    print(f"   ✓ Chat deployment: {config['openai_chat_deployment']}")
    
    # Initialize Azure OpenAI client
    print("\n2. Initializing Azure OpenAI client...")
    client = AzureOpenAI(
        api_key=config["openai_key"],
        azure_endpoint=config["openai_endpoint"],
        api_version="2024-02-15-preview"
    )
    print("   ✓ Client initialized")
    
    # Load scraped speeches
    print("\n3. Loading scraped speeches from bsp_speeches.xlsx...")
    input_file = "bsp_speeches.xlsx"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found!")
    
    df = pd.read_excel(input_file)
    print(f"   ✓ Loaded {len(df)} speeches")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    
    # Classify each speech
    print(f"\n4. Classifying {len(df)} speeches and determining governors...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Determine Governor
        governor = normalize_governor_name(str(row.get('Speaker', '')))
        if not governor:
            governor = determine_governor_by_date(str(row.get('Date', '')))
        
        # Classify the speech
        classification = classify_speech_audience(
            speech_text=str(row.get('Content', '')),
            title=str(row.get('Title', '')),
            occasion=str(row.get('Occasion', '')),
            place=str(row.get('Place', '')),
            speaker=str(row.get('Speaker', '')),
            client=client,
            deployment_name=config["openai_chat_deployment"],
        )
        
        # Create metadata item
        item = create_metadata_item(row, classification, config["tenant_id"], governor)
        results.append(item)
        
        # Small delay to avoid rate limits
        time.sleep(0.2)
    
    # Convert to DataFrame
    print("\n5. Creating output DataFrame...")
    results_df = pd.DataFrame(results)
    
    # Display classification summary
    print("\n6. Classification Summary:")
    print("   " + "=" * 66)
    classification_counts = results_df['audience_setting_classification'].value_counts()
    for label, count in classification_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"   {label:30s}: {count:3d} ({pct:5.1f}%)")
    print("   " + "=" * 66)
    
    # Governor summary
    print("\n7. Governor Distribution:")
    print("   " + "=" * 66)
    governor_counts = results_df['Governor'].value_counts()
    for governor, count in governor_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"   {governor:40s}: {count:3d} ({pct:5.1f}%)")
    print("   " + "=" * 66)
    
    # Average confidence
    avg_confidence = results_df['classification_confidence'].mean()
    print(f"\n   Average Classification Confidence: {avg_confidence:.2%}")
    
    # Save to Excel
    output_file = "bsp_speeches_classified.xlsx"
    print(f"\n8. Saving results to {output_file}...")
    results_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"   ✓ Saved {len(results_df)} classified speeches")
    
    # Also save as JSON for potential Cosmos DB upload
    output_json = "bsp_speeches_classified.json"
    print(f"\n9. Saving JSON for Cosmos DB to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved JSON with {len(results)} documents")
    
    # Upload to Cosmos DB
    if config.get("cosmos_endpoint") and config.get("cosmos_container"):
        print(f"\n10. Uploading to Cosmos DB...")
        try:
            # Initialize Cosmos client - prefer using endpoint + key for reliability
            cosmos_client = CosmosClient(
                config["cosmos_endpoint"].rstrip('/'),
                credential=config["cosmos_key"]
            )
            
            # Get database and container
            database = cosmos_client.get_database_client(config["cosmos_database"])
            container = database.get_container_client(config["cosmos_container"])
            
            print(f"   Connected to: {config['cosmos_database']}/{config['cosmos_container']}")
            
            # Upload each document
            uploaded = 0
            failed = 0
            for item in tqdm(results, desc="Uploading to Cosmos"):
                try:
                    container.upsert_item(item)
                    uploaded += 1
                except Exception as e:
                    failed += 1
                    if failed <= 5:  # Only show first 5 errors
                        print(f"   ⚠ Failed to upload {item['id']}: {str(e)[:100]}")
            
            print(f"   ✓ Uploaded {uploaded} documents successfully")
            if failed > 0:
                print(f"   ⚠ Failed to upload {failed} documents")
        except Exception as e:
            print(f"   ✗ Error connecting to Cosmos DB: {e}")
            print(f"   Data saved locally - you can upload manually later")
    else:
        print(f"\n10. Cosmos DB credentials not configured - skipping upload")
        print(f"   Data saved locally in JSON format for manual upload")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 70)
    print(f"Output files:")
    print(f"  - {output_file} (Excel format)")
    print(f"  - {output_json} (JSON format for Cosmos DB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
