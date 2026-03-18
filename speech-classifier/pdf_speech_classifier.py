#!/usr/bin/env python3
"""
End-to-end ingestion pipeline for BSP speeches:
Azure Blob Storage -> Document Intelligence parse -> metadata extraction ->
audience classification -> Cosmos DB storage (NO chunking/embedding).
 
Flow:
1. DocIntelligence -> parse document
2. Extract information (title, date, place, occasion, speaker, content)
3. Classify document by audience_setting_classification using Azure OpenAI
4. Store to Cosmos
 
Required env vars:
- AZURE_STORAGE_CONNECTION_STRING (or AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_CREDENTIAL)
- AZURE_STORAGE_CONTAINER
- AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
- AZURE_DOCUMENT_INTELLIGENCE_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_KEY
- AZURE_OPENAI_CHAT_DEPLOYMENT (for audience classification)
- COSMOS_ENDPOINT
- COSMOS_KEY
- COSMOS_DATABASE
- COSMOS_CONTAINER
 
Optional env vars:
- TENANT_ID
- TAGS_JSON (e.g., ["speeches","bsp"])
- GROUP_IDS_JSON (e.g., ["policy_team"])
- SCORE_BOOST (default 1.0)
- MAX_DOCUMENTS (default 58, limits number of documents to process)
"""
 
from __future__ import annotations
 
import argparse
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
 
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm
 
 
# Audience/Setting Classification Labels
LABEL_DEFS = {
    "INTERNAL_BSP": {
        "description": "Internal BSP staff/leadership audience; culture, morale, internal milestones.",
        "cues": ["fellow BSPers", "turnover", "anniversary", "our colleagues", "within the BSP", "staff", "employees"],
    },
    "GOV_OVERSIGHT": {
        "description": "Government oversight/accountability forum (Congress, DBCC, executive briefings).",
        "cues": ["Honorable", "House of Representatives", "Senate", "DBCC", "budget", "assumptions", "committee hearing"],
    },
    "INDUSTRY_MARKET": {
        "description": "Regulated industry / market stakeholders (banks, associations, FM participants, infrastructures).",
        "cues": ["association", "banks", "bankers", "industry", "convention", "payments", "trade finance", "correspondent banking"],
    },
    "INTERNATIONAL_OFFICIAL": {
        "description": "International/multilateral policy community (IMF/BIS/World Bank etc.).",
        "cues": ["IMF", "BIS", "World Bank", "OECD", "international cooperation", "cross-border", "delegates"],
    },
    "PUBLIC_REGIONAL": {
        "description": "General public / regional outreach / broad economic briefings.",
        "cues": ["economic briefing", "public", "community", "region", "local", "Iloilo", "Cebu", "Davao", "inclusive growth"],
    },
}
 
 
@dataclass
class Config:
    storage_connection_string: Optional[str]
    storage_account_url: Optional[str]
    storage_credential: Optional[str]
    storage_container: str
 
    docint_endpoint: str
    docint_key: str
 
    openai_endpoint: str
    openai_key: str
    openai_chat_deployment: str
 
    cosmos_connection_string: Optional[str]
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str
    cosmos_container: str
 
    tenant_id: Optional[str]
    tags: List[str]
    group_ids: List[str]
    score_boost: float
    max_documents: int
 
 
def load_config() -> Config:
    # Load .env from the writer folder (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    load_dotenv(dotenv_path=env_path, override=False)
 
    def getenv(name: str, default: Optional[str] = None) -> Optional[str]:
        value = os.getenv(name, default)
        if isinstance(value, str):
            cleaned = value.strip()
            if (cleaned.startswith("\"") and cleaned.endswith("\"")) or (
                cleaned.startswith("'") and cleaned.endswith("'")
            ):
                cleaned = cleaned[1:-1]
            return cleaned
        return value
 
    def get_json_list(name: str) -> List[str]:
        raw = getenv(name)
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return [str(item) for item in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
 
    return Config(
        storage_connection_string=getenv("AZURE_STORAGE_CONNECTION_STRING"),
        storage_account_url=getenv("AZURE_STORAGE_ACCOUNT_URL"),
        storage_credential=getenv("AZURE_STORAGE_CREDENTIAL"),
        storage_container=getenv("AZURE_STORAGE_CONTAINER") or "",
        docint_endpoint=getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or "",
        docint_key=getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY") or "",
        openai_endpoint=getenv("AZURE_OPENAI_ENDPOINT") or "",
        openai_key=getenv("AZURE_OPENAI_KEY") or "",
        openai_chat_deployment=getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or "",
        cosmos_connection_string=getenv("COSMOS_CONNECTION_STRING"),
        cosmos_endpoint=getenv("COSMOS_ENDPOINT") or "",
        cosmos_key=getenv("COSMOS_KEY") or "",
        cosmos_database=getenv("COSMOS_DATABASE") or "",
        cosmos_container=getenv("COSMOS_CONTAINER") or "",
        tenant_id=getenv("TENANT_ID"),
        tags=get_json_list("TAGS_JSON"),
        group_ids=get_json_list("GROUP_IDS_JSON"),
        score_boost=float(getenv("SCORE_BOOST", "1.0")),
        max_documents=int(getenv("MAX_DOCUMENTS", "58")),
    )
 
 
def validate_config(cfg: Config) -> None:
    missing = []
    if not (cfg.storage_connection_string or (cfg.storage_account_url and cfg.storage_credential)):
        missing.append("AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_CREDENTIAL")
    if not cfg.storage_container:
        missing.append("AZURE_STORAGE_CONTAINER")
    if not cfg.docint_endpoint or not cfg.docint_key:
        missing.append("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT/AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not cfg.openai_endpoint or not cfg.openai_key or not cfg.openai_chat_deployment:
        missing.append("AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY/AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not cfg.cosmos_connection_string and (not cfg.cosmos_endpoint or not cfg.cosmos_key):
        missing.append("COSMOS_CONNECTION_STRING or COSMOS_ENDPOINT/COSMOS_KEY")
    if not cfg.cosmos_database or not cfg.cosmos_container:
        missing.append("COSMOS_DATABASE/COSMOS_CONTAINER")
 
    if missing:
        raise SystemExit("Missing required configuration: " + ", ".join(missing))
 
 
def get_blob_service_client(cfg: Config) -> BlobServiceClient:
    if cfg.storage_connection_string:
        return BlobServiceClient.from_connection_string(cfg.storage_connection_string)
    return BlobServiceClient(account_url=cfg.storage_account_url, credential=cfg.storage_credential)
 
 
def get_docint_client(cfg: Config) -> DocumentIntelligenceClient:
    return DocumentIntelligenceClient(cfg.docint_endpoint, AzureKeyCredential(cfg.docint_key))
 
 
def get_openai_client(cfg: Config) -> AzureOpenAI:
    return AzureOpenAI(api_key=cfg.openai_key, azure_endpoint=cfg.openai_endpoint, api_version="2024-02-15-preview")
 
 
def get_cosmos_container(cfg: Config):
    if cfg.cosmos_connection_string:
        client = CosmosClient.from_connection_string(cfg.cosmos_connection_string)
    else:
        client = CosmosClient(cfg.cosmos_endpoint, credential=cfg.cosmos_key)
    database = client.create_database_if_not_exists(cfg.cosmos_database)
   
    # No vector indexing needed - simple document storage
    container = database.create_container_if_not_exists(
        id=cfg.cosmos_container,
        partition_key=PartitionKey(path="/PartitionKey"),
    )
 
    return container
 
 
def guess_mime_type(name: str) -> str:
    mime, _ = mimetypes.guess_type(name)
    return mime or "application/octet-stream"
 
 
def extract_text_docint(doc_client: DocumentIntelligenceClient, file_bytes: bytes) -> str:
    """Extract full text from document using Document Intelligence."""
    poller = doc_client.begin_analyze_document(model_id="prebuilt-read", body=file_bytes)
    result = poller.result()
    lines = []
    for page in result.pages:
        if page.lines:
            lines.extend([line.content for line in page.lines])
    return "\n".join(lines).strip()
 
 
def extract_metadata_from_text(text: str, blob_name: str) -> Dict[str, Any]:
    """
    Extract structured metadata from speech text based on the pattern:
    - Title (first substantial line or from "Suggested Title:")
    - Date: <value>
    - Place: <value>
    - Occasion: <value>
    - Speaker: <value>
   
    Returns dict with extracted fields.
    """
    metadata = {
        "title": "",
        "date": "",
        "place": "",
        "occasion": "",
        "speaker": "",
    }
   
    lines = text.split("\n")
   
    # Extract title (look for "Suggested Title:" or use first substantial line)
    title_found = False
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if "suggested title:" in line_stripped.lower():
            # Title is on the next line or same line after colon
            if ":" in line_stripped:
                title = line_stripped.split(":", 1)[1].strip()
                if title:
                    metadata["title"] = title
                    title_found = True
                elif i + 1 < len(lines):
                    metadata["title"] = lines[i + 1].strip()
                    title_found = True
            break
   
    # If no "Suggested Title:", use the first substantial line (likely the actual title)
    if not title_found:
        for line in lines[:10]:  # Check first 10 lines
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) > 10 and not line_stripped.lower().startswith("media and research"):
                metadata["title"] = line_stripped
                break
   
    # Fallback to filename if no title found
    if not metadata["title"]:
        metadata["title"] = os.path.splitext(os.path.basename(blob_name))[0]
   
    # Extract other metadata using regex patterns
    for line in lines[:50]:  # Check first 50 lines where metadata typically appears
        line_stripped = line.strip()
       
        # Date pattern: "Date: <value>"
        if re.match(r"^date:\s*.+", line_stripped, re.IGNORECASE):
            metadata["date"] = re.sub(r"^date:\s*", "", line_stripped, flags=re.IGNORECASE).strip()
       
        # Place pattern: "Place: <value>"
        elif re.match(r"^place:\s*.+", line_stripped, re.IGNORECASE):
            metadata["place"] = re.sub(r"^place:\s*", "", line_stripped, flags=re.IGNORECASE).strip()
       
        # Occasion pattern: "Occasion: <value>"
        elif re.match(r"^occasion:\s*.+", line_stripped, re.IGNORECASE):
            metadata["occasion"] = re.sub(r"^occasion:\s*", "", line_stripped, flags=re.IGNORECASE).strip()
       
        # Speaker pattern: "Speaker: <value>"
        elif re.match(r"^speaker:\s*.+", line_stripped, re.IGNORECASE):
            metadata["speaker"] = re.sub(r"^speaker:\s*", "", line_stripped, flags=re.IGNORECASE).strip()
   
    return metadata
 
 
def classify_speech_audience(
    speech_text: str,
    deployment_name: str,
    client: AzureOpenAI,
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
    text = (speech_text or "").strip()
    if not text:
        return {"label": "PUBLIC_REGIONAL", "confidence": 0.2, "reasons": ["Empty input"], "evidence": []}
 
    # Cap length to keep calls efficient - prefer opening + closing + mid slice
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
Use strongest cues: addressed titles, venue, event type, stakeholders.
 
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
 
 
def store_document(
    container,
    doc_id: str,
    source_url: str,
    source_type: str,
    title: str,
    date: str,
    place: str,
    occasion: str,
    speaker: str,
    content: str,
    audience_classification: Dict[str, Any],
    tags: List[str],
    group_ids: List[str],
    tenant_id: Optional[str],
    last_modified: Optional[str],
    score_boost: float,
    author_name: str = "",
) -> None:
    """
    Store a single document to Cosmos DB with the target schema.
    PartitionKey format: <Filename>_<sourceType>_<authorName>_<timestamp>
    """
    timestamp = int(time.time())
    filename = os.path.basename(source_url.split("?")[0])  # Remove query params if any
   
    # Extract author name from speaker field (simplified)
    if not author_name and speaker:
        # Try to extract just the name part (e.g., "BSP Governor Eli M. Remolona, Jr" -> "Eli M. Remolona")
        author_name = speaker.replace("BSP Governor", "").replace("Jr.", "").replace(",", "").strip()
   
    partition_key = f"{filename}_{source_type}_{author_name}_{timestamp}"
   
    item = {
        "id": doc_id,
        "PartitionKey": partition_key,
        "title": title,
        "sourceType": source_type,
        "sourcePath": source_url,
        "date": date,
        "place": place,
        "Occasion": occasion,  # Note: Capital O to match schema
        "audience_setting_classification": audience_classification.get("label", "PUBLIC_REGIONAL"),
        "Speaker": speaker,  # Note: Capital S to match schema
        "content": content,
        "language": "en",
        "tags": tags,
        "group_ids": group_ids,
        "tenantId": tenant_id,
        "lastModified": last_modified,
        "summary": "",
        "isHeader": False,
        "scoreBoost": score_boost,
        "_ts": timestamp,
        # Additional classification metadata (optional, for debugging/tracking)
        "classification_confidence": audience_classification.get("confidence"),
        "classification_reasons": audience_classification.get("reasons", []),
        "classification_evidence": audience_classification.get("evidence", []),
    }
   
    container.upsert_item(item)
    print(f"  Stored document with audience classification: {audience_classification.get('label')}")
    print(f"  Confidence: {audience_classification.get('confidence', 0):.2f}")
 
 
def process_blob(
    cfg: Config,
    blob_client,
    doc_client: DocumentIntelligenceClient,
    openai: AzureOpenAI,
    container,
    skip_existing: bool,
) -> None:
    """
    Process a single blob through the ETL pipeline:
    1. Extract text using Document Intelligence
    2. Extract metadata (title, date, place, occasion, speaker)
    3. Classify audience/setting using Azure OpenAI
    4. Store to Cosmos DB
    """
    blob_name = blob_client.blob_name
    doc_id = blob_name.replace("/", "_")
    source_url = blob_client.url
    blob_props = blob_client.get_blob_properties()
    last_modified = blob_props.last_modified.isoformat() if blob_props and blob_props.last_modified else None
 
    print(f"\n{'='*80}")
    print(f"Processing: {blob_name}")
    print(f"{'='*80}")
 
    # Check if already exists
    if skip_existing:
        try:
            existing = container.read_item(item=doc_id, partition_key=doc_id)
            if existing:
                print(f"✓ Skipping already indexed: {blob_name}")
                return
        except Exception:
            pass  # Document doesn't exist, continue processing
 
    # Download blob
    print("1. Downloading blob...")
    data = blob_client.download_blob().readall()
   
    # Extract text using Document Intelligence
    print("2. Extracting text with Document Intelligence...")
    text = extract_text_docint(doc_client, data)
   
    if not text or len(text) < 50:
        print(f"✗ No sufficient text extracted for: {blob_name}")
        return
   
    print(f"  Extracted {len(text)} characters")
   
    # Extract metadata from text
    print("3. Extracting metadata...")
    metadata = extract_metadata_from_text(text, blob_name)
    print(f"  Title: {metadata['title']}")
    print(f"  Date: {metadata['date']}")
    print(f"  Place: {metadata['place']}")
    print(f"  Occasion: {metadata['occasion']}")
    print(f"  Speaker: {metadata['speaker']}")
   
    # Classify audience/setting
    print("4. Classifying audience/setting with Azure OpenAI...")
    classification = classify_speech_audience(
        speech_text=text,
        deployment_name=cfg.openai_chat_deployment,
        client=openai,
    )
    print(f"  Classification: {classification.get('label')} (confidence: {classification.get('confidence', 0):.2f})")
    if classification.get('reasons'):
        print(f"  Reasons: {', '.join(classification['reasons'][:2])}")
   
    # Determine source type from file extension
    ext = os.path.splitext(blob_name)[1].lstrip(".").upper()
    source_type = ext if ext else "PDF"
   
    # Store to Cosmos DB
    print("5. Storing to Cosmos DB...")
    store_document(
        container=container,
        doc_id=doc_id,
        source_url=source_url,
        source_type=source_type,
        title=metadata["title"],
        date=metadata["date"],
        place=metadata["place"],
        occasion=metadata["occasion"],
        speaker=metadata["speaker"],
        content=text,
        audience_classification=classification,
        tags=cfg.tags,
        group_ids=cfg.group_ids,
        tenant_id=cfg.tenant_id,
        last_modified=last_modified,
        score_boost=cfg.score_boost,
    )
   
    print(f"✓ Successfully processed: {blob_name}")
    print(f"{'='*80}\n")
 
 
def ingest_all(cfg: Config, prefix: Optional[str], skip_existing: bool) -> None:
    """
    Ingest all blobs from Azure Blob Storage through the ETL pipeline.
    """
    print("\n" + "="*80)
    print("BSP Speech ETL Pipeline - Starting Ingestion")
    print("="*80)
   
    blob_service = get_blob_service_client(cfg)
    container_client = blob_service.get_container_client(cfg.storage_container)
 
    doc_client = get_docint_client(cfg)
    openai = get_openai_client(cfg)
    cosmos_container = get_cosmos_container(cfg)
 
    print(f"\nScanning blobs in container: {cfg.storage_container}")
    if prefix:
        print(f"With prefix filter: {prefix}")
   
    blob_list = list(container_client.list_blobs(name_starts_with=prefix))
   
    # Apply document limit
    if len(blob_list) > cfg.max_documents:
        print(f"Found {len(blob_list)} blobs, limiting to {cfg.max_documents} (MAX_DOCUMENTS setting)")
        blob_list = blob_list[:cfg.max_documents]
    else:
        print(f"Found {len(blob_list)} blobs to process")
    print()
   
    success_count = 0
    error_count = 0
   
    for blob in tqdm(blob_list, desc="Processing blobs", unit="file"):
        blob_client = container_client.get_blob_client(blob.name)
        try:
            process_blob(cfg, blob_client, doc_client, openai, cosmos_container, skip_existing)
            success_count += 1
        except Exception as exc:
            error_count += 1
            print(f"✗ Error processing {blob.name}: {exc}")
            import traceback
            traceback.print_exc()
   
    print("\n" + "="*80)
    print("ETL Pipeline Complete")
    print("="*80)
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {len(blob_list)}")
    print("="*80 + "\n")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest BSP speech PDFs from Azure Blob Storage into Cosmos DB with audience classification."
    )
    parser.add_argument("--prefix", help="Optional blob prefix to filter", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip blobs already indexed")
    args = parser.parse_args()
 
    cfg = load_config()
    validate_config(cfg)
 
    ingest_all(cfg, args.prefix, args.skip_existing)
 
 
if __name__ == "__main__":
    main()
 