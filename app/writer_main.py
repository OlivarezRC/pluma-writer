import os
import json
import time
import asyncio
import re
import requests
import pandas as pd
import streamlit as st

from datetime import datetime
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from dotenv import load_dotenv
import hashlib
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# Load environment variables
load_dotenv()

# Import deep_research components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_research.pipeline import run_deep_research

# Import for link processing
from tavily import AsyncTavilyClient
from azure.core.credentials import AzureKeyCredential
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage

# Import plagiarism checker
from app.plagiarism_checker import check_plagiarism

# Import policy checker
from app.policy_checker import check_speech_policy_alignment


# Model initialization for link processing - lazy loaded
_link_processing_model = None

def _get_link_processing_model():
    """
    Lazy initialization of model for link processing.
    
    Returns:
        AzureAIChatCompletionsModel: Initialized model instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    global _link_processing_model
    
    if _link_processing_model is not None:
        return _link_processing_model
    
    # Get environment variables
    _endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    _model_name = os.getenv("AZURE_DEEPSEEK_DEPLOYMENT")
    _key = os.getenv("AZURE_AI_API_KEY")
    
    # Validate required environment variables
    if not _endpoint or not _model_name or not _key:
        missing = []
        if not _endpoint: missing.append("AZURE_INFERENCE_ENDPOINT")
        if not _model_name: missing.append("AZURE_DEEPSEEK_DEPLOYMENT")
        if not _key: missing.append("AZURE_AI_API_KEY")
        raise ValueError(
            f"Missing required environment variables for link processing: {', '.join(missing)}. "
            "Please set these in your .env file."
        )
    
    # Initialize model
    _link_processing_model = AzureAIChatCompletionsModel(
        endpoint=_endpoint,
        credential=AzureKeyCredential(_key),
        model=_model_name,
    )
    
    return _link_processing_model


async def generate_summary_from_evidence(query: str, evidence_store: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary with forced [ENN] citations.
    
    Args:
        query: The user's original query
        evidence_store: List of atomic evidence items with id, claim, quote_span
    
    Returns:
        Dictionary with summary, citations metadata, and validation results
    """
    if not evidence_store or len(evidence_store) == 0:
        return {
            "success": False,
            "summary": "",
            "error": "No evidence available to generate summary",
            "evidence_count": 0
        }
    
    try:
        import re
        
        # Get LLM model
        model = _get_link_processing_model()
        
        # Build the evidence context with atomic claims
        evidence_context = ""
        allowed_ids = []
        for evidence in evidence_store:
            eid = evidence.get("id", "")
            claim = evidence.get("claim", "")
            source = evidence.get("source_url", evidence.get("source", ""))
            confidence = evidence.get("confidence", 0.0)
            
            if eid:
                allowed_ids.append(eid)
                evidence_context += f"{eid}: {claim}\n"
                evidence_context += f"Source: {source} (confidence: {confidence:.2f})\n\n"
        
        if not allowed_ids:
            return {
                "success": False,
                "summary": "",
                "error": "No valid evidence IDs found",
                "evidence_count": len(evidence_store)
            }
        
        # Create allowed IDs string
        allowed_ids_str = ", ".join(allowed_ids)
        
        # Create the summary generation prompt with STRICT citation rules
        summary_prompt = f"""You are a research assistant synthesizing information to answer a query.

<USER_QUERY>
{query}
</USER_QUERY>

<EVIDENCE>
{evidence_context}
</EVIDENCE>

<STRICT CITATION RULES>
1. EVERY factual sentence MUST end with a citation in format [ENN] or [E1,E3,E5]
2. You may ONLY cite these evidence IDs: {allowed_ids_str}
3. NEVER generate citations to non-existent evidence
4. Multiple claims in one sentence = multiple citations: [E1,E2,E5]
5. General statements without specific facts do NOT need citations
6. Cite the EXACT evidence that supports each claim

EXAMPLES:
✓ "Transformers use attention mechanisms to process sequential data [E1]."
✓ "The model achieved 95% accuracy on the benchmark dataset [E3,E7]."
✗ "Transformers are important." (too general, missing citation)
✗ "Deep learning has many applications [E99]." (E99 doesn't exist)
</STRICT CITATION RULES>

<TASK>
Write a comprehensive summary that:
1. Directly answers the user's query
2. Synthesizes insights from the evidence
3. Cites EVERY factual claim with [ENN] format
4. Organizes information logically
5. Only uses valid evidence IDs from the list above

Generate your summary with citations:"""
        
        messages = [
            SystemMessage(content="You are an expert research synthesizer who ALWAYS cites sources using [ENN] format for every factual claim."),
            HumanMessage(content=summary_prompt)
        ]
        
        result = await model.ainvoke(messages)
        summary_text = result.content.strip()
        
        # Strip thinking tokens if present
        if "<think>" in summary_text and "</think>" in summary_text:
            while "<think>" in summary_text and "</think>" in summary_text:
                start = summary_text.find("<think>")
                end = summary_text.find("</think>") + 8
                summary_text = summary_text[:start] + summary_text[end:]
            summary_text = summary_text.strip()
        
        # Validate citations
        citation_pattern = r'\[E\d+(?:,E\d+)*\]'
        citations_found = re.findall(citation_pattern, summary_text)
        
        # Extract individual evidence IDs from citations
        cited_ids = set()
        invalid_citations = []
        for citation in citations_found:
            # Extract E1, E2, etc. from [E1,E2]
            ids_in_citation = re.findall(r'E\d+', citation)
            for eid in ids_in_citation:
                cited_ids.add(eid)
                if eid not in allowed_ids:
                    invalid_citations.append(eid)
        
        # Calculate coverage
        cited_count = len(cited_ids & set(allowed_ids))
        total_evidence = len(allowed_ids)
        coverage = (cited_count / total_evidence * 100) if total_evidence > 0 else 0
        
        return {
            "success": True,
            "summary": summary_text,
            "evidence_count": len(evidence_store),
            "query": query,
            "citations_found": len(citations_found),
            "unique_evidence_cited": cited_count,
            "citation_coverage": f"{coverage:.1f}%",
            "invalid_citations": invalid_citations,
            "validation": {
                "all_citations_valid": len(invalid_citations) == 0,
                "cited_ids": sorted(list(cited_ids)),
                "uncited_ids": sorted(list(set(allowed_ids) - cited_ids))
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "error": str(e),
            "evidence_count": len(evidence_store)
        }


async def critique_summary(query: str, summary: str, evidence_store: List[Dict[str, Any]], iteration: int) -> Dict[str, Any]:
    """
    Critique the current summary to identify gaps and areas for improvement.
    
    Args:
        query: The user's original query
        summary: The current generated summary
        evidence_store: The evidence used to create the summary
        iteration: Current iteration number
    
    Returns:
        Dictionary with critique and suggested adjustments
    """
    try:
        # Get LLM model
        model = _get_link_processing_model()
        
        critique_prompt = f"""You are a research critic tasked with evaluating a research summary and identifying areas for improvement.

<USER_QUERY>
{query}
</USER_QUERY>

<CURRENT_SUMMARY>
{summary}
</CURRENT_SUMMARY>

<ITERATION>
This is iteration {iteration} of the research process.
</ITERATION>

<TASK>
Analyze the summary and provide constructive criticism in these areas:

1. **Research Gaps**: What important aspects of the query are not adequately addressed? What questions remain unanswered?

2. **Topic Alignment**: How well does the summary align with the user's original query? Are there tangential topics that should be removed or refocused?

3. **Information Enrichment**: What specific details, examples, or explanations would make the summary more comprehensive and valuable?

4. **Source Coverage**: Based on the {len(evidence_store)} evidence sources, are there perspectives or information not yet incorporated?

5. **Clarity and Organization**: How could the structure or presentation be improved?

Provide your critique in a structured format with:
- **Gaps Identified**: List specific missing information
- **Alignment Issues**: Note any misalignments with the query
- **Enrichment Opportunities**: Suggest specific improvements
- **Recommended Focus**: What should the next iteration prioritize?

Be specific and actionable in your recommendations.
</TASK>

Provide your critique:"""
        
        messages = [
            SystemMessage(content="You are an expert research critic who identifies gaps and suggests improvements in research summaries."),
            HumanMessage(content=critique_prompt)
        ]
        
        result = await model.ainvoke(messages)
        critique_text = result.content.strip()
        
        # Strip thinking tokens
        if "<think>" in critique_text and "</think>" in critique_text:
            while "<think>" in critique_text and "</think>" in critique_text:
                start = critique_text.find("<think>")
                end = critique_text.find("</think>") + 8
                critique_text = critique_text[:start] + critique_text[end:]
            critique_text = critique_text.strip()
        
        return {
            "success": True,
            "critique": critique_text,
            "iteration": iteration
        }
        
    except Exception as e:
        return {
            "success": False,
            "critique": "",
            "error": str(e),
            "iteration": iteration
        }


async def generate_adjustments_from_critique(query: str, critique: str) -> str:
    """
    Generate query adjustments based on critique to guide the next iteration.
    
    Args:
        query: Original user query
        critique: The critique from previous iteration
    
    Returns:
        Adjusted query string with additional focus areas
    """
    try:
        model = _get_link_processing_model()
        
        adjustment_prompt = f"""Based on a critique of the current research, generate specific adjustments to refine the research focus.

<ORIGINAL_QUERY>
{query}
</ORIGINAL_QUERY>

<CRITIQUE>
{critique}
</CRITIQUE>

<TASK>
Create a refined query that:
1. Maintains the core intent of the original query
2. Adds specific focus areas identified in the critique
3. Addresses the gaps and enrichment opportunities mentioned
4. Guides the next research iteration toward completeness

Provide ONLY the enhanced query text without any preamble or explanation.
</TASK>

Enhanced query:"""
        
        messages = [
            SystemMessage(content="You are a research assistant who refines queries based on identified gaps."),
            HumanMessage(content=adjustment_prompt)
        ]
        
        result = await model.ainvoke(messages)
        adjusted_query = result.content.strip()
        
        # Strip thinking tokens
        if "<think>" in adjusted_query and "</think>" in adjusted_query:
            while "<think>" in adjusted_query and "</think>" in adjusted_query:
                start = adjusted_query.find("<think>")
                end = adjusted_query.find("</think>") + 8
                adjusted_query = adjusted_query[:start] + adjusted_query[end:]
            adjusted_query = adjusted_query.strip()
        
        return adjusted_query
        
    except Exception as e:
        # If adjustment generation fails, return original query
        return query


def split_sources(sources: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split the sources into topics, links, and attachments.
    
    Args:
        sources: JSON object containing:
            - topics: string
            - links: array of strings
            - attachments: array of file paths/objects
    
    Returns:
        Dictionary with separated sources
    """
    return {
        "topics": sources.get("topics", ""),
        "links": sources.get("links", []),
        "attachments": sources.get("attachments", [])
    }


async def topic_processing(topic: str, query: str, evidence_id_start: int = 1) -> Dict[str, Any]:
    """
    Process a topic by calling deep_research module.
    Extracts atomic claims as evidence with stable IDs and exact quotes.
    
    Args:
        topic: The research topic string
        query: The user's query/question
        evidence_id_start: Starting ID number for evidence items
    
    Returns:
        Research results with atomic evidence store
    """
    if not topic or not topic.strip():
        return {
            "success": False,
            "error": "Empty topic provided",
            "evidence_store": [],
            "next_evidence_id": evidence_id_start
        }
    
    try:
        from datetime import datetime
        
        # Call the deep_research pipeline
        research_summary = await run_deep_research(
            topic=topic,
            notify=None
        )
        
        # Extract atomic claims from summary
        evidence_store = []
        evidence_id = evidence_id_start
        
        # Use LLM to extract atomic claims
        model = _get_link_processing_model()
        
        extraction_prompt = f"""Extract atomic factual claims from this research summary with APA citation metadata. Each claim should be:
- One clear factual statement
- Self-contained and verifiable
- Include the exact quote span that supports it
- Include bibliographic metadata for APA citations

Research Summary:
{research_summary[:4000]}

Return a JSON array where each item has:
- "claim": the atomic factual statement
- "quote_span": exact excerpt from text that supports this claim (20-100 words)
- "confidence": your confidence (0.0-1.0) that this is factually grounded
- "author": author name(s) if mentioned in text (e.g., "Smith", "Jones & Brown", or "n.d." if not found)
- "year": publication year if mentioned (e.g., "2023" or "n.d." if not found)
- "publication": publication/source name if mentioned (e.g., journal, website, organization)
- "publisher": publisher or organization name if mentioned

Return ONLY the JSON array, no other text."""
        
        try:
            messages = [
                SystemMessage(content="You are a precise fact extraction system. Extract atomic claims with exact quotes."),
                HumanMessage(content=extraction_prompt)
            ]
            
            response = await model.ainvoke(messages)
            content = response.content
            
            # Remove markdown code blocks if present
            content = content.replace('```json', '').replace('```', '').strip()
            if '<think>' in content:
                content = content.split('</think>')[-1].strip()
            
            claims = json.loads(content)
            
            # Create evidence items with stable IDs and APA metadata
            for claim_data in claims:
                evidence_store.append({
                    "id": f"E{evidence_id}",
                    "claim": claim_data.get("claim", ""),
                    "source": f"Deep Research: {topic}",
                    "source_url": None,
                    "source_title": topic,
                    "quote_span": claim_data.get("quote_span", ""),
                    "author": claim_data.get("author", "n.d."),
                    "year": claim_data.get("year", "n.d."),
                    "publication": claim_data.get("publication", topic),
                    "publisher": claim_data.get("publisher", "n.d."),
                    "retrieval_context": "deep_research_pipeline",
                    "confidence": claim_data.get("confidence", 0.8),
                    "timestamp_accessed": datetime.now().isoformat()
                })
                evidence_id += 1
                
        except Exception as e:
            print(f"Error extracting atomic claims: {e}")
            # Fallback: create evidence from substantial paragraphs
            paragraphs = [p.strip() for p in research_summary.split('\n\n') if len(p.strip()) > 100]
            for para in paragraphs[:5]:
                if not para.startswith('#'):
                    evidence_store.append({
                        "id": f"E{evidence_id}",
                        "claim": para[:200] + "..." if len(para) > 200 else para,
                        "source": f"Deep Research: {topic}",
                        "source_url": None,
                        "source_title": topic,
                        "quote_span": para,
                        "retrieval_context": "deep_research_pipeline",
                        "confidence": 0.7,
                        "timestamp_accessed": datetime.now().isoformat()
                    })
                    evidence_id += 1
        
        return {
            "success": True,
            "topic": topic,
            "query": query,
            "summary": research_summary,
            "evidence_store": evidence_store,
            "next_evidence_id": evidence_id
        }
    
    except Exception as e:
        return {
            "success": False,
            "topic": topic,
            "error": str(e),
            "evidence_store": [],
            "next_evidence_id": evidence_id_start
        }


async def process_links(links: List[str], query: str, evidence_id_start: int = 1) -> Dict[str, Any]:
    """
    Process links to extract atomic claims with validated citations.
    
    Args:
        links: Array of URL strings
        query: The user's query
        evidence_id_start: Starting ID number for evidence items
    
    Returns:
        Results with atomic evidence store and stable IDs
    """
    from datetime import datetime
    
    if not links:
        return {
            "type": "links",
            "count": 0,
            "items": [],
            "query": query,
            "evidence_store": [],
            "next_evidence_id": evidence_id_start
        }
    
    processed_links = []
    evidence_store = []
    evidence_id = evidence_id_start
    tavily_client = None
    
    try:
        # Initialize Tavily client
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {
                "type": "links",
                "count": len(links),
                "error": "TAVILY_API_KEY not found in environment variables",
                "items": [{"url": link, "status": "skipped"} for link in links],
                "query": query,
                "evidence_store": [],
                "next_evidence_id": evidence_id_start
            }
        
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        
        # Get LLM model
        try:
            model = _get_link_processing_model()
        except ValueError as e:
            return {
                "type": "links",
                "count": len(links),
                "error": str(e),
                "items": [{"url": link, "status": "skipped"} for link in links],
                "query": query,
                "evidence_store": [],
                "next_evidence_id": evidence_id_start
            }
        
        # Process each link
        for link in links:
            try:
                # Use Tavily to extract content
                search_result = await tavily_client.extract(urls=[link])
                
                if not search_result or not search_result.get("results") or len(search_result.get("results", [])) == 0:
                    processed_links.append({
                        "url": link,
                        "status": "error",
                        "error": "Could not extract content"
                    })
                    continue
                
                # Get extracted content
                result = search_result["results"][0]
                raw_content = result.get("raw_content", "")
                title = result.get("title", "Unknown")
                
                if not raw_content or len(raw_content.strip()) == 0:
                    processed_links.append({
                        "url": link,
                        "status": "error",
                        "error": "Empty content"
                    })
                    continue
                
                # Truncate if too long
                if len(raw_content) > 5000:
                    raw_content = raw_content[:5000] + "..."
                
                # Extract atomic claims with exact quotes and APA metadata
                extraction_prompt = f"""Extract atomic factual claims from this source that are relevant to the query.

Query: {query}

Source: {title}
URL: {link}

Content:
{raw_content}

For each relevant fact, provide:
- "claim": one clear, self-contained factual statement
- "quote_span": the EXACT excerpt from the content (20-150 words) that supports this claim
- "confidence": your confidence (0.0-1.0) this is accurately represented
- "author": author name(s) from the content (e.g., "Smith", "Jones & Brown", or extract from URL/metadata if not in text)
- "year": publication year from content or URL (e.g., "2023", or "n.d." if not found)
- "publication": publication/source name (journal, website, organization name)
- "publisher": publisher or organization name if identifiable

Return ONLY a JSON array of claims. Extract 3-8 most relevant atomic facts."""
                
                messages = [
                    SystemMessage(content="You are a precise fact extraction system. Extract atomic claims with exact verbatim quotes."),
                    HumanMessage(content=extraction_prompt)
                ]
                
                response = await model.ainvoke(messages)
                response_content = response.content
                
                # Strip thinking tags
                if "<think>" in response_content:
                    response_content = response_content.split("</think>")[-1].strip()
                
                # Remove markdown code blocks
                response_content = response_content.replace('```json', '').replace('```', '').strip()
                
                try:
                    claims = json.loads(response_content)
                    
                    # Create evidence items with stable IDs and APA metadata
                    for claim_data in claims:
                        evidence_store.append({
                            "id": f"E{evidence_id}",
                            "claim": claim_data.get("claim", ""),
                            "source": link,
                            "source_url": link,
                            "source_title": title,
                            "quote_span": claim_data.get("quote_span", ""),
                            "author": claim_data.get("author", "n.d."),
                            "year": claim_data.get("year", "n.d."),
                            "publication": claim_data.get("publication", title),
                            "publisher": claim_data.get("publisher", "n.d."),
                            "retrieval_context": "link_processing",
                            "confidence": claim_data.get("confidence", 0.85),
                            "timestamp_accessed": datetime.now().isoformat()
                        })
                        evidence_id += 1
                    
                    processed_links.append({
                        "url": link,
                        "status": "success",
                        "claims_extracted": len(claims),
                        "title": title
                    })
                    
                    print(f"✓ Extracted {len(claims)} atomic claims from {link}")
                    
                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse claims from {link}: {e}")
                    processed_links.append({
                        "url": link,
                        "status": "error",
                        "error": "Failed to parse claims"
                    })
                    
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Error processing {link}: {error_msg}")
                processed_links.append({
                    "url": link,
                    "status": "error",
                    "error": f"Could not extract content: {error_msg}"
                })
    
    finally:
        # Close Tavily client
        if tavily_client:
            await tavily_client.close()
    
    return {
        "type": "links",
        "count": len(links),
        "items": processed_links,
        "query": query,
        "evidence_store": evidence_store,
        "next_evidence_id": evidence_id
    }


async def process_attachments(attachments: List[Any], query: str, evidence_id_start: int = 1) -> Dict[str, Any]:
    """
    Process file attachments provided by the user.
    
    Args:
        attachments: Array of file paths or file objects
        query: The user's query
    
    Returns:
        Processed attachment information
    """
    from datetime import datetime

    if not attachments:
        return {
            "type": "attachments",
            "count": 0,
            "items": [],
            "query": query,
            "evidence_store": [],
            "next_evidence_id": evidence_id_start,
        }

    async def extract_docint_text(raw_bytes: bytes) -> str:
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        if not endpoint or not key or not raw_bytes:
            return ""

        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/octet-stream",
        }

        analyze_urls = [
            f"{endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30",
            f"{endpoint}/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31",
        ]

        op_url = None
        for url in analyze_urls:
            try:
                resp = requests.post(url, headers=headers, data=raw_bytes, timeout=60)
                if resp.status_code in [200, 202]:
                    op_url = resp.headers.get("operation-location") or resp.headers.get("Operation-Location")
                    if op_url:
                        break
            except Exception:
                continue

        if not op_url:
            return ""

        poll_headers = {"Ocp-Apim-Subscription-Key": key}
        for _ in range(25):
            try:
                poll = requests.get(op_url, headers=poll_headers, timeout=60)
                data = poll.json() if poll.content else {}
                status = (data.get("status") or "").lower()
                if status == "succeeded":
                    result = data.get("analyzeResult", {})
                    lines = []
                    for paragraph in result.get("paragraphs", []):
                        text = paragraph.get("content", "")
                        if text:
                            lines.append(text)
                    if not lines:
                        for page in result.get("pages", []):
                            for line in page.get("lines", []):
                                text = line.get("content", "")
                                if text:
                                    lines.append(text)
                    return "\n".join(lines)
                if status in ["failed", "error"]:
                    return ""
            except Exception:
                return ""
            await asyncio.sleep(1)

        return ""

    async def extract_metadata_with_llm(text: str, filename: str, fallback_title: str) -> Dict[str, str]:
        metadata = {
            "author": "Unknown",
            "year": "Unknown",
            "publication": "Unknown",
            "source_title": fallback_title,
        }
        if not text:
            return metadata

        try:
            model = _get_link_processing_model()
            prompt = f"""Extract bibliographic metadata from this document text and filename.

Filename: {filename}
Fallback title: {fallback_title}

Document excerpt:
{text[:3500]}

Return ONLY JSON object with keys:
- author
- year
- publication
- source_title

Rules:
- If missing, return "Unknown" for author/year/publication.
- source_title should be meaningful title if identifiable, else use fallback title.
"""

            response = await model.ainvoke([
                SystemMessage(content="You extract bibliographic metadata from text."),
                HumanMessage(content=prompt),
            ])
            content = response.content.replace("```json", "").replace("```", "").strip()
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            parsed = json.loads(content)
            for key in metadata:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    metadata[key] = value.strip()
        except Exception:
            pass

        return metadata

    async def extract_claims_with_llm(text: str, query_text: str, metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        if not text:
            return []
        try:
            model = _get_link_processing_model()
            prompt = f"""Extract atomic factual claims from this attachment relevant to the query.

Query: {query_text}
Document metadata:
- author: {metadata.get('author')}
- year: {metadata.get('year')}
- publication: {metadata.get('publication')}
- source_title: {metadata.get('source_title')}

Document content:
{text[:5000]}

Return ONLY JSON array with objects:
- claim
- quote_span
- confidence (0.0-1.0)

Extract 3-8 claims if possible.
"""

            response = await model.ainvoke([
                SystemMessage(content="You extract atomic factual claims with exact supporting quote spans."),
                HumanMessage(content=prompt),
            ])
            content = response.content.replace("```json", "").replace("```", "").strip()
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
        return []

    processed_items = []
    evidence_store = []
    evidence_id = evidence_id_start

    for index, attachment in enumerate(attachments, start=1):
        filename = f"attachment_{index}"
        file_type = "unknown"
        raw_bytes = b""
        provided_text = ""

        if isinstance(attachment, dict):
            filename = attachment.get("filename") or filename
            file_type = attachment.get("file_type") or "unknown"
            provided_text = attachment.get("content", "") if isinstance(attachment.get("content", ""), str) else ""
            raw_bytes = attachment.get("bytes", b"") if isinstance(attachment.get("bytes", b""), (bytes, bytearray)) else b""
        elif isinstance(attachment, str):
            provided_text = attachment

        docint_text = await extract_docint_text(bytes(raw_bytes)) if raw_bytes else ""
        final_text = docint_text if docint_text and len(docint_text.strip()) > 50 else provided_text

        fallback_title = filename if filename else f"attachment_{index}"
        metadata = await extract_metadata_with_llm(final_text, filename, fallback_title)
        claims = await extract_claims_with_llm(final_text, query, metadata)

        if not claims and final_text.strip():
            # Minimal fallback claim when extraction fails
            snippet = final_text.strip()[:400]
            claims = [{
                "claim": f"Attachment {index} contains content relevant to the query.",
                "quote_span": snippet,
                "confidence": 0.6,
            }]

        for claim_data in claims:
            evidence_store.append({
                "id": f"E{evidence_id}",
                "claim": claim_data.get("claim", ""),
                "source": fallback_title,
                "source_url": None,
                "source_title": metadata.get("source_title", fallback_title),
                "quote_span": claim_data.get("quote_span", ""),
                "author": metadata.get("author", "Unknown"),
                "year": metadata.get("year", "Unknown"),
                "publication": metadata.get("publication", "Unknown"),
                "publisher": "Unknown",
                "retrieval_context": "attachment_processing",
                "confidence": claim_data.get("confidence", 0.75),
                "timestamp_accessed": datetime.now().isoformat(),
            })
            evidence_id += 1

        processed_items.append({
            "name": filename,
            "file_type": file_type,
            "status": "success" if claims else "no_claims",
            "claims_extracted": len(claims),
            "used_document_intelligence": bool(docint_text),
            "metadata": metadata,
        })

        print(f"✓ Extracted {len(claims)} atomic claims from attachment: {filename}")

    return {
        "type": "attachments",
        "count": len(processed_items),
        "items": processed_items,
        "query": query,
        "evidence_store": evidence_store,
        "next_evidence_id": evidence_id,
    }


async def process_user_input(query: str, sources: Dict[str, Any], evidence_id_start: int = 1) -> Dict[str, Any]:
    """
    Main function to process user query and sources with atomic evidence tracking.
    
    Args:
        query: User's research query/question
        sources: JSON object containing topics, links, and attachments
            Example: {
                "topics": "machine learning transformers",
                "links": ["https://arxiv.org/abs/1706.03762"],
                "attachments": ["/path/to/file.pdf"]
            }
        evidence_id_start: Starting ID for evidence items (default: 1)
    
    Returns:
        Combined results from all processing with stable evidence IDs
    """
    # Split sources
    split_data = split_sources(sources)
    
    results = {
        "query": query,
        "timestamp": datetime.now().strftime("%B %d, %Y"),
        "topic_results": None,
        "link_results": None,
        "attachment_results": None,
        "evidence_store": [],
        "next_evidence_id": evidence_id_start
    }
    
    # Track evidence ID continuity across all sources
    next_evidence_id = evidence_id_start
    
    # Process topics through deep_research
    if split_data["topics"]:
        print(f"\n[Evidence IDs starting at E{next_evidence_id}]")
        results["topic_results"] = await topic_processing(
            split_data["topics"],
            query,
            evidence_id_start=next_evidence_id
        )
        # Add topic evidence to combined evidence store
        if results["topic_results"] and results["topic_results"].get("evidence_store"):
            results["evidence_store"].extend(results["topic_results"]["evidence_store"])
            next_evidence_id = results["topic_results"].get("next_evidence_id", next_evidence_id)
            print(f"[Topic processing: added {len(results['topic_results']['evidence_store'])} evidence items]")
    
    # Process links
    if split_data["links"]:
        print(f"\n[Evidence IDs starting at E{next_evidence_id}]")
        results["link_results"] = await process_links(
            split_data["links"],
            query,
            evidence_id_start=next_evidence_id
        )
        # Add link evidence to combined evidence store
        if results["link_results"] and results["link_results"].get("evidence_store"):
            results["evidence_store"].extend(results["link_results"]["evidence_store"])
            next_evidence_id = results["link_results"].get("next_evidence_id", next_evidence_id)
            print(f"[Link processing: added {len(results['link_results']['evidence_store'])} evidence items]")
    
    # Process attachments
    if split_data["attachments"]:
        print(f"\n[Evidence IDs starting at E{next_evidence_id}]")
        results["attachment_results"] = await process_attachments(
            split_data["attachments"],
            query,
            evidence_id_start=next_evidence_id
        )
        # Add attachment evidence to combined evidence store (future implementation)
        if results["attachment_results"] and results["attachment_results"].get("evidence_store"):
            results["evidence_store"].extend(results["attachment_results"]["evidence_store"])
            next_evidence_id = results["attachment_results"].get("next_evidence_id", next_evidence_id)
            print(f"[Attachment processing: added {len(results['attachment_results']['evidence_store'])} evidence items]")
    
    # Print evidence ID summary
    if results["evidence_store"]:
        print(f"\n[Total evidence collected: {len(results['evidence_store'])} items (E{evidence_id_start} through E{next_evidence_id-1})]")
    
    # Store the next evidence ID for continuation
    results["next_evidence_id"] = next_evidence_id
    
    # Generate comprehensive summary from all collected evidence
    if results["evidence_store"] and len(results["evidence_store"]) > 0:
        summary_result = await generate_summary_from_evidence(query, results["evidence_store"])
        results["generated_summary"] = summary_result
    else:
        results["generated_summary"] = {
            "success": False,
            "summary": "",
            "error": "No evidence collected to generate summary",
            "evidence_count": 0
        }
    
    return results


async def process_with_iterative_refinement(
    query: str,
    sources: Dict[str, Any],
    max_iterations: int = 3,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Process user query with iterative refinement - runs multiple iterations with critiques and adjustments.
    
    Args:
        query: User's research query/question
        sources: JSON object containing topics, links, and attachments
        max_iterations: Number of refinement iterations (default: 3)
    
    Returns:
        Dictionary containing all iterations with summaries, critiques, and adjustments
    """
    def emit(event_type: str, **payload):
        if progress_callback:
            try:
                progress_callback({"type": event_type, **payload})
            except Exception:
                pass

    iterations = []
    current_query = query
    cumulative_evidence = []
    next_evidence_id = 1  # Track evidence ID across all iterations
    
    for iteration in range(1, max_iterations + 1):
        emit("stage_text", stage=1, text=f"Iteration {iteration}/{max_iterations}")
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        # Process the current iteration with continuing evidence IDs
        iteration_results = await process_user_input(current_query, sources, evidence_id_start=next_evidence_id)
        
        # Update next_evidence_id for the next iteration
        next_evidence_id = iteration_results.get("next_evidence_id", next_evidence_id)
        
        # Add new evidence to cumulative store
        new_evidence = iteration_results.get("evidence_store", [])
        cumulative_evidence.extend(new_evidence)
        
        # Update iteration results with cumulative evidence
        iteration_results["cumulative_evidence_store"] = cumulative_evidence.copy()
        iteration_results["iteration"] = iteration
        iteration_results["query_used"] = current_query
        
        # Generate or regenerate summary with all cumulative evidence
        if cumulative_evidence:
            summary_result = await generate_summary_from_evidence(query, cumulative_evidence)
            iteration_results["generated_summary"] = summary_result
        
        iteration_data = {
            "iteration": iteration,
            "query": current_query,
            "results": iteration_results,
            "new_evidence_count": len(new_evidence),
            "cumulative_evidence_count": len(cumulative_evidence)
        }

        emit(
            "stage_metric",
            stage=1,
            key="Cumulative Evidence",
            value=len(cumulative_evidence),
        )
        
        # Only critique if not the last iteration
        if iteration < max_iterations:
            if iteration_results.get("generated_summary", {}).get("success"):
                print(f"\n[Iteration {iteration}] Generating critique...")
                
                critique_result = await critique_summary(
                    query,
                    iteration_results["generated_summary"]["summary"],
                    cumulative_evidence,
                    iteration
                )
                
                iteration_data["critique"] = critique_result
                
                if critique_result.get("success"):
                    print(f"[Iteration {iteration}] Generating adjustments for next iteration...")
                    
                    # Generate adjustments for next iteration
                    adjusted_query = await generate_adjustments_from_critique(
                        query,
                        critique_result["critique"]
                    )
                    
                    iteration_data["adjustments"] = {
                        "original_query": query,
                        "adjusted_query": adjusted_query
                    }
                    
                    # Use adjusted query for next iteration
                    current_query = adjusted_query
                    print(f"[Iteration {iteration}] Next iteration will use refined query")
                else:
                    print(f"[Iteration {iteration}] Critique failed, using original query")
                    iteration_data["adjustments"] = {
                        "original_query": query,
                        "adjusted_query": query
                    }
        else:
            print(f"\n[Iteration {iteration}] Final iteration - no critique needed")
            iteration_data["critique"] = None
            iteration_data["adjustments"] = None
        
        iterations.append(iteration_data)
    
    # Build final results
    final_results = {
        "original_query": query,
        "sources": sources,
        "total_iterations": max_iterations,
        "iterations": iterations,
        "final_summary": iterations[-1]["results"].get("generated_summary", {}),
        "final_evidence_count": len(cumulative_evidence),
        "cumulative_evidence_store": cumulative_evidence
    }
    
    return final_results


def get_random_style_from_db():
    """
    Get a random writing style from Cosmos DB for testing.
    Also fetches associated global rulebook and extracts style rules properly.
    
    Returns:
        Dictionary with style information including:
        - style_description: JSON string of style rules
        - global_rules: Text from global rulebook
        - example: Example text from evidence_spans or example field
        - Other metadata: speaker, audience, name, etc.
    """
    try:
        # Initialize Cosmos DB client
        cosmos_client = CosmosClient(
            url=os.getenv("AZURE_COSMOS_ENDPOINT"),
            credential=os.getenv("AZURE_COSMOS_KEY")
        )
        database = cosmos_client.get_database_client(os.getenv("AZURE_COSMOS_DATABASE"))
        styles_container = database.get_container_client("styles_from_speeches")
        
        # Query for style fingerprints
        query = """
        SELECT *
        FROM c
        WHERE c.doc_kind = 'style_fingerprint'
        """
        items = list(styles_container.query_items(
            query=query,
            parameters=[],
            enable_cross_partition_query=True
        ))
        
        if not items:
            print("No style fingerprint documents found in Cosmos DB")
            return None
        
        # Select a random style for testing
        import random
        selected_style = random.choice(items)
        
        # Extract style rules from the document
        style_rules = _extract_style_rules_from_doc(selected_style)
        selected_style["style_description"] = json.dumps(style_rules, ensure_ascii=False, indent=2) if style_rules else ""
        
        # Fetch associated global rulebook if available
        rulebook_id = selected_style.get("global_rulebook_id")
        if rulebook_id:
            try:
                rulebook_query = "SELECT * FROM c WHERE c.id = @id OR c.container_key = @id"
                rulebook_params = [{"name": "@id", "value": rulebook_id}]
                rulebook_items = list(styles_container.query_items(
                    query=rulebook_query,
                    parameters=rulebook_params,
                    enable_cross_partition_query=True
                ))
                if rulebook_items:
                    rulebook = rulebook_items[0]
                    selected_style["global_rules"] = _extract_rulebook_text(rulebook)
            except Exception as e:
                print(f"Error fetching global rulebook: {e}")
                selected_style["global_rules"] = ""
        
        # Extract example text
        example = selected_style.get("example")
        if not example:
            evidence = selected_style.get("evidence_spans") or []
            example = "\n".join([str(e) for e in evidence if e]) if evidence else ""
        selected_style["example"] = example
        
        return selected_style
        
    except Exception as e:
        print(f"Error getting style from Cosmos DB: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_style_rules_from_doc(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract style rules from a style_fingerprint document."""
    rules = item.get("style") or item.get("style_instructions")
    if rules and isinstance(rules, dict):
        return rules
    
    # Check nested properties
    props = item.get("properties") or {}
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}
    
    if isinstance(props, dict):
        nested = props.get("style_instructions") or props.get("style")
        if isinstance(nested, dict):
            return nested
    
    # Fallback: build from known top-level fields
    fallback_keys = [
        "register_profile",
        "stylistics",
        "pragmatics",
        "semantics_frames",
        "discourse_structure",
    ]
    fallback = {}
    for key in fallback_keys:
        if key in item and item[key]:
            fallback[key] = item[key]
    
    return fallback if fallback else {}


def _extract_rulebook_text(rulebook: Dict[str, Any]) -> str:
    """Extract readable text from a global_rulebook document."""
    if not rulebook:
        return ""
    
    for key in ["global_rules", "rules", "rulebook", "rulebook_text"]:
        if key in rulebook and rulebook[key]:
            return rulebook[key]
    
    try:
        return json.dumps(rulebook, ensure_ascii=False, indent=2)
    except Exception:
        return str(rulebook)


def get_sample_speeches_from_db(speaker_name: str, max_speeches: int = 3) -> List[str]:
    """
    Fetch sample speeches from the speeches database for the given speaker.
    
    Args:
        speaker_name: The name of the speaker (e.g., "Benjamin E. Diokno", "Eli M. Remolona Jr.")
        max_speeches: Maximum number of speeches to fetch (default: 3)
    
    Returns:
        List of speech content strings
    """
    try:
        # Initialize Cosmos DB client for speeches database
        cosmos_client = CosmosClient(
            url=os.getenv("AZURE_SPEECHES_ENDPOINT"),
            credential=os.getenv("AZURE_SPEECHES_KEY")
        )
        database = cosmos_client.get_database_client(os.getenv("AZURE_SPEECHES_DATABASE"))
        
        # Try common container names
        container_names = ["speeches", "Speeches", "speech_data", "documents"]
        container = None
        
        for name in container_names:
            try:
                container = database.get_container_client(name)
                # Test if container exists by trying to read properties
                container.read()
                break
            except:
                continue
        
        if not container:
            print(f"[WARNING] No speeches container found in database")
            return []
        
        # Query for speeches by speaker
        # Try different speaker field variations
        speaker_queries = [
            f"SELECT TOP {max_speeches} * FROM c WHERE CONTAINS(c.Speaker, @speaker) ORDER BY c._ts DESC",
            f"SELECT TOP {max_speeches} * FROM c WHERE CONTAINS(c.speaker, @speaker) ORDER BY c._ts DESC",
            f"SELECT TOP {max_speeches} * FROM c WHERE c.Speaker = @speaker ORDER BY c._ts DESC",
            f"SELECT TOP {max_speeches} * FROM c WHERE c.speaker = @speaker ORDER BY c._ts DESC"
        ]
        
        items = []
        for query in speaker_queries:
            try:
                items = list(container.query_items(
                    query=query,
                    parameters=[{"name": "@speaker", "value": speaker_name}],
                    enable_cross_partition_query=True
                ))
                if items:
                    break
            except Exception as e:
                continue
        
        if not items:
            print(f"[WARNING] No speeches found for speaker: {speaker_name}")
            return []
        
        # Extract speech content
        speeches = []
        for item in items[:max_speeches]:
            content = item.get("content", "") or item.get("text", "") or item.get("speech_text", "")
            if content and len(content) > 200:  # Only use substantial speeches
                # Clean up the content - find where actual speech starts
                # Look for common greeting patterns
                greeting_start_patterns = [
                    'magandang', 'good morning', 'good afternoon', 'good evening',
                    'ladies and gentlemen', 'thank you', 'it is'
                ]
                
                lines = content.split('\n')
                speech_start_idx = None
                
                for i, line in enumerate(lines):
                    line_lower = line.lower().strip()
                    if any(greeting in line_lower for greeting in greeting_start_patterns):
                        speech_start_idx = i
                        break
                
                if speech_start_idx is not None:
                    cleaned_content = '\n'.join(lines[speech_start_idx:]).strip()
                else:
                    # Fallback: skip obvious metadata lines
                    cleaned_lines = []
                    for line in lines:
                        line_lower = line.lower().strip()
                        if not any(x in line_lower for x in ['http', '.pdf', 'bangko sentral ng pilipinas media']):
                            cleaned_lines.append(line)
                    cleaned_content = '\n'.join(cleaned_lines).strip()
                
                if len(cleaned_content) > 200:
                    speeches.append(cleaned_content)
        
        print(f"[INFO] Fetched {len(speeches)} sample speech(es) for {speaker_name}")
        for i, speech in enumerate(speeches, 1):
            print(f"[INFO]   Speech {i}: {len(speech)} characters")
        
        return speeches
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch speeches: {e}")
        import traceback
        traceback.print_exc()
        return []


def enforce_citation_discipline(text: str, max_ids_per_sentence: int = 2) -> str:
    """
    IMPROVEMENT #3: Enforce citation discipline - max N evidence IDs per sentence.
    
    If a sentence has more than max_ids_per_sentence citations, keep only the first N.
    This prevents "citation spraying" where models cite many IDs hoping one fits.
    
    Args:
        text: Styled output text with citations
        max_ids_per_sentence: Maximum evidence IDs allowed per sentence (default: 2)
        
    Returns:
        Text with citation discipline enforced
    """
    # Split into sentences (preserve terminators)
    # Pattern splits on .!? followed by space, so we need to add them back
    parts = re.split(r'([.!?]+\s+)', text)
    
    corrected_parts = []
    violations_fixed = 0
    
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
            
        sentence = parts[i]
        terminator = parts[i+1] if i+1 < len(parts) else ''
        
        # Find all citations in this sentence
        citations = re.findall(r'\[E\d+(?:,E\d+)*\]', sentence)
        
        if not citations:
            corrected_parts.append(sentence + terminator)
            continue
        
        # Extract all unique evidence IDs from all citations in the sentence
        all_ids = []
        for citation in citations:
            ids = re.findall(r'E\d+', citation)
            all_ids.extend(ids)
        
        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(all_ids))
        
        if len(unique_ids) <= max_ids_per_sentence:
            # Already compliant
            corrected_parts.append(sentence + terminator)
        else:
            # Violation: too many IDs
            violations_fixed += 1
            
            # Keep only first max_ids_per_sentence IDs
            kept_ids = unique_ids[:max_ids_per_sentence]
            
            # Replace all citations in this sentence with a single citation
            corrected_sentence = sentence
            for citation in citations:
                corrected_sentence = corrected_sentence.replace(citation, '', 1)
            
            # Add the corrected citation (clean up extra spaces)
            new_citation = f"[{','.join(kept_ids)}]"
            corrected_sentence = corrected_sentence.strip() + " " + new_citation
            
            corrected_parts.append(corrected_sentence + terminator)
    
    if violations_fixed > 0:
        print(f"  ⚠️  Fixed {violations_fixed} citation discipline violations (reduced to ≤{max_ids_per_sentence} IDs/sentence)")
    else:
        print(f"  ✓ Citation discipline already maintained (≤{max_ids_per_sentence} IDs/sentence)")
    
    return ''.join(corrected_parts)


async def fix_speech_issues(
    speech_text: str,
    issues: List[Dict[str, Any]],
    evidence_store: List[Dict[str, Any]] = None,
    issue_type: str = "generic"
) -> Dict[str, Any]:
    """
    Use LLM to automatically fix flagged issues in speech segments.
    
    Args:
        speech_text: The full speech text
        issues: List of issues to fix, each with:
            - segment: The problematic text segment
            - issue_description: What's wrong
            - suggestion: Optional suggestion for fix
            - severity: CRITICAL/HIGH/MEDIUM/LOW
        evidence_store: List of evidence for citation context
        issue_type: Type of issues ("citation", "plagiarism", "policy", "generic")
    
    Returns:
        {
            "success": bool,
            "fixed_speech": str (revised text),
            "fixes_applied": int,
            "fix_details": List of what was changed
        }
    """
    if not issues:
        return {
            "success": True,
            "fixed_speech": speech_text,
            "fixes_applied": 0,
            "fix_details": []
        }
    
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        
        model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        # Build issue-specific system prompt
        system_prompts = {
            "citation": """Fix claims not fully supported by evidence. Soften exaggerated claims, add hedging ("may", "appears to"), remove unsupported specifics. Keep citations [ENN] intact. Maintain speaker's voice.""",
            
            "plagiarism": """Rewrite text with high similarity to sources. Restructure sentences, use synonyms, change word order. Keep citations [ENN] intact. Preserve factual accuracy.""",
            
            "policy": """Fix BSP policy violations. Add disclaimers for forward-looking statements, soften absolutes, add context. Use "BSP/we" not "I". Keep citations [ENN] intact. Maintain professional tone.""",
            
            "generic": """Fix flagged issues. Preserve citations [ENN], maintain voice and facts."""
        }
        
        system_prompt = system_prompts.get(issue_type, system_prompts["generic"])
        
        # Build evidence context if available
        evidence_context = ""
        if evidence_store and issue_type == "citation":
            evidence_context = "\n<EVIDENCE_STORE>\n"
            for ev in evidence_store[:10]:  # Reduced from 20 to 10 to save tokens
                # Truncate claim to 150 chars to reduce token usage
                evidence_context += f"[{ev.get('id')}]: {ev.get('claim', '')[:150]}\n"
            evidence_context += "</EVIDENCE_STORE>\n\n"
        
        # Build issues list
        issues_text = "\n<ISSUES_TO_FIX>\n"
        for i, issue in enumerate(issues[:5], 1):  # Reduced from 10 to 5 to save tokens
            issues_text += f"\nISSUE #{i}:\n"
            issues_text += f"Segment: \"{issue.get('segment', '')[:200]}\"\n"  # Reduced from 300
            issues_text += f"Problem: {issue.get('issue_description', 'Not specified')[:150]}\n"  # Added limit
            if issue.get('suggestion'):
                issues_text += f"Fix: {issue.get('suggestion')[:150]}\n"  # Reduced from no limit
            issues_text += f"Severity: {issue.get('severity', 'MEDIUM')}\n"
        issues_text += "</ISSUES_TO_FIX>\n"
        
        user_prompt = f"""{evidence_context}{issues_text}

<SPEECH_TO_REVISE>
{speech_text}
</SPEECH_TO_REVISE>

TASK: Revise the speech to fix the {len(issues)} issue(s) listed above. Return ONLY the full revised speech text (no explanations, no markup)."""
        
        print(f"[INFO] 🔧 Calling LLM to fix {len(issues)} {issue_type} issue(s)...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=8000  # Sufficient for speech revision (GPT-5 compatible)
        )
        
        fixed_speech = response.choices[0].message.content.strip()
        
        # Analyze what changed
        fix_details = []
        if fixed_speech != speech_text:
            # Simple diff: count sentence changes
            original_sentences = [s.strip() for s in re.split(r'[.!?]+', speech_text) if s.strip()]
            fixed_sentences = [s.strip() for s in re.split(r'[.!?]+', fixed_speech) if s.strip()]
            
            changes = sum(1 for orig, fixed in zip(original_sentences, fixed_sentences) if orig != fixed)
            fix_details.append({
                "sentences_modified": changes,
                "original_length": len(speech_text),
                "fixed_length": len(fixed_speech)
            })
            
            print(f"[SUCCESS] ✅ Fixed {changes} sentence(s), preserved {len(fixed_sentences) - changes} sentence(s)")
        else:
            print(f"[INFO] No changes needed")
        
        return {
            "success": True,
            "fixed_speech": fixed_speech,
            "fixes_applied": len(issues),
            "fix_details": fix_details
        }
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to fix issues: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "fixed_speech": speech_text,  # Return original on failure
            "fixes_applied": 0,
            "fix_details": [],
            "error": str(e)
        }


async def generate_styled_output(
    summary: str, 
    query: str, 
    style: Dict[str, Any], 
    context_details: str = "",
    evidence_store: List[Dict[str, Any]] = None, 
    max_output_length: int = 16000,
    use_claim_outline: bool = True  # IMPROVEMENT #1: Use claim outlines by default
) -> Dict[str, Any]:
    """
    Generate final styled output with STRICT citation enforcement.
    Follows the same pattern as prompts.rewrite_content() but enforces [ENN] citations.
    
    Args:
        summary: The comprehensive summary from iterative refinement
        query: Original user query
        style: Writing style dictionary from Cosmos DB (can be None for default formatting)
        context_details: Optional event/setting/partner context provided by user
        evidence_store: List of atomic evidence items for citation validation
        max_output_length: Maximum completion tokens for output (default: 16000 for GPT-5 reasoning)
        use_claim_outline: If True, generate claim outline first to prevent style drift
    
    Returns:
        Dictionary with styled output, validation metrics, and metadata
    """
    # Allow None style - will use default formatting
    if style is None:
        style = {}
        print("[INFO] No style provided, using default formatting")
    
    try:
        import re
        
        # Initialize Azure OpenAI client (GPT-5 or configured model)
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        
        model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        # Extract style components (matching existing app structure)
        def normalize_prompt_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                try:
                    return json.dumps(value, ensure_ascii=False, indent=2)
                except Exception:
                    return str(value)
            return str(value)

        style_text_value = (
            style.get("style_description")
            or style.get("style")
            or style.get("style_instructions")
            or ((style.get("properties") or {}).get("style_instructions") if isinstance(style.get("properties"), dict) else "")
        )
        global_rules_value = (
            style.get("global_rules")
            or style.get("global_rulebook")
            or style.get("rulebook")
        )
        guidelines_value = style.get("guidelines") or ""
        example_value = style.get("example")
        if not example_value:
            evidence_spans = style.get("evidence_spans")
            if isinstance(evidence_spans, list) and evidence_spans:
                example_value = "\n".join(str(item) for item in evidence_spans if item)

        style_text = normalize_prompt_text(style_text_value)
        global_rules = normalize_prompt_text(global_rules_value)
        guidelines = normalize_prompt_text(guidelines_value)
        example = normalize_prompt_text(example_value)
        
        # Build allowed evidence IDs list
        allowed_ids = []
        if evidence_store:
            allowed_ids = [e.get("id") for e in evidence_store if e.get("id")]
        allowed_ids_str = ", ".join(allowed_ids) if allowed_ids else "None available"
        
        # IMPROVEMENT #1: Generate claim outline to prevent style drift
        claim_outline_data = None
        claim_count = 0
        claim_outline = None
        
        print(f"[DEBUG] Claim outline parameters: use_claim_outline={use_claim_outline}, evidence_store_size={len(evidence_store) if evidence_store else 0}")
        print(f"[DEBUG] Summary length for extraction: {len(summary)} chars")
        
        if use_claim_outline and evidence_store:
            try:
                print("[INFO] ✨ Generating claim outline to prevent style drift...")
                print(f"[INFO] Evidence store has {len(evidence_store)} items")
                
                # Build evidence map
                evidence_map = {e['id']: e for e in evidence_store if 'id' in e}
                
                # Create extraction prompt
                extraction_prompt = f"""Extract atomic factual claims from this research summary. Each claim should be:

1. **One clear factual statement** (not multiple facts combined)
2. **Tied to 1-2 evidence IDs** (from the cited [ENN] references)
3. **Classified by type**:
   - "factual": Concrete fact, statistic, or research finding  
   - "interpretation": Analysis or implication drawn from facts
   - "transition": Logical connection (no citations needed)

<SUMMARY>
{summary[:3000]}  
</SUMMARY>

<TASK>
Return a JSON array where each item has:
- "claim_text": the atomic factual statement (one sentence)
- "evidence_ids": array of 1-2 evidence IDs that support this claim (e.g., ["E1", "E5"])
- "claim_type": "factual", "interpretation", or "transition"
- "confidence": your confidence (0.0-1.0) that evidence supports this claim

IMPORTANT RULES:
- One fact per claim (don't combine multiple statistics)
- Maximum 2 evidence IDs per claim
- Only use evidence IDs that actually appear in the summary

Return ONLY the JSON array (no markdown, no explanation).
</TASK>

JSON:"""
                
                # Call LLM to extract claims (GPT-5 needs more tokens for reasoning)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise claim extraction system. Return ONLY valid JSON array, no other text."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    max_completion_tokens=8000  # GPT-5 uses lots of reasoning tokens, needs higher limit (default temp=1 only)
                )
                
                # Check finish reason first
                finish_reason = response.choices[0].finish_reason
                print(f"[DEBUG] Finish reason: {finish_reason}")
                
                # Check token usage
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    print(f"[DEBUG] Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}")
                    if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                        details = usage.completion_tokens_details
                        if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                            print(f"[DEBUG] Reasoning tokens: {details.reasoning_tokens}, Output tokens: {usage.completion_tokens - details.reasoning_tokens}")
                
                raw_content = response.choices[0].message.content
                print(f"[DEBUG] Raw LLM response: {raw_content[:500] if raw_content else 'NONE'}")
                
                if not raw_content:
                    error_msg = f"LLM returned empty response (finish_reason: {finish_reason})"
                    if finish_reason == 'length':
                        error_msg += " - Hit token limit, increase max_completion_tokens"
                    raise ValueError(error_msg)
                
                content = raw_content.strip()
                print(f"[DEBUG] After strip: {len(content)} chars")
                
                # IMPROVEMENT #1: Better JSON extraction handling
                # Strip thinking tokens (both <think> and <thinking>)
                if '<think>' in content or '<thinking>' in content:
                    if '</think>' in content:
                        content = content.split('</think>')[-1].strip()
                    elif '</thinking>' in content:
                        content = content.split('</thinking>')[-1].strip()
                    print(f"[DEBUG] After thinking removal: {len(content)} chars")
                
                # Strip markdown code blocks
                content = content.replace('```json', '').replace('```', '').strip()
                print(f"[DEBUG] After markdown removal: {len(content)} chars")
                
                # Remove any leading/trailing text before/after JSON
                # Extract JSON array (find first [ to last ])
                start_idx = content.find('[')
                end_idx = content.rfind(']')
                print(f"[DEBUG] JSON array search: start={start_idx}, end={end_idx}")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    content = content[start_idx:end_idx+1]
                    print(f"[DEBUG] Extracted array: {len(content)} chars, starts with: {content[:100]}")
                else:
                    # Try to find JSON object if no array
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    print(f"[DEBUG] JSON object search: start={start_idx}, end={end_idx}")
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        # Wrap single object in array
                        content = '[' + content[start_idx:end_idx+1] + ']'
                        print(f"[DEBUG] Wrapped single object in array: {len(content)} chars")
                    else:
                        raise ValueError(f"No JSON found. Content length: {len(content)}, Raw: {raw_content[:500]}")
                
                if len(content) < 5:
                    raise ValueError(f"Extracted JSON too short ({len(content)} chars): '{content}'")
                
                print(f"[DEBUG] About to parse JSON: {content[:200]}")
                claims_data = json.loads(content)
                
                if not isinstance(claims_data, list):
                    raise ValueError(f"Expected JSON array, got {type(claims_data).__name__}")
                
                # Filter to valid claims with evidence
                claim_outline_data = []
                for i, claim in enumerate(claims_data[:15], 1):  # Max 15 claims
                    evidence_ids = claim.get('evidence_ids', [])
                    valid_ids = [eid for eid in evidence_ids if eid in evidence_map]
                    if len(valid_ids) > 2:
                        valid_ids = valid_ids[:2]  # Max 2 IDs per claim
                    
                    if valid_ids or claim.get('claim_type') == 'transition':
                        claim_outline_data.append({
                            'claim_id': f"C{i}",
                            'claim_text': claim.get('claim_text', ''),
                            'evidence_ids': valid_ids,
                            'claim_type': claim.get('claim_type', 'factual')
                        })
                
                claim_count = len(claim_outline_data)
                print(f"[SUCCESS] ✅ Extracted {claim_count} claims from summary")
                print(f"[SUCCESS] ✅ Claim outline will be used for styling")
                
                # Create prompt text from claims
                claim_prompt_lines = ["CLAIM OUTLINE (Rephrase ONLY these claims):", ""]
                for claim in claim_outline_data:
                    evidence_str = ", ".join(claim['evidence_ids'])
                    claim_prompt_lines.append(f"{claim['claim_id']}. {claim['claim_text']}")
                    if claim['evidence_ids']:
                        claim_prompt_lines.append(f"   Evidence: [{evidence_str}]")
                    claim_prompt_lines.append(f"   Type: {claim['claim_type']}")
                    claim_prompt_lines.append("")
                
                claim_outline = "\n".join(claim_prompt_lines)
                
            except Exception as e:
                print(f"[ERROR] ❌ Claim outline generation failed: {e}")
                import traceback
                print(f"[ERROR] Traceback:")
                traceback.print_exc()
                print(f"[INFO] Falling back to freeform prose styling")
                claim_outline = None
                claim_outline_data = None
        else:
            if not use_claim_outline:
                print(f"[INFO] Claim outline disabled (use_claim_outline=False)")
            elif not evidence_store:
                print(f"[INFO] No evidence store available for claim outline generation")
        
        # Build the system prompt with STRICT citation rules and claim outline
        system_parts = [
            "You are an expert writer assistant. Rewrite the user input based on the following writing style, global rules, writing guidelines and writing example.\n",
            f"<writingStyle>{style_text}</writingStyle>\n",
            f"<globalRules>{global_rules}</globalRules>\n",
            f"<writingGuidelines>{guidelines}</writingGuidelines>\n",
        ]
        
        # Add original example if available
        if example:
            system_parts.append(f"<writingExample>{example}</writingExample>\n")
        
        # ENHANCEMENT: Fetch real speech examples from speeches database
        speaker_name = style.get("speaker") or style.get("Speaker")
        if speaker_name:
            print(f"[INFO] Fetching sample speeches for speaker: {speaker_name}")
            sample_speeches = get_sample_speeches_from_db(speaker_name, max_speeches=3)
            
            if sample_speeches:
                system_parts.append("\n<realSpeechExamples>\n")
                system_parts.append(f"Below are {len(sample_speeches)} actual speeches by {speaker_name}. Study their:\n")
                system_parts.append("- Sentence structure and rhythm\n")
                system_parts.append("- Vocabulary choices and phrasing\n")
                system_parts.append("- Opening/closing patterns\n")
                system_parts.append("- Transition phrases\n")
                system_parts.append("- Rhetorical devices\n\n")
                
                for i, speech in enumerate(sample_speeches, 1):
                    # Truncate very long speeches to fit in context
                    speech_excerpt = speech[:2000] if len(speech) > 2000 else speech
                    system_parts.append(f"=== SPEECH EXAMPLE {i} ===\n")
                    system_parts.append(f"{speech_excerpt}\n")
                    if len(speech) > 2000:
                        system_parts.append("... (excerpt from longer speech)\n")
                    system_parts.append("\n")
                
                system_parts.append("</realSpeechExamples>\n\n")
                print(f"[SUCCESS] ✅ Added {len(sample_speeches)} real speech examples to style prompt")
            else:
                print(f"[WARNING] No speech examples found for {speaker_name}, using style description only")
        
        system_parts.append("Make sure to emulate the writing style, global rules, guidelines and examples provided above.\n\n")
        
        # Add claim outline instructions if available
        if claim_outline and claim_outline_data:
            system_parts.extend([
                "CLAIM-BASED REWRITING MODE:\n",
                "You are given a CLAIM OUTLINE below. You must:\n",
                "1. Rephrase ONLY the claims provided (do not add new facts or methods)\n",
                "2. Preserve all evidence citations exactly as shown\n",
                "3. You may adjust wording for flow, but NOT add new factual content\n",
                "4. Transitions must be non-factual (e.g., 'This suggests...', 'A key implication is...')\n",
                "5. Do NOT introduce named methods, numbers, or entities not in the claim outline\n\n",
                claim_outline + "\n\n"
            ])
        
        system_parts.extend([
            "CRITICAL CITATION REQUIREMENTS:\n",
            "1. EVERY factual claim MUST end with [ENN] citation format (e.g., [E1] or [E2,E5])\n",
            "2. You may ONLY cite these evidence IDs: " + allowed_ids_str + "\n",
            "3. Do NOT generate citations to non-existent evidence IDs\n",
            "4. Maintain ALL citations from the original summary\n",
            "5. If you rephrase a sentence, keep its citation intact\n",
            "6. Multiple claims in one sentence = multiple citations [E1,E3,E7]\n",
            "7. End with a short closing reflection (1-3 sentences) on why this topic matters to BSP, financial institutions, and the nation as a whole. Keep this aligned with available evidence and avoid unsupported claims.\n\n",
            f"YOU CAN ONLY OUTPUT A MAXIMUM OF {max_output_length} CHARACTERS"
        ])
        
        system_prompt = "".join(system_parts)
        
        # User prompt with citation examples
        user_prompt = f"""Original Query: {query}

    Setting / Location / Conference / Partners (optional context):
    {context_details.strip() if context_details and context_details.strip() else "Not provided"}

Research Summary to Rewrite:
{summary}

STRICT REQUIREMENTS:
- Rewrite in the specified style while preserving ALL factual content
- EVERY factual sentence MUST have [ENN] citations
- Only use valid evidence IDs: {allowed_ids_str}
- Do NOT remove, modify, or invent citations
    - Use the optional setting/location/partners context naturally when relevant
    - Include a short closing reflection on importance to BSP, financial institutions, and the nation as a whole

Example correct format:
"Transformers use attention mechanisms [E1]. They achieve better performance than RNNs [E3,E7]."

Now rewrite the summary in the specified style with ALL citations preserved:"""

        # Keep completion token budget separate from output character cap.
        # max_output_length limits final characters (enforced after generation),
        # while completion tokens must leave room for GPT-5 reasoning + visible output.
        completion_token_budget = max(8000, min(16000, max_output_length * 8))

        # Log prompt lengths for debugging
        print(f"[DEBUG] System prompt length: {len(system_prompt)} chars")
        print(f"[DEBUG] User prompt length: {len(user_prompt)} chars")
        print(f"[DEBUG] Summary length: {len(summary)} chars")
        print(f"[DEBUG] Max completion tokens: {completion_token_budget}")

        # Generate styled output (GPT-5 needs more tokens due to reasoning)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=completion_token_budget
            )
            print(f"[DEBUG] API call successful, response choices: {len(response.choices)}")
        except Exception as api_error:
            print(f"[DEBUG] API call failed: {str(api_error)}")
            return {
                "success": False,
                "error": f"API call failed: {str(api_error)}",
                "styled_output": summary,
                "style_name": style.get("name", "Unknown"),
                "speaker": style.get("speaker", "Unknown"),
                "audience": style.get("audience_setting_classification", "General"),
                "model_used": model
            }
        
        # Extract and truncate styled output (matching existing app behavior)
        styled_output = response.choices[0].message.content
        print(f"[DEBUG] Response content length: {len(styled_output) if styled_output else 0} chars")
        print(f"[DEBUG] Response content is None: {styled_output is None}")
        print(f"[DEBUG] Response finish_reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
        
        # Additional GPT-5 debugging
        if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
            print(f"[DEBUG] ⚠️ Response refusal: {response.choices[0].message.refusal}")
        
        # Show token usage for GPT-5 reasoning models
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            print(f"[DEBUG] Token usage:")
            print(f"[DEBUG]   Prompt tokens: {usage.prompt_tokens}")
            print(f"[DEBUG]   Completion tokens: {usage.completion_tokens}")
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                    print(f"[DEBUG]   Reasoning tokens: {details.reasoning_tokens}")
                    print(f"[DEBUG]   Output tokens: {usage.completion_tokens - details.reasoning_tokens}")
        
        if styled_output:
            print(f"[DEBUG] ✓ Content received, first 100 chars: {styled_output[:100]}")
        else:
            print(f"[DEBUG] ✗ Content is empty or None - investigating...")
            print(f"[DEBUG]   Response ID: {response.id if hasattr(response, 'id') else 'N/A'}")
            print(f"[DEBUG]   Response model: {response.model if hasattr(response, 'model') else 'N/A'}")
            print(f"[DEBUG]   Finish reason: {response.choices[0].finish_reason}")
            if response.choices[0].finish_reason == 'length':
                print(f"[DEBUG]   ⚠️ ISSUE: Model hit token limit! Increase max_completion_tokens.")
        
        # Handle None or empty response
        if not styled_output or len(styled_output.strip()) == 0:
            return {
                "success": False,
                "error": "GPT model returned empty response",
                "styled_output": summary,  # Fallback to original summary
                "style_name": style.get("name", "Unknown"),
                "speaker": style.get("speaker", "Unknown"),
                "audience": style.get("audience_setting_classification", "General"),
                "model_used": model
            }
        
        # Truncate if needed (boundary-aware to avoid partial sentence/citation artifacts)
        if len(styled_output) > max_output_length:
            raw_truncated = styled_output[:max_output_length]

            # If citation is cut mid-way, drop from the unmatched '[' onward
            if raw_truncated.count('[') > raw_truncated.count(']'):
                last_open_bracket = raw_truncated.rfind('[')
                if last_open_bracket > 0:
                    raw_truncated = raw_truncated[:last_open_bracket].rstrip()

            # Prefer ending on a natural sentence/citation boundary
            candidate_boundaries = [
                raw_truncated.rfind("]. "),
                raw_truncated.rfind("].\n"),
                raw_truncated.rfind(". "),
                raw_truncated.rfind("? "),
                raw_truncated.rfind("! "),
                raw_truncated.rfind("]\n"),
            ]
            best_boundary = max(candidate_boundaries)

            if best_boundary >= int(max_output_length * 0.6):
                styled_output = raw_truncated[:best_boundary + 1].rstrip()
            else:
                styled_output = raw_truncated.rstrip()

            print(f"[INFO] Output truncated to max length with boundary-safe trim ({len(styled_output)} chars)")
        
        # IMPROVEMENT #5: Enforce citation discipline (max 2 IDs per sentence)
        print("[INFO] Enforcing citation discipline (max 2 IDs/sentence)...")
        styled_output = enforce_citation_discipline(styled_output, max_ids_per_sentence=2)
        
        # Validate citations in styled output
        citation_pattern = r'\[E\d+(?:,E\d+)*\]'
        citations_found = re.findall(citation_pattern, styled_output)
        
        # Extract individual evidence IDs from citations
        cited_ids = set()
        invalid_citations = []
        for citation in citations_found:
            ids_in_citation = re.findall(r'E\d+', citation)
            for eid in ids_in_citation:
                cited_ids.add(eid)
                if allowed_ids and eid not in allowed_ids:
                    invalid_citations.append(eid)
        
        # Calculate coverage
        cited_count = len(cited_ids & set(allowed_ids)) if allowed_ids else len(cited_ids)
        total_evidence = len(allowed_ids) if allowed_ids else 0
        coverage = (cited_count / total_evidence * 100) if total_evidence > 0 else 0
        
        return {
            "success": True,
            "styled_output": styled_output,
            "style_name": style.get("name", "Unknown"),
            "speaker": style.get("speaker", "Unknown"),
            "audience": style.get("audience_setting_classification", "General"),
            "model_used": model,
            "output_length": len(styled_output) if styled_output else 0,
            "max_length": max_output_length,
            "citations_found": len(citations_found),
            "unique_evidence_cited": cited_count,
            "citation_coverage": f"{coverage:.1f}%",
            "invalid_citations": invalid_citations,
            # IMPROVEMENT #1: Track claim outline usage
            "claim_outline_used": claim_outline_data is not None,
            "claim_count": claim_count,
            "validation": {
                "all_citations_valid": len(invalid_citations) == 0,
                "cited_ids": sorted(list(cited_ids)),
                "uncited_ids": sorted(list(set(allowed_ids) - cited_ids)) if allowed_ids else []
            }
        }
        
        # Debug: Confirm claim outline status being returned
        print(f"[DEBUG] Returning styled_result with claim_outline_used={claim_outline_data is not None}, claim_count={claim_count}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Style generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "styled_output": summary,  # Fallback to unstyled summary
            "claim_outline_used": False,
            "claim_count": 0
        }


async def convert_styled_output_to_apa(
    styled_output: str,
    evidence_store: List[Dict[str, Any]],
    azure_llm_client = None
) -> Dict[str, Any]:
    """
    Convert styled output with [ENN] citations to APA format with (Author, Year) citations
    and add a References section following APA 7th edition guidelines.
    
    Args:
        styled_output: The styled text with [ENN] citations
        evidence_store: List of evidence dicts with id, source_url, claim, etc.
        azure_llm_client: Optional Azure OpenAI client
        
    Returns:
        {
            "success": bool,
            "apa_output": str with (Author, Year) citations,
            "references": list of APA-formatted references,
            "citation_map": dict mapping evidence IDs to (Author, Year)
        }
    """
    
    print("\n" + "="*70)
    print("CONVERTING TO APA FORMAT (7th Edition)")
    print("="*70)
    
    # Build evidence lookup
    evidence_dict = {}
    for evidence in evidence_store:
        eid = evidence.get("id")
        if eid:
            evidence_dict[eid] = evidence
    
    print(f"Evidence store: {len(evidence_dict)} items")
    
    # Step 1: Build citation map and group by unique source
    print("\nBuilding APA citations from evidence metadata...")
    
    citation_map = {}
    source_groups = {}  # Group evidence by unique source (author, year, URL)
    
    for eid, evidence in evidence_dict.items():
        author = evidence.get("author", "")
        year = evidence.get("year", "n.d.")
        source_url = evidence.get("source_url") or evidence.get("source") or ""
        publication = evidence.get("publication", "")
        publisher = evidence.get("publisher", "")
        source_title = evidence.get("source_title", "")
        
        # Determine display author for citations
        display_author = author
        if not display_author or display_author == "n.d.":
            if publisher and publisher != "n.d.":
                display_author = publisher
            elif publication:
                display_author = publication
            else:
                display_author = "Unknown"
        
        # For multiple authors, use "et al." format if already present, otherwise use first author
        if display_author and ", " in display_author:
            authors_list = display_author.split(", ")
            if len(authors_list) > 2:
                # Extract first author's last name for et al. format
                first_author = authors_list[0]
                if " " in first_author:
                    # "Ashish Vaswani" -> "Vaswani"
                    last_name = first_author.split()[-1]
                else:
                    last_name = first_author
                display_author_short = f"{last_name} et al."
            else:
                display_author_short = display_author
        else:
            display_author_short = display_author
        
        citation_map[eid] = {
            "author": display_author,
            "author_short": display_author_short,
            "year": year,
            "source_url": source_url,
            "publication": publication,
            "publisher": publisher,
            "source_title": source_title
        }
        
        # Group by unique source for deduplication
        source_key = (display_author, year, source_url if source_url else publication)
        if source_key not in source_groups:
            source_groups[source_key] = {
                "author": display_author,
                "year": year,
                "source_url": source_url,
                "publication": publication,
                "publisher": publisher,
                "source_title": source_title,
                "evidence_ids": []
            }
        source_groups[source_key]["evidence_ids"].append(eid)
    
    print(f"Built citation map for {len(citation_map)} evidence items")
    print(f"Unique sources: {len(source_groups)}")
    
    # Step 2: Replace [ENN] with (Author, Year) in the text
    print("\nConverting [ENN] citations to (Author, Year) format...")
    
    apa_output = styled_output
    
    # Find all [ENN] or [E1,E2] citations
    citation_pattern = r'\[E\d+(?:,E\d+)*\]'
    matches = list(re.finditer(citation_pattern, apa_output))
    
    # Replace from end to start to preserve positions
    for match in reversed(matches):
        citation_str = match.group(0)
        citation_ids = re.findall(r'E\d+', citation_str)
        
        # Convert to (Author, Year) format, removing duplicates
        seen_citations = set()
        apa_citations = []
        for cid in citation_ids:
            if cid in citation_map:
                info = citation_map[cid]
                citation_key = (info['author_short'], info['year'])
                if citation_key not in seen_citations:
                    apa_citations.append(f"{info['author_short']}, {info['year']}")
                    seen_citations.add(citation_key)
        
        if apa_citations:
            # Format as (Author1, Year1; Author2, Year2)
            apa_format = f"({'; '.join(apa_citations)})"
            apa_output = apa_output[:match.start()] + apa_format + apa_output[match.end():]
    
    # Step 3: Generate References section with proper APA 7th edition format
    print("\nGenerating APA 7th edition references list...")
    
    # Get unique citations used in the text
    all_citations_in_text = re.findall(citation_pattern, styled_output)
    cited_ids = set()
    for citation in all_citations_in_text:
        ids = re.findall(r'E\d+', citation)
        cited_ids.update(ids)
    
    # Find which unique sources were actually cited
    cited_sources = set()
    for eid in cited_ids:
        if eid in citation_map:
            info = citation_map[eid]
            source_key = (info['author'], info['year'], info['source_url'] if info['source_url'] else info['publication'])
            cited_sources.add(source_key)
    
    # Build references list (one entry per unique source)
    references = []
    for source_key in sorted(cited_sources, key=lambda x: (x[0], x[1])):
        source_info = source_groups[source_key]
        
        author = source_info['author']
        year = source_info['year']
        source_url = source_info['source_url']
        publication = source_info['publication']
        source_title = source_info['source_title']
        
        # Format author for reference list (APA 7th edition)
        if author and ", " in author and "et al." not in author:
            # Convert "First Last, First Last" to "Last, F., & Last, F."
            authors_list = author.split(", ")
            if len(authors_list) <= 20:  # APA shows up to 20 authors
                formatted_authors = []
                for auth in authors_list:
                    parts = auth.strip().split()
                    if len(parts) >= 2:
                        last_name = parts[-1]
                        initials = ". ".join([p[0] for p in parts[:-1]]) + "."
                        formatted_authors.append(f"{last_name}, {initials}")
                    else:
                        formatted_authors.append(auth.strip())
                
                if len(formatted_authors) > 1:
                    author_display = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
                else:
                    author_display = formatted_authors[0]
            else:
                author_display = author  # Keep original if too many
        else:
            author_display = author
        
        # Extract title from source_title or use publication
        title = ""
        if source_title:
            # Clean up title (remove site name if present)
            title = source_title.replace(" - arXiv", "").replace("[1706.03762] ", "")
            # Remove trailing site names
            for site in [" - Medium", " - YouTube", " - ResearchGate"]:
                if title.endswith(site):
                    title = title[:-len(site)]
        elif publication:
            title = publication
        
        # Generate APA 7th edition reference entry
        if source_url:
            # Online source with URL
            if "arxiv.org" in source_url:
                # arXiv format: Author. (Year). Title. arXiv. URL
                apa_ref = f"{author_display} ({year}). {title}. arXiv. {source_url}"
            else:
                # General web source: Author. (Year). Title. Site Name. URL
                site_name = publication if publication else "Website"
                if year and year != "n.d.":
                    apa_ref = f"{author_display} ({year}). {title}. {site_name}. {source_url}"
                else:
                    # No date format
                    apa_ref = f"{author_display} (n.d.). {title}. {site_name}. Retrieved from {source_url}"
        else:
            # No URL available
            if year and year != "n.d.":
                apa_ref = f"{author_display} ({year}). {title}. {publication if publication else 'Unknown source'}."
            else:
                apa_ref = f"{author_display} (n.d.). {title}. {publication if publication else 'Unknown source'}."
        
        references.append(apa_ref)
    
    # Add References section
    apa_output += "\n\n" + "="*70 + "\n"
    apa_output += "REFERENCES\n"
    apa_output += "="*70 + "\n\n"
    
    for ref in references:
        apa_output += f"{ref}\n\n"
    
    print(f"✓ APA conversion complete")
    print(f"  Converted {len(matches)} citation instances")
    print(f"  Generated {len(references)} unique reference entries (removed duplicates)")
    
    return {
        "success": True,
        "apa_output": apa_output,
        "references": references,
        "citation_map": citation_map,
        "citations_converted": len(matches),
        "references_generated": len(references)
    }


async def verify_styled_citations(
    styled_output: str, 
    evidence_store: List[Dict[str, Any]],
    azure_llm_client = None,
    sentence_level: bool = True  # IMPROVEMENT #4: Enable sentence-level verification by default
) -> Dict[str, Any]:
    """
    Parse styled output by citations and verify each claim matches its cited evidence.
    
    Args:
        styled_output: The styled text with citations like [E1] or [E1,E2]
        evidence_store: List of evidence dicts with id, claim, quote_span, etc.
        azure_llm_client: Optional Azure OpenAI client for verification
        sentence_level: If True, verify at sentence level (more precise). If False, use segment level.
        
    Returns:
        {
            "segments": [
                {
                    "segment_number": 1,
                    "text": "...",
                    "citations": ["E1", "E2"],
                    "cited_claims": ["claim text...", "claim text..."],
                    "verified": "Yes" | "No",
                    "verification_reason": "...",
                    "is_sentence_level": True|False
                }
            ],
            "total_segments": 10,
            "verified_segments": 8,
            "unverified_segments": 2,
            "verification_rate": "80.0%",
            "is_sentence_level": True|False
        }
    """
    
    print("\n" + "="*70)
    print(f"VERIFYING STYLED OUTPUT CITATIONS ({'SENTENCE-LEVEL' if sentence_level else 'SEGMENT-LEVEL'})")
    print("="*70)
    
    # Build evidence lookup dict
    evidence_dict = {}
    for evidence in evidence_store:
        eid = evidence.get("id")
        if eid:
            evidence_dict[eid] = evidence
    
    print(f"Evidence store: {len(evidence_dict)} items")
    
    # IMPROVEMENT #4: Split text into segments
    citation_pattern = r'\[E\d+(?:,E\d+)*\]'
    
    segments = []
    segment_number = 0
    
    if sentence_level:
        # Sentence-level verification: Split into sentences first, then check each
        sentences = re.split(r'(?<=[.!?])\s+', styled_output)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if this sentence has citations
            citations_in_sentence = re.findall(citation_pattern, sentence)
            if not citations_in_sentence:
                # Skip sentences without citations (non-factual content)
                continue
            
            segment_number += 1
            
            # Extract all citation IDs from this sentence
            citation_ids = []
            for citation_match in citations_in_sentence:
                citation_ids.extend(re.findall(r'E\d+', citation_match))
            
            # Remove duplicates while preserving order
            citation_ids = list(dict.fromkeys(citation_ids))
            
            # Look up cited claims
            cited_claims = []
            missing_ids = []
            for cid in citation_ids:
                if cid in evidence_dict:
                    cited_claims.append({
                        "id": cid,
                        "claim": evidence_dict[cid].get("claim", ""),
                        "quote_span": evidence_dict[cid].get("quote_span", "")
                    })
                else:
                    missing_ids.append(cid)
            
            segments.append({
                "segment_number": segment_number,
                "text": sentence,
                "citations": citation_ids,
                "cited_claims": cited_claims,
                "missing_evidence_ids": missing_ids,
                "verified": None,
                "verification_reason": None,
                "is_sentence_level": True
            })
    else:
        # Original segment-level verification: Split by citation patterns
        current_pos = 0
        
        # Find all citations and their positions
        for match in re.finditer(citation_pattern, styled_output):
            segment_number += 1
            
            # Extract text from last position to end of this citation
            segment_end = match.end()
            segment_text = styled_output[current_pos:segment_end].strip()
            
            # Extract citation IDs
            citation_str = match.group(0)
            citation_ids = re.findall(r'E\d+', citation_str)
            
            # Look up cited claims
            cited_claims = []
            missing_ids = []
            for cid in citation_ids:
                if cid in evidence_dict:
                    cited_claims.append({
                        "id": cid,
                        "claim": evidence_dict[cid].get("claim", ""),
                        "quote_span": evidence_dict[cid].get("quote_span", "")
                    })
                else:
                    missing_ids.append(cid)
            
            segments.append({
                "segment_number": segment_number,
                "text": segment_text,
                "citations": citation_ids,
                "cited_claims": cited_claims,
                "missing_evidence_ids": missing_ids,
                "verified": None,
                "verification_reason": None,
                "is_sentence_level": False
            })
            
            current_pos = segment_end
    
    print(f"Parsed {len(segments)} {'sentences' if sentence_level else 'segments'} with citations")
    
    # Initialize Azure client if not provided
    if not azure_llm_client:
        azure_llm_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
    
    # Use gpt-4o for verification (more reliable than gpt-5 for this task)
    model = "gpt-4o"  # Hardcoded for reliability
    
    # Verify each segment with LLM
    verified_count = 0
    unverified_count = 0
    
    print(f"\nVerifying segments using {model}...")
    
    for i, segment in enumerate(segments, 1):
        # Skip if missing evidence
        if segment["missing_evidence_ids"]:
            segment["verified"] = "No"
            segment["verification_reason"] = f"Missing evidence IDs: {', '.join(segment['missing_evidence_ids'])}"
            unverified_count += 1
            continue
        
        # Build verification prompt
        claims_text = "\n".join([
            f"- [{c['id']}] {c['claim']}" 
            for c in segment["cited_claims"]
        ])
        
        verification_prompt = f"""You are a precise fact-checking AI. Your task is to verify if the OUTPUT TEXT accurately reflects the CITED EVIDENCE.

OUTPUT TEXT:
"{segment['text']}"

CITED EVIDENCE CLAIMS:
{claims_text}

INSTRUCTIONS:
- Check if the factual claims in the OUTPUT TEXT are supported by the CITED EVIDENCE
- The wording doesn't need to be identical - paraphrasing is acceptable if the meaning is preserved
- Verify that no contradicting or unsupported claims are made
- Focus on semantic accuracy, not exact word matching

IMPORTANT: Be reasonable - if the output text is a valid paraphrase or accurate representation of the evidence, it should be VERIFIED.

Respond with EXACTLY ONE of these formats:
- "VERIFIED" (if the text accurately reflects the evidence, even if paraphrased)
- "UNVERIFIED: [specific reason]" (only if there are clear inaccuracies or unsupported claims)

Your response:"""

        try:
            response = azure_llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": verification_prompt}],
                max_completion_tokens=200
            )
            
            verification_result = response.choices[0].message.content.strip()
            
            # Remove surrounding quotes if present
            if verification_result.startswith('"') and verification_result.endswith('"'):
                verification_result = verification_result[1:-1]
            
            # Debug: Print first few responses
            if i <= 3:
                print(f"    [DEBUG] Segment {i} LLM response: {verification_result[:150]}")
            
            if verification_result.upper().startswith("VERIFIED"):
                segment["verified"] = "Yes"
                segment["verification_reason"] = "Claims match cited evidence"
                verified_count += 1
            else:
                segment["verified"] = "No"
                # Extract reason after "UNVERIFIED:"
                if "UNVERIFIED:" in verification_result.upper():
                    reason = verification_result.split(":", 1)[1].strip() if ":" in verification_result else verification_result
                else:
                    reason = verification_result
                segment["verification_reason"] = reason if reason else "Claims do not match cited evidence"
                unverified_count += 1
                
        except Exception as e:
            segment["verified"] = "Error"
            segment["verification_reason"] = f"Verification failed: {str(e)}"
            unverified_count += 1
        
        # Progress indicator
        if i % 5 == 0:
            print(f"  Verified {i}/{len(segments)} segments...")
    
    print(f"\nVerification complete!")
    print(f"  Verified: {verified_count}")
    print(f"  Unverified: {unverified_count}")
    
    # Calculate verification rate
    total = len(segments)
    verification_rate = (verified_count / total * 100) if total > 0 else 0
    
    return {
        "total_segments": total,
        "verified_segments": verified_count,
        "unverified_segments": unverified_count,
        "verification_rate": f"{verification_rate:.1f}%",
        "segments": segments,
        "is_sentence_level": sentence_level  # IMPROVEMENT #4: Flag for sentence-level verification
    }


async def process_with_iterative_refinement_and_style(
    query: str,
    sources: Dict[str, Any],
    max_iterations: int = 3,
    max_output_length: int = 16000,
    context_details: str = "",
    style: Optional[Dict[str, Any]] = None,
    enable_policy_check: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Complete pipeline: Iterative refinement + Style-based output generation + Policy check.
    
    Args:
        query: User's research query
        sources: Sources dictionary (topics, links, attachments)
        max_iterations: Number of refinement iterations
        max_output_length: Maximum styled output character length
        context_details: Optional event/location/partner context
        style: Writing style dictionary (if None, will fetch random from DB)
        enable_policy_check: Whether to run BSP policy alignment check (default: True)
    
    Returns:
        Complete results including refined summary, styled output, and policy check
    """
    def emit(event_type: str, **payload):
        if progress_callback:
            try:
                progress_callback({"type": event_type, **payload})
            except Exception:
                pass

    # Step 1: Iterative refinement
    emit("stage_started", stage=1)
    print(f"\n{'='*70}")
    print("STEP 1: ITERATIVE REFINEMENT")
    print('='*70)
    refinement_results = await process_with_iterative_refinement(
        query,
        sources,
        max_iterations,
        progress_callback=progress_callback,
    )

    stage1_evidence = refinement_results.get("cumulative_evidence_store", [])
    topic_evidence_count = 0
    link_evidence_count = 0
    attachment_evidence_count = 0
    other_evidence_count = 0

    for evidence_item in stage1_evidence:
        retrieval_context = str(evidence_item.get("retrieval_context", "")).strip().lower()
        if retrieval_context == "deep_research_pipeline":
            topic_evidence_count += 1
        elif retrieval_context == "link_processing":
            link_evidence_count += 1
        elif retrieval_context == "attachment_processing":
            attachment_evidence_count += 1
        else:
            other_evidence_count += 1

    emit("stage_metric", stage=1, key="Topic Evidence", value=topic_evidence_count)
    emit("stage_metric", stage=1, key="Link Evidence", value=link_evidence_count)
    emit("stage_metric", stage=1, key="Attachment Evidence", value=attachment_evidence_count)
    if other_evidence_count > 0:
        emit("stage_metric", stage=1, key="Other Evidence", value=other_evidence_count)

    emit("stage_done", stage=1)
    
    # Get final summary
    final_summary = refinement_results["final_summary"].get("summary", "")
    
    if not final_summary:
        return {
            **refinement_results,
            "styled_output": {
                "success": False,
                "error": "No final summary generated from iterative refinement"
            }
        }
    
    # Step 2: Get writing style
    emit("stage_started", stage=2)
    print(f"\n{'='*70}")
    print("STEP 2: RETRIEVING WRITING STYLE")
    print('='*70)
    
    if style is None:
        style = get_random_style_from_db()
        if style:
            print(f"✓ Retrieved style: {style.get('name', 'Unknown')}")
            print(f"  Speaker: {style.get('speaker', 'Unknown')}")
            print(f"  Audience: {style.get('audience_setting_classification', 'General')}")
            emit("stage_text", stage=2, text=f"Style: {style.get('name', 'Unknown')}")
            emit("stage_text", stage=2, text=f"Speaker: {style.get('speaker', 'Unknown')}")
            emit("stage_text", stage=2, text=f"Audience: {style.get('audience_setting_classification', 'General')}")
        else:
            print("✗ No style found in database, using default formatting")
            emit("stage_text", stage=2, text="No style found in database, using default formatting")
    emit("stage_done", stage=2)
    
    # Step 3: Generate styled output
    emit("stage_started", stage=3)
    print(f"\n{'='*70}")
    print("STEP 3: GENERATING STYLED OUTPUT")
    print('='*70)
    
    # Get cumulative evidence store for citation validation
    cumulative_evidence = refinement_results.get("cumulative_evidence_store", [])
    print(f"[DEBUG] Evidence store size: {len(cumulative_evidence)} items")
    print(f"[DEBUG] Final summary length: {len(final_summary)} chars")
    
    styled_result = await generate_styled_output(
        final_summary, 
        query, 
        style,
        context_details=context_details,
        max_output_length=max_output_length,
        evidence_store=cumulative_evidence,
        use_claim_outline=True  # IMPROVEMENT #1: Explicitly enable claim-based styling
    )
    
    if styled_result.get("success"):
        print(f"✓ Styled output generated successfully")
        print(f"  Style applied: {styled_result.get('style_name', 'Unknown')}")
        print(f"  Output length: {len(styled_result.get('styled_output', ''))} characters")
        
        # Show citation validation results
        if styled_result.get("citations_found") is not None:
            print(f"  Citations: {styled_result.get('citations_found')} instances")
            print(f"  Evidence cited: {styled_result.get('unique_evidence_cited')} unique IDs")
            print(f"  Coverage: {styled_result.get('citation_coverage')}")
            
            if styled_result.get("invalid_citations"):
                print(f"  ⚠️ Invalid citations: {styled_result.get('invalid_citations')}")
            else:
                print(f"  ✓ All citations valid")
        emit("stage_text", stage=3, text="Styled output generated successfully")
        emit("stage_metric", stage=3, key="Output Length", value=len(styled_result.get("styled_output", "")))
        emit("stage_metric", stage=3, key="Max Length", value=max_output_length)
        emit("stage_metric", stage=3, key="Citations", value=styled_result.get("citations_found", 0))
        emit("stage_metric", stage=3, key="Evidence IDs", value=styled_result.get("unique_evidence_cited", 0))
    else:
        print(f"✗ Style generation failed: {styled_result.get('error', 'Unknown')}")
        emit("stage_text", stage=3, text=f"Style generation failed: {styled_result.get('error', 'Unknown')}")
    emit("stage_done", stage=3)
    
    # Step 4: Verify citations in styled output
    verification_result = None
    if styled_result.get("success") and styled_result.get("styled_output"):
        emit("stage_started", stage=4)
        print(f"\n{'='*70}")
        print("STEP 4: VERIFYING STYLED OUTPUT CITATIONS")
        print('='*70)
        
        try:
            verification_result = await verify_styled_citations(
                styled_output=styled_result.get("styled_output", ""),
                evidence_store=cumulative_evidence,
                sentence_level=True  # IMPROVEMENT #4: Use sentence-level verification
            )
            
            print(f"\n✓ Citation verification complete")
            print(f"  Total segments: {verification_result.get('total_segments')}")
            print(f"  Verified: {verification_result.get('verified_segments')}")
            print(f"  Unverified: {verification_result.get('unverified_segments')}")
            print(f"  Verification rate: {verification_result.get('verification_rate')}")
            
            # Show first few unverified segments if any
            unverified_segments = [
                s for s in verification_result.get("segments", []) 
                if s.get("verified") == "No"
            ]
            if unverified_segments:
                print(f"\n  ⚠️ Sample unverified segments:")
                for seg in unverified_segments[:3]:
                    print(f"    Segment {seg['segment_number']}: {seg.get('verification_reason', 'Unknown')}")
                
                # AUTO-FIX: Revise unverified segments
                print(f"\n[AUTO-FIX] 🔧 Attempting to fix {len(unverified_segments)} unverified segments...")
                
                # Build issues list for fix_speech_issues
                citation_issues = []
                for seg in unverified_segments:
                    citation_issues.append({
                        "segment": seg.get("sentence", ""),
                        "issue_description": seg.get("verification_reason", "Claim not fully supported by evidence"),
                        "suggestion": "Soften claim language or add hedging (e.g., 'may suggest', 'appears to indicate')",
                        "severity": "HIGH"
                    })
                
                # Call fix function
                fix_result = await fix_speech_issues(
                    speech_text=styled_result.get("styled_output", ""),
                    issues=citation_issues,
                    evidence_store=cumulative_evidence,
                    issue_type="citation"
                )
                
                if fix_result.get("success") and fix_result.get("fixes_applied") > 0:
                    print(f"[AUTO-FIX] ✅ Applied {fix_result['fixes_applied']} citation fixes")
                    # Update styled result with fixed speech
                    styled_result["styled_output"] = fix_result["fixed_speech"]
                    
                    # Re-verify to confirm fixes
                    print(f"[AUTO-FIX] Rechecking citations after fixes...")
                    verification_result = await verify_styled_citations(
                        styled_output=fix_result["fixed_speech"],
                        evidence_store=cumulative_evidence,
                        sentence_level=True
                    )
                    print(f"[AUTO-FIX] After fixes - Verified: {verification_result.get('verified_segments')}/{verification_result.get('total_segments')} segments")
                else:
                    print(f"[AUTO-FIX] ⚠️ Could not apply fixes: {fix_result.get('error', 'Unknown')}")
            emit("stage_metric", stage=4, key="Verified", value=verification_result.get('verified_segments'))
            emit("stage_metric", stage=4, key="Unverified", value=verification_result.get('unverified_segments'))
            emit("stage_text", stage=4, text=f"Verification rate: {verification_result.get('verification_rate')}")
            
        except Exception as e:
            print(f"✗ Verification failed: {str(e)}")
            verification_result = {
                "error": str(e),
                "total_segments": 0,
                "verified_segments": 0,
                "unverified_segments": 0
            }
            emit("stage_done", stage=4)
    
    # Step 5: Convert to APA format
    apa_result = None
    if styled_result.get("success") and styled_result.get("styled_output"):
        emit("stage_started", stage=5)
        print(f"\n{'='*70}")
        print("STEP 5: CONVERTING TO APA FORMAT")
        print('='*70)
        
        try:
            apa_result = await convert_styled_output_to_apa(
                styled_output=styled_result.get("styled_output", ""),
                evidence_store=cumulative_evidence
            )
            
            if apa_result.get("success"):
                print(f"\n✓ APA conversion complete")
                print(f"  Citations converted: {apa_result.get('citations_converted')}")
                print(f"  References generated: {apa_result.get('references_generated')}")
                print(f"  Output length: {len(apa_result.get('apa_output', ''))} characters")
                emit("stage_metric", stage=5, key="Citations Converted", value=apa_result.get('citations_converted', 0))
                emit("stage_metric", stage=5, key="References", value=apa_result.get('references_generated', 0))
            else:
                print(f"✗ APA conversion failed: {apa_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"✗ APA conversion failed: {str(e)}")
            apa_result = {
                "success": False,
                "error": str(e),
                "apa_output": styled_result.get("styled_output", "")  # Fallback to original
            }
            emit("stage_done", stage=5)
    
    # Step 6: Plagiarism Detection
    plagiarism_result = None
    if styled_result.get("success") and styled_result.get("styled_output"):
        emit("stage_started", stage=6)
        print(f"\n{'='*70}")
        print("STEP 6: PLAGIARISM DETECTION")
        print('='*70)
        
        # Initialize clients for plagiarism check
        azure_llm_client = None
        tavily_client = None
        
        try:
            # Initialize clients for plagiarism check
            from openai import AsyncAzureOpenAI
            azure_llm_client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            tavily_client = AsyncTavilyClient(api_key=tavily_api_key) if tavily_api_key else None
            
            # Prepare speech metadata
            speech_metadata = {
                "id": hashlib.md5(styled_result.get("styled_output", "").encode()).hexdigest()[:16],
                "query": query,
                "speaker": style.get("speaker") if style else None,
                "institution": "Analysis",
                "date": datetime.now().isoformat()
            }
            
            # Run plagiarism check
            plagiarism_result = await check_plagiarism(
                speech_text=styled_result.get("styled_output", ""),
                azure_client=azure_llm_client,
                tavily_client=tavily_client,
                speech_metadata=speech_metadata,
                use_hf_classifier=False  # Can enable if transformers is installed
            )
            
            if plagiarism_result.get("success"):
                print(f"\n✓ Plagiarism analysis complete")
                print(f"  Overall risk: {plagiarism_result.get('overall_risk_level')} ({plagiarism_result.get('overall_risk_score'):.3f})")
                stats = plagiarism_result.get('statistics', {})
                print(f"  Total chunks: {stats.get('total_chunks', 0)}")
                print(f"  High risk: {stats.get('high_risk_chunks', 0)}")
                print(f"  Medium risk: {stats.get('medium_risk_chunks', 0)}")
                print(f"  Low risk: {stats.get('low_risk_chunks', 0)}")
                print(f"  Clean: {stats.get('clean_chunks', 0)}")
                emit("stage_metric", stage=6, key="Risk Level", value=plagiarism_result.get('overall_risk_level', 'UNKNOWN'))
                
                # Show top sources if any
                top_sources = plagiarism_result.get('top_sources', [])
                if top_sources:
                    print(f"\n  Top matching sources:")
                    for i, src in enumerate(top_sources[:3], 1):
                        print(f"    {i}. {src.get('title', 'Unknown')} ({src.get('match_count', 0)} matches)")
                
                # AUTO-FIX: Revise elevated-risk plagiarism chunks (MEDIUM/HIGH/CRITICAL)
                chunks = plagiarism_result.get('chunks', [])
                elevated_risk_chunks = [
                    c for c in chunks
                    if str(c.get('risk_level', '')).strip().upper() in ['MEDIUM', 'HIGH', 'CRITICAL']
                ]
                
                if elevated_risk_chunks:
                    print(f"\n[AUTO-FIX] 🔧 Attempting to rephrase {len(elevated_risk_chunks)} medium/high/critical chunks...")
                    
                    # Build issues list for fix_speech_issues
                    plagiarism_issues = []
                    for chunk in elevated_risk_chunks[:12]:  # Limit to avoid token overflow
                        chunk_risk = str(chunk.get('risk_level', '')).strip().upper()
                        plagiarism_issues.append({
                            "segment": chunk.get("text", ""),
                            "issue_description": f"{chunk_risk.title()} similarity to source material (risk score: {chunk.get('overall_risk_score', 0):.2f})",
                            "suggestion": "Restructure sentences completely, use different word order and synonyms",
                            "severity": "CRITICAL" if chunk_risk == 'CRITICAL' else ("HIGH" if chunk_risk == 'HIGH' else "MEDIUM")
                        })
                    
                    # Call fix function
                    fix_result = await fix_speech_issues(
                        speech_text=styled_result.get("styled_output", ""),
                        issues=plagiarism_issues,
                        evidence_store=cumulative_evidence,
                        issue_type="plagiarism"
                    )
                    
                    if fix_result.get("success") and fix_result.get("fixes_applied") > 0:
                        print(f"[AUTO-FIX] ✅ Applied {fix_result['fixes_applied']} plagiarism fixes")
                        # Update styled result with fixed speech
                        styled_result["styled_output"] = fix_result["fixed_speech"]
                        
                        # Note: Full re-check would require repeating plagiarism analysis (expensive)
                        # We trust the LLM fix for now
                        print(f"[AUTO-FIX] Speech rephrased to reduce similarity")
                    else:
                        print(f"[AUTO-FIX] ⚠️ Could not apply plagiarism fixes: {fix_result.get('error', 'Unknown')}")
            else:
                print(f"✗ Plagiarism analysis failed: {plagiarism_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"✗ Plagiarism detection failed: {str(e)}")
            plagiarism_result = {
                "success": False,
                "error": str(e),
                "overall_risk_level": "unknown"
            }
        finally:
            # Clean up clients
            if azure_llm_client:
                await azure_llm_client.close()
            if tavily_client:
                await tavily_client.close()
        emit("stage_done", stage=6)
    
    # Step 7: BSP Policy Alignment Check
    policy_check_result = None
    if enable_policy_check and styled_result.get("success"):
        emit("stage_started", stage=7)
        print(f"\n{'='*70}")
        print("STEP 7: BSP POLICY ALIGNMENT CHECK")
        print('='*70)
        
        try:
            # Prepare speech metadata
            speech_metadata = {
                "topic": query,
                "speaker": style.get("speaker") if style else "BSP Official",
                "audience": style.get("audience_setting_classification") if style else "General audience",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query
            }
            max_policy_fix_rounds = 3
            policy_fix_round = 0
            total_policy_fixes = 0

            while True:
                # Determine which version of the speech to check (prefer APA if available)
                speech_to_check = (
                    apa_result.get("apa_output") if apa_result and apa_result.get("success")
                    else styled_result.get("styled_output", "")
                )

                print(f"\n[POLICY] Verification pass {policy_fix_round + 1}/{max_policy_fix_rounds + 1}")
                emit("stage_text", stage=7, text=f"Policy verification pass {policy_fix_round + 1}")

                # Run policy alignment check using Azure Agent
                policy_check_result = await check_speech_policy_alignment(
                    speech_content=speech_to_check,
                    speech_metadata=speech_metadata
                )

                if not policy_check_result.get("success"):
                    print(f"✗ Policy check failed: {policy_check_result.get('error', 'Unknown')}")
                    emit("stage_text", stage=7, text=f"Policy check failed: {policy_check_result.get('error', 'Unknown')}")
                    break

                # Normalize policy decision to avoid conflicting records.
                # Rule: score >= 95% => no revision needed; otherwise requires revision.
                compliance_score = policy_check_result.get('compliance_score')
                normalized_requires_revision = None
                if isinstance(compliance_score, (int, float)):
                    score_percent_rounded = round(compliance_score * 100, 1)
                    normalized_requires_revision = score_percent_rounded < 95.0
                    policy_check_result['requires_revision'] = normalized_requires_revision
                    policy_check_result['overall_compliance'] = (
                        'needs_revision' if normalized_requires_revision else 'approved'
                    )

                effective_requires_revision = (
                    normalized_requires_revision
                    if normalized_requires_revision is not None
                    else bool(policy_check_result.get('requires_revision'))
                )

                # Ensure displayed compliance label is always consistent with final decision.
                policy_check_result['overall_compliance'] = (
                    'needs_revision' if effective_requires_revision else 'approved'
                )

                print(f"\n✓ Policy alignment check complete")
                print(f"  Compliance: {policy_check_result.get('overall_compliance').upper()}")
                if isinstance(compliance_score, (int, float)):
                    print(f"  Score: {compliance_score:.1%}")
                emit("stage_metric", stage=7, key="Compliance", value=policy_check_result.get('overall_compliance', 'unknown').upper())

                # Show violations summary
                violations_count = policy_check_result.get('violations_count', 0)
                if violations_count > 0:
                    print(f"  Violations: {violations_count}")
                    print(f"    Critical: {policy_check_result.get('critical_violations', 0)}")
                    print(f"    High: {policy_check_result.get('high_violations', 0)}")
                    violations = policy_check_result.get('violations', [])
                    if violations:
                        print("  Violation details (top 5):")
                        emit("stage_text", stage=7, text="Policy violations detected (showing top 5)")
                        for idx, violation in enumerate(violations[:5], 1):
                            violation_type = violation.get('violation_type', 'Unknown')
                            severity = str(violation.get('severity', 'MEDIUM')).upper()
                            description = violation.get('description', '')
                            suggestion = violation.get('suggested_fix', '')
                            problematic_text = (violation.get('problematic_text', '') or '').strip().replace('\n', ' ')
                            if len(problematic_text) > 140:
                                problematic_text = problematic_text[:140].rstrip() + "..."

                            print(f"    {idx}. [{severity}] {violation_type}")
                            if description:
                                print(f"       Cause: {description}")
                            if problematic_text:
                                print(f"       Text: \"{problematic_text}\"")
                            if suggestion:
                                print(f"       Suggested fix: {suggestion}")

                            summary_line = f"[{severity}] {violation_type}: {description}" if description else f"[{severity}] {violation_type}"
                            emit("stage_text", stage=7, text=summary_line)

                # Show commendations
                commendations = policy_check_result.get('commendations', [])
                if commendations:
                    print(f"  Commendations: {len(commendations)} positive findings")

                # Show circulars referenced
                circulars = policy_check_result.get('circular_references', [])
                if circulars:
                    print(f"  BSP Circulars Referenced: {len(circulars)}")
                    for i, circ in enumerate(circulars[:3], 1):
                        print(f"    {i}. {circ}")

                # Exit loop if approved
                if not effective_requires_revision:
                    print(f"\n  ✅ RECOMMENDATION: APPROVED FOR USE")
                    emit("stage_text", stage=7, text="Policy check: Approved for use")
                    break

                print(f"\n  ⚠️ RECOMMENDATION: REVISION REQUIRED")
                emit("stage_text", stage=7, text="Policy check: Needs revision")

                # Stop if max fix rounds reached
                if policy_fix_round >= max_policy_fix_rounds:
                    print(f"[AUTO-FIX] ⚠️ Max policy fix rounds reached ({max_policy_fix_rounds}). Stopping recheck loop.")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Max policy fix rounds reached ({max_policy_fix_rounds})")
                    break

                # AUTO-FIX: Revise policy violations
                violations = policy_check_result.get('violations', [])

                # Build issues list for fix_speech_issues (always attempt fix when revision is required)
                policy_issues = []
                if violations:
                    print(f"\n[AUTO-FIX] 🔧 Attempting to fix {len(violations)} policy violations...")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Attempting policy fix for {len(violations)} violations")
                    for violation in violations[:10]:  # Limit to top 10
                        policy_issues.append({
                            "segment": violation.get("problematic_text", "") or speech_to_check[:700],
                            "issue_description": f"{violation.get('violation_type', 'Unknown')}: {violation.get('description', '')}",
                            "suggestion": violation.get("suggested_fix", "Revise to align with BSP policy guidelines"),
                            "severity": violation.get("severity", "MEDIUM").upper()
                        })
                else:
                    print("\n[AUTO-FIX] 🔧 Needs revision flagged without detailed violations; attempting holistic policy fix...")
                    emit("stage_text", stage=7, text="[AUTO-FIX] Needs revision flagged; attempting holistic policy fix")
                    policy_issues.append({
                        "segment": speech_to_check[:900],
                        "issue_description": f"Policy check flagged needs revision (compliance: {policy_check_result.get('overall_compliance', 'unknown')})",
                        "suggestion": policy_check_result.get("recommendation", "Revise wording to align with BSP policy requirements and reduce non-compliant phrasing."),
                        "severity": "MEDIUM"
                    })

                # Call fix function
                fix_result = await fix_speech_issues(
                    speech_text=speech_to_check,
                    issues=policy_issues,
                    evidence_store=cumulative_evidence,
                    issue_type="policy"
                )

                if fix_result.get("success") and fix_result.get("fixes_applied") > 0:
                    policy_fix_round += 1
                    total_policy_fixes += fix_result.get("fixes_applied", 0)
                    print(f"[AUTO-FIX] ✅ Applied {fix_result['fixes_applied']} policy fixes (round {policy_fix_round})")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Applied {fix_result['fixes_applied']} policy fixes (round {policy_fix_round})")
                    emit("stage_metric", stage=7, key="Policy Fixes", value=total_policy_fixes)
                    emit("stage_metric", stage=7, key="Fix Rounds", value=policy_fix_round)

                    # Update the appropriate result before re-verifying
                    if apa_result and apa_result.get("success"):
                        apa_result["apa_output"] = fix_result["fixed_speech"]
                        print(f"[AUTO-FIX] Updated APA output with policy fixes")
                        emit("stage_text", stage=7, text="[AUTO-FIX] Updated APA output with policy fixes")
                    else:
                        styled_result["styled_output"] = fix_result["fixed_speech"]
                        print(f"[AUTO-FIX] Updated styled output with policy fixes")
                        emit("stage_text", stage=7, text="[AUTO-FIX] Updated styled output with policy fixes")

                    print(f"[AUTO-FIX] Re-running policy verification after fixes...")
                    emit("stage_text", stage=7, text="[AUTO-FIX] Re-running policy verification after fixes")
                    continue

                print(f"[AUTO-FIX] ⚠️ Could not apply policy fixes: {fix_result.get('error', 'Unknown')}")
                emit("stage_text", stage=7, text=f"[AUTO-FIX] Could not apply policy fixes: {fix_result.get('error', 'Unknown')}")
                break
                
        except Exception as e:
            print(f"✗ Policy alignment check failed: {str(e)}")
            import traceback
            traceback.print_exc()
            policy_check_result = {
                "success": False,
                "error": str(e),
                "overall_compliance": "error",
                "requires_revision": True
            }
            emit("stage_text", stage=7, text=f"Policy alignment check failed: {str(e)}")
        emit("stage_done", stage=7)
    elif not enable_policy_check:
        print(f"\n{'='*70}")
        print("STEP 7: BSP POLICY ALIGNMENT CHECK - SKIPPED")
        print('='*70)
        print("  (Set enable_policy_check=True to enable)")
    
    # Combine results
    complete_results = {
        **refinement_results,
        "styled_output": styled_result,
        "styled_output_apa": apa_result,
        "citation_verification": verification_result,
        "plagiarism_analysis": plagiarism_result,
        "policy_check": policy_check_result,
        "style_used": {
            "name": style.get("name", "None") if style else "None",
            "speaker": style.get("speaker", "None") if style else "None",
            "audience": style.get("audience_setting_classification", "None") if style else "None"
        }
    }
    
    return complete_results


# Example usage
async def main():
    """
    Example usage of the writer_main functionality.
    """
    # Example input
    user_query = "What are the latest developments in transformer architectures?"
    
    user_sources = {
        "topics": "machine learning transformers",
        "links": [
            "https://arxiv.org/abs/1706.03762",
            "https://huggingface.co/docs/transformers"
        ],
        "attachments": [
            "/path/to/research_paper.pdf",
            "/path/to/notes.txt"
        ]
    }
    
    # Process the input
    results = await process_user_input(user_query, user_sources)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
