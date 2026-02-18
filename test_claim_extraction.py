#!/usr/bin/env python3
"""Quick test to diagnose claim extraction issues"""
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Initialize client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

model = os.getenv("AZURE_OPENAI_MODEL_WRITER", "gpt-5")

# Simple test summary
summary = """
AI systems analyze market trends using machine learning algorithms [E1]. 
Neural networks can predict portfolio performance with 85% accuracy [E2,E3]. 
This enables better investment decisions [E5].
"""

extraction_prompt = f"""Extract atomic factual claims from this research summary. Each claim should be:

1. **One clear factual statement** (not multiple facts combined)
2. **Tied to 1-2 evidence IDs** (from the cited [ENN] references)

<SUMMARY>
{summary}
</SUMMARY>

Return a JSON array where each item has:
- "claim_text": the atomic factual statement
- "evidence_ids": array of 1-2 evidence IDs (e.g., ["E1", "E5"])

Return ONLY the JSON array, no markdown, no explanation.

JSON:"""

print("Calling LLM for claim extraction...")
print(f"Model: {model}")
print("=" * 70)

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a precise claim extraction system. Return ONLY valid JSON array, no other text."},
        {"role": "user", "content": extraction_prompt}
    ],
    max_completion_tokens=8000  # GPT-5 needs more tokens for reasoning (default temperature=1 only)
)

print(f"\nResponse ID: {response.id}")
print(f"Model: {response.model}")
print(f"Finish reason: {response.choices[0].finish_reason}")
print(f"Has refusal: {response.choices[0].message.refusal is not None}")

# Show token usage
if hasattr(response, 'usage') and response.usage:
    usage = response.usage
    print(f"\nToken usage:")
    print(f"  Prompt: {usage.prompt_tokens}")
    print(f"  Completion: {usage.completion_tokens}")
    if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
        details = usage.completion_tokens_details
        if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
            print(f"  Reasoning: {details.reasoning_tokens}")
            print(f"  Output: {usage.completion_tokens - details.reasoning_tokens}")

raw_content = response.choices[0].message.content
print(f"\nRaw content type: {type(raw_content)}")
print(f"Raw content is None: {raw_content is None}")
print(f"Raw content length: {len(raw_content) if raw_content else 0}")

if raw_content:
    print("\n" + "=" * 70)
    print("RAW CONTENT:")
    print("=" * 70)
    print(raw_content)
    print("=" * 70)
    
    # Try to extract JSON
    content = raw_content.strip()
    
    # Strip thinking tags
    if '<think>' in content:
        content = content.split('</think>')[-1].strip()
    if '<thinking>' in content:
        content = content.split('</thinking>')[-1].strip()
    
    # Strip markdown
    content = content.replace('```json', '').replace('```', '').strip()
    
    # Find JSON
    start = content.find('[')
    end = content.rfind(']')
    
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end+1]
        print(f"\nExtracted JSON ({len(json_str)} chars):")
        print(json_str)
        
        try:
            data = json.loads(json_str)
            print(f"\n✅ Successfully parsed {len(data)} claims")
            for i, claim in enumerate(data, 1):
                print(f"{i}. {claim.get('claim_text', 'N/A')}")
                print(f"   Evidence: {claim.get('evidence_ids', [])}")
        except Exception as e:
            print(f"\n❌ JSON parsing failed: {e}")
    else:
        print(f"\n❌ No JSON array found (start={start}, end={end})")
else:
    print("\n❌ Raw content is empty!")
