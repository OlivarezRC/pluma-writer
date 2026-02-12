import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import generate_styled_output

# Load existing test data from complete_pipeline_output
with open('complete_pipeline_output.json', 'r') as f:
    data = json.load(f)

# Extract the final summary and evidence store
final_summary_data = data.get('final_summary', {})
final_summary = final_summary_data.get('summary', '')
evidence_store = data.get('cumulative_evidence_store', [])

print(f"Loaded summary length: {len(final_summary)}")
print(f"Loaded evidence store items: {len(evidence_store)}")
print(f"First 200 chars of summary: {final_summary[:200]}")

# Load style
with open('data/db_styles.json', 'r') as f:
    styles = json.load(f)

# Use first style
style = styles[0] if styles else {
    "name": "Academic",
    "speaker": "Researcher",
    "audience_setting_classification": "Academic Conference"
}

print(f"\nStyle: {style.get('name')}")
print(f"Speaker: {style.get('speaker')}")
print(f"Audience: {style.get('audience_setting_classification')}")

# Test with the actual data
async def test():
    query = data.get('query', 'test query')
    print(f"\n{'='*60}")
    print("GENERATING STYLED OUTPUT...")
    print(f"{'='*60}\n")
    
    result = await generate_styled_output(
        summary=final_summary,
        query=query,
        style=style,
        evidence_store=evidence_store
    )
    
    print(f"\n{'='*60}")
    print("STYLED OUTPUT RESULT:")
    print(f"{'='*60}")
    print(f"Success: {result.get('success')}")
    
    if 'error' in result:
        print(f"Error: {result.get('error')}")
    
    output = result.get('styled_output', '')
    print(f"Output length: {len(output)}")
    print(f"Citations found: {result.get('citations_found', 0)}")
    print(f"Citation coverage: {result.get('citation_coverage', '0%')}")
    
    if output:
        print(f"\nFirst 500 chars of output:")
        print(output[:500])
    else:
        print("\nNO OUTPUT GENERATED!")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test())
