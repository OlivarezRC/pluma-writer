#!/usr/bin/env python3
"""
Quick test for styled output with strict citation enforcement
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import generate_styled_output, get_random_style_from_db


async def test_styled_citations():
    """Test that styled output enforces strict citations"""
    
    print("="*70)
    print("TESTING STYLED OUTPUT WITH STRICT CITATIONS")
    print("="*70)
    
    # Sample summary with citations
    sample_summary = """Transformer neural networks introduced several key innovations [E1,E2]. 
The attention mechanism allows the model to weigh relationships between all tokens [E3]. 
Multi-head attention enables parallel processing of different representation subspaces [E4,E5]. 
Positional encodings inject sequence order information [E6]. 
These innovations led to superior performance on translation tasks [E7,E8]."""
    
    # Sample evidence store
    evidence_store = [
        {"id": "E1", "claim": "Transformers use attention mechanisms", "source_url": "https://example.com"},
        {"id": "E2", "claim": "Transformers eliminate recurrence", "source_url": "https://example.com"},
        {"id": "E3", "claim": "Attention weighs token relationships", "source_url": "https://example.com"},
        {"id": "E4", "claim": "Multi-head attention used", "source_url": "https://example.com"},
        {"id": "E5", "claim": "Parallel processing enabled", "source_url": "https://example.com"},
        {"id": "E6", "claim": "Positional encodings used", "source_url": "https://example.com"},
        {"id": "E7", "claim": "Superior performance achieved", "source_url": "https://example.com"},
        {"id": "E8", "claim": "Translation tasks improved", "source_url": "https://example.com"},
    ]
    
    query = "What are the key innovations in transformer neural networks?"
    
    print(f"\nQuery: {query}")
    print(f"Evidence items: {len(evidence_store)}")
    print(f"Allowed IDs: E1-E8")
    print(f"\nOriginal summary length: {len(sample_summary)} chars")
    print(f"Original citations: 7 instances")
    
    # Get a random style
    print(f"\nRetrieving writing style...")
    style = get_random_style_from_db()
    
    if not style:
        print("✗ No style available")
        return
    
    print(f"✓ Style retrieved: {style.get('speaker', 'Unknown')}")
    
    # Generate styled output
    print(f"\nGenerating styled output with strict citation enforcement...")
    
    result = await generate_styled_output(
        summary=sample_summary,
        query=query,
        style=style,
        evidence_store=evidence_store,
        max_output_length=5000
    )
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    if result.get("success"):
        print(f"\n✓ Success!")
        print(f"  Style: {result.get('style_name', 'Unknown')}")
        print(f"  Speaker: {result.get('speaker', 'Unknown')}")
        print(f"  Output length: {result.get('output_length')} chars")
        
        print(f"\n📊 CITATION VALIDATION:")
        print(f"  Citations found: {result.get('citations_found')}")
        print(f"  Unique evidence cited: {result.get('unique_evidence_cited')}")
        print(f"  Coverage: {result.get('citation_coverage')}")
        
        if result.get('invalid_citations'):
            print(f"  ⚠️ Invalid citations: {result.get('invalid_citations')}")
        else:
            print(f"  ✓ All citations valid")
        
        validation = result.get('validation', {})
        print(f"\n  Cited IDs: {validation.get('cited_ids', [])}")
        uncited = validation.get('uncited_ids', [])
        if uncited:
            print(f"  ⚠️ Uncited IDs: {uncited}")
        else:
            print(f"  ✓ All evidence cited")
        
        print(f"\n{'='*70}")
        print("STYLED OUTPUT")
        print(f"{'='*70}")
        print(result.get('styled_output', ''))
        print(f"{'='*70}")
        
    else:
        print(f"\n✗ Failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(test_styled_citations())
