#!/usr/bin/env python3
"""
Test to diagnose link processing errors
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_links


async def test_link_errors():
    """Test links individually to see which one fails"""
    
    query = "What are transformer architectures?"
    links = [
        "https://arxiv.org/abs/1706.03762",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_model)"
    ]
    
    print("=" * 70)
    print("TESTING LINK PROCESSING - INDIVIDUAL RESULTS")
    print("=" * 70)
    
    results = await process_links(links, query)
    
    print(f"\nQuery: {query}")
    print(f"Total Links: {results['count']}")
    print(f"Success: {results.get('success_count', 0)}")
    print(f"Errors: {results.get('error_count', 0)}")
    
    print("\n" + "=" * 70)
    print("INDIVIDUAL LINK DETAILS")
    print("=" * 70)
    
    for i, item in enumerate(results['items'], 1):
        print(f"\n[Link {i}]")
        print(f"URL: {item['url']}")
        print(f"Status: {item['status']}")
        
        if item['status'] == 'success':
            print(f"✓ Title: {item.get('title', 'N/A')}")
            print(f"✓ Content Length: {item.get('raw_content_length', 0)} chars")
            print(f"✓ Extracted: {len(item.get('extracted_content', ''))} chars")
        elif item['status'] == 'error':
            print(f"✗ Error: {item.get('error', 'Unknown error')}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("EVIDENCE STORE")
    print("=" * 70)
    print(f"Evidence items created: {len(results.get('evidence_store', []))}")
    
    for i, evidence in enumerate(results.get('evidence_store', []), 1):
        print(f"\n[Evidence {i}]")
        print(f"Source: {evidence['Source']}")
        info = evidence['Information'][:200] + "..." if len(evidence['Information']) > 200 else evidence['Information']
        print(f"Info: {info}")


if __name__ == "__main__":
    asyncio.run(test_link_errors())
