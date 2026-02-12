#!/usr/bin/env python3
"""
Comprehensive test showing evidence_store with full JSON output
"""
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the functions
from app.writer_main import process_user_input


async def comprehensive_test():
    """Complete test showing evidence_store structure"""
    
    print("=" * 70)
    print("COMPREHENSIVE EVIDENCE STORE TEST")
    print("=" * 70)
    
    user_query = "What are the main innovations in transformer architecture?"
    
    user_sources = {
        "topics": "transformer neural networks attention mechanism",  # Include topic for deep research
        "links": [
            "https://arxiv.org/abs/1706.03762",
            "https://en.wikipedia.org/wiki/Transformer_(deep_learning_model)"
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Processing {len(user_sources['links'])} links...")
    print("\nThis will combine evidence from both deep research AND link analysis...")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        print("\n" + "=" * 70)
        print("COMPLETE RESULTS")
        print("=" * 70)
        
        # Show high-level summary
        print(f"\nQuery: {results['query']}")
        print(f"Timestamp: {results['timestamp']}")
        
        # Evidence Store Overview
        if results.get('evidence_store'):
            print(f"\n{'='*70}")
            print(f"EVIDENCE STORE - {len(results['evidence_store'])} items")
            print('='*70)
            
            for i, evidence in enumerate(results['evidence_store'], 1):
                print(f"\n[Evidence Item #{i}]")
                print(f"Source: {evidence['Source']}")
                print(f"\nInformation:")
                info = evidence['Information']
                # Show first 400 chars of information
                if len(info) > 400:
                    print(info[:400] + "...\n[truncated]")
                else:
                    print(info)
                print("-" * 70)
        
        # Show detailed JSON structure (first item only)
        print("\n" + "=" * 70)
        print("EVIDENCE STORE JSON STRUCTURE (First Item)")
        print("=" * 70)
        if results.get('evidence_store') and len(results['evidence_store']) > 0:
            print(json.dumps(results['evidence_store'][0], indent=2))
        
        # Show processing statistics
        print("\n" + "=" * 70)
        print("PROCESSING STATISTICS")
        print("=" * 70)
        
        if results.get('link_results'):
            lr = results['link_results']
            print(f"\nLinks Processed: {lr['count']}")
            print(f"  - Success: {lr.get('success_count', 0)}")
            print(f"  - Errors: {lr.get('error_count', 0)}")
            print(f"  - Evidence Items Generated: {len(lr.get('evidence_store', []))}")
        
        if results.get('topic_results'):
            tr = results['topic_results']
            if tr.get('success'):
                print(f"\nTopic Research: Success")
                print(f"  - Evidence Items Generated: {len(tr.get('evidence_store', []))}")
            else:
                print(f"\nTopic Research: {tr.get('error', 'Failed')}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nTotal Evidence Items: {len(results.get('evidence_store', []))}")
        print(f"All evidence has been structured with 'Information' and 'Source' fields")
        print(f"Ready for use in downstream processing and document generation")
        
        # Save to file for inspection
        with open('evidence_store_output.json', 'w') as f:
            json.dump({
                'query': results['query'],
                'timestamp': results['timestamp'],
                'evidence_store': results['evidence_store']
            }, f, indent=2)
        
        print(f"\n✓ Full evidence store saved to: evidence_store_output.json")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(comprehensive_test())
