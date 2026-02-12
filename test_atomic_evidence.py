#!/usr/bin/env python3
"""
Test script for atomic evidence store with forced citations
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_user_input, generate_summary_from_evidence


async def test_atomic_evidence():
    """Test atomic evidence extraction and forced citations"""
    
    print("=" * 70)
    print("ATOMIC EVIDENCE & CITATION TEST")
    print("=" * 70)
    
    user_query = "What are the key architectural components of transformer neural networks?"
    
    user_sources = {
        "topics": "transformer architecture components",
        "links": [
            "https://arxiv.org/abs/1706.03762"
        ],
        "attachments": []
    }
    
    print(f"\nQuery: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Links: {len(user_sources['links'])}")
    
    try:
        # Step 1: Process sources with atomic claim extraction
        print("\n" + "=" * 70)
        print("STEP 1: EXTRACTING ATOMIC CLAIMS")
        print("=" * 70)
        
        results = await process_user_input(user_query, user_sources)
        
        evidence_store = results.get("evidence_store", [])
        
        print(f"\n✓ Extracted {len(evidence_store)} atomic claims with stable IDs")
        
        # Show sample evidence
        print("\n" + "=" * 70)
        print("SAMPLE ATOMIC EVIDENCE")
        print("=" * 70)
        
        for i, evidence in enumerate(evidence_store[:5]):  # Show first 5
            print(f"\nID: {evidence.get('id', 'N/A')}")
            print(f"Claim: {evidence.get('claim', 'N/A')[:150]}...")
            print(f"Source: {evidence.get('source_title', 'N/A')}")
            print(f"Confidence: {evidence.get('confidence', 0.0):.2f}")
            if evidence.get('quote_span'):
                print(f"Quote: \"{evidence.get('quote_span', '')[:100]}...\"")
        
        # Step 2: Generate summary with FORCED citations
        print("\n" + "=" * 70)
        print("STEP 2: GENERATING SUMMARY WITH FORCED CITATIONS")
        print("=" * 70)
        
        summary_result = await generate_summary_from_evidence(user_query, evidence_store)
        
        if summary_result.get("success"):
            summary = summary_result.get("summary", "")
            print(f"\n✓ Summary generated with {summary_result.get('citations_found', 0)} citation instances")
            
            if summary_result.get("invalid_citations"):
                print(f"\n⚠ Invalid citations found: {summary_result['invalid_citations']}")
            else:
                print("\n✓ All citations validated against evidence store")
            
            # Show summary with citations
            print("\n" + "=" * 70)
            print("SUMMARY WITH [ENN] CITATIONS")
            print("=" * 70)
            print(summary[:2000])
            if len(summary) > 2000:
                print("\n... (truncated)")
            print("=" * 70)
            
            # Save results
            output_data = {
                "query": user_query,
                "evidence_count": len(evidence_store),
                "evidence_store": evidence_store,
                "summary": summary,
                "citations_found": summary_result.get("citations_found", 0),
                "invalid_citations": summary_result.get("invalid_citations")
            }
            
            with open('atomic_evidence_test.json', 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n✓ Results saved to: atomic_evidence_test.json")
            
            # Analyze citation coverage
            import re
            cited_ids = set()
            citations = re.findall(r'\[E\d+(?:,E\d+)*\]', summary)
            for citation in citations:
                ids = re.findall(r'E\d+', citation)
                cited_ids.update(ids)
            
            all_ids = set([e.get('id') for e in evidence_store])
            uncited = all_ids - cited_ids
            
            print(f"\n" + "=" * 70)
            print("CITATION COVERAGE ANALYSIS")
            print("=" * 70)
            print(f"Total evidence items: {len(evidence_store)}")
            print(f"Evidence items cited: {len(cited_ids)}")
            print(f"Citation coverage: {len(cited_ids)/len(evidence_store)*100:.1f}%")
            if uncited:
                print(f"Uncited evidence: {', '.join(sorted(uncited))}")
            
        else:
            print(f"\n✗ Summary generation failed: {summary_result.get('error')}")
        
        print("\n" + "=" * 70)
        print("✓ ATOMIC EVIDENCE TEST COMPLETED")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_atomic_evidence())
