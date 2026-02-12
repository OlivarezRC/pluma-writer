#!/usr/bin/env python3
"""
Test with multiple sources - both topics and multiple links
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_user_input


async def test_multiple_sources():
    """Test with a topic and multiple links"""
    
    print("=" * 70)
    print("TESTING WITH MULTIPLE SOURCES")
    print("=" * 70)
    
    user_query = "How do attention mechanisms work in transformer models and what are their benefits?"
    
    user_sources = {
        "topics": "attention mechanism in transformers self-attention multi-head attention",
        "links": [
            "https://arxiv.org/abs/1706.03762",  # Original Transformer paper
            "https://arxiv.org/abs/2004.05150",  # Longformer
            "https://arxiv.org/abs/1810.04805",  # BERT
            "https://arxiv.org/abs/2005.14165",  # GPT-3
            "https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"  # Attention is All You Need full paper
        ],
        "attachments": []
    }
    
    print(f"\nUser Query:")
    print(f"  {user_query}\n")
    print(f"Topic for Deep Research:")
    print(f"  {user_sources['topics']}\n")
    print(f"Links to Process: {len(user_sources['links'])}")
    for i, link in enumerate(user_sources['links'], 1):
        print(f"  {i}. {link}")
    
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    print("This will:")
    print("  1. Run deep research on the topic (3-4 web searches)")
    print("  2. Extract and analyze content from each link")
    print("  3. Aggregate all evidence into a unified knowledge base\n")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\nQuery: {results['query']}")
        print(f"Timestamp: {results['timestamp']}")
        
        # Topic Research Results
        if results.get('topic_results'):
            tr = results['topic_results']
            print("\n--- TOPIC RESEARCH ---")
            if tr.get('success'):
                print(f"✓ Status: Success")
                print(f"✓ Topic: {tr['topic']}")
                print(f"✓ Evidence Items: {len(tr.get('evidence_store', []))}")
            else:
                print(f"✗ Status: Failed")
                print(f"✗ Error: {tr.get('error')}")
        
        # Link Processing Results
        if results.get('link_results'):
            lr = results['link_results']
            print("\n--- LINK PROCESSING ---")
            print(f"Total Links: {lr['count']}")
            print(f"✓ Successfully Processed: {lr.get('success_count', 0)}")
            print(f"✗ Failed/Blocked: {lr.get('error_count', 0)}")
            print(f"✓ Evidence Items: {len(lr.get('evidence_store', []))}")
            
            print("\n  Individual Link Status:")
            for i, item in enumerate(lr.get('items', []), 1):
                status_icon = "✓" if item['status'] == 'success' else "✗"
                title = item.get('title', '') or 'N/A'
                title = title[:60] if title else 'N/A'
                print(f"  {status_icon} Link {i}: {item['status']}")
                if item['status'] == 'success':
                    print(f"     Title: {title}")
                    print(f"     Extracted: {item.get('raw_content_length', 0)} chars")
                else:
                    error_msg = item.get('error', 'Unknown')[:60] if item.get('error') else 'Unknown'
                    print(f"     Error: {error_msg}")
        
        # Combined Evidence Store
        print("\n" + "=" * 70)
        print("COMBINED EVIDENCE STORE")
        print("=" * 70)
        
        total_evidence = len(results.get('evidence_store', []))
        print(f"\n📚 Total Evidence Items: {total_evidence}")
        
        if total_evidence > 0:
            print(f"\nEvidence Breakdown:")
            topic_count = len(results.get('topic_results', {}).get('evidence_store', []))
            link_count = len(results.get('link_results', {}).get('evidence_store', []))
            print(f"  • From Topic Research: {topic_count} items")
            print(f"  • From Link Processing: {link_count} items")
            
            print(f"\n📖 Sample Evidence Items:")
            for i, evidence in enumerate(results['evidence_store'][:3], 1):
                print(f"\n[Evidence #{i}]")
                print(f"Source: {evidence['Source'][:80]}...")
                info = evidence['Information']
                if len(info) > 250:
                    print(f"Info: {info[:250]}...")
                else:
                    print(f"Info: {info}")
                print("-" * 70)
            
            if total_evidence > 3:
                print(f"\n... and {total_evidence - 3} more evidence items")
        
        # Save detailed results
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        output_data = {
            'query': results['query'],
            'timestamp': results['timestamp'],
            'statistics': {
                'total_evidence_items': total_evidence,
                'topic_evidence_items': len(results.get('topic_results', {}).get('evidence_store', [])),
                'link_evidence_items': len(results.get('link_results', {}).get('evidence_store', [])),
                'links_processed': results.get('link_results', {}).get('count', 0),
                'links_successful': results.get('link_results', {}).get('success_count', 0),
                'links_failed': results.get('link_results', {}).get('error_count', 0)
            },
            'evidence_store': results['evidence_store']
        }
        
        with open('evidence_store_output.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Full results saved to: evidence_store_output.json")
        print(f"✓ Total evidence items: {total_evidence}")
        print(f"✓ Ready for downstream processing\n")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_multiple_sources())
