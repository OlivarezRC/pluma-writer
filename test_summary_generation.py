#!/usr/bin/env python3
"""
Test script demonstrating LLM-generated summary from evidence sources
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_user_input


async def test_summary_generation():
    """Test the complete workflow including summary generation"""
    
    print("=" * 70)
    print("TESTING: LLM SUMMARY GENERATION FROM EVIDENCE")
    print("=" * 70)
    
    user_query = "What are the key innovations in transformer neural networks?"
    
    user_sources = {
        "topics": "transformer architecture innovations",
        "links": [
            "https://arxiv.org/abs/1706.03762",  # Attention is All You Need
            "https://arxiv.org/abs/2004.05150",  # Longformer
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Links: {len(user_sources['links'])} research papers")
    
    print("\n" + "=" * 70)
    print("STEP 1: Collecting Evidence")
    print("=" * 70)
    print("Processing topic research and links...")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        # Show evidence collection results
        print(f"\n✓ Evidence Collection Complete")
        print(f"  • Total Evidence Items: {len(results.get('evidence_store', []))}")
        
        if results.get('topic_results'):
            tr = results['topic_results']
            if tr.get('success'):
                print(f"  • From Topic Research: {len(tr.get('evidence_store', []))} items")
        
        if results.get('link_results'):
            lr = results['link_results']
            print(f"  • From Links: {lr.get('success_count', 0)}/{lr.get('count', 0)} successful")
        
        # Show evidence sources
        print(f"\n📚 Evidence Sources:")
        for i, evidence in enumerate(results['evidence_store'], 1):
            source = evidence['Source']
            if len(source) > 80:
                source = source[:77] + "..."
            print(f"  {i}. {source}")
        
        print("\n" + "=" * 70)
        print("STEP 2: Generating Comprehensive Summary")
        print("=" * 70)
        
        # Show generated summary
        if results.get('generated_summary'):
            summary_result = results['generated_summary']
            
            if summary_result.get('success'):
                print(f"\n✓ Summary Generated Successfully")
                print(f"  • Based on {summary_result.get('evidence_count', 0)} evidence items")
                print(f"  • Query: {summary_result.get('query', '')}")
                
                print("\n" + "=" * 70)
                print("GENERATED SUMMARY")
                print("=" * 70)
                print(f"\n{summary_result['summary']}\n")
                print("=" * 70)
                
            else:
                print(f"\n✗ Summary Generation Failed")
                print(f"  Error: {summary_result.get('error', 'Unknown error')}")
        else:
            print("\n✗ No summary generated")
        
        # Save complete results
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        output_data = {
            'query': results['query'],
            'timestamp': results['timestamp'],
            'evidence_count': len(results.get('evidence_store', [])),
            'generated_summary': results.get('generated_summary', {}),
            'evidence_store': results['evidence_store']
        }
        
        with open('summary_generation_output.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Complete results saved to: summary_generation_output.json")
        print(f"✓ Evidence items: {len(results.get('evidence_store', []))}")
        print(f"✓ Summary length: {len(results.get('generated_summary', {}).get('summary', ''))} characters")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_summary_generation())
