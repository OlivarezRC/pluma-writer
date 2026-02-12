#!/usr/bin/env python3
"""
Test script for evidence_store functionality in writer_main.py
"""
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the functions
from app.writer_main import process_user_input


async def test_evidence_store_links_only():
    """Test evidence store with links only"""
    
    print("=" * 70)
    print("TEST 1: EVIDENCE STORE - LINKS ONLY")
    print("=" * 70)
    
    user_query = "What are transformers in machine learning?"
    
    user_sources = {
        "topics": "",
        "links": [
            "https://arxiv.org/abs/1706.03762"
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Links: {user_sources['links']}")
    print("\nProcessing...\n")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        print("=" * 70)
        print("EVIDENCE STORE RESULTS")
        print("=" * 70)
        
        if results.get('evidence_store'):
            print(f"\n✓ Total Evidence Items: {len(results['evidence_store'])}\n")
            
            for i, evidence in enumerate(results['evidence_store'], 1):
                print(f"--- Evidence #{i} ---")
                print(f"Source: {evidence.get('Source', 'N/A')}")
                print(f"\nInformation:")
                print(evidence.get('Information', 'N/A')[:300] + "..." if len(evidence.get('Information', '')) > 300 else evidence.get('Information', 'N/A'))
                print()
        else:
            print("✗ No evidence store found")
        
        # Show link results summary
        if results.get('link_results'):
            print("\n--- Link Processing Summary ---")
            print(f"Success: {results['link_results'].get('success_count', 0)}")
            print(f"Errors: {results['link_results'].get('error_count', 0)}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_evidence_store_combined():
    """Test evidence store with both topic and links"""
    
    print("\n\n" + "=" * 70)
    print("TEST 2: EVIDENCE STORE - TOPIC + LINKS COMBINED")
    print("=" * 70)
    
    user_query = "Explain attention mechanisms in deep learning"
    
    user_sources = {
        "topics": "attention mechanism neural networks",
        "links": [
            "https://arxiv.org/abs/1706.03762"
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Links: {user_sources['links']}")
    print("\nProcessing...\n")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        print("=" * 70)
        print("COMBINED EVIDENCE STORE")
        print("=" * 70)
        
        if results.get('evidence_store'):
            print(f"\n✓ Total Evidence Items: {len(results['evidence_store'])}\n")
            
            # Separate by source type
            topic_evidence = [e for e in results['evidence_store'] if 'Deep Research' in e.get('Source', '')]
            link_evidence = [e for e in results['evidence_store'] if 'http' in e.get('Source', '')]
            
            print(f"From Topic Research: {len(topic_evidence)} items")
            print(f"From Link Processing: {len(link_evidence)} items")
            
            print("\n--- TOPIC RESEARCH EVIDENCE ---")
            for i, evidence in enumerate(topic_evidence[:3], 1):  # Show first 3
                print(f"\n{i}. Source: {evidence.get('Source', 'N/A')}")
                info = evidence.get('Information', 'N/A')
                print(f"   Info: {info[:200]}..." if len(info) > 200 else f"   Info: {info}")
            
            print("\n--- LINK PROCESSING EVIDENCE ---")
            for i, evidence in enumerate(link_evidence[:3], 1):  # Show first 3
                print(f"\n{i}. Source: {evidence.get('Source', 'N/A')[:100]}...")
                info = evidence.get('Information', 'N/A')
                print(f"   Info: {info[:200]}..." if len(info) > 200 else f"   Info: {info}")
        else:
            print("✗ No evidence store found")
        
        # Show processing summary
        print("\n--- PROCESSING SUMMARY ---")
        if results.get('topic_results'):
            status = "✓ Success" if results['topic_results'].get('success') else f"✗ Error: {results['topic_results'].get('error')}"
            print(f"Topic Processing: {status}")
            if results['topic_results'].get('evidence_store'):
                print(f"  Evidence Items: {len(results['topic_results']['evidence_store'])}")
        
        if results.get('link_results'):
            print(f"Link Processing: {results['link_results'].get('success_count', 0)} success, {results['link_results'].get('error_count', 0)} errors")
            if results['link_results'].get('evidence_store'):
                print(f"  Evidence Items: {len(results['link_results']['evidence_store'])}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_evidence_store_structure():
    """Test and display the exact structure of evidence_store"""
    
    print("\n\n" + "=" * 70)
    print("TEST 3: EVIDENCE STORE STRUCTURE (JSON Format)")
    print("=" * 70)
    
    user_query = "What is deep learning?"
    
    user_sources = {
        "topics": "",
        "links": [
            "https://arxiv.org/abs/1706.03762"
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print("\nProcessing...\n")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        if results.get('evidence_store'):
            print("Evidence Store Structure:")
            print("-" * 70)
            print(json.dumps({
                "evidence_store": results['evidence_store'][:2]  # Show first 2 items
            }, indent=2))
            print("-" * 70)
            print(f"\n... and {len(results['evidence_store']) - 2} more items" if len(results['evidence_store']) > 2 else "")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run all tests
    asyncio.run(test_evidence_store_links_only())
    
    # Uncomment to run additional tests
    # asyncio.run(test_evidence_store_combined())
    # asyncio.run(test_evidence_store_structure())
