#!/usr/bin/env python3
"""
Test script for process_links functionality in writer_main.py
"""
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the functions
from app.writer_main import process_user_input


async def test_links_processing():
    """Test with links and a query"""
    
    print("=" * 60)
    print("Testing LINKS PROCESSING with Tavily + LLM")
    print("=" * 60)
    
    # Define test inputs
    user_query = "What are the key features of transformer architectures?"
    
    user_sources = {
        "topics": "",  # No topic
        "links": [
            "https://en.wikipedia.org/wiki/Transformer_(deep_learning_model)",
            "https://arxiv.org/abs/1706.03762"
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Links to process: {len(user_sources['links'])}")
    for i, link in enumerate(user_sources['links'], 1):
        print(f"  {i}. {link}")
    
    print("\nProcessing links...\n")
    
    try:
        # Process the input
        results = await process_user_input(user_query, user_sources)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        # Display results
        print(f"\nQuery: {results['query']}")
        print(f"Timestamp: {results['timestamp']}")
        
        if results['link_results']:
            print("\n--- LINK PROCESSING RESULTS ---")
            link_res = results['link_results']
            
            print(f"Total Links: {link_res['count']}")
            print(f"Successfully Processed: {link_res.get('success_count', 0)}")
            print(f"Errors: {link_res.get('error_count', 0)}")
            
            if 'error' in link_res:
                print(f"\n✗ Overall Error: {link_res['error']}")
            
            print("\n--- Individual Link Results ---")
            for i, item in enumerate(link_res['items'], 1):
                print(f"\n{i}. URL: {item['url']}")
                print(f"   Status: {item['status']}")
                
                if item['status'] == 'success':
                    print(f"   Title: {item.get('title', 'N/A')}")
                    print(f"   Content Length: {item.get('raw_content_length', 0)} characters")
                    print(f"\n   Extracted Information:")
                    print("-" * 60)
                    print(item.get('extracted_content', 'No content'))
                    print("-" * 60)
                elif item['status'] == 'error':
                    print(f"   Error: {item.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_combined_topic_and_links():
    """Test with both topic and links"""
    
    print("\n\n" + "=" * 60)
    print("Testing COMBINED: TOPIC + LINKS")
    print("=" * 60)
    
    user_query = "Explain attention mechanisms in transformers"
    
    user_sources = {
        "topics": "attention mechanism transformers",
        "links": [
            "https://en.wikipedia.org/wiki/Attention_(machine_learning)"
        ],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Links: {user_sources['links']}")
    
    print("\nProcessing...\n")
    
    try:
        results = await process_user_input(user_query, user_sources)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        if results['topic_results']:
            print("\n--- TOPIC RESEARCH ---")
            if results['topic_results'].get('success'):
                print("✓ Deep research completed")
                summary = results['topic_results'].get('summary', 'No summary')
                print(f"Summary length: {len(summary)} characters")
            else:
                print(f"✗ Error: {results['topic_results'].get('error')}")
        
        if results['link_results']:
            print("\n--- LINK PROCESSING ---")
            print(f"Processed: {results['link_results']['count']} links")
            print(f"Success: {results['link_results'].get('success_count', 0)}")
        
        print("\n" + "=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run links-only test
    asyncio.run(test_links_processing())
    
    # Uncomment to test combined topic + links
    # asyncio.run(test_combined_topic_and_links())
