#!/usr/bin/env python3
"""
Test script for writer_main.py with only topic source
"""
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the main function
from app.writer_main import process_user_input


async def test_topic_only():
    """Test with only a topic, no links or attachments"""
    
    print("=" * 60)
    print("Testing writer_main.py with TOPIC ONLY")
    print("=" * 60)
    
    # Define test inputs
    user_query = "What are transformer neural networks?"
    
    user_sources = {
        "topics": "transformer neural networks architecture",
        "links": [],
        "attachments": []
    }
    
    print(f"\nUser Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print("\nStarting deep research...\n")
    
    try:
        # Process the input
        results = await process_user_input(user_query, user_sources)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        # Display results in a readable format
        print(f"\nQuery: {results['query']}")
        print(f"Timestamp: {results['timestamp']}")
        
        if results['topic_results']:
            print("\n--- TOPIC PROCESSING RESULTS ---")
            topic_res = results['topic_results']
            
            if topic_res['success']:
                print(f"✓ Success!")
                print(f"Topic: {topic_res['topic']}")
                print(f"\nSummary:\n{topic_res['summary']}")
            else:
                print(f"✗ Error: {topic_res['error']}")
        
        if results['link_results']:
            print("\n--- LINK RESULTS ---")
            print(json.dumps(results['link_results'], indent=2))
        
        if results['attachment_results']:
            print("\n--- ATTACHMENT RESULTS ---")
            print(json.dumps(results['attachment_results'], indent=2))
        
        print("\n" + "=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_topic_only())
