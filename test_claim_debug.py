#!/usr/bin/env python3
"""Quick test to debug claim outline generation"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_with_iterative_refinement_and_style


async def debug_claims():
    """Run just the style generation part and check claim outline"""
    
    print("Running pipeline with claim outline enabled...")
    
    sources = {
        "topics": "artificial intelligence machine learning risk prediction",
        "links": ["https://arxiv.org/abs/2106.03072"],
        "attachments": []
    }
    
    results = await process_with_iterative_refinement_and_style(
        query="AI in financial risk prediction",
        sources=sources,
        max_iterations=1,  # Just 1 iteration for speed
        enable_policy_check=False  # Skip policy check
    )
    
    styled = results.get('styled_output', {})
    
    print("\n" + "="*70)
    print("CLAIM OUTLINE DEBUG")
    print("="*70)
    print(f"Success: {styled.get('success')}")
    print(f"claim_outline_used: {styled.get('claim_outline_used')}")
    print(f"claim_count: {styled.get('claim_count')}")
    print(f"Error (if any): {styled.get('error')}")
    
    # Check all keys in styled output
    print(f"\nAll keys in styled_output:")
    for key in sorted(styled.keys()):
        value = styled[key]
        if isinstance(value, (str, int, float, bool, type(None))):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    return results


if __name__ == "__main__":
    asyncio.run(debug_claims())
