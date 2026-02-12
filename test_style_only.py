#!/usr/bin/env python3
"""
Simple test for style retrieval and styled output generation
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import get_random_style_from_db, generate_styled_output


async def test_style_functions():
    """Test style retrieval and generation separately"""
    
    print("=" * 70)
    print("TESTING STYLE FUNCTIONS")
    print("=" * 70)
    
    # Test 1: Get random style
    print("\n1. Testing style retrieval from Cosmos DB...")
    style = get_random_style_from_db()
    
    if style:
        print(f"✓ Style retrieved successfully")
        print(f"  • Name: {style.get('name', 'Unknown')}")
        print(f"  • Speaker: {style.get('speaker', 'Unknown')}")
        print(f"  • Audience: {style.get('audience_setting_classification', 'Unknown')}")
        print(f"  • Has style description: {'Yes' if style.get('style_description') else 'No'}")
        print(f"  • Has global rules: {'Yes' if style.get('global_rules') else 'No'}")
        print(f"  • Has guidelines: {'Yes' if style.get('guidelines') else 'No'}")
        print(f"  • Has example: {'Yes' if style.get('example') else 'No'}")
        print(f"\n  Available keys in returned style:")
        for key in sorted(style.keys()):
            value = style[key]
            if isinstance(value, str) and len(value) > 100:
                print(f"    - {key}: <string, {len(value)} chars>")
            elif isinstance(value, (dict, list)):
                elem_count = len(value) if hasattr(value, '__len__') else '?'
                print(f"    - {key}: <{type(value).__name__}, {elem_count} elements>")
            else:
                print(f"    - {key}: {value}")
    else:
        print("✗ No style found in database")
        return
    
    # Test 2: Generate styled output with sample summary
    print("\n2. Testing styled output generation...")
    
    sample_summary = """**Summary of Transformer Neural Networks**

Transformers are a revolutionary neural network architecture introduced in 2017 that rely entirely on attention mechanisms. Key innovations include:

1. **Self-Attention Mechanism**: Allows the model to weigh the importance of different words in a sequence when processing each word.

2. **Multi-Head Attention**: Uses multiple attention layers in parallel to capture different types of relationships in the data.

3. **Positional Encoding**: Injects information about the position of words in the sequence since transformers don't have inherent sequential processing.

4. **Scalability**: Can be parallelized effectively, making training much faster than recurrent neural networks.

The transformer architecture has become the foundation for modern language models like BERT, GPT, and many others."""

    query = "What are transformer neural networks?"
    
    result = await generate_styled_output(sample_summary, query, style)
    
    if result.get('success'):
        print(f"✓ Styled output generated successfully")
        print(f"  • Style applied: {result.get('style_name', 'Unknown')}")
        print(f"  • Model used: {result.get('model_used', 'Unknown')}")
        print(f"  • Output length: {len(result.get('styled_output', ''))} characters")
        
        print(f"\n{'='*70}")
        print("STYLED OUTPUT")
        print('='*70)
        print(result.get('styled_output', ''))
        print('='*70)
    else:
        print(f"✗ Style generation failed")
        print(f"  • Error: {result.get('error', 'Unknown')}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_style_functions())
