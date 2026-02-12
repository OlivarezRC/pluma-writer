#!/usr/bin/env python3
"""
Test APA citation conversion
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import convert_styled_output_to_apa


async def test_apa_conversion():
    """Test the APA conversion function"""
    
    print("="*70)
    print("TESTING APA CITATION CONVERSION")
    print("="*70)
    
    # Sample styled output with [ENN] citations
    styled_output = """Transformer neural networks introduced revolutionary innovations in sequence modeling [E1]. The architecture eliminates recurrent connections entirely, relying solely on attention mechanisms [E2,E3]. This design choice enables parallel processing during training, significantly reducing computational time compared to RNNs [E4].

The core innovation is the multi-head attention mechanism, which allows the model to attend to different representation subspaces simultaneously [E5]. Each attention head computes scaled dot-product attention using the established formula [E6]. This formulation ensures stable gradients during training [E7].

Positional encodings are injected into the input embeddings to provide sequence order information, since the model lacks recurrence [E8]. The original paper used sinusoidal functions for this purpose [E9]. These innovations collectively led to superior performance on machine translation benchmarks, outperforming previous state-of-the-art models [E10]."""

    # Sample evidence store
    evidence_store = [
        {
            "id": "E1",
            "claim": "Transformers introduced revolutionary innovations in sequence modeling",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "The Transformer model architecture eschews recurrence and instead relies entirely on an attention mechanism"
        },
        {
            "id": "E2",
            "claim": "Architecture eliminates recurrent connections",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "eliminates recurrent connections"
        },
        {
            "id": "E3",
            "claim": "Relies solely on attention mechanisms",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "relies entirely on an attention mechanism"
        },
        {
            "id": "E4",
            "claim": "Enables parallel processing during training",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "This allows for significantly more parallelization"
        },
        {
            "id": "E5",
            "claim": "Multi-head attention attends to different representation subspaces",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "Multi-head attention allows the model to jointly attend to information from different representation subspaces"
        },
        {
            "id": "E6",
            "claim": "Uses scaled dot-product attention formula",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "We call our particular attention 'Scaled Dot-Product Attention'"
        },
        {
            "id": "E7",
            "claim": "Ensures stable gradients during training",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients"
        },
        {
            "id": "E8",
            "claim": "Positional encodings provide sequence order information",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position"
        },
        {
            "id": "E9",
            "claim": "Used sinusoidal functions for positional encoding",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "In this work, we use sine and cosine functions of different frequencies"
        },
        {
            "id": "E10",
            "claim": "Superior performance on machine translation benchmarks",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "achieves a new single-model state-of-the-art BLEU score"
        }
    ]
    
    print(f"\nOriginal styled output length: {len(styled_output)} chars")
    print(f"Evidence store items: {len(evidence_store)}")
    print(f"[ENN] citations in original: {styled_output.count('[E')}")
    
    print(f"\n{'='*70}")
    print("RUNNING APA CONVERSION...")
    print(f"{'='*70}")
    
    # Run conversion
    result = await convert_styled_output_to_apa(
        styled_output=styled_output,
        evidence_store=evidence_store
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("APA CONVERSION RESULTS")
    print(f"{'='*70}")
    
    if result.get("success"):
        print(f"\n✓ Conversion successful!")
        print(f"  Citations converted: {result.get('citations_converted')}")
        print(f"  References generated: {result.get('references_generated')}")
        print(f"  Output length: {len(result.get('apa_output', ''))} chars")
        
        apa_output = result.get('apa_output', '')
        
        # Show sample of converted text
        print(f"\nFirst 500 characters of APA output:")
        print("-" * 70)
        print(apa_output[:500])
        print("...\n")
        
        # Show references section
        if "REFERENCES" in apa_output:
            refs_section = apa_output.split("REFERENCES")[1]
            print(f"References section (first 500 chars):")
            print("-" * 70)
            print("REFERENCES" + refs_section[:500])
            print("...")
        
        # Save to file
        with open('test_apa_output.txt', 'w') as f:
            f.write(apa_output)
        print(f"\n✓ Full APA output saved to: test_apa_output.txt")
        
    else:
        print(f"\n✗ Conversion failed: {result.get('error')}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_apa_conversion())
