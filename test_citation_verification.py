#!/usr/bin/env python3
"""
Test citation verification system
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import verify_styled_citations


async def test_citation_verification():
    """Test the citation verification function with sample data"""
    
    print("="*70)
    print("TESTING CITATION VERIFICATION SYSTEM")
    print("="*70)
    
    # Sample styled output with citations
    styled_output = """Transformer neural networks introduced revolutionary innovations in sequence modeling. The architecture eliminates recurrent connections entirely, relying solely on attention mechanisms [E1]. This design choice enables parallel processing during training, significantly reducing computational time compared to RNNs [E2,E3].

The core innovation is the multi-head attention mechanism, which allows the model to attend to different representation subspaces simultaneously [E4]. Each attention head computes scaled dot-product attention using the formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V [E5]. This formulation ensures stable gradients during training [E6].

Positional encodings are injected into the input embeddings to provide sequence order information, since the model lacks recurrence [E7]. The original paper used sinusoidal functions for this purpose [E8]. These innovations collectively led to superior performance on machine translation benchmarks, outperforming previous state-of-the-art models [E9,E10]."""

    # Sample evidence store that matches some but not all claims
    evidence_store = [
        {
            "id": "E1",
            "claim": "Transformers use only attention mechanisms, no recurrence",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "The Transformer model architecture eschews recurrence and instead relies entirely on an attention mechanism",
            "confidence": 0.98
        },
        {
            "id": "E2",
            "claim": "Parallel processing is enabled by removing recurrence",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "This allows for significantly more parallelization",
            "confidence": 0.95
        },
        {
            "id": "E3",
            "claim": "Training time is reduced compared to recurrent models",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "and can reach a new state of the art in translation quality after being trained for as little as twelve hours",
            "confidence": 0.92
        },
        {
            "id": "E4",
            "claim": "Multi-head attention attends to different representation subspaces",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "Multi-head attention allows the model to jointly attend to information from different representation subspaces",
            "confidence": 0.97
        },
        {
            "id": "E5",
            "claim": "Attention uses scaled dot-product formula",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "We call our particular attention 'Scaled Dot-Product Attention'",
            "confidence": 0.99
        },
        {
            "id": "E6",
            "claim": "Scaling factor ensures stable gradients",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients",
            "confidence": 0.94
        },
        {
            "id": "E7",
            "claim": "Positional encodings provide sequence order information",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position",
            "confidence": 0.98
        },
        {
            "id": "E8",
            "claim": "Sinusoidal functions used for positional encoding",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "In this work, we use sine and cosine functions of different frequencies",
            "confidence": 0.96
        },
        {
            "id": "E9",
            "claim": "Superior performance on translation benchmarks",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "achieves a new single-model state-of-the-art BLEU score of 41.8 on the WMT 2014 English-to-German translation task",
            "confidence": 0.97
        },
        {
            "id": "E10",
            "claim": "Outperforms previous state-of-the-art models",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "quote_span": "improving over the existing best results, including ensembles, by over 2 BLEU",
            "confidence": 0.95
        }
    ]
    
    print(f"\nStyled output length: {len(styled_output)} chars")
    print(f"Evidence store items: {len(evidence_store)}")
    print(f"\nFirst 200 chars of styled output:")
    print(styled_output[:200] + "...")
    
    print(f"\n{'='*70}")
    print("RUNNING VERIFICATION...")
    print(f"{'='*70}")
    
    # Run verification
    result = await verify_styled_citations(
        styled_output=styled_output,
        evidence_store=evidence_store
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nSUMMARY:")
    print(f"  Total segments: {result.get('total_segments')}")
    print(f"  Verified: {result.get('verified_segments')}")
    print(f"  Unverified: {result.get('unverified_segments')}")
    print(f"  Verification rate: {result.get('verification_rate')}")
    
    # Show all segments
    print(f"\nDETAILED SEGMENTS:")
    for segment in result.get("segments", []):
        print(f"\n  Segment {segment['segment_number']}:")
        print(f"    Text: {segment['text'][:100]}{'...' if len(segment['text']) > 100 else ''}")
        print(f"    Citations: {segment['citations']}")
        print(f"    Verified: {segment['verified']}")
        print(f"    Reason: {segment['verification_reason']}")
        
        # Show cited claims
        if segment['cited_claims']:
            print(f"    Cited claims:")
            for claim_info in segment['cited_claims']:
                print(f"      [{claim_info['id']}] {claim_info['claim'][:80]}...")
    
    # Save results to JSON
    output_file = "citation_verification_output.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Display JSON format example
    print(f"\nJSON FORMAT EXAMPLE (First segment):")
    if result.get("segments"):
        first_segment = result["segments"][0]
        print(json.dumps(first_segment, indent=2))
    
    return result


if __name__ == "__main__":
    asyncio.run(test_citation_verification())
