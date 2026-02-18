#!/usr/bin/env python3
"""
Test script for plagiarism checker
"""
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from app.plagiarism_checker import check_plagiarism, TextNormalizer
from openai import AsyncAzureOpenAI
from tavily import AsyncTavilyClient


async def test_plagiarism_checker():
    """Test the plagiarism checker with a sample speech"""
    
    print("=" * 70)
    print("PLAGIARISM CHECKER TEST")
    print("=" * 70)
    
    # Sample speech text (using transformer content as example)
    sample_speech = """
Good afternoon, ladies and gentlemen. I am pleased to address you today on the topic 
of transformer neural networks and their revolutionary impact on artificial intelligence.

Transformer neural networks have fundamentally changed the landscape of deep learning 
since their introduction in 2017. The key innovation lies in the self-attention mechanism, 
which allows the model to weigh the importance of different parts of the input sequence 
when making predictions.

The original Transformer architecture, introduced in the paper "Attention Is All You Need" 
by Vaswani and colleagues, achieved remarkable results on machine translation tasks. 
The model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task, 
surpassing previous best results by over 2 BLEU points.

One of the most significant advantages of transformers is their ability to process sequences 
in parallel, unlike recurrent neural networks which must process sequentially. This 
parallelization enables much faster training times while maintaining or exceeding the 
performance of previous architectures.

The encoder-decoder structure with multi-head attention allows the model to capture long-range 
dependencies in the data without the vanishing gradient problems that plague traditional RNNs. 
This architectural innovation has enabled transformers to scale to billions of parameters and 
achieve human-level performance on many natural language tasks.

Today, transformer architectures power most state-of-the-art language models, including GPT, 
BERT, and their countless variants. The impact extends beyond natural language processing 
to computer vision, speech recognition, and even protein structure prediction.

Thank you for your attention, and I look forward to discussing these developments further.
"""
    
    # Initialize clients
    print("\nInitializing clients...")
    azure_client = None
    tavily_client = None
    
    try:
        azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        print("  ✓ Azure OpenAI client initialized")
    except Exception as e:
        print(f"  ⚠️ Azure client initialization failed: {e}")
    
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
            print("  ✓ Tavily client initialized")
        else:
            print("  ⚠️ Tavily API key not found")
    except Exception as e:
        print(f"  ⚠️ Tavily client initialization failed: {e}")
    
    # Prepare metadata
    speech_metadata = {
        "id": "test_speech_001",
        "speaker": "Test Speaker",
        "institution": "Test Institution",
        "date": "2026-02-12"
    }
    
    print(f"\nSpeech metadata:")
    print(f"  Speaker: {speech_metadata['speaker']}")
    print(f"  Institution: {speech_metadata['institution']}")
    print(f"  Date: {speech_metadata['date']}")
    print(f"  Speech length: {len(sample_speech)} characters")
    
    # Test text normalization first
    print(f"\n{'='*70}")
    print("TESTING TEXT NORMALIZATION & CHUNKING")
    print('='*70)
    
    chunks = TextNormalizer.create_chunks(
        sample_speech,
        speech_id=speech_metadata['id'],
        chunk_by="paragraph"
    )
    
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  Chunk {i} ({chunk.chunk_id}):")
        print(f"    Words: {chunk.word_count}")
        print(f"    Chars: {chunk.char_count}")
        print(f"    Text preview: {chunk.text[:100]}...")
    
    if len(chunks) > 3:
        print(f"\n  ... and {len(chunks) - 3} more chunks")
    
    # Run plagiarism check
    print(f"\n{'='*70}")
    print("RUNNING PLAGIARISM ANALYSIS")
    print('='*70)
    print(f"\nThis process includes:")
    print(f"  [0] Text normalization and chunking")
    print(f"  [1] Web search via Tavily")
    print(f"  [2] Source document processing")
    print(f"  [3] Optional HuggingFace classification")
    print(f"  [4] Similarity detection (semantic + lexical)")
    print(f"  [5] AI-powered explanations")
    print(f"  [6] Risk aggregation")
    print(f"\n{'-'*70}\\n")
    
    try:
        result = await check_plagiarism(
            speech_text=sample_speech,
            azure_client=azure_client,
            tavily_client=tavily_client,
            speech_metadata=speech_metadata,
            use_hf_classifier=False  # Set to True if transformers is installed
        )
        
        # Display results
        print(f"\n{'='*70}")
        print("PLAGIARISM ANALYSIS RESULTS")
        print('='*70)
        
        if result.get("success"):
            print(f"\n✓ Analysis completed successfully")
            print(f"\nOVERALL ASSESSMENT:")
            print(f"  Risk Level: {result.get('overall_risk_level').upper()}")
            print(f"  Risk Score: {result.get('overall_risk_score'):.3f} / 1.0")
            
            stats = result.get('statistics', {})
            print(f"\nSTATISTICS:")
            print(f"  Total chunks: {stats.get('total_chunks', 0)}")
            print(f"  Total words: {stats.get('total_words', 0)}")
            print(f"  High risk chunks: {stats.get('high_risk_chunks', 0)}")
            print(f"  Medium risk chunks: {stats.get('medium_risk_chunks', 0)}")
            print(f"  Low risk chunks: {stats.get('low_risk_chunks', 0)}")
            print(f"  Clean chunks: {stats.get('clean_chunks', 0)}")
            
            # Show top sources
            top_sources = result.get('top_sources', [])
            if top_sources:
                print(f"\nTOP MATCHING SOURCES:")
                for i, source in enumerate(top_sources, 1):
                    print(f"\n  {i}. {source.get('title', 'Unknown')}")
                    print(f"     URL: {source.get('url', 'N/A')}")
                    print(f"     Matches: {source.get('match_count', 0)}")
                    print(f"     Max similarity: {source.get('max_similarity', 0):.3f}")
                    if source.get('date'):
                        print(f"     Date: {source.get('date')}")
            else:
                print(f"\nNo matching sources found (or search not available)")
            
            # Show flagged chunks
            flagged = result.get('flagged_chunks', [])
            if flagged:
                print(f"\nFLAGGED CHUNKS ({len(flagged)} total):")
                for i, chunk in enumerate(flagged[:5], 1):
                    print(f"\n  Chunk {i} ({chunk.get('chunk_id')}):")
                    print(f"    Risk: {chunk.get('risk_level').upper()} ({chunk.get('risk_score'):.3f})")
                    print(f"    Text: {chunk.get('text', '')[:150]}...")
                    
                    if chunk.get('is_boilerplate'):
                        print(f"    ℹ️ Flagged as boilerplate (greeting/closing)")
                    
                    top_match = chunk.get('top_match')
                    if top_match:
                        print(f"    Top match:")
                        print(f"      Source: {top_match.get('source_title', 'Unknown')}")
                        print(f"      Semantic similarity: {top_match.get('similarity_semantic', 0):.3f}")
                        print(f"      Lexical similarity: {top_match.get('similarity_lexical', 0):.3f}")
                        print(f"      Match type: {top_match.get('match_type', 'unknown')}")
                    
                    if chunk.get('explanation'):
                        print(f"    Explanation: {chunk.get('explanation')}")
                
                if len(flagged) > 5:
                    print(f"\n  ... and {len(flagged) - 5} more flagged chunks")
            else:
                print(f"\n✓ No high or medium risk chunks detected")
            
            # Save detailed results
            output_file = "plagiarism_analysis_output.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n{'='*70}")
            print(f"✓ Detailed results saved to: {output_file}")
            print('='*70)
            
        else:
            print(f"\n✗ Analysis failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n✗ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_plagiarism_checker())
