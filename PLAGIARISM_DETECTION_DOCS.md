# Plagiarism Detection System Documentation

## Overview

The plagiarism detection system provides comprehensive analysis of speech documents to identify potential similarity with existing web content. It uses a multi-stage approach combining web search, semantic similarity, lexical analysis, and AI-powered explanations.

## Architecture

### Components

1. **TextNormalizer** - Text preprocessing and chunking
2. **SearchQueryGenerator** - Query generation from speech chunks  
3. **SimilarityDetector** - Semantic and lexical similarity computation
4. **PlagiarismClassifier** - Match type classification and risk scoring
5. **PlagiarismChecker** - Main orchestrator for the detection pipeline

## Pipeline Workflow

### Step 0: Speech Ingestion

**Input:** Raw speech text + optional metadata

**Process:**
- Remove headers/footers and normalize whitespace
- Segment into paragraphs or sentences
- Assign stable chunk IDs (`speech_id_chunk_NNN`)
- Store chunk metadata (word count, character count, indices)

**Output:** List of `SpeechChunk` objects

### Step 1: Web Search (Tavily)

**Input:** Speech chunks

**Process:**
- Generate 1-2 search queries per chunk using:
  - Keyword extraction (top 5 keywords)
  - Entity extraction (proper nouns, numbers, dates)
  - Key phrase extraction (3-5 word sequences)
- Execute Tavily searches targeting:
  - Central bank speeches and statements
  - News commentary (Bloomberg, Reuters)
  - Official transcripts and Q&A sessions
- Deduplicate results by URL

**Output:** Corpus of candidate source documents

### Step 2: Source Processing

**Input:** Raw search results

**Process:**
- Extract content from each source
- Chunk source documents (paragraphs)
- Normalize text for fair comparison
- Attach source metadata (URL, title, date)

**Output:** List of source chunks with metadata

### Step 3: Classification (Optional)

**Input:** Speech chunks + source chunks

**Process:**
- Use HuggingFace zero-shot classifier (`facebook/bart-large-mnli`)
- Apply labels: `monetary_policy`, `economic_outlook`, `financial_stability`, etc.
- Enable topic-based matching (compare only same-topic chunks)
- Weight credibility (policy signals > procedural text)

**Output:** Classification labels for each chunk

### Step 4: Plagiarism Detection

**Two-stage matching:**

#### Stage 1: Candidate Retrieval (Fast)
- Compute embeddings using Azure OpenAI (`text-embedding-3-small`)
- Calculate cosine similarity between speech and source chunks
- Select top-k candidates (threshold: 0.5) per speech chunk

**Fallback:** If embeddings unavailable, use Jaccard similarity (threshold: 0.3)

#### Stage 2: Verification (Precise)
- Compute lexical metrics:
  - **Jaccard similarity:** Word-level overlap
  - **N-gram overlap:** Trigram and 5-gram matching
  - **Containment score:** % of suspect n-grams in source (asymmetric)
- Classify match type:
  - `direct_quote`: High lexical (>0.7) + high containment (>0.6)
  - `close_paraphrase`: High semantic (>0.8) + moderate lexical (>0.4)
  - `common_phrase`: Moderate similarities (>0.6 or >0.5)
  - `unique`: Below similarity thresholds
- Detect boilerplate patterns (greetings, standard closings)

**Output:** `SourceMatch` objects with similarity scores and match types

### Step 5: Explanation Generation

**Input:** Flagged chunks (medium/high risk)

**Process:**
- Use Azure OpenAI (GPT-4o) to analyze matches
- Generate 1-2 sentence explanations:
  - Direct copying
  - Paraphrasing
  - Common phrasing (acceptable)
  - Coincidental similarity

**Output:** Human-readable explanations per flagged chunk

### Step 6: Aggregation

**Per-chunk risk scoring:**
```python
risk_score = (
    base_score(match_type) * 0.5 +
    similarity_score * 0.3 +
    length_weight * 0.2
) * boilerplate_discount(0.3 if boilerplate else 1.0)
```

**Risk levels:**
- `high`: score > 0.7
- `medium`: score > 0.4
- `low`: score > 0.15
- `none`: score ≤ 0.15

**Overall risk:**
- Weighted by chunk word count
- Considers distribution of risk levels
- Flags speeches with >20% high-risk chunks as overall high risk

**Output:** Comprehensive plagiarism report

## Data Models

### SpeechChunk
```python
@dataclass
class SpeechChunk:
    chunk_id: str              # "speech_id_chunk_001"
    speech_id: str             # Unique speech identifier
    text: str                  # Original text
    paragraph_index: int       # Paragraph number
    sentence_indices: List[int] # Sentence numbers (if chunked by sentence)
    normalized_text: str       # Lowercased, cleaned
    word_count: int
    char_count: int
```

### SourceMatch
```python
@dataclass
class SourceMatch:
    source_url: str
    source_title: str
    source_date: Optional[str]
    source_speaker: Optional[str]
    source_chunk: str          # Matched text from source
    similarity_semantic: float # 0-1 cosine similarity
    similarity_lexical: float  # 0-1 combined lexical score
    overlap_ratio: float       # Containment score
    match_type: str           # "direct_quote" | "close_paraphrase" | ...
    classification_label: Optional[str]
    evidence_snippets: List[str]
```

### Analysis Report
```python
{
    "success": bool,
    "timestamp": str,
    "speech_metadata": {},
    "overall_risk_score": float,      # 0-1
    "overall_risk_level": str,        # "high" | "medium" | "low" | "minimal"
    "statistics": {
        "total_chunks": int,
        "total_words": int,
        "high_risk_chunks": int,
        "medium_risk_chunks": int,
        "low_risk_chunks": int,
        "clean_chunks": int
    },
    "top_sources": [
        {
            "url": str,
            "title": str,
            "date": str,
            "match_count": int,
            "max_similarity": float
        }
    ],
    "flagged_chunks": [
        {
            "chunk_id": str,
            "text": str,
            "risk_score": float,
            "risk_level": str,
            "is_boilerplate": bool,
            "top_match": {},
            "explanation": str
        }
    ],
    "all_chunk_analyses": [...]
}
```

## Integration

### In Pipeline

The plagiarism checker runs as **Step 6** in the complete pipeline, after:
1. Iterative refinement
2. Retrieving writing style
3. Generating styled output
4. Citation verification
5. APA format conversion

### Usage Example

```python
from app.plagiarism_checker import check_plagiarism
from openai import AsyncAzureOpenAI
from tavily import AsyncTavilyClient

# Initialize clients
azure_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

tavily_client = AsyncTavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Run analysis
result = await check_plagiarism(
    speech_text=my_speech,
    azure_client=azure_client,
    tavily_client=tavily_client,
    speech_metadata={
        "id": "speech_001",
        "speaker": "John Doe",
        "institution": "Central Bank",
        "date": "2026-02-12"
    },
    use_hf_classifier=False
)

# Access results
print(f"Risk Level: {result['overall_risk_level']}")
print(f"Risk Score: {result['overall_risk_score']:.3f}")
```

## Configuration

### Environment Variables

```bash
# Required
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
TAVILY_API_KEY=your_key

# Optional
TRANSFORMERS_CACHE=./cache  # For HuggingFace models
```

### Parameters

#### TextNormalizer.create_chunks
- `chunk_by`: `"paragraph"` (default) or `"sentence"`
- Paragraph chunking recommended for speeches (better context)

#### SearchQueryGenerator.generate_search_queries
- `num_queries`: 1-3 queries per chunk (default: 2)
- More queries = more coverage, but slower

#### PlagiarismChecker._detect_matches
- `top_k`: Number of matches to keep per chunk (default: 5)
- Semantic similarity threshold: 0.5
- Lexical similarity threshold: 0.3

#### Risk Scoring Thresholds
- High risk: > 0.7
- Medium risk: > 0.4
- Low risk: > 0.15
- Boilerplate discount: 0.3x

## Testing

### Unit Test

```bash
python test_plagiarism_checker.py
```

Tests:
- Text normalization and chunking
- Query generation
- Similarity detection
- Full pipeline with sample speech

### Integration Test

```bash
python test_complete_pipeline.py
```

Runs full pipeline including plagiarism detection (Step 6)

## Performance Considerations

### Scalability

**Chunking:**
- Paragraph-level: ~5-10 chunks per typical speech
- Sentence-level: ~30-50 chunks (slower but more granular)

**Search:**
- Samples first 10 chunks for query generation
- 2 queries per chunk = 20 Tavily searches
- Time: ~30-60 seconds

**Embeddings:**
- Batches of 16 chunks
- ~100 embeddings total (speech + sources)
- Time: ~5-10 seconds

**Total runtime:** ~60-90 seconds per speech

### Optimization Tips

1. **Reduce search scope:** Sample fewer chunks (3-5 instead of 10)
2. **Cache embeddings:** Store in database for repeated analysis
3. **Disable explanations:** Skip GPT-4o calls for medium-risk chunks
4. **Use sentence chunking selectively:** Only for suspected plagiarism areas
5. **Adjust thresholds:** Raise similarity thresholds to reduce false positives

## Limitations

1. **Language:** English only (due to stopwords, boilerplate patterns)
2. **Domain:** Optimized for central bank speeches
3. **Search coverage:** Limited to publicly indexed web content
4. **Paraphrase detection:** May miss heavily reworded content
5. **Common knowledge:** May flag standard economic terminology

## Future Enhancements

### Planned
- [ ] Cross-encoder for improved paraphrase detection
- [ ] Multi-language support
- [ ] Custom boilerplate templates per institution
- [ ] Historical analysis (track similarity over time)
- [ ] Configurable risk thresholds per organization

### Possible
- [ ] Graph-based citation analysis
- [ ] Speaker attribution (who originated this phrasing?)
- [ ] Policy path scoring integration
- [ ] Real-time monitoring (alert on new similar content)

## Troubleshooting

### Common Issues

**No search results:**
- Verify `TAVILY_API_KEY` is set
- Check query generation (may be too specific)
- Try broader keywords

**Low similarity scores despite copying:**
- Text preprocessing may over-normalize
- Try sentence-level chunking
- Check if source is in search results

**Many false positives:**
- Increase similarity thresholds
- Enable HF classifier for topic filtering
- Add domain-specific boilerplate patterns

**Slow performance:**
- Reduce `num_queries`
- Limit chunk sampling
- Disable explanations for medium-risk chunks

## Support

For issues or questions:
- Check test outputs: `plagiarism_analysis_output.json`
- Review chunk analysis: `all_chunk_analyses` field
- Enable debug logging in `PlagiarismChecker`

## License

Part of the Pluma Writer system.
