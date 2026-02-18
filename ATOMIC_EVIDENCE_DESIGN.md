# Atomic Evidence Store Design

## Overview
Transformed the evidence collection system from paragraph-based to atomic-claim-based storage with validated citations. This ensures all generated content is information-grounded with traceable sources.

## Core Principle
**Citations are not generated, they're validated** - Every claim in the system has a stable ID (E1, E2, ...) and the LLM can only cite IDs that exist in the evidence store.

---

## Evidence Store Structure

Each evidence item is now an atomic claim with the following fields:

```json
{
  "id": "E17",                          // Stable index for citation
  "claim": "Company X announced Y on 2026-01-12.",  // One factual statement
  "source": "https://example.com/article",  // URL or source identifier
  "source_url": "https://example.com/article",  // Full URL if available
  "source_title": "Article Title",     // Human-readable source name
  "quote_span": "On Jan 12, 2026, Company X announced...",  // Exact excerpt (verbatim)
  "retrieval_context": "link_processing",  // Where it came from
  "confidence": 0.95,                   // Optional verifier score (0.0-1.0)
  "timestamp_accessed": "2026-02-11T01:30:00"  // When retrieved
}
```

### Field Descriptions

- **id**: Stable identifier in format `E{number}`. Used for citations like `[E17]` or `[E3,E17]`
- **claim**: Single factual statement, minimal and self-contained
- **source**: Primary source identifier (URL for links, "Deep Research: Topic" for topics)
- **source_url**: Full URL when available (null for aggregated sources)
- **source_title**: Human-readable source name for display
- **quote_span**: Exact verbatim text from source (20-150 words typical)
- **retrieval_context**: Origin tracker (`deep_research_pipeline`, `link_processing`, `attachment_processing`)
- **confidence**: Optional score from verification pipeline (0.0-1.0)
- **timestamp_accessed**: ISO 8601 timestamp of retrieval

---

## Implementation in Processing Functions

### 1. Topic Processing (Deep Research)

**Function**: `topic_processing(topic, query, evidence_id_start)`

**Process**:
1. Run deep research pipeline to get comprehensive summary
2. Use LLM to extract atomic claims from summary
3. For each claim, extract:
   - The factual statement
   - Verbatim quote span supporting it
   - Confidence score
4. Assign stable IDs starting from `evidence_id_start`
5. Return evidence store with `next_evidence_id` for continuity

**Example Output**:
```python
{
  "id": "E1",
  "claim": "Transformers use self-attention mechanisms instead of recurrence",
  "source": "Deep Research: transformer architecture",
  "quote_span": "The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolution altogether.",
  "confidence": 0.9,
  "retrieval_context": "deep_research_pipeline"
}
```

### 2. Link Processing (Tavily Extraction)

**Function**: `process_links(links, query, evidence_id_start)`

**Process**:
1. Use Tavily to extract raw content from each URL
2. Send content + query to LLM with strict instructions
3. LLM extracts 3-8 most relevant atomic facts
4. For EACH fact, LLM provides:
   - Atomic claim statement
   - EXACT verbatim quote from source (20-150 words)
   - Confidence score
5. Create evidence items with stable IDs
6. Track successful vs failed extractions

**Example Output**:
```python
{
  "id": "E7",
  "claim": "Multi-head attention allows parallel processing of 8 attention heads",
  "source": "https://arxiv.org/abs/1706.03762",
  "source_url": "https://arxiv.org/abs/1706.03762",
  "source_title": "Attention Is All You Need",
  "quote_span": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. We employ h=8 parallel attention heads.",
  "confidence": 0.95,
  "retrieval_context": "link_processing"
}
```

### 3. Evidence ID Continuity

Evidence IDs are sequential across ALL sources within a single processing run:

```python
# Start with E1
next_id = 1

# Topic processing creates E1-E5
topic_result = await topic_processing(topic, query, evidence_id_start=next_id)
next_id = topic_result["next_evidence_id"]  # Now 6

# Link processing creates E6-E12
link_result = await process_links(links, query, evidence_id_start=next_id)
next_id = link_result["next_evidence_id"]  # Now 13

# Final evidence store has E1 through E12
```

---

## Forced Citation Discipline

### Summary Generation with Citations

**Function**: `generate_summary_from_evidence(query, evidence_store)`

**Strict Rules**:
1. Every factual sentence MUST end with `[ENN]` citations
2. Only cite IDs that exist in provided evidence_store
3. Multi-claim sentences can cite multiple: `[E1,E5,E9]`
4. Introductory/transition sentences without facts don't need citations
5. System validates all citations against allowed IDs

**Prompt Structure**:
```
=== ALLOWED EVIDENCE ===
E1: <claim>
Source: <source> (confidence: 0.95)

E2: <claim>
Source: <source> (confidence: 0.88)
...

=== STRICT CITATION RULES ===
1. EVERY factual sentence MUST end with [ENN]
2. You may ONLY cite: E1, E2, E3, ..., E15
3. Do NOT make claims without citation
...
```

**Example Output**:
```
Transformers revolutionized NLP by replacing recurrence with attention mechanisms [E1,E3]. 
The architecture uses multi-head attention with 8 parallel heads for diverse relationship 
capture [E7]. Positional encoding injects sequence information since transformers don't 
process sequentially [E5].
```

### Styled Output with Citation Preservation

**Function**: `generate_styled_output(summary, query, style, max_output_length)`

**Additional Rule**:
```
CRITICAL: Preserve ALL [ENN] citations from the research summary. 
Every factual claim must keep its original citation.
```

The styled output maintains all evidence references while applying the writing style.

---

## Citation Validation

### Automatic Validation

After summary generation, the system:

1. Extracts all citation patterns: `[E\d+(?:,E\d+)*]`
2. Parses individual IDs from each citation
3. Checks each ID against allowed evidence store IDs
4. Reports invalid citations as warnings

**Example**:
```python
allowed_ids = ["E1", "E2", "E3", "E4", "E5"]
summary = "Transformers use attention [E1]. They scale well [E99]."

# Validation finds: E99 is invalid (not in allowed_ids)
invalid_citations = ["E99"]
```

### Citation Coverage Analysis

Track which evidence was actually used:

```python
all_ids = {e['id'] for e in evidence_store}  # {E1, E2, ..., E15}
cited_ids = {extracted from summary}          # {E1, E3, E5, E7, E9}
uncited = all_ids - cited_ids                 # {E2, E4, E6, E8, ...}

coverage = len(cited_ids) / len(all_ids) * 100  # e.g., 60%
```

---

## Benefits

### 1. **Verifiable Claims**
Every statement traces back to exact source quote - no hallucination possible if citations are validated.

### 2. **Audit Trail**
Full provenance: Query → Evidence Extraction → Atomic Claim → Citation → Styled Output

### 3. **Quality Metrics**
- Citation coverage %
- Confidence scores per claim
- Invalid citation detection

### 4. **Iterative Refinement Compatible**
- Evidence accumulates with stable IDs: E1-E5 (iter 1) → E1-E10 (iter 2) → E1-E15 (iter 3)
- Citations remain valid across iterations
- Critiques can reference specific evidence IDs

### 5. **Multi-Source Integration**
- Topics, links, attachments all produce atomic claims
- Unified evidence store with continuous ID sequence
- Source attribution preserved per claim

---

## Testing

### Test File: `test_atomic_evidence.py`

Tests:
1. Atomic claim extraction from both topics and links
2. Evidence store structure validation
3. Summary generation with forced citations
4. Citation validation (allowed vs invalid IDs)
5. Citation coverage analysis

**Run**:
```bash
python test_atomic_evidence.py
```

**Output**:
- Displays extracted atomic claims with IDs
- Shows generated summary with [ENN] citations
- Validates all citations against evidence store
- Reports coverage statistics

---

## Migration Notes

### Old Format (Paragraph-Based)
```python
{
  "Information": "Long paragraph with multiple facts mixed together...",
  "Source": "https://example.com"
}
```

**Problems**:
- Multiple facts per item = ambiguous citations
- No quote spans = can't verify claims
- No stable IDs = citations like [Evidence 3] are positional

### New Format (Atomic Claims)
```python
{
  "id": "E7",
  "claim": "Single factual statement",
  "quote_span": "Exact verbatim text from source",
  "source_url": "https://example.com",
  "confidence": 0.95
}
```

**Improvements**:
- One fact per item = precise citations
- Quote spans enable verification
- Stable IDs = citations remain valid even if evidence store reordered

---

## Future Enhancements

1. **Confidence Threshold Filtering**: Only use claims above 0.8 confidence
2. **Contradiction Detection**: Flag conflicting claims from different sources
3. **Quote Verification**: Hash quote_spans and validate against original sources
4. **Citation Network Analysis**: Track which evidence items are most cited
5. **Source Quality Scoring**: Weight claims by source reliability
6. **Real-time Fact Checking**: API integration for live claim verification

---

## Example End-to-End Flow

```
User Query: "What are key transformer innovations?"

1. TOPIC PROCESSING (Deep Research)
   → Summary: "Transformers use attention mechanisms..."
   → Extract: E1: "Uses self-attention" [confidence: 0.9]
             E2: "Has multi-head architecture" [confidence: 0.85]
             E3: "Employs positional encoding" [confidence: 0.88]

2. LINK PROCESSING (arXiv paper)
   → Extract from URL: E4: "Introduced in 2017 paper" [confidence: 0.95]
                      E5: "Achieved 28.4 BLEU score" [confidence: 0.92]
                      E6: "Uses scaled dot-product attention" [confidence: 0.90]

3. EVIDENCE STORE
   → Combined: [E1, E2, E3, E4, E5, E6] with sources and quotes

4. SUMMARY GENERATION
   → LLM receives: Allowed IDs = E1-E6 with claims
   → Generates: "Transformers, introduced in 2017 [E4], revolutionized 
                 NLP with self-attention [E1] and multi-head architecture [E2].
                 They use scaled dot-product attention [E6] and positional 
                 encoding [E3], achieving 28.4 BLEU [E5]."

5. VALIDATION
   → All citations [E1-E6] valid ✓
   → Coverage: 100% (6/6 evidence items cited)

6. STYLED OUTPUT
   → Applies writing style while preserving [E1-E6] citations
```

---

## Status

✅ Topic processing with atomic extraction
✅ Link processing with atomic extraction  
✅ Evidence ID continuity across sources
✅ Forced citation in summary generation
✅ Citation validation system
✅ Citation preservation in styled output
✅ Test script created

Ready for production use with full citation traceability.
