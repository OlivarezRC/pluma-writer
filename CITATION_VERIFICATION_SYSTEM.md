# Citation Verification System

## Overview

The citation verification system automatically validates that citations in the styled output accurately reflect the evidence claims they reference. This ensures factual grounding and citation integrity in the final output.

## How It Works

### 1. **Segment Parsing**
The system splits the styled output into segments based on citation patterns:
```
Text before citation [E1] → Segment 1
Text before next citation [E2,E3] → Segment 2
...
```

Each segment includes:
- The text content
- Citation IDs (e.g., E1, E2, E3)
- Referenced evidence claims from the evidence store

### 2. **Evidence Lookup**
For each citation, the system:
- Looks up the evidence ID in the evidence store
- Retrieves the atomic claim and quote_span
- Builds a list of cited claims for the segment

### 3. **LLM Verification**
An LLM (GPT-4o) verifies each segment by:
- Comparing the segment text to the cited evidence claims
- Checking for semantic accuracy (not exact word matching)
- Allowing reasonable paraphrasing
- Identifying unsupported or contradicting claims

### 4. **Result Classification**
Each segment is classified as:
- **Verified (Yes)**: Text accurately reflects cited evidence
- **Unverified (No)**: Mismatches, unsupported claims, or inaccuracies detected
- **Error**: Verification process failed (missing evidence, API issues, etc.)

## Output Format

The verification results are returned in JSON format:

```json
{
  "total_segments": 8,
  "verified_segments": 7,
  "unverified_segments": 1,
  "verification_rate": "87.5%",
  "segments": [
    {
      "segment_number": 1,
      "text": "Transformer networks use attention mechanisms...",
      "citations": ["E1", "E2"],
      "cited_claims": [
        {
          "id": "E1",
          "claim": "Transformers use attention mechanisms",
          "quote_span": "The architecture relies on attention..."
        }
      ],
      "verified": "Yes",
      "verification_reason": "Claims match cited evidence"
    }
  ]
}
```

## Integration

### In Complete Pipeline
The verification runs automatically in `process_with_iterative_refinement_and_style()`:

```python
# Step 4: Verify citations in styled output
verification_result = await verify_styled_citations(
    styled_output=styled_result.get("styled_output", ""),
    evidence_store=cumulative_evidence
)
```

Results are automatically saved to `complete_pipeline_output.json` under the `citation_verification` field:

```json
{
  "query": "...",
  "iterations": [...],
  "styled_output": {...},
  "citation_verification": {
    "total_segments": 8,
    "verified_segments": 7,
    "unverified_segments": 1,
    "verification_rate": "87.5%",
    "segments": [...]
  }
}
```

### Standalone Usage
You can verify any text with citations:

```python
from app.writer_main import verify_styled_citations

result = await verify_styled_citations(
    styled_output="Your text with citations [E1]...",
    evidence_store=[
        {"id": "E1", "claim": "...", "quote_span": "..."},
        {"id": "E2", "claim": "...", "quote_span": "..."}
    ]
)

print(f"Verification rate: {result['verification_rate']}")
```

## Example Results

### High Verification Rate (Good)
```
Citation Verification Results:
  • Total segments: 10
  • Verified segments: 10
  • Unverified segments: 0
  • Verification rate: 100.0%
  ✓ All citations verified successfully!
```

### Low Verification Rate (Needs Review)
```
Citation Verification Results:
  • Total segments: 10
  • Verified segments: 6
  • Unverified segments: 4
  • Verification rate: 60.0%
  
  ⚠️ Sample unverified segments:
    Segment 3: Claim contradicts cited evidence about training time...
    Segment 7: Unsupported generalization beyond cited evidence...
```

## Technical Details

### Model Used
- **GPT-4o** (Azure OpenAI deployment)
- GPT-5 returns empty responses for this task, so GPT-4o is hardcoded
- Uses default temperature (1.0)
- Max completion tokens: 200

### Citation Pattern
- Regex: `\[E\d+(?:,E\d+)*\]`
- Matches: `[E1]`, `[E1,E2]`, `[E10,E25,E30]`
- Invalid: `[1]`, `[E1 E2]`, `(E1)`

### Verification Criteria
The LLM checks:
1. **Semantic accuracy**: Does the text match the evidence meaning?
2. **Paraphrasing tolerance**: Reasonable rewording is acceptable
3. **No unsupported claims**: Text shouldn't make claims beyond cited evidence
4. **No contradictions**: Text shouldn't contradict the evidence

## Test Files

### test_citation_verification.py
Standalone test with sample data:
```bash
python test_citation_verification.py
```

Output:
- Console summary with verification rates
- `citation_verification_output.json` with full results (for debugging only)
- Debug logging for first 3 segments

**Note**: This standalone test saves to `citation_verification_output.json` for debugging purposes. The complete pipeline saves verification results to `complete_pipeline_output.json` under the `citation_verification` field.

### test_complete_pipeline.py
Includes verification in full pipeline:
```bash
python test_complete_pipeline.py
```

Displays:
- Verification results in final summary
- Sample unverified segments if any
- Overall verification rate

Results saved to: `complete_pipeline_output.json` (includes full verification data)

## Benefits

1. **Quality Assurance**: Ensures citations are accurate and justified
2. **Fact-Checking**: Catches hallucinations or unsupported claims
3. **Citation Integrity**: Verifies the styled output maintains factual grounding
4. **Transparency**: Provides detailed breakdown of which claims are verified
5. **Debugging**: Identifies problematic segments for review

## Limitations

1. **LLM Dependency**: Verification quality depends on GPT-4o capabilities
2. **Semantic Judgment**: Some edge cases may be subjective
3. **Performance**: Each segment requires an API call (can be slow for large outputs)
4. **Cost**: Additional API calls for verification

## Future Enhancements

Potential improvements:
- [ ] Batch verification requests for efficiency
- [ ] Confidence scores instead of binary Yes/No
- [ ] Support for multiple verification models
- [ ] Citation correction suggestions
- [ ] Integration with human review workflow
- [ ] Caching of verification results
