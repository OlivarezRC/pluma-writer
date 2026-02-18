# Pipeline Architecture Improvements

## Implementation Summary

This document outlines the 6 major architectural improvements to the speech generation pipeline.

## 1. Style Drift Prevention - Claim Outline System

**Problem**: Style model adds new factual claims not in evidence (SHAP/LIME, comparative statistics).

**Solution**: Introduce "Claim Outline" generation step before styling.

### Implementation:
- **New Function**: `generate_claim_outline(summary, evidence_store)` → structured claim plan
- **Each claim has**:
  - Claim text (factual assertion)
  - Evidence IDs (1-3 max)
  - Claim type (factual/transition/interpretation)
- **Style model updated**:
  - Receives claim outline, not freeform summary
  - Prompt: "Rephrase ONLY these claims. No new methods, numbers, or entities."
  - Allowed transitions: "This suggests...", "A key implication is..."

### Files Changed:
- `app/writer_main.py`: New `generate_claim_outline()` function
- `app/writer_main.py`: Updated `generate_styled_output()` to use claim outline

---

## 2. Cumulative Evidence Store

**Problem**: Evidence is unstable across iterations - sources drop claims, parse errors lose data.

**Solution**: Idempotent, cumulative evidence store with deduplication.

### Implementation:
- **Store keyed by**: `(url, claim_hash)` for deduplication
- **Across iterations**:
  - Add new evidence items
  - Never drop valid items unless explicitly marked as deprecated
  - If source fails to parse in iteration N, keep last good extraction from N-1
- **Evidence metadata**:
  - `iteration_discovered`: Which iteration this was first seen
  - `last_verified_iteration`: Last iteration where it was validated
  - `parse_status`: "valid", "failed", "deprecated"

### Files Changed:
- `app/writer_main.py`: Updated `process_with_iterative_refinement()` to deduplicate
- New utility: `deduplicate_evidence_store(evidence_list)` → keyed dict

---

## 3. Citation Discipline - Max 1-2 IDs per Sentence

**Problem**: Citation spraying (multiple IDs per claim hoping one fits), causing verification failures.

**Solution**: Enforce tight citation limits in summarization prompt.

### Implementation:
- **Updated prompt**:
  - "Use maximum 1-2 evidence IDs per sentence."
  - "Each [ENN] should strongly support the specific claim."
  - "Penalize citation spraying: multiple weak citations worse than one strong citation."
- **Post-generation validation**:
  - Flag sentences with >2 citations
  - Show warning in output: "Over-citation detected in N sentences"
  
### Files Changed:
- `app/writer_main.py`: Updated `generate_summary_from_evidence()` prompt
- `app/writer_main.py`: Added citation count validation

---

## 4. Sentence-Level Verification

**Problem**: Segment-level verification (21 segments) misses sentence-level hallucinations.

**Solution**: Verify each factual sentence independently against its specific citations.

### Implementation:
- **New function**: `verify_citations_sentence_level(styled_output, evidence_store)`
- **Process**:
  1. Split output into sentences
  2. For each sentence: extract citations at end
  3. Verify sentence against ONLY those cited evidence IDs
  4. Return per-sentence verification results
- **Benefits**:
  - Higher precision: can pinpoint exact problematic sentence
  - Easier repair: rewrite one sentence, not whole segment
  
### Files Changed:
- `app/writer_main.py`: New `verify_citations_sentence_level()` function
- Replace `verify_styled_citations()` calls with new function

---

## 5. Plagiarism Boilerplate Filter

**Problem**: Wasting computation on "Good morning", "Thank you", etc. → 1353 chunks embedded.

**Solution**: Filter boilerplate before plagiarism check.

### Implementation:
- **Boilerplate classifier**:
  - Greetings: "Good morning", "Thank you", "Dear colleagues"
  - Common phrases: "It is my pleasure", "I would like to", "In conclusion"
  - Short sentences: < 8 words
- **Filter strategy**:
  - Classify each sentence as boilerplate/content before chunking
  - Skip plagiarism checks on boilerplate sentences
  - Still include them in final output (just don't verify)
- **Expected impact**: Reduce plagiarism check chunks by ~30-40%

### Files Changed:
- `app/plagiarism_checker.py`: New `is_boilerplate(sentence)` function
- `app/plagiarism_checker.py`: Updated `create_chunks()` to mark boilerplate
- `app/plagiarism_checker.py`: Skip web search for boilerplate chunks

---

## 6. Policy Compliance Strict Enum

**Problem**: Policy agent returns "UNKNOWN" but we treat it as "Approved" → reporting risk.

**Solution**: Use strict enum and require human review for ambiguous cases.

### Implementation:
- **New enum**: PolicyCompliance = "COMPLIANT" | "NEEDS_REVISION" | "NON_COMPLIANT" | "NEEDS_HUMAN_REVIEW"
- **Mapping**:
  - Agent returns "compliant" → COMPLIANT
  - Agent returns "minor_issues" → NEEDS_REVISION
  - Agent returns "major_issues" or "non_compliant" → NON_COMPLIANT
  - Agent returns "UNKNOWN" or fails → NEEDS_HUMAN_REVIEW
- **Updated reporting**:
  - Never show "APPROVED" for NEEDS_HUMAN_REVIEW
  - Clear warnings for manual review required
  
### Files Changed:
- `app/policy_checker.py`: Added `PolicyComplianceLevel` enum
- `app/policy_checker.py`: Updated result parsing to map to enum
- `app/writer_main.py`: Updated policy check display logic

---

## Testing Strategy

### Unit Tests
- Test claim outline generation with sample evidence
- Test evidence deduplication with duplicate claims
- Test citation validation with over-citing examples
- Test boilerplate classification accuracy
- Test policy enum mapping

### Integration Tests
- Run complete pipeline with all improvements
- Compare before/after metrics:
  - Verification rate (target: >95%)
  - Citation density (target: 1.5 IDs/sentence average)
  - Plagiarism chunks (target: <1000 for typical speech)
  - Policy clarity (target: 0% UNKNOWN responses)

### Regression Tests
- Ensure evidence continuity across iterations
- Verify no valid evidence is lost
- Check that styling preserves factual accuracy

---

## Rollout Plan

### Phase 1: Foundation (Immediate)
- ✅ Implement policy compliance enum
- ✅ Add boilerplate filter
- ✅ Update citation discipline prompt

### Phase 2: Evidence Management (Week 1)
- ✅ Implement cumulative evidence store
- ✅ Add deduplication logic
- ✅ Test across multiple iterations

### Phase 3: Verification Enhancement (Week 1)
- ✅ Implement sentence-level verification
- ✅ Update reporting

### Phase 4: Style Control (Week 2)
- ✅ Implement claim outline system
- ✅ Update styling prompts
- ✅ Test style drift prevention

### Phase 5: Validation & Metrics (Week 2)
- ✅ Run comprehensive tests
- ✅ Measure improvement metrics
- ✅ Document results

---

## Expected Outcomes

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Verification Rate | 85.7% | >95% | +9.3% |
| Invalid Citations | 3/21 seg | 0/100 sent | ~100% |
| Evidence Waste | 32% | <10% | -22% |
| Plagiarism Chunks | 1353 | <900 | -33% |
| Policy Clarity | "UNKNOWN" | Enum | 100% |
| Citation Density | Variable | 1-2/sent | Consistent |

---

## Maintenance Notes

- Monitor verification rates after each iteration
- Review claim outline quality weekly
- Adjust boilerplate patterns as needed
- Update policy enum if agent API changes
- Track evidence deduplication stats
