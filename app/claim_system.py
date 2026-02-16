"""
Advanced pipeline components for claim-based styling and sentence-level verification.

Contains:
1. Claim outline generation (prevents style drift)
2. Sentence-level citation verification
3. Cumulative evidence management
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class Claim:
    """
    Represents an atomic factual claim with evidence support.
    """
    claim_id: str
    claim_text: str
    evidence_ids: List[str]  # 1-3 evidence IDs max
    claim_type: str  # "factual", "interpretation", "transition"
    confidence: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClaimOutline:
    """
    Structured claim plan for style-based rewriting.
    Prevents style model from inventing new claims.
    """
    
    def __init__(self, claims: List[Claim]):
        self.claims = claims
    
    def to_json(self) -> str:
        """Serialize to JSON for LLM consumption."""
        return json.dumps([c.to_dict() for c in self.claims], indent=2)
    
    def to_prompt_text(self) -> str:
        """Convert to readable prompt format."""
        lines = ["CLAIM OUTLINE (Rephrase ONLY these claims):", ""]
        
        for i, claim in enumerate(self.claims, 1):
            evidence_str = ", ".join(claim.evidence_ids)
            lines.append(f"{i}. {claim.claim_text}")
            lines.append(f"   Evidence: [{evidence_str}]")
            lines.append(f"   Type: {claim.claim_type}")
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    async def from_summary(
        cls, 
        summary: str, 
        evidence_store: List[Dict[str, Any]],
        model
    ) -> 'ClaimOutline':
        """
        Extract claim outline from summary text.
        
        Args:
            summary: Summary text with [ENN] citations
            evidence_store: Available evidence items
            model: LLM model instance
            
        Returns:
            ClaimOutline instance
        """
        evidence_map = {e['id']: e for e in evidence_store if 'id' in e}
        
        extraction_prompt = f"""Extract atomic factual claims from this research summary. Each claim should be:

1. **One clear factual statement** (not multiple facts combined)
2. **Tied to 1-3 evidence IDs** (from the cited [ENN] references)
3. **Classified by type**:
   - "factual": Concrete fact, statistic, or research finding
   - "interpretation": Analysis or implication drawn from facts
   - "transition": Logical connection (no citations needed)

<SUMMARY>
{summary}
</SUMMARY>

<TASK>
Return a JSON array where each item has:
- "claim_text": the atomic factual statement (one sentence)
- "evidence_ids": array of 1-3 evidence IDs that support this claim (e.g., ["E1", "E5"])
- "claim_type": "factual", "interpretation", or "transition"
- "confidence": your confidence (0.0-1.0) that evidence supports this claim

IMPORTANT RULES:
- One fact per claim (don't combine multiple statistics)
- Maximum 3 evidence IDs per claim (prefer 1-2)
- Transition claims can have empty evidence_ids []
- Each evidence ID must be from the summary's [ENN] citations

Return ONLY the JSON array.
</TASK>

JSON:"""
        
        messages = [
            SystemMessage(content="You are a precise claim extraction system. Extract atomic, evidence-backed claims."),
            HumanMessage(content=extraction_prompt)
        ]
        
        result = await model.ainvoke(messages)
        content = result.content.strip()
        
        # Strip thinking tokens and markdown
        if '<think>' in content:
            content = content.split('</think>')[-1].strip()
        content = content.replace('```json', '').replace('```', '').strip()
        
        try:
            claims_data = json.loads(content)
            
            # Validate and create Claim objects
            claims = []
            for i, claim_data in enumerate(claims_data, 1):
                # Validate evidence IDs exist
                evidence_ids = claim_data.get('evidence_ids', [])
                valid_ids = [eid for eid in evidence_ids if eid in evidence_map]
                
                # Enforce max 3 evidence IDs
                if len(valid_ids) > 3:
                    valid_ids = valid_ids[:3]
                
                claims.append(Claim(
                    claim_id=f"C{i}",
                    claim_text=claim_data.get('claim_text', ''),
                    evidence_ids=valid_ids,
                    claim_type=claim_data.get('claim_type', 'factual'),
                    confidence=claim_data.get('confidence', 0.8)
                ))
            
            print(f"✓ Extracted {len(claims)} atomic claims from summary")
            
            # Show claim type breakdown
            factual_count = sum(1 for c in claims if c.claim_type == 'factual')
            interp_count = sum(1 for c in claims if c.claim_type == 'interpretation')
            trans_count = sum(1 for c in claims if c.claim_type == 'transition')
            print(f"  Factual: {factual_count}, Interpretation: {interp_count}, Transition: {trans_count}")
            
            return cls(claims)
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse claims: {e}")
            # Fallback: create basic outline from summary paragraphs
            paragraphs = [p.strip() for p in summary.split('\n\n') if p.strip()]
            claims = []
            for i, para in enumerate(paragraphs[:10], 1):
                # Extract citations from paragraph
                citations = re.findall(r'E\d+', para)[:3]
                claims.append(Claim(
                    claim_id=f"C{i}",
                    claim_text=para[:200] + "..." if len(para) > 200 else para,
                    evidence_ids=citations,
                    claim_type='factual',
                    confidence=0.7
                ))
            return cls(claims)


async def generate_claim_outline(
    summary: str,
    evidence_store: List[Dict[str, Any]],
    model
) -> ClaimOutline:
    """
    Generate structured claim outline from summary to prevent style drift.
    
    Args:
        summary: Summary text with citations
        evidence_store: Evidence items
        model: LLM model instance
        
    Returns:
        ClaimOutline object
    """
    return await ClaimOutline.from_summary(summary, evidence_store, model)


async def verify_sentence_citations(
    text: str,
    evidence_store: List[Dict[str, Any]],
    azure_llm_client,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Verify citations at sentence level (more precise than segment level).
    
    Each factual sentence should:
    1. Have citations at the end
    2. Be verifiable against cited evidence only
    
    Args:
        text: Styled text with [ENN] citations
        evidence_store: Evidence items
        azure_llm_client: Azure OpenAI client
        model: Model name for verification
        
    Returns:
        {
            "sentences": List of per-sentence verification results,
            "total_sentences": int,
            "factual_sentences": int,
            "verified_sentences": int,
            "unverified_sentences": int,
            "verification_rate": str
        }
    """
    from app.pipeline_enhancements import (
        split_into_sentences, 
        extract_sentence_citations, 
        is_factual_sentence,
        is_boilerplate
    )
    
    print("\n" + "="*70)
    print("SENTENCE-LEVEL CITATION VERIFICATION")
    print("="*70)
    
    # Build evidence lookup
    evidence_map = {e['id']: e for e in evidence_store if 'id' in e}
    print(f"Evidence store: {len(evidence_map)} items")
    
    # Split into sentences
    sentences = split_into_sentences(text)
    print(f"Parsed {len(sentences)} sentences")
    
    sentence_results = []
    verified_count = 0
    unverified_count = 0
    factual_count = 0
    
    for i, sentence in enumerate(sentences, 1):
        # Skip boilerplate
        if is_boilerplate(sentence):
            sentence_results.append({
                "sentence_number": i,
                "text": sentence,
                "is_boilerplate": True,
                "is_factual": False,
                "verified": "N/A",
                "verification_reason": "Boilerplate - no verification needed"
            })
            continue
        
        # Extract citations
        sentence_clean, evidence_ids = extract_sentence_citations(sentence)
        
        # Check if factual
        is_fact = is_factual_sentence(sentence_clean)
        
        if not is_fact:
            # Non-factual (transition, opinion) - no citation needed
            sentence_results.append({
                "sentence_number": i,
                "text": sentence,
                "is_boilerplate": False,
                "is_factual": False,
                "is_transition": True,
                "verified": "N/A",
                "verification_reason": "Non-factual sentence - no verification needed"
            })
            continue
        
        factual_count += 1
        
        # Factual sentence - must have citations
        if not evidence_ids:
            unverified_count += 1
            sentence_results.append({
                "sentence_number": i,
                "text": sentence,
                "is_boilerplate": False,
                "is_factual": True,
                "citations": [],
                "verified": "No",
                "verification_reason": "Factual claim without citation"
            })
            continue
        
        # Look up cited evidence
        cited_claims = []
        missing_ids = []
        for eid in evidence_ids:
            if eid in evidence_map:
                cited_claims.append({
                    "id": eid,
                    "claim": evidence_map[eid].get("claim", ""),
                    "quote_span": evidence_map[eid].get("quote_span", "")
                })
            else:
                missing_ids.append(eid)
        
        if missing_ids:
            unverified_count += 1
            sentence_results.append({
                "sentence_number": i,
                "text": sentence,
                "is_boilerplate": False,
                "is_factual": True,
                "citations": evidence_ids,
                "missing_evidence": missing_ids,
                "verified": "No",
                "verification_reason": f"Evidence IDs not found: {', '.join(missing_ids)}"
            })
            continue
        
        # Verify sentence against cited evidence using LLM
        verification_prompt = f"""Verify if this sentence is supported by the cited evidence.

<SENTENCE>
{sentence_clean}
</SENTENCE>

<CITED_EVIDENCE>
{json.dumps(cited_claims, indent=2)}
</CITED_EVIDENCE>

<TASK>
Determine if the sentence's claims are adequately supported by the cited evidence.

Return JSON:
{{
  "verified": "Yes" or "No",
  "reason": "Brief explanation (1 sentence)"
}}
</TASK>

JSON:"""
        
        try:
            response = await azure_llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant. Verify claims against evidence."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            
            verification_data = json.loads(content)
            is_verified = verification_data.get("verified", "No") == "Yes"
            reason = verification_data.get("reason", "Unknown")
            
            if is_verified:
                verified_count += 1
            else:
                unverified_count += 1
            
            sentence_results.append({
                "sentence_number": i,
                "text": sentence,
                "is_boilerplate": False,
                "is_factual": True,
                "citations": evidence_ids,
                "cited_claims": cited_claims,
                "verified": "Yes" if is_verified else "No",
                "verification_reason": reason
            })
            
        except Exception as e:
            print(f"  ✗ Verification error for sentence {i}: {str(e)}")
            unverified_count += 1
            sentence_results.append({
                "sentence_number": i,
                "text": sentence,
                "is_boilerplate": False,
                "is_factual": True,
                "citations": evidence_ids,
                "cited_claims": cited_claims,
                "verified": "Error",
                "verification_reason": f"Verification failed: {str(e)}"
            })
    
    verification_rate = (verified_count / factual_count * 100) if factual_count > 0 else 0
    
    print(f"\n✓ Sentence verification complete:")
    print(f"  Total sentences: {len(sentences)}")
    print(f"  Factual sentences: {factual_count}")
    print(f"  Verified: {verified_count}")
    print(f"  Unverified: {unverified_count}")
    print(f"  Verification rate: {verification_rate:.1f}%")
    
    return {
        "sentences": sentence_results,
        "total_sentences": len(sentences),
        "factual_sentences": factual_count,
        "verified_sentences": verified_count,
        "unverified_sentences": unverified_count,
        "verification_rate": f"{verification_rate:.1f}%"
    }
