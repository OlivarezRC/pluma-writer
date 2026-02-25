"""
Enhanced pipeline utilities for improving speech generation quality.

Contains:
1. Policy compliance strict enum
2. Boilerplate detection for plagiarism filtering
3. Evidence store deduplication
4. Citation discipline validators
5. Sentence-level verification helpers
"""

from enum import Enum
from typing import List, Dict, Any, Set, Tuple
import hashlib
import re


class PolicyComplianceLevel(Enum):
    """
    Strict policy compliance levels.
    
    - COMPLIANT: Speech fully aligns with BSP policies
    - NEEDS_REVISION: Minor issues requiring fixes before approval
    - NON_COMPLIANT: Major violations, cannot be approved
    - NEEDS_HUMAN_REVIEW: Ambiguous or system error, requires manual review
    """
    COMPLIANT = "compliant"
    NEEDS_REVISION = "needs_revision"
    NON_COMPLIANT = "non_compliant"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    
    @staticmethod
    def from_agent_response(agent_compliance: str, requires_revision: bool) -> 'PolicyComplianceLevel':
        """
        Map agent response to strict enum.
        
        Args:
            agent_compliance: Agent's compliance assessment (e.g., "compliant", "minor_issues", "UNKNOWN")
            requires_revision: Whether agent flagged revision requirement
            
        Returns:
            PolicyComplianceLevel enum value
        """
        if not agent_compliance or agent_compliance.upper() == "UNKNOWN":
            return PolicyComplianceLevel.NEEDS_HUMAN_REVIEW
        
        compliance_lower = agent_compliance.lower().replace(" ", "_")
        
        if "major" in compliance_lower or "non_compliant" in compliance_lower or "non-compliant" in compliance_lower:
            return PolicyComplianceLevel.NON_COMPLIANT
        if requires_revision:
            return PolicyComplianceLevel.NEEDS_REVISION

        # No revision required: treat compliant/minor_issues as approved.
        if "compliant" in compliance_lower or "minor" in compliance_lower:
            return PolicyComplianceLevel.COMPLIANT

        if "issues" not in compliance_lower:
            return PolicyComplianceLevel.COMPLIANT
        else:
            # Ambiguous case - require human review
            return PolicyComplianceLevel.NEEDS_HUMAN_REVIEW
    
    def is_approved(self) -> bool:
        """Check if this compliance level allows approval without human review."""
        return self == PolicyComplianceLevel.COMPLIANT
    
    def display_str(self) -> str:
        """Human-readable display string."""
        return {
            PolicyComplianceLevel.COMPLIANT: "✅ APPROVED - Fully Compliant",
            PolicyComplianceLevel.NEEDS_REVISION: "⚠️ REQUIRES REVISION - Minor Issues",
            PolicyComplianceLevel.NON_COMPLIANT: "❌ REJECTED - Major Violations",
            PolicyComplianceLevel.NEEDS_HUMAN_REVIEW: "🔍 NEEDS HUMAN REVIEW - Ambiguous Result"
        }[self]


# Boilerplate patterns for filtering
BOILERPLATE_PATTERNS = [
    # Greetings
    r"^good (morning|afternoon|evening)",
    r"^(dear|esteemed) (colleagues|guests|friends|ladies and gentlemen)",
    r"^(thank you|thanks) (for|to|everyone)",
    r"^it('s| is) (my|a) (pleasure|honor|privilege)",
    
    # Closings
    r"^(thank you|thanks)( very much| so much)?[\.!]?\s*$",
    r"^(in conclusion|to conclude|finally)",
    r"^(maraming|salamat)( po)?",
    r"^that concludes my",
    
    # Transitional
    r"^(as|in) (mentioned|discussed|noted) (earlier|before|previously)",
    r"^(let me|allow me|i (would like|want) to)",
    r"^(we|i) (thank|appreciate|welcome)",
    r"^(our focus|we focus|today we) (is|on)",
    
    # Common institutional
    r"^(the|our) (bangko sentral|bsp|central bank)",
    r"^(ladies and gentlemen|colleagues and friends)",
]

BOILERPLATE_PHRASES = {
    "good morning", "good afternoon", "good evening",
    "thank you", "thanks", "maraming salamat",
    "in conclusion", "to conclude", "finally",
    "it is my pleasure", "it is an honor",
    "dear colleagues", "esteemed guests",
    "ladies and gentlemen",
}


def is_boilerplate(sentence: str, min_words: int = 8) -> bool:
    """
    Determine if a sentence is boilerplate (greeting, closing, common phrase).
    
    Args:
        sentence: The sentence to check
        min_words: Minimum word count threshold (sentences shorter than this are more likely boilerplate)
        
    Returns:
        True if sentence is likely boilerplate
    """
    if not sentence or not sentence.strip():
        return True
    
    sentence_clean = sentence.strip().lower()
    
    # Check word count - very short sentences often boilerplate
    word_count = len(sentence_clean.split())
    if word_count < min_words:
        # But check if it contains substantive content
        # If it has numbers, citations, or technical terms, it's likely not boilerplate
        if re.search(r'\d+(\.\d+)?%?', sentence_clean) or '[e' in sentence_clean.lower():
            return False
        return True
    
    # Check against patterns
    for pattern in BOILERPLATE_PATTERNS:
        if re.match(pattern, sentence_clean, re.IGNORECASE):
            return True
    
    # Check if entire sentence is in known boilerplate phrases
    if sentence_clean.rstrip('.,!?') in BOILERPLATE_PHRASES:
        return True
    
    return False


def deduplicate_evidence_store(
    evidence_list: List[Dict[str, Any]],
    key_fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Deduplicate evidence items based on content hash.
    
    Keeps the most recent version of each unique evidence item.
    Evidence is considered duplicate if it has the same source + claim content.
    
    Args:
        evidence_list: List of evidence dictionaries
        key_fields: Fields to use for deduplication (default: ['source_url', 'claim'])
        
    Returns:
        Deduplicated evidence list with stable IDs
    """
    if key_fields is None:
        key_fields = ['source_url', 'claim']
    
    seen_hashes = {}
    seen_ids = set()
    deduplicated = []
    
    for evidence in evidence_list:
        # Create hash key from specified fields
        hash_parts = []
        for field in key_fields:
            value = evidence.get(field, '')
            if value:
                hash_parts.append(str(value).strip().lower())
        
        if not hash_parts:
            # No valid key fields, skip
            continue
        
        content_hash = hashlib.sha256('|'.join(hash_parts).encode()).hexdigest()
        
        # Check if we've seen this content before
        if content_hash in seen_hashes:
            # Keep the one with more complete data (more non-empty fields)
            existing = seen_hashes[content_hash]
            current_completeness = sum(1 for v in evidence.values() if v)
            existing_completeness = sum(1 for v in existing.values() if v)
            
            if current_completeness > existing_completeness:
                # Replace with more complete version
                seen_hashes[content_hash] = evidence
        else:
            # New unique evidence
            seen_hashes[content_hash] = evidence
            seen_ids.add(evidence.get('id', ''))
    
    # Convert back to list, preserving order and renumbering IDs if needed
    deduplicated = list(seen_hashes.values())
    
    # Renumber evidence IDs sequentially if there are gaps
    for i, evidence in enumerate(deduplicated, start=1):
        old_id = evidence.get('id', '')
        if not old_id or not old_id.startswith('E'):
            evidence['id'] = f"E{i}"
        # Mark as deduplicated
        if 'dedup_info' not in evidence:
            evidence['dedup_info'] = {
                'original_id': old_id,
                'content_hash': hashlib.sha256(str(evidence.get('claim', '')).encode()).hexdigest()[:12]
            }
    
    print(f"Evidence deduplication: {len(evidence_list)} → {len(deduplicated)} items (-{len(evidence_list) - len(deduplicated)} duplicates)")
    
    return deduplicated


def validate_citation_discipline(text: str, max_citations_per_sentence: int = 2) -> Dict[str, Any]:
    """
    Validate that citations follow discipline rules (max 1-2 IDs per sentence).
    
    Args:
        text: Text with [ENN] citations
        max_citations_per_sentence: Maximum allowed citations per sentence
        
    Returns:
        {
            "compliant": bool,
            "over_cited_sentences": List[str],
            "total_sentences": int,
            "violations": int
        }
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    violations = []
    citation_pattern = r'\[E\d+(?:,E\d+)*\]'
    
    for sentence in sentences:
        # Find all citation blocks in this sentence
        citations = re.findall(citation_pattern, sentence)
        
        if not citations:
            continue
        
        # Count individual evidence IDs across all citation blocks
        all_ids = []
        for citation in citations:
            ids = re.findall(r'E\d+', citation)
            all_ids.extend(ids)
        
        if len(all_ids) > max_citations_per_sentence:
            violations.append({
                "sentence": sentence,
                "citation_count": len(all_ids),
                "citations": all_ids
            })
    
    return {
        "compliant": len(violations) == 0,
        "over_cited_sentences": violations,
        "total_sentences": len(sentences),
        "violations": len(violations),
        "violation_rate": f"{len(violations) / len(sentences) * 100:.1f}%" if sentences else "0%"
    }


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling common abbreviations.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Protect common abbreviations
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|viz|i\.e|e\.g)\.',
                  lambda m: m.group(0).replace('.', '<PERIOD>'), text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore periods
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences if s.strip()]
    
    return sentences


def extract_sentence_citations(sentence: str) -> Tuple[str, List[str]]:
    """
    Extract citations from end of sentence.
    
    Args:
        sentence: Sentence potentially ending with citations
        
    Returns:
        Tuple of (sentence_without_citations, list_of_evidence_ids)
    """
    citation_pattern = r'\[E\d+(?:,E\d+)*\]'
    citations = re.findall(citation_pattern, sentence)
    
    evidence_ids = []
    for citation in citations:
        ids = re.findall(r'E\d+', citation)
        evidence_ids.extend(ids)
    
    # Remove citations from sentence text
    sentence_clean = re.sub(citation_pattern, '', sentence).strip()
    
    return sentence_clean, evidence_ids


def is_factual_sentence(sentence: str) -> bool:
    """
    Determine if sentence contains factual claims requiring citation.
    
    Non-factual sentences (transitions, opinions, questions) don't need citations.
    
    Args:
        sentence: The sentence to check
        
    Returns:
        True if sentence makes factual claims
    """
    sentence_lower = sentence.lower().strip()
    
    # Skip questions
    if sentence_lower.endswith('?'):
        return False
    
    # Skip pure transitions/connectives
    transition_patterns = [
        r'^(however|moreover|furthermore|therefore|thus|hence|consequently|accordingly)',
        r'^(this suggests|this implies|this indicates|this means|a key implication)',
        r'^(in addition|additionally|in contrast|on the other hand|similarly)',
        r'^(first|second|third|finally|lastly|in conclusion)',
    ]
    
    for pattern in transition_patterns:
        if re.match(pattern, sentence_lower):
            # Still might be factual if it continues with claims
            # Check if it has specific numbers, names, or technical terms
            if not re.search(r'\d+(\.\d+)?%?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', sentence):
                return False
    
    # Sentences with numbers, percentages, specific data are factual
    if re.search(r'\d+(\.\d+)?%?', sentence):
        return True
    
    # Sentences with proper nouns (likely referencing studies/sources)
    if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', sentence):
        return True
    
    # Check for claim indicators
    claim_patterns = [
        r'\b(research|study|analysis|data|statistics|findings|evidence)\b',
        r'\b(shows?|demonstrates?|indicates?|reveals?|suggests?|finds?|reports?)\b',
        r'\b(achieves?|reaches?|attains?|produces?|generates?)\b',
    ]
    
    for pattern in claim_patterns:
        if re.search(pattern, sentence_lower):
            return True
    
    # Default: if sentence is substantial (not boilerplate), consider it factual
    if not is_boilerplate(sentence):
        return True
    
    return False
