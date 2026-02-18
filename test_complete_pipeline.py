#!/usr/bin/env python3
"""
Test script for complete pipeline: Iterative Refinement + Style-based Output

Enhanced with 6 architectural improvements:
1. Claim-based style generation (prevents style drift)
2. Cumulative evidence store (prevents evidence loss)
3. Tight citation discipline (1-2 IDs per sentence)
4. Sentence-level verification (more precise)
5. Boilerplate filtering in plagiarism check
6. Strict policy compliance enum (no UNKNOWN)
"""
import asyncio
import json
import re
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_with_iterative_refinement_and_style


def validate_citation_discipline(text: str) -> dict:
    """
    Improvement #3: Validate tight citation discipline (1-2 IDs per sentence)
    """
    sentences = re.split(r'[.!?]+\s+', text)
    
    stats = {
        'total_sentences': len(sentences),
        'sentences_with_citations': 0,
        'citation_violations': [],  # More than 2 IDs in one sentence
        'max_ids_per_sentence': 0,
        'avg_ids_per_sentence': 0,
        'disciplined': True
    }
    
    total_ids = 0
    
    for i, sentence in enumerate(sentences, 1):
        citations = re.findall(r'\[E\d+(?:,E\d+)*\]', sentence)
        if citations:
            stats['sentences_with_citations'] += 1
            
            # Count unique IDs in this sentence
            ids_in_sentence = set()
            for citation in citations:
                ids = re.findall(r'E\d+', citation)
                ids_in_sentence.update(ids)
            
            id_count = len(ids_in_sentence)
            total_ids += id_count
            stats['max_ids_per_sentence'] = max(stats['max_ids_per_sentence'], id_count)
            
            # Violation: more than 2 IDs per sentence (citation spraying)
            if id_count > 2:
                stats['citation_violations'].append({
                    'sentence_num': i,
                    'id_count': id_count,
                    'ids': list(ids_in_sentence),
                    'preview': sentence[:100] + '...' if len(sentence) > 100 else sentence
                })
                stats['disciplined'] = False
    
    if stats['sentences_with_citations'] > 0:
        stats['avg_ids_per_sentence'] = total_ids / stats['sentences_with_citations']
    
    return stats


def validate_sentence_level_verification(verification_result: dict) -> dict:
    """
    Improvement #4: Analyze sentence-level verification instead of segment-level
    """
    if not verification_result:
        return {'enabled': False}
    
    segments = verification_result.get('segments', [])
    
    # Check if we're using sentence-level (flag is at root level, not segment level)
    sentence_level = verification_result.get('is_sentence_level', False)
    
    return {
        'enabled': True,
        'is_sentence_level': sentence_level,
        'total_segments': len(segments),
        'verified_count': sum(1 for s in segments if s.get('verified') == 'Yes'),
        'unverified_count': sum(1 for s in segments if s.get('verified') != 'Yes'),
        'verification_rate': verification_result.get('verification_rate', '0%')
    }


def validate_boilerplate_filtering(plagiarism_result: dict) -> dict:
    """
    Improvement #5: Check if boilerplate greetings are filtered from plagiarism check
    """
    if not plagiarism_result or not plagiarism_result.get('success'):
        return {'enabled': False}
    
    stats = plagiarism_result.get('statistics', {})
    
    # Check the new boilerplate_filtered stat
    boilerplate_filtered = stats.get('boilerplate_filtered', 0)
    total_chunks = stats.get('total_chunks', 0)
    
    return {
        'enabled': True,
        'boilerplate_patterns_filtered': boilerplate_filtered,
        'total_chunks_analyzed': total_chunks,
        'filtering_active': boilerplate_filtered > 0
    }


def validate_policy_compliance_enum(policy_result: dict) -> dict:
    """
    Improvement #6: Ensure policy compliance returns strict enum (not UNKNOWN)
    """
    if not policy_result or not policy_result.get('success'):
        return {'enabled': False}
    
    overall_compliance = policy_result.get('overall_compliance', '').upper()
    
    # Valid strict enums
    valid_enums = ['COMPLIANT', 'NEEDS_REVISION', 'NON_COMPLIANT', 'MINOR_ISSUES', 'MAJOR_ISSUES']
    
    return {
        'enabled': True,
        'overall_compliance': overall_compliance,
        'is_strict_enum': overall_compliance in valid_enums,
        'is_unknown': overall_compliance == 'UNKNOWN',
        'requires_revision': policy_result.get('requires_revision', False),
        'compliance_score': policy_result.get('compliance_score', 0.0)
    }


def validate_evidence_stability(iterations: list) -> dict:
    """
    Improvement #2: Check if evidence extraction is cumulative (no loss across iterations)
    """
    if not iterations:
        return {'enabled': False}
    
    evidence_counts = []
    evidence_ids_per_iter = []
    
    for iteration in iterations:
        count = iteration.get('cumulative_evidence_count', 0)
        evidence_counts.append(count)
        
        # Extract evidence IDs
        evidence_store = iteration.get('results', {}).get('cumulative_evidence_store', [])
        ids = {ev.get('id') for ev in evidence_store if ev.get('id')}
        evidence_ids_per_iter.append(ids)
    
    # Check for evidence loss (decreasing count)
    evidence_lost = any(
        evidence_counts[i] < evidence_counts[i-1] 
        for i in range(1, len(evidence_counts))
    )
    
    # Check for ID loss
    ids_lost = []
    for i in range(1, len(evidence_ids_per_iter)):
        prev_ids = evidence_ids_per_iter[i-1]
        curr_ids = evidence_ids_per_iter[i]
        lost = prev_ids - curr_ids
        if lost:
            ids_lost.append({
                'iteration': i + 1,
                'lost_ids': list(lost)
            })
    
    return {
        'enabled': True,
        'evidence_counts': evidence_counts,
        'is_cumulative': not evidence_lost and not ids_lost,
        'evidence_lost': evidence_lost,
        'ids_lost': ids_lost
    }


def validate_claim_outline(results: dict) -> dict:
    """
    Improvement #1: Check if style generation uses claim outlines instead of freeform prose
    """
    styled_output = results.get('styled_output', {})
    
    # Check if claim outline was used in the process
    has_claim_outline = styled_output.get('claim_outline_used', False)
    claim_count = styled_output.get('claim_count', 0)
    
    # Look for evidence of structured claims
    styled_text = styled_output.get('styled_output', '')
    
    # Parse to see if claims are well-attributed (1-2 IDs per claim)
    sentences = re.split(r'[.!?]+\s+', styled_text)
    factual_sentences = [s for s in sentences if re.search(r'\[E\d+', s)]
    
    return {
        'enabled': True,
        'claim_outline_used': has_claim_outline,
        'claim_count': claim_count,
        'factual_sentences': len(factual_sentences),
        'total_sentences': len(sentences)
    }


async def test_complete_pipeline():
    """Test the complete workflow with iterative refinement and styling"""
    
    print("=" * 70)
    print("COMPLETE PIPELINE TEST")
    print("Iterative Refinement → Style-Based Output Generation")
    print("=" * 70)
    
    print("\n🔧 ARCHITECTURAL IMPROVEMENTS ENABLED:")
    print("  1. Evidence Stability: Cumulative store prevents loss")
    print("  2. Claim-Based Style: Prevents style drift & hallucinations")
    print("  3. Citation Discipline: Max 1-2 IDs per sentence")
    print("  4. Sentence-Level Verification: More precise than segments")
    print("  5. Boilerplate Filtering: Skip greetings in plagiarism check")
    print("  6. Strict Policy Enum: COMPLIANT/NEEDS_REVISION/NON_COMPLIANT")
    print("=" * 70)
    
    user_query = "Use of AI models for financial portfolio management and risk predictions"
    
    user_sources = {
        "topics": "AI artificial intelligence machine learning financial portfolio management risk prediction asset allocation quantitative finance deep learning",
        "links": [
            "https://arxiv.org/abs/2106.03072",  # Deep Learning for Portfolio Optimization
            "https://arxiv.org/abs/2012.10377",  # Machine Learning in Finance
            "https://www.sciencedirect.com/science/article/abs/pii/S0957417421006448",  # AI risk management
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3420952",  # Deep Portfolio Theory
            "https://www.bis.org/publ/work1022.htm",  # BIS - AI in finance
            "https://www.imf.org/en/Publications/fintech-notes/Issues/2021/10/01/Fintech-and-Financial-Services-Initial-Considerations-464925"  # IMF on Fintech
        ],
        "attachments": []
    }
    
    print(f"\nOriginal Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Links: {len(user_sources['links'])}")
    print(f"\nPipeline Steps:")
    print("  1. Iterative refinement (3 iterations with critiques)")
    print("  2. Retrieve random writing style from Cosmos DB")
    print("  3. Generate styled output using GPT model")
    
    try:
        # Run complete pipeline
        results = await process_with_iterative_refinement_and_style(
            query=user_query,
            sources=user_sources,
            max_iterations=3
        )
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        # Refinement results
        print(f"\n📚 ITERATIVE REFINEMENT:")
        print(f"  • Total iterations: {results['total_iterations']}")
        print(f"  • Final evidence count: {results['final_evidence_count']}")
        print(f"  • Final summary length: {len(results['final_summary'].get('summary', ''))} chars")
        
        # === NEW: ARCHITECTURAL IMPROVEMENTS VALIDATION ===
        print(f"\n" + "=" * 70)
        print("🔧 ARCHITECTURAL IMPROVEMENTS VALIDATION")
        print("=" * 70)
        
        # Improvement #2: Evidence Stability
        print(f"\n1️⃣  EVIDENCE STABILITY (Cumulative Store):")
        evidence_stability = validate_evidence_stability(results['iterations'])
        if evidence_stability['enabled']:
            print(f"    Evidence counts: {' → '.join(map(str, evidence_stability['evidence_counts']))}")
            if evidence_stability['is_cumulative']:
                print(f"    ✅ Evidence is cumulative (no loss across iterations)")
            else:
                print(f"    ❌ Evidence loss detected!")
                if evidence_stability['ids_lost']:
                    for loss in evidence_stability['ids_lost']:
                        print(f"       Iteration {loss['iteration']}: Lost {len(loss['lost_ids'])} IDs")
        
        # Improvement #1: Claim Outline
        print(f"\n2️⃣  CLAIM-BASED STYLE (Prevents Style Drift):")
        claim_validation = validate_claim_outline(results)
        if claim_validation['enabled']:
            if claim_validation['claim_outline_used']:
                print(f"    ✅ Claim outline used: {claim_validation['claim_count']} claims")
                print(f"    Factual sentences: {claim_validation['factual_sentences']}/{claim_validation['total_sentences']}")
            else:
                print(f"    ⚠️  Freeform prose used (may cause style drift)")
        
        # Improvement #3: Citation Discipline
        styled_result = results.get('styled_output', {})
        citation_discipline = {}
        if styled_result.get('success'):
            print(f"\n3️⃣  CITATION DISCIPLINE (1-2 IDs per sentence):")
            styled_text = styled_result.get('styled_output', '')
            citation_discipline = validate_citation_discipline(styled_text)
            print(f"    Total sentences: {citation_discipline['total_sentences']}")
            print(f"    Sentences with citations: {citation_discipline['sentences_with_citations']}")
            print(f"    Avg IDs per sentence: {citation_discipline['avg_ids_per_sentence']:.2f}")
            print(f"    Max IDs per sentence: {citation_discipline['max_ids_per_sentence']}")
            
            if citation_discipline['disciplined']:
                print(f"    ✅ Citation discipline maintained (≤2 IDs/sentence)")
            else:
                print(f"    ⚠️  Citation violations detected: {len(citation_discipline['citation_violations'])}")
                for violation in citation_discipline['citation_violations'][:2]:
                    print(f"       Sentence {violation['sentence_num']}: {violation['id_count']} IDs (citation spraying)")
        
        # Improvement #4: Sentence-Level Verification
        print(f"\n4️⃣  SENTENCE-LEVEL VERIFICATION:")
        verification_result = results.get('citation_verification', {})
        sentence_verify = validate_sentence_level_verification(verification_result)
        if sentence_verify['enabled']:
            if sentence_verify['is_sentence_level']:
                print(f"    ✅ Sentence-level verification active")
            else:
                print(f"    ⚠️  Segment-level verification (less precise)")
            print(f"    Verified: {sentence_verify['verified_count']}/{sentence_verify['total_segments']}")
            print(f"    Verification rate: {sentence_verify['verification_rate']}")
        
        # Improvement #5: Boilerplate Filtering
        print(f"\n5️⃣  BOILERPLATE FILTERING (Plagiarism Check):")
        plagiarism_result = results.get('plagiarism_analysis', {})
        boilerplate = validate_boilerplate_filtering(plagiarism_result)
        if boilerplate['enabled']:
            if boilerplate['filtering_active']:
                print(f"    ✅ Boilerplate filtering active")
                print(f"    Filtered: {boilerplate['boilerplate_patterns_filtered']} chunks")
                print(f"    Total analyzed: {boilerplate['total_chunks_analyzed']}")
            else:
                print(f"    ⚠️  No boilerplate filtering detected")
                print(f"    Total analyzed: {boilerplate['total_chunks_analyzed']}")
        
        # Improvement #6: Strict Policy Enum
        print(f"\n6️⃣  STRICT POLICY COMPLIANCE ENUM:")
        policy_result = results.get('policy_check', {})
        policy_enum = validate_policy_compliance_enum(policy_result)
        if policy_enum['enabled']:
            print(f"    Compliance status: {policy_enum['overall_compliance']}")
            if policy_enum['is_strict_enum']:
                print(f"    ✅ Strict enum used (COMPLIANT/NEEDS_REVISION/NON_COMPLIANT)")
                print(f"    Score: {policy_enum['compliance_score']:.1%}")
                print(f"    Requires revision: {'Yes' if policy_enum['requires_revision'] else 'No'}")
            elif policy_enum['is_unknown']:
                print(f"    ❌ UNKNOWN status detected (should require human review)")
            else:
                print(f"    ⚠️  Non-standard enum: {policy_enum['overall_compliance']}")
        
        print(f"\n" + "=" * 70)
        
        # === CALCULATE IMPROVEMENT SCORECARD (before saving) ===
        score = 0
        max_score = 6
        
        # Score each improvement silently (will display later)
        if evidence_stability.get('is_cumulative'):
            score += 1
        if claim_validation.get('claim_outline_used'):
            score += 1
        if citation_discipline.get('disciplined'):
            score += 1
        if sentence_verify.get('is_sentence_level'):
            score += 1
        if boilerplate.get('filtering_active'):
            score += 1
        if policy_enum.get('is_strict_enum') and not policy_enum.get('is_unknown'):
            score += 1
        
        grade_percent = (score / max_score) * 100
        grade_letter = 'A+' if grade_percent >= 95 else 'A' if grade_percent >= 90 else 'B+' if grade_percent >= 85 else 'B' if grade_percent >= 80 else 'C+'
        
        # Continue with style information display
        print(f"\n✍️ WRITING STYLE APPLIED:")
        style_info = results.get('style_used', {})
        print(f"  • Style name: {style_info.get('name', 'None')}")
        print(f"  • Speaker: {style_info.get('speaker', 'None')}")
        print(f"  • Audience: {style_info.get('audience', 'None')}")
        
        # Styled output
        styled_result = results.get('styled_output', {})
        if styled_result.get('success'):
            print(f"\n📄 STYLED OUTPUT:")
            print(f"  • Status: Success ✓")
            print(f"  • Model used: {styled_result.get('model_used', 'Unknown')}")
            print(f"  • Output length: {len(styled_result.get('styled_output', ''))} chars")
            
            # Show citation validation
            if styled_result.get('citations_found') is not None:
                print(f"  • Citations: {styled_result.get('citations_found')} instances")
                print(f"  • Evidence cited: {styled_result.get('unique_evidence_cited')} unique IDs")
                print(f"  • Coverage: {styled_result.get('citation_coverage')}")
                
                if styled_result.get('invalid_citations'):
                    print(f"  • ⚠️ Invalid citations: {styled_result.get('invalid_citations')}")
                else:
                    print(f"  • ✓ All citations valid")
            
            # Show preview of styled output
            styled_text = styled_result.get('styled_output', '')
            print(f"\n" + "=" * 70)
            print("STYLED OUTPUT PREVIEW (first 1000 characters)")
            print("=" * 70)
            print(styled_text[:1000])
            if len(styled_text) > 1000:
                print("\n... (truncated)")
            print("=" * 70)
        else:
            print(f"\n📄 STYLED OUTPUT:")
            print(f"  • Status: Failed ✗")
            print(f"  • Error: {styled_result.get('error', 'Unknown')}")
        
        # Save results
        print(f"\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save complete results including all iterations with evidence_store and summaries
        iterations_data = []
        for iteration in results['iterations']:
            iter_data = {
                "iteration": iteration['iteration'],
                "query": iteration['query'],
                "new_evidence_count": iteration['new_evidence_count'],
                "cumulative_evidence_count": iteration['cumulative_evidence_count'],
                "evidence_store": iteration['results'].get('cumulative_evidence_store', []),
                "summary": iteration['results'].get('generated_summary', {}).get('summary', ''),
                "critique": iteration.get('critique', {}),
                "adjustments": iteration.get('adjustments', {})
            }
            iterations_data.append(iter_data)
        
        output_data = {
            "query": results['original_query'],
            "total_iterations": results['total_iterations'],
            "final_evidence_count": results['final_evidence_count'],
            "iterations": iterations_data,
            "cumulative_evidence_store": results.get('cumulative_evidence_store', []),
            "style_used": results.get('style_used', {}),
            "styled_output": results.get('styled_output', {}),
            "styled_output_apa": results.get('styled_output_apa', {}),
            "citation_verification": results.get('citation_verification', {}),
            "plagiarism_analysis": results.get('plagiarism_analysis', {}),
            "policy_check": results.get('policy_check', {}),
            "final_summary": results['final_summary'],
            # NEW: Add improvement metrics
            "improvement_metrics": {
                "evidence_stability": evidence_stability,
                "claim_outline": claim_validation,
                "citation_discipline": citation_discipline,
                "sentence_verification": sentence_verify,
                "boilerplate_filtering": boilerplate,
                "policy_enum": policy_enum,
                "overall_score": f"{score}/{max_score}",
                "grade_percent": grade_percent,
                "grade_letter": grade_letter
            }
        }
        
        with open('complete_pipeline_output.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Complete results saved to: complete_pipeline_output.json")
        
        # Display what was saved
        verification_data = results.get('citation_verification')
        if verification_data:
            print(f"  ├─ Citation verification: {verification_data.get('total_segments', 0)} segments analyzed")
        print(f"  ├─ Evidence store: {len(results.get('cumulative_evidence_store', []))} items")
        print(f"  ├─ Iterations: {results['total_iterations']} refinement cycles")
        print(f"  └─ Styled output: {'✓' if styled_result.get('success') else '✗'}")
        
        # Also save just the styled output for easy reading
        if styled_result.get('success'):
            with open('styled_output.txt', 'w') as f:
                f.write(f"Query: {user_query}\n\n")
                f.write(f"Style: {style_info.get('name', 'Unknown')}\n")
                f.write(f"Speaker: {style_info.get('speaker', 'Unknown')}\n")
                f.write(f"Audience: {style_info.get('audience', 'Unknown')}\n")
                f.write(f"\n{'='*70}\n")
                f.write(f"STYLED OUTPUT\n")
                f.write(f"{'='*70}\n\n")
                f.write(styled_result.get('styled_output', ''))
            
            print(f"✓ Styled output saved to: styled_output.txt")
        
        # Also save APA formatted output
        apa_result = results.get('styled_output_apa')
        if apa_result and apa_result.get('success'):
            with open('styled_output_apa.txt', 'w') as f:
                f.write(f"Query: {user_query}\n\n")
                f.write(f"Style: {style_info.get('name', 'Unknown')}\n")
                f.write(f"Speaker: {style_info.get('speaker', 'Unknown')}\n")
                f.write(f"Audience: {style_info.get('audience', 'Unknown')}\n")
                f.write(f"\n{'='*70}\n")
                f.write(f"STYLED OUTPUT (APA FORMAT)\n")
                f.write(f"{'='*70}\n\n")
                f.write(apa_result.get('apa_output', ''))
            
            print(f"✓ APA formatted output saved to: styled_output_apa.txt")
        
        # Statistics
        print(f"\n" + "=" * 70)
        print("PIPELINE STATISTICS")
        print("=" * 70)
        
        print(f"\nIterations completed: {results['total_iterations']}")
        print(f"Total evidence collected: {results['final_evidence_count']}")
        
        # Validate atomic evidence format for each iteration
        import re
        print(f"\n{'─'*70}")
        print("ATOMIC EVIDENCE & CITATION VALIDATION")
        print(f"{'─'*70}")
        
        for i, iteration in enumerate(results['iterations'], 1):
            evidence_count = iteration['cumulative_evidence_count']
            has_critique = iteration.get('critique') is not None
            
            print(f"\nIteration {i}:")
            print(f"  Evidence items: {evidence_count}")
            print(f"  Has critique: {'Yes' if has_critique else 'No'}")
            
            # Check evidence format
            evidence_store = iteration['results'].get('cumulative_evidence_store', [])
            if evidence_store:
                first_evidence = evidence_store[0]
                is_atomic = 'id' in first_evidence and 'claim' in first_evidence and 'quote_span' in first_evidence
                print(f"  Format: {'✓ ATOMIC' if is_atomic else '✗ OLD PARAGRAPH'}")
                
                if is_atomic:
                    last_evidence = evidence_store[-1]
                    print(f"  ID Range: {first_evidence.get('id')} → {last_evidence.get('id')}")
                    print(f"  Sample: {first_evidence.get('claim', '')[:60]}...")
                
                # Check citations in summary
                summary = iteration['results'].get('generated_summary', {}).get('summary', '')
                citations = re.findall(r'\[E\d+(?:,E\d+)*\]', summary)
                
                if citations:
                    cited_ids = set()
                    for citation in citations:
                        ids = re.findall(r'E\d+', citation)
                        cited_ids.update(ids)
                    
                    print(f"  Citations: {len(citations)} instances, {len(cited_ids)} unique IDs")
                    print(f"  Sample citations: {citations[:3]}")
                else:
                    print(f"  ⚠️ No [ENN] citations found in summary")
        
        print(f"\n{'─'*70}")
        print(f"\nFinal output:")
        print(f"  • Summary: {len(results['final_summary'].get('summary', ''))} characters")
        if styled_result.get('success'):
            print(f"  • Styled version: {len(styled_result.get('styled_output', ''))} characters")
            print(f"  • Style: {style_info.get('name', 'Unknown')}")
            
            # Check if styled output preserves citations
            styled_text = styled_result.get('styled_output', '')
            styled_citations = re.findall(r'\[E\d+(?:,E\d+)*\]', styled_text)
            if styled_citations:
                print(f"  • Citations preserved in styled output: {len(styled_citations)} instances ✓")
            else:
                print(f"  • ⚠️ Citations NOT preserved in styled output")
                
            # Display citation verification results
            verification_result = results.get('citation_verification')
            if verification_result:
                print(f"\n{'─'*70}")
                print("Citation Verification Results:")
                print(f"  • Total segments: {verification_result.get('total_segments', 0)}")
                print(f"  • Verified segments: {verification_result.get('verified_segments', 0)}")
                print(f"  • Unverified segments: {verification_result.get('unverified_segments', 0)}")
                print(f"  • Verification rate: {verification_result.get('verification_rate', '0%')}")
                
                # Show sample unverified segments if any
                if verification_result.get('unverified_segments', 0) > 0:
                    unverified = [
                        s for s in verification_result.get('segments', [])
                        if s.get('verified') != 'Yes'
                    ]
                    if unverified:
                        print(f"\n  ⚠️ Sample unverified segments:")
                        for seg in unverified[:2]:
                            print(f"    Segment {seg['segment_number']}: {seg.get('verification_reason', 'Unknown')[:60]}...")
                else:
                    print(f"  ✓ All citations verified successfully!")
            
            # Display APA conversion results
            apa_result = results.get('styled_output_apa')
            if apa_result and apa_result.get('success'):
                print(f"\n{'─'*70}")
                print("APA Format Conversion Results:")
                print(f"  • Citations converted: {apa_result.get('citations_converted', 0)}")
                print(f"  • References generated: {apa_result.get('references_generated', 0)}")
                print(f"  • Output length: {len(apa_result.get('apa_output', ''))} characters")
                print(f"  ✓ Saved to: styled_output_apa.txt")
            
            # Display plagiarism analysis results
            plagiarism_result = results.get('plagiarism_analysis')
            if plagiarism_result and plagiarism_result.get('success'):
                print(f"\n{'─'*70}")
                print("Plagiarism Analysis Results:")
                print(f"  • Overall risk: {plagiarism_result.get('overall_risk_level').upper()} ({plagiarism_result.get('overall_risk_score'):.3f})")
                stats = plagiarism_result.get('statistics', {})
                print(f"  • Total chunks: {stats.get('total_chunks', 0)}")
                print(f"  • High risk: {stats.get('high_risk_chunks', 0)}")
                print(f"  • Medium risk: {stats.get('medium_risk_chunks', 0)}")
                print(f"  • Low risk: {stats.get('low_risk_chunks', 0)}")
                print(f"  • Clean: {stats.get('clean_chunks', 0)}")
                
                top_sources = plagiarism_result.get('top_sources', [])
                if top_sources:
                    print(f"  • Matching sources found: {len(top_sources)}")
                    for i, src in enumerate(top_sources[:3], 1):
                        print(f"    {i}. {src.get('title', 'Unknown')[:60]}... ({src.get('match_count', 0)} matches)")
                else:
                    print(f"  • No significant matches found")
                
                flagged_count = len(plagiarism_result.get('flagged_chunks', []))
                if flagged_count > 0:
                    print(f"  ⚠️  {flagged_count} chunks flagged for review")
                else:
                    print(f"  ✓ No problematic content detected")
            elif plagiarism_result:
                print(f"\n{'─'*70}")
                print("Plagiarism Analysis Results:")
                print(f"  ✗ Analysis failed: {plagiarism_result.get('error', 'Unknown')}")
            
            # Display BSP Policy Alignment results
            policy_result = results.get('policy_check')
            if policy_result and policy_result.get('success'):
                print(f"\n{'─'*70}")
                print("BSP Policy Alignment Results:")
                print(f"  • Overall compliance: {policy_result.get('overall_compliance').upper()}")
                print(f"  • Compliance score: {policy_result.get('compliance_score'):.1%}")
                
                violations_count = policy_result.get('violations_count', 0)
                if violations_count > 0:
                    print(f"  • Total violations: {violations_count}")
                    print(f"    🔴 Critical: {policy_result.get('critical_violations', 0)}")
                    print(f"    🟠 High: {policy_result.get('high_violations', 0)}")
                    print(f"    🟡 Medium/Low: {violations_count - policy_result.get('critical_violations', 0) - policy_result.get('high_violations', 0)}")
                    
                    # Show sample violations
                    violations = policy_result.get('violations', [])
                    if violations:
                        print(f"\n  Sample violations:")
                        for v in violations[:2]:  # Show first 2
                            print(f"    [{v['severity'].upper()}] {v['category']}")
                            print(f"      • {v['issue'][:80]}...")
                            print(f"      • Circular: {v.get('circular_reference', 'N/A')}")
                else:
                    print(f"  ✓ No policy violations detected")
                
                # Show commendations
                commendations = policy_result.get('commendations', [])
                if commendations:
                    print(f"  • Commendations: {len(commendations)} positive findings")
                
                # Show BSP circulars referenced
                circulars = policy_result.get('circular_references', [])
                if circulars:
                    print(f"  • BSP Circulars Referenced: {len(circulars)}")
                    for i, circ in enumerate(circulars[:3], 1):
                        print(f"    {i}. {circ}")
                    if len(circulars) > 3:
                        print(f"    ... and {len(circulars) - 3} more")
                
                # Final recommendation
                if policy_result.get('requires_revision'):
                    print(f"\n  ⚠️  RECOMMENDATION: REVISION REQUIRED BEFORE PUBLICATION")
                else:
                    print(f"\n  ✅ RECOMMENDATION: APPROVED FOR USE")
            elif policy_result and not policy_result.get('success'):
                print(f"\n{'─'*70}")
                print("BSP Policy Alignment Results:")
                print(f"  ✗ Policy check failed: {policy_result.get('error', 'Unknown')}")
            else:
                print(f"\n{'─'*70}")
                print("BSP Policy Alignment Results:")
                print(f"  ⊘ Policy check was skipped or not configured")
        
        # === DISPLAY IMPROVEMENTS SCORECARD (already calculated) ===
        print("\n" + "=" * 70)
        print("📊 IMPROVEMENTS SCORECARD")
        print("=" * 70)
        
        # Display scores using already-calculated values
        if evidence_stability.get('is_cumulative'):
            print("✅ [1/1] Evidence Stability: Cumulative store working")
        else:
            print("❌ [0/1] Evidence Stability: Evidence loss detected")
        
        if claim_validation.get('claim_outline_used'):
            print("✅ [1/1] Claim-Based Style: Active")
        else:
            print("⚠️  [0/1] Claim-Based Style: Not active")
        
        if citation_discipline.get('disciplined'):
            print("✅ [1/1] Citation Discipline: Maintained (≤2 IDs/sentence)")
        else:
            print(f"⚠️  [0/1] Citation Discipline: {len(citation_discipline.get('citation_violations', []))} violations")
        
        if sentence_verify.get('is_sentence_level'):
            print("✅ [1/1] Sentence-Level Verification: Active")
        else:
            print("⚠️  [0/1] Sentence-Level Verification: Using segment-level")
        
        if boilerplate.get('filtering_active'):
            print("✅ [1/1] Boilerplate Filtering: Active")
        else:
            print("⚠️  [0/1] Boilerplate Filtering: Not detected")
        
        if policy_enum.get('is_strict_enum') and not policy_enum.get('is_unknown'):
            print("✅ [1/1] Strict Policy Enum: Enforced")
        elif policy_enum.get('is_unknown'):
            print("❌ [0/1] Strict Policy Enum: UNKNOWN status found")
        else:
            print("⚠️  [0/1] Strict Policy Enum: Non-standard enum")
        
        print(f"\n{'─'*70}")
        print(f"OVERALL GRADE: {score}/{max_score} improvements active ({grade_percent:.0f}%) - {grade_letter}")
        print(f"{'─'*70}")
        
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())
