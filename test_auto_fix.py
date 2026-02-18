"""
Test script for the LLM-based auto-fix system.

This script tests the fix_speech_issues() function which automatically revises
problematic speech segments based on validation feedback.

Usage:
    python test_auto_fix.py
"""

import asyncio
import os
from dotenv import load_dotenv
from app.writer_main import fix_speech_issues

# Load environment variables
load_dotenv()

async def test_citation_fix():
    """Test auto-fixing of unverified citation claims."""
    print("\n" + "="*70)
    print("TEST 1: CITATION FIX - Unverified Claims")
    print("="*70)
    
    # Sample speech with unverified claims
    speech_text = """
    Good morning everyone. The Philippine economy has grown by 15% this year [E1], 
    making it the fastest growing economy in Asia. Inflation has been completely 
    eliminated [E2], and unemployment is now at zero percent [E3]. These remarkable 
    achievements demonstrate BSP's world-leading monetary policy.
    """
    
    # Sample evidence store (simplified)
    evidence_store = [
        {
            "id": "E1",
            "claim": "The Philippine economy grew by 6.3% in Q2 2024",
            "source": "BSP Economic Report"
        },
        {
            "id": "E2",
            "claim": "Inflation rate decreased to 3.7% in July 2024",
            "source": "PSA Inflation Report"
        },
        {
            "id": "E3",
            "claim": "Unemployment rate fell to 4.5% in Q2 2024",
            "source": "PSA Labor Force Survey"
        }
    ]
    
    # Issues detected by verification
    issues = [
        {
            "segment": "The Philippine economy has grown by 15% this year [E1]",
            "issue_description": "Claim overstates growth rate. Evidence shows 6.3% growth, not 15%.",
            "suggestion": "Revise to match evidence: 'has grown by 6.3% in Q2 2024'",
            "severity": "CRITICAL"
        },
        {
            "segment": "Inflation has been completely eliminated [E2]",
            "issue_description": "Claim is too absolute. Evidence shows 3.7% inflation rate, not zero.",
            "suggestion": "Soften claim: 'Inflation has decreased to 3.7%'",
            "severity": "HIGH"
        },
        {
            "segment": "unemployment is now at zero percent [E3]",
            "issue_description": "Claim is factually incorrect. Evidence shows 4.5% unemployment.",
            "suggestion": "Correct to: 'unemployment has fallen to 4.5%'",
            "severity": "CRITICAL"
        }
    ]
    
    print(f"\nOriginal speech ({len(speech_text)} chars):")
    print(speech_text.strip())
    
    print(f"\n\n{len(issues)} issues detected:")
    for i, issue in enumerate(issues, 1):
        print(f"\n  Issue #{i} [{issue['severity']}]:")
        print(f"    Segment: {issue['segment']}")
        print(f"    Problem: {issue['issue_description']}")
    
    # Call auto-fix
    print("\n\nCalling fix_speech_issues()...")
    result = await fix_speech_issues(
        speech_text=speech_text,
        issues=issues,
        evidence_store=evidence_store,
        issue_type="citation"
    )
    
    if result.get("success"):
        print(f"\n✅ Auto-fix successful!")
        print(f"   Fixes applied: {result['fixes_applied']}")
        
        print(f"\n\nFixed speech ({len(result['fixed_speech'])} chars):")
        print(result['fixed_speech'].strip())
        
        # Show differences
        print(f"\n\nChanges made:")
        for detail in result.get('fix_details', []):
            print(f"  • Modified {detail.get('sentences_modified', 0)} sentence(s)")
            print(f"  • Length change: {detail.get('original_length', 0)} → {detail.get('fixed_length', 0)} chars")
    else:
        print(f"\n❌ Auto-fix failed: {result.get('error', 'Unknown')}")
    
    return result


async def test_plagiarism_fix():
    """Test auto-fixing of high-similarity text."""
    print("\n\n" + "="*70)
    print("TEST 2: PLAGIARISM FIX - High Similarity Text")
    print("="*70)
    
    # Sample speech with plagiarism risk
    speech_text = """
    According to the World Bank report, economic growth is essential for poverty 
    reduction and shared prosperity. Sustained economic growth is the key driver 
    of poverty reduction and helps to generate resources for development.
    """
    
    # Issues detected by plagiarism checker
    issues = [
        {
            "segment": "economic growth is essential for poverty reduction and shared prosperity",
            "issue_description": "High similarity to World Bank source (similarity score: 0.92)",
            "suggestion": "Restructure completely using different phrasing",
            "severity": "CRITICAL"
        },
        {
            "segment": "Sustained economic growth is the key driver of poverty reduction",
            "issue_description": "Near-exact match to source material (similarity score: 0.89)",
            "suggestion": "Paraphrase with different sentence structure",
            "severity": "HIGH"
        }
    ]
    
    print(f"\nOriginal speech ({len(speech_text)} chars):")
    print(speech_text.strip())
    
    print(f"\n\n{len(issues)} plagiarism risks detected:")
    for i, issue in enumerate(issues, 1):
        print(f"\n  Risk #{i} [{issue['severity']}]:")
        print(f"    Segment: {issue['segment']}")
        print(f"    Problem: {issue['issue_description']}")
    
    # Call auto-fix
    print("\n\nCalling fix_speech_issues()...")
    result = await fix_speech_issues(
        speech_text=speech_text,
        issues=issues,
        evidence_store=None,
        issue_type="plagiarism"
    )
    
    if result.get("success"):
        print(f"\n✅ Auto-fix successful!")
        print(f"   Fixes applied: {result['fixes_applied']}")
        
        print(f"\n\nFixed speech ({len(result['fixed_speech'])} chars):")
        print(result['fixed_speech'].strip())
        
        # Show differences
        print(f"\n\nChanges made:")
        for detail in result.get('fix_details', []):
            print(f"  • Modified {detail.get('sentences_modified', 0)} sentence(s)")
            print(f"  • Length change: {detail.get('original_length', 0)} → {detail.get('fixed_length', 0)} chars")
    else:
        print(f"\n❌ Auto-fix failed: {result.get('error', 'Unknown')}")
    
    return result


async def test_policy_fix():
    """Test auto-fixing of BSP policy violations."""
    print("\n\n" + "="*70)
    print("TEST 3: POLICY FIX - BSP Guideline Violations")
    print("="*70)
    
    # Sample speech with policy violations
    speech_text = """
    I personally guarantee that inflation will never exceed 2% again. BSP's strategies 
    have completely solved all economic problems in the Philippines. Our policies are 
    perfect and will definitely ensure prosperity for everyone without any conditions.
    """
    
    # Issues detected by policy checker
    issues = [
        {
            "segment": "I personally guarantee that inflation will never exceed 2% again",
            "issue_description": "Unqualified absolute statement about future performance violates BSP Circular No. 1011",
            "suggestion": "Add disclaimer: 'BSP targets to maintain inflation within 2-4% range, subject to various economic factors'",
            "severity": "CRITICAL"
        },
        {
            "segment": "BSP's strategies have completely solved all economic problems",
            "issue_description": "Absolute claim without context or qualifications",
            "suggestion": "Soften claim: 'BSP's strategies have contributed to addressing several economic challenges'",
            "severity": "HIGH"
        },
        {
            "segment": "Our policies are perfect and will definitely ensure prosperity",
            "issue_description": "Unhedged forward-looking statement requires disclaimer",
            "suggestion": "Add qualifier: 'aim to support sustainable economic growth, though outcomes depend on various factors'",
            "severity": "MEDIUM"
        }
    ]
    
    print(f"\nOriginal speech ({len(speech_text)} chars):")
    print(speech_text.strip())
    
    print(f"\n\n{len(issues)} policy violations detected:")
    for i, issue in enumerate(issues, 1):
        print(f"\n  Violation #{i} [{issue['severity']}]:")
        print(f"    Segment: {issue['segment']}")
        print(f"    Problem: {issue['issue_description']}")
    
    # Call auto-fix
    print("\n\nCalling fix_speech_issues()...")
    result = await fix_speech_issues(
        speech_text=speech_text,
        issues=issues,
        evidence_store=None,
        issue_type="policy"
    )
    
    if result.get("success"):
        print(f"\n✅ Auto-fix successful!")
        print(f"   Fixes applied: {result['fixes_applied']}")
        
        print(f"\n\nFixed speech ({len(result['fixed_speech'])} chars):")
        print(result['fixed_speech'].strip())
        
        # Show differences
        print(f"\n\nChanges made:")
        for detail in result.get('fix_details', []):
            print(f"  • Modified {detail.get('sentences_modified', 0)} sentence(s)")
            print(f"  • Length change: {detail.get('original_length', 0)} → {detail.get('fixed_length', 0)} chars")
    else:
        print(f"\n❌ Auto-fix failed: {result.get('error', 'Unknown')}")
    
    return result


async def main():
    """Run all auto-fix tests."""
    print("\n" + "="*70)
    print("AUTO-FIX SYSTEM TEST SUITE")
    print("="*70)
    print("Testing the LLM-based auto-revision capability for:")
    print("  1. Citation verification issues (unverified claims)")
    print("  2. Plagiarism detection issues (high similarity)")
    print("  3. Policy compliance issues (BSP guideline violations)")
    print("="*70)
    
    results = {
        "citation_fix": await test_citation_fix(),
        "plagiarism_fix": await test_plagiarism_fix(),
        "policy_fix": await test_policy_fix()
    }
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        fixes = result.get("fixes_applied", 0)
        print(f"{status} | {test_name}: {fixes} fixes applied")
    
    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
