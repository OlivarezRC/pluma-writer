#!/usr/bin/env python3
"""
Test script for BSP Policy Alignment Checker using Azure AI Agent
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.policy_checker import check_speech_policy_alignment


# Sample speech for testing
SAMPLE_SPEECH = """
Good morning, esteemed colleagues from the banking sector and financial institutions.

Today, I want to discuss the transformative role of artificial intelligence in financial portfolio management and risk prediction. The Bangko Sentral ng Pilipinas recognizes that AI and machine learning are fundamentally reshaping how we approach investment strategies and risk assessment in the Philippine financial system.

Recent developments show that deep learning models can analyze vast amounts of market data, identifying patterns that human analysts might miss. These AI-driven systems are being deployed for automated trading, portfolio optimization, and real-time risk monitoring [E1,E2]. Financial institutions worldwide are reporting significant improvements in prediction accuracy and operational efficiency.

However, we must balance innovation with prudent oversight. The BSP maintains its commitment to ensuring that AI applications in finance comply with our regulatory framework, particularly regarding consumer protection, data privacy, and systemic risk management. Any AI system used in portfolio management must be transparent, explainable, and subject to appropriate human oversight [E5,E8].

Looking ahead, we are exploring how machine learning can enhance our own supervisory capabilities. Predictive models show promise in detecting early warning signals of banking stress and potential fraud. We believe these tools can complement our existing frameworks while maintaining the stability and integrity of our financial system.

The BSP will continue to support responsible innovation in fintech, including AI applications, through our regulatory sandbox and ongoing dialogue with industry stakeholders. We encourage financial institutions to explore these technologies while adhering to sound risk management principles and BSP guidelines on digital financial services.

In closing, artificial intelligence represents both opportunity and responsibility. We must harness its potential while safeguarding the interests of Filipino consumers and the stability of our financial system. The BSP remains committed to fostering an environment where innovation and prudence go hand in hand.

Maraming salamat po. Thank you for your attention.
"""


async def test_policy_checker():
    """Test the BSP Policy Alignment Checker"""
    
    print("=" * 70)
    print("BSP POLICY ALIGNMENT CHECKER TEST")
    print("=" * 70)
    
    # Speech metadata
    speech_metadata = {
        "topic": "Use of AI models for financial portfolio management and risk predictions",
        "speaker": "BSP Deputy Governor",
        "audience": "Banking Sector Forum",
        "date": "February 2026",
        "query": "Discuss AI in financial risk management and portfolio optimization"
    }
    
    print(f"\nSpeech Metadata:")
    print(f"  Topic: {speech_metadata['topic']}")
    print(f"  Speaker: {speech_metadata['speaker']}")
    print(f"  Audience: {speech_metadata['audience']}")
    print(f"  Date: {speech_metadata['date']}")
    print(f"\nSpeech Length: {len(SAMPLE_SPEECH)} characters")
    print(f"Speech Preview: {SAMPLE_SPEECH[:200]}...\n")
    
    try:
        # Run policy alignment check
        result = await check_speech_policy_alignment(
            speech_content=SAMPLE_SPEECH,
            speech_metadata=speech_metadata
        )
        
        print("\n" + "=" * 70)
        print("POLICY CHECK RESULTS")
        print("=" * 70)
        
        if result.get('success'):
            print(f"\n🏛️ Overall Compliance: {result['overall_compliance'].upper()}")
            print(f"📊 Compliance Score: {result['compliance_score']:.1%}")
            print(f"🔍 Total Violations: {result['violations_count']}")
            
            if result['violations_count'] > 0:
                print(f"\n⚠️ VIOLATIONS FOUND:")
                print(f"  🔴 Critical: {result['critical_violations']}")
                print(f"  🟠 High: {result['high_violations']}")
                print(f"  🟡 Medium/Low: {result['violations_count'] - result['critical_violations'] - result['high_violations']}")
                
                print(f"\nDetailed Violations:")
                for i, v in enumerate(result['violations'][:5], 1):  # Show first 5
                    print(f"\n  {i}. [{v['severity'].upper()}] {v['category']}")
                    print(f"     Location: {v['location']}")
                    print(f"     Issue: {v['issue'][:150]}...")
                    print(f"     BSP Reference: {v['circular_reference']}")
                    if v.get('recommendation'):
                        print(f"     Fix: {v['recommendation'][:150]}...")
            else:
                print(f"\n✅ No policy violations detected!")
            
            if result.get('commendations'):
                print(f"\n✨ COMMENDATIONS ({len(result['commendations'])}):")
                for i, comm in enumerate(result['commendations'][:3], 1):
                    print(f"  {i}. {comm.get('finding', '')[:150]}...")
            
            if result.get('circular_references'):
                print(f"\n📋 BSP CIRCULARS REFERENCED ({len(result['circular_references'])}):")
                for i, circ in enumerate(result['circular_references'], 1):
                    print(f"  {i}. {circ}")
            
            print(f"\n📝 AGENT ANALYSIS PREVIEW:")
            print(f"{result['agent_analysis'][:500]}...")
            
            print(f"\n{'─'*70}")
            if result['requires_revision']:
                print("⚠️ FINAL RECOMMENDATION: REVISION REQUIRED")
                print("Speech should be revised to address policy violations before publication.")
            else:
                print("✅ FINAL RECOMMENDATION: APPROVED FOR USE")
                print("Speech is aligned with BSP policies and ready for publication.")
            
            # Save full results
            output = {
                "speech_metadata": speech_metadata,
                "policy_check": result,
                "speech_content": SAMPLE_SPEECH
            }
            
            with open('policy_check_test_result.json', 'w') as f:
                json.dump(output, f, indent=2, default=str)
            
            print(f"\n✓ Full results saved to: policy_check_test_result.json")
            
        else:
            print(f"\n✗ Policy check failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error during policy check: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_policy_checker())
