#!/usr/bin/env python3
"""
Test script for iterative refinement with critiques and adjustments
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_with_iterative_refinement


async def test_iterative_refinement():
    """Test the complete iterative refinement workflow"""
    
    print("=" * 70)
    print("TESTING: ITERATIVE REFINEMENT WITH CRITIQUES")
    print("=" * 70)
    
    user_query = "How do attention mechanisms work in transformers?"
    
    user_sources = {
        "topics": "attention mechanisms transformers",
        "links": [
            "https://arxiv.org/abs/1706.03762"
        ],
        "attachments": []
    }
    
    print(f"\nOriginal Query: {user_query}")
    print(f"Topic: {user_sources['topics']}")
    print(f"Links: {len(user_sources['links'])}")
    print(f"\nRunning 3 iterations with critiques and refinements...")
    
    try:
        # Run the iterative refinement process
        results = await process_with_iterative_refinement(user_query, user_sources, max_iterations=3)
        
        print("\n" + "=" * 70)
        print("ITERATION SUMMARY")
        print("=" * 70)
        
        # Display summary of each iteration
        for iter_data in results["iterations"]:
            iteration = iter_data["iteration"]
            print(f"\n--- ITERATION {iteration} ---")
            print(f"Query Used: {iter_data['query'][:100]}...")
            print(f"New Evidence: {iter_data['new_evidence_count']} items")
            print(f"Cumulative Evidence: {iter_data['cumulative_evidence_count']} items")
            
            # Show summary
            summary_result = iter_data["results"].get("generated_summary", {})
            if summary_result.get("success"):
                summary = summary_result["summary"]
                print(f"✓ Summary Generated: {len(summary)} characters")
                print(f"\n  Summary Preview:")
                print(f"  {summary[:300]}...")
            else:
                print(f"✗ Summary Failed: {summary_result.get('error', 'Unknown')}")
            
            # Show critique
            if iter_data.get("critique"):
                critique = iter_data["critique"]
                if critique.get("success"):
                    print(f"\n  ✓ Critique Generated:")
                    critique_text = critique["critique"]
                    print(f"  {critique_text[:400]}...")
                else:
                    print(f"\n  ✗ Critique Failed")
            
            # Show adjustments
            if iter_data.get("adjustments"):
                adj = iter_data["adjustments"]
                print(f"\n  → Adjusted Query for Next Iteration:")
                print(f"  {adj['adjusted_query'][:200]}...")
            
            print("-" * 70)
        
        # Show final results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        final_summary = results.get("final_summary", {})
        if final_summary.get("success"):
            print(f"\n✓ Final Summary (Iteration 3)")
            print(f"  Length: {len(final_summary['summary'])} characters")
            print(f"  Based on: {results['final_evidence_count']} evidence items")
            print(f"\n{'-'*70}")
            print(final_summary["summary"])
            print(f"{'-'*70}")
        
        # Save detailed results
        print("\n" + "=" * 70)
        print("SAVING ITERATION RESULTS")
        print("=" * 70)
        
        # Save each iteration's summary separately
        for iter_data in results["iterations"]:
            iteration = iter_data["iteration"]
            summary_result = iter_data["results"].get("generated_summary", {})
            
            if summary_result.get("success"):
                # Safely extract critique
                critique_obj = iter_data.get("critique")
                critique_text = "N/A"
                if critique_obj and isinstance(critique_obj, dict):
                    critique_text = critique_obj.get("critique", "N/A")
                
                iter_output = {
                    "iteration": iteration,
                    "query_used": iter_data["query"],
                    "summary": summary_result["summary"],
                    "evidence_count": iter_data["cumulative_evidence_count"],
                    "critique": critique_text,
                    "adjustments": iter_data.get("adjustments", {})
                }
                
                filename = f"iteration_{iteration}_summary.json"
                with open(filename, 'w') as f:
                    json.dump(iter_output, f, indent=2)
                
                print(f"✓ Iteration {iteration} saved to: {filename}")
        
        # Save complete results
        complete_output = {
            "original_query": results["original_query"],
            "total_iterations": results["total_iterations"],
            "final_evidence_count": results["final_evidence_count"],
            "iterations_summary": [
                {
                    "iteration": iter_data["iteration"],
                    "query": iter_data["query"],
                    "new_evidence": iter_data["new_evidence_count"],
                    "cumulative_evidence": iter_data["cumulative_evidence_count"],
                    "summary_length": len(iter_data["results"].get("generated_summary", {}).get("summary", "")),
                    "has_critique": iter_data.get("critique") is not None,
                    "has_adjustments": iter_data.get("adjustments") is not None
                }
                for iter_data in results["iterations"]
            ],
            "final_summary": results["final_summary"],
            "cumulative_evidence_store": results["cumulative_evidence_store"]
        }
        
        with open('iterative_refinement_output.json', 'w') as f:
            json.dump(complete_output, f, indent=2)
        
        print(f"\n✓ Complete results saved to: iterative_refinement_output.json")
        
        # Show statistics
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"\nTotal Iterations: {results['total_iterations']}")
        print(f"Final Evidence Items: {results['final_evidence_count']}")
        print(f"Final Summary Length: {len(results['final_summary'].get('summary', ''))} characters")
        
        print("\nIteration Progression:")
        for iter_data in results["iterations"]:
            print(f"  Iteration {iter_data['iteration']}: "
                  f"{iter_data['new_evidence_count']} new evidence → "
                  f"{iter_data['cumulative_evidence_count']} total")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(test_iterative_refinement())
