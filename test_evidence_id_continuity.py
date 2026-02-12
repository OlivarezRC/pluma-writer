#!/usr/bin/env python3
"""
Quick test to verify evidence IDs continue across iterations
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import process_with_iterative_refinement


async def test_evidence_id_continuity():
    """Test that evidence IDs continue properly across iterations"""
    
    print("="*70)
    print("TESTING EVIDENCE ID CONTINUITY ACROSS ITERATIONS")
    print("="*70)
    
    user_query = "What are the key innovations in transformer neural networks?"
    
    user_sources = {
        "topics": "transformer innovations",
        "links": ["https://arxiv.org/abs/1706.03762"],
        "attachments": []
    }
    
    print(f"\nQuery: {user_query}")
    print(f"Running 3 iterations...\n")
    
    try:
        results = await process_with_iterative_refinement(
            query=user_query,
            sources=user_sources,
            max_iterations=3
        )
        
        print("\n" + "="*70)
        print("EVIDENCE ID ANALYSIS")
        print("="*70)
        
        for i, iteration in enumerate(results['iterations'], 1):
            evidence_store = iteration['results'].get('cumulative_evidence_store', [])
            
            if evidence_store:
                # Get all IDs
                ids = [e.get('id') for e in evidence_store if e.get('id')]
                
                if ids:
                    first_id = ids[0]
                    last_id = ids[-1]
                    
                    print(f"\nIteration {i}:")
                    print(f"  Total evidence: {len(evidence_store)}")
                    print(f"  ID Range: {first_id} → {last_id}")
                    print(f"  New evidence this iteration: {iteration.get('new_evidence_count', 0)}")
                    
                    # Check if IDs are sequential
                    id_numbers = []
                    for id_str in ids:
                        if id_str and id_str.startswith('E'):
                            try:
                                num = int(id_str[1:])
                                id_numbers.append(num)
                            except ValueError:
                                pass
                    
                    if id_numbers:
                        min_id = min(id_numbers)
                        max_id = max(id_numbers)
                        print(f"  Numeric range: {min_id} to {max_id}")
                        
                        # Check for duplicates
                        unique_ids = set(ids)
                        if len(unique_ids) < len(ids):
                            print(f"  ⚠️ WARNING: Duplicate IDs found!")
                            print(f"     Total IDs: {len(ids)}, Unique: {len(unique_ids)}")
                        else:
                            print(f"  ✓ All IDs are unique")
        
        # Final check: are IDs continuing across iterations or resetting?
        print("\n" + "="*70)
        print("CONTINUITY CHECK")
        print("="*70)
        
        iteration_ranges = []
        for i, iteration in enumerate(results['iterations'], 1):
            evidence_store = iteration['results'].get('evidence_store', [])  # New evidence only
            if evidence_store:
                ids = [e.get('id') for e in evidence_store if e.get('id')]
                if ids:
                    id_numbers = []
                    for id_str in ids:
                        if id_str and id_str.startswith('E'):
                            try:
                                num = int(id_str[1:])
                                id_numbers.append(num)
                            except ValueError:
                                pass
                    if id_numbers:
                        iteration_ranges.append((i, min(id_numbers), max(id_numbers)))
        
        print(f"\nNew evidence ID ranges per iteration:")
        for iter_num, min_id, max_id in iteration_ranges:
            print(f"  Iteration {iter_num}: E{min_id} to E{max_id}")
        
        # Check if they continue
        if len(iteration_ranges) >= 2:
            iter1_max = iteration_ranges[0][2]
            iter2_min = iteration_ranges[1][2]
            
            if iter2_min == iter1_max + 1 or iter2_min > iter1_max:
                print(f"\n✅ SUCCESS: Evidence IDs continue across iterations!")
            else:
                print(f"\n✗ PROBLEM: Evidence IDs reset or overlap!")
                print(f"   Iteration 1 ends at E{iter1_max}")
                print(f"   Iteration 2 starts at E{iteration_ranges[1][1]}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_evidence_id_continuity())
