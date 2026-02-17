#!/usr/bin/env python3
"""Test fetching sample speeches from Cosmos DB"""
import os
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import get_sample_speeches_from_db

print("=" * 70)
print("TESTING SPEECH FETCHING FROM COSMOS DB")
print("=" * 70)

# Test with different speaker names
test_speakers = [
    "Benjamin E. Diokno",
    "Eli M. Remolona Jr.",
    "Felipe M. Medalla"
]

for speaker in test_speakers:
    print(f"\n{'=' * 70}")
    print(f"Fetching speeches for: {speaker}")
    print("=" * 70)
    
    speeches = get_sample_speeches_from_db(speaker, max_speeches=3)
    
    if speeches:
        print(f"\n✅ Found {len(speeches)} speech(es)")
        for i, speech in enumerate(speeches, 1):
            print(f"\n--- Speech {i} Preview (first 500 chars) ---")
            print(speech[:500])
            print(f"\n... ({len(speech)} total characters)")
    else:
        print(f"\n❌ No speeches found for {speaker}")

print(f"\n{'=' * 70}")
print("TEST COMPLETE")
print("=" * 70)
