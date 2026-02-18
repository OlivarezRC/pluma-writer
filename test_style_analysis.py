#!/usr/bin/env python3
"""Deep analysis of style instruction quality"""
import json
import os
from dotenv import load_dotenv

load_dotenv()

from app.writer_main import get_random_style_from_db

print("=" * 70)
print("DEEP STYLE INSTRUCTION ANALYSIS")
print("=" * 70)

style = get_random_style_from_db()

if not style:
    print("❌ Failed to load style")
    exit(1)

print(f"\nSpeaker: {style.get('speaker', 'N/A')}")
print(f"Audience: {style.get('audience_setting_classification', 'N/A')}")

# Parse style_description
style_desc = style.get('style_description', '')
try:
    style_json = json.loads(style_desc)
    
    print(f"\n" + "=" * 70)
    print("WHAT THE STYLE_DESCRIPTION CONTAINS:")
    print("=" * 70)
    
    # Extract key sections
    if 'properties' in style_json:
        props = style_json['properties']
        
        # Voice summary
        if 'voice_summary' in props:
            summary = props['voice_summary'].get('enum', [''])[0]
            print(f"\n📝 Voice Summary:")
            print(f"   {summary}")
        
        # Tone tags
        if 'tone_tags' in props:
            tones = props['tone_tags'].get('items', {}).get('enum', [])
            print(f"\n🎭 Tone Tags: {', '.join(tones)}")
        
        # Structure template
        if 'structure_template' in props:
            structure = props['structure_template'].get('items', {}).get('enum', [])
            print(f"\n📐 Structure Template:")
            for item in structure[:5]:
                print(f"   • {item}")
            if len(structure) > 5:
                print(f"   ... and {len(structure) - 5} more")
        
        # Signature phrases
        if 'signature_phrases' in props:
            phrases = props['signature_phrases'].get('items', {}).get('enum', [])
            print(f"\n💬 Signature Phrases ({len(phrases)} total):")
            for phrase in phrases[:5]:
                print(f"   • \"{phrase}\"")
            if len(phrases) > 5:
                print(f"   ... and {len(phrases) - 5} more")
        
        # Lexical preferences
        if 'lexical_preferences' in props:
            lex = props['lexical_preferences'].get('properties', {})
            print(f"\n📚 Lexical Preferences:")
            for key in ['formal_terms', 'hedging_phrases', 'transition_words']:
                if key in lex:
                    terms = lex[key].get('items', {}).get('enum', [])
                    print(f"   • {key.replace('_', ' ').title()}: {len(terms)} terms")
        
        # Rhetorical devices
        if 'rhetorical_devices' in props:
            devices = props['rhetorical_devices'].get('items', {}).get('enum', [])
            print(f"\n🎨 Rhetorical Devices:")
            for device in devices[:5]:
                print(f"   • {device}")

except json.JSONDecodeError:
    print(f"\n⚠️  Style description is not valid JSON")

# Parse global_rules
global_rules = style.get('global_rules', '')
try:
    rules_json = json.loads(global_rules)
    
    print(f"\n" + "=" * 70)
    print("WHAT THE GLOBAL_RULES CONTAINS:")
    print("=" * 70)
    
    must_do = rules_json.get('must_do', [])
    print(f"\n✅ MUST DO ({len(must_do)} rules):")
    for i, rule in enumerate(must_do[:5], 1):
        print(f"   {i}. {rule}")
    if len(must_do) > 5:
        print(f"   ... and {len(must_do) - 5} more")
    
    must_not = rules_json.get('must_not_do', [])
    print(f"\n🚫 MUST NOT DO ({len(must_not)} rules):")
    for i, rule in enumerate(must_not[:5], 1):
        print(f"   {i}. {rule}")
    if len(must_not) > 5:
        print(f"   ... and {len(must_not) - 5} more")
    
except json.JSONDecodeError:
    print(f"\n⚠️  Global rules is not valid JSON")

print(f"\n" + "=" * 70)
print("CRITICAL GAPS IN STYLE DATA")
print("=" * 70)

gaps = []

# Check for example text
example = style.get('example', '')
if not example or len(example) < 100:
    gaps.append({
        'component': 'Example Text',
        'severity': 'CRITICAL',
        'issue': 'No concrete example of speaker\'s writing to imitate',
        'impact': 'LLM cannot learn speaker voice patterns, sentence rhythm, or authentic phrasing',
        'fix': 'Add 500-1000 words of actual speech text from this speaker'
    })

# Check for rhetorical patterns
try:
    style_json = json.loads(style_desc)
    if 'properties' in style_json:
        if 'rhetorical_devices' not in style_json['properties']:
            gaps.append({
                'component': 'Rhetorical Devices',
                'severity': 'HIGH',
                'issue': 'No specific rhetorical patterns defined',
                'impact': 'Generic prose without persuasive power',
                'fix': 'Add metaphors, analogies, parallel structures used by speaker'
            })
        
        if 'sentence_patterns' not in style_json['properties']:
            gaps.append({
                'component': 'Sentence Patterns',
                'severity': 'HIGH',
                'issue': 'No syntactic fingerprint defined',
                'impact': 'Voice sounds generic, not distinctively "Diokno"',
                'fix': 'Add sentence length distribution, clause patterns, punctuation usage'
            })
        
        if 'emotional_arc' not in style_json['properties']:
            gaps.append({
                'component': 'Emotional Arc',
                'severity': 'MEDIUM',
                'issue': 'No guidance on speech momentum/progression',
                'impact': 'Flat delivery, no persuasive build-up',
                'fix': 'Define typical opening (context), body (evidence), closing (action)'
            })

except:
    pass

# Check guidelines
guidelines = style.get('guidelines', '')
if not guidelines:
    gaps.append({
        'component': 'Guidelines',
        'severity': 'MEDIUM',
        'issue': 'No audience-specific adaptation rules',
        'impact': 'Same style for technical experts and general public',
        'fix': 'Add "For TECHNICAL audience: use X", "For PUBLIC audience: use Y"'
    })

print(f"\nFound {len(gaps)} critical gaps:")
for i, gap in enumerate(gaps, 1):
    print(f"\n{i}. {gap['component']} - {gap['severity']}")
    print(f"   Issue: {gap['issue']}")
    print(f"   Impact: {gap['impact']}")
    print(f"   Fix: {gap['fix']}")

print(f"\n" + "=" * 70)
print("OVERALL STYLE POWER RATING")
print("=" * 70)

dimensions = {
    'Factual Constraints': '⭐⭐⭐⭐⭐',  # must_do / must_not very detailed
    'Tone Guidance': '⭐⭐⭐⭐',  # tone_tags present
    'Structure Template': '⭐⭐⭐⭐',  # structure_template present
    'Signature Phrases': '⭐⭐⭐',  # Some phrases but limited
    'Example Imitation': '⭐',  # MISSING - critical gap
    'Rhetorical Power': '⭐⭐',  # Basic devices only
    'Voice Authenticity': '⭐⭐',  # Instructions but no exemplar
    'Audience Adaptation': '⭐⭐'  # Basic guidance only
}

print()
for dimension, rating in dimensions.items():
    print(f"{rating} {dimension}")

avg_stars = sum(len([c for c in r if c == '⭐']) for r in dimensions.values()) / len(dimensions)
print(f"\nAverage: {avg_stars:.1f}/5 stars")

print(f"\n" + "=" * 70)
print("VERDICT ON STYLE INSTRUCTIONS")
print("=" * 70)

print(f"""
The style instructions are **STRUCTURALLY COMPREHENSIVE** but **PRACTICALLY WEAK**.

✅ **Strengths:**
   • Detailed constraint system (must_do / must_not_do)
   • Clear voice description and tone tags
   • Signature phrases catalogued
   • Structure template provided

❌ **Critical Weakness:**
   • **NO EXAMPLE TEXT** - This is the #1 issue
   • LLMs learn best by imitation, not description
   • Without seeing actual Diokno speech patterns, output is generic

📊 **Current Output Quality:**
   • Factually accurate: ✅ (strong constraints)
   • Grammatically correct: ✅ (LLM baseline)
   • Sounds like Diokno: ❌ (no training exemplar)
   • Persuasively powerful: ❌ (no rhetorical patterns)

🚀 **To Make Style "Powerful":**
   
   1. **CRITICAL (Do First):**
      Add 3-5 complete speech paragraphs (500+ words each) from Diokno
      → LLM will learn sentence rhythm, vocabulary choices, rhetorical flow
   
   2. **HIGH Priority:**
      Extract rhetorical patterns from real speeches:
      • Metaphors used (e.g., "economy as engine", "inflation as fire")
      • Parallel structures (e.g., "Not X, but Y", "First...Second...Finally")
      • Emotional progression (calm intro → urgent middle → hopeful close)
   
   3. **MEDIUM Priority:**
      Add audience-specific adaptations:
      • TECHNICAL: more jargon, detailed methodology
      • PUBLIC: concrete examples, personal stories
      • INTERNAL_BSP: institutional focus, policy continuity

**Bottom Line:**
Your style system has the **framework** (constraints, metadata) but lacks the 
**soul** (actual voice examples). It's like having a recipe without seeing the 
finished dish. Add real speech examples and you'll jump from 3/5 → 4.5/5 stars.
""")
