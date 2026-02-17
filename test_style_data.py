#!/usr/bin/env python3
"""Quick test to see what style data is actually loaded from Cosmos DB"""
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Import the function that loads styles
from app.writer_main import get_random_style_from_db

print("=" * 70)
print("TESTING COSMOS DB STYLE LOADING")
print("=" * 70)

# Get a style from Cosmos DB
style = get_random_style_from_db()

if not style:
    print("❌ No style returned from Cosmos DB")
    exit(1)

print(f"\n✅ Successfully loaded style")
print(f"\nStyle Metadata:")
print(f"  • ID: {style.get('id', 'N/A')}")
print(f"  • Name: {style.get('name', 'N/A')}")
print(f"  • Speaker: {style.get('speaker', 'N/A')}")
print(f"  • Audience: {style.get('audience_setting_classification', 'N/A')}")

print(f"\n" + "=" * 70)
print("STYLE COMPONENTS")
print("=" * 70)

# Check style_description
style_desc = style.get('style_description', '')
print(f"\n1. STYLE DESCRIPTION ({len(style_desc)} chars):")
print("-" * 70)
if style_desc:
    # Try to parse as JSON
    try:
        style_json = json.loads(style_desc)
        print(json.dumps(style_json, indent=2)[:1000])
        if len(json.dumps(style_json, indent=2)) > 1000:
            print("\n... (truncated)")
    except:
        print(style_desc[:1000])
        if len(style_desc) > 1000:
            print("\n... (truncated)")
else:
    print("❌ EMPTY - No style description!")

# Check global_rules
global_rules = style.get('global_rules', '')
print(f"\n2. GLOBAL RULES ({len(global_rules)} chars):")
print("-" * 70)
if global_rules:
    print(global_rules[:1000])
    if len(global_rules) > 1000:
        print("\n... (truncated)")
else:
    print("❌ EMPTY - No global rules!")

# Check guidelines
guidelines = style.get('guidelines', '')
print(f"\n3. GUIDELINES ({len(guidelines)} chars):")
print("-" * 70)
if guidelines:
    print(guidelines[:500])
    if len(guidelines) > 500:
        print("\n... (truncated)")
else:
    print("⚠️  EMPTY - No guidelines")

# Check example
example = style.get('example', '')
print(f"\n4. EXAMPLE ({len(example)} chars):")
print("-" * 70)
if example:
    print(example[:800])
    if len(example) > 800:
        print("\n... (truncated)")
else:
    print("❌ EMPTY - No example text!")

print(f"\n" + "=" * 70)
print("STYLE INSTRUCTION QUALITY ASSESSMENT")
print("=" * 70)

# Calculate quality score
quality_checks = {
    "Has style_description": bool(style_desc),
    "Style description is detailed (>100 chars)": len(style_desc) > 100,
    "Has global_rules": bool(global_rules),
    "Global rules are detailed (>100 chars)": len(global_rules) > 100,
    "Has example text": bool(example),
    "Example is substantial (>500 chars)": len(example) > 500,
    "Has speaker info": bool(style.get('speaker')),
    "Has audience classification": bool(style.get('audience_setting_classification'))
}

passed = sum(quality_checks.values())
total = len(quality_checks)

print(f"\nQuality Score: {passed}/{total} ({passed/total*100:.0f}%)")
print(f"\nDetailed Checks:")
for check, result in quality_checks.items():
    symbol = "✅" if result else "❌"
    print(f"  {symbol} {check}")

# Overall assessment
print(f"\n" + "=" * 70)
print("VERDICT:")
print("=" * 70)
if passed >= 6:
    print("✅ GOOD - Style instructions are comprehensive")
elif passed >= 4:
    print("⚠️  MODERATE - Style instructions need improvement")
else:
    print("❌ POOR - Style instructions are insufficient for quality output")

print(f"\n💡 Recommendation:")
if not style_desc or len(style_desc) < 100:
    print("   - Add detailed style_description with rhetorical patterns")
if not global_rules or len(global_rules) < 100:
    print("   - Add comprehensive global_rules for consistency")
if not example or len(example) < 500:
    print("   - Add substantial example text (500+ chars) for style imitation")
