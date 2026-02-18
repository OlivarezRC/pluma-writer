#!/usr/bin/env python3
"""
Test Azure Policy Agent connection with current credentials
"""
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()


async def test_connection_with_version(api_version):
    """Test connection to Azure Policy Agent with specific API version"""
    
    # Load credentials
    endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
    api_key = os.getenv("AZURE_AI_API_KEY")  # This is what's in .env
    agent_id = os.getenv("AZURE_POLICY_AGENT_ID")
    deployment = os.getenv("AZURE_POLICY_DEPLOYMENT")
    
    # Check for missing credentials
    if not all([endpoint, api_key, agent_id, deployment]):
        return False, "Missing credentials"
    
    # Test: Create a thread
    base_url = endpoint.rstrip('/')
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Try with and without api-version parameter
    if api_version == "NONE":
        url = f"{base_url}/threads"
    else:
        url = f"{base_url}/threads?api-version={api_version}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response_text = await response.text()
                
                if response.status in [200, 201]:
                    try:
                        data = await response.json()
                        thread_id = data.get('id')
                        return True, f"Thread created: {thread_id}"
                    except:
                        return True, "Success (200/201)"
                else:
                    return False, f"{response.status}: {response_text[:150]}"
                    
    except Exception as e:
        return False, str(e)[:150]


async def test_connection():
    """Test connection to Azure Policy Agent"""
    
    print("=" * 70)
    print("AZURE POLICY AGENT CONNECTION TEST")
    print("=" * 70)
    
    # Load credentials
    endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
    api_key = os.getenv("AZURE_AI_API_KEY")  # This is what's in .env
    agent_id = os.getenv("AZURE_POLICY_AGENT_ID")
    deployment = os.getenv("AZURE_POLICY_DEPLOYMENT")
    
    print("\n📋 Configuration:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Agent ID: {agent_id}")
    print(f"  Deployment: {deployment}")
    print(f"  API Key: {'***' + api_key[-8:] if api_key else 'NOT SET'}")
    
    # Check for missing credentials
    missing = []
    if not endpoint:
        missing.append("AZURE_POLICY_ENDPOINT")
    if not api_key:
        missing.append("AZURE_AI_API_KEY")
    if not agent_id:
        missing.append("AZURE_POLICY_AGENT_ID")
    if not deployment:
        missing.append("AZURE_POLICY_DEPLOYMENT")
    
    if missing:
        print(f"\n❌ Missing credentials: {', '.join(missing)}")
        return False
    
    # Test different API versions
    print("\n" + "-" * 70)
    print("Testing different API versions:")
    print("-" * 70)
    
    api_versions = [
        "NONE",  # Try without api-version parameter
        "2024-02-15-preview",
        "2024-08-06",
        "2024-07-18",
        "2024-04-01-preview",
        "2024-03-01-preview",
        "2024-02-01",
        "v1",
        "2023-11-01-preview",
    ]
    
    success = False
    for version in api_versions:
        print(f"\n  Testing {version}...", end=" ")
        result, message = await test_connection_with_version(version)
        
        if result:
            print(f"✅ {message}")
            success = True
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS! Use API version: {version}")
            print(f"{'='*70}")
            break
        else:
            print(f"❌ {message}")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(test_connection())
    print("\n" + "=" * 70)
    if result:
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed!")
    print("=" * 70)
