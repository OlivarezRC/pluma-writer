#!/usr/bin/env python3
"""
Test Azure Policy Agent connection with Azure AD authentication
"""
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
import msal

load_dotenv()


async def get_azure_ad_token():
    """Get Azure AD access token using client credentials"""
    
    client_id = os.getenv("AZURE_CLIENT_ID")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    if not all([client_id, tenant_id, client_secret]):
        print("❌ Missing Azure AD credentials")
        return None
    
    # Create MSAL confidential client
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = msal.ConfidentialClientApplication(
        client_id,
        authority=authority,
        client_credential=client_secret,
    )
    
    # Get token for Azure AI services
    # The endpoint requires https://ai.azure.com audience
    scopes = [
        "https://ai.azure.com/.default",
        "https://cognitiveservices.azure.com/.default",
        "https://ml.azure.com/.default",
    ]
    
    for scope in scopes:
        print(f"  Trying scope: {scope}")
        result = app.acquire_token_for_client(scopes=[scope])
        
        if "access_token" in result:
            print(f"  ✅ Got token with scope: {scope}")
            return result["access_token"]
        else:
            print(f"  ❌ Failed: {result.get('error_description', result.get('error'))}")
    
    return None


async def test_with_bearer_token():
    """Test connection using Azure AD bearer token"""
    
    print("=" * 70)
    print("AZURE POLICY AGENT - AZURE AD AUTHENTICATION TEST")
    print("=" * 70)
    
    # Load credentials
    endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
    agent_id = os.getenv("AZURE_POLICY_AGENT_ID")
    deployment = os.getenv("AZURE_POLICY_DEPLOYMENT")
    
    print("\n📋 Configuration:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Agent ID: {agent_id}")
    print(f"  Deployment: {deployment}")
    
    # Get Azure AD token
    print("\n" + "-" * 70)
    print("Step 1: Getting Azure AD token")
    print("-" * 70)
    
    token = await get_azure_ad_token()
    
    if not token:
        print("❌ Failed to get Azure AD token")
        return False
    
    print(f"✅ Got token: ***{token[-20:]}")
    
    # Test connection with bearer token
    print("\n" + "-" * 70)
    print("Step 2: Testing connection with bearer token")
    print("-" * 70)
    
    api_versions = ["v1", "2024-02-15-preview", "2024-07-01-preview"]
    
    for api_version in api_versions:
        print(f"\n  Testing API version: {api_version}")
        
        base_url = endpoint.rstrip('/')
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        url = f"{base_url}/threads?api-version={api_version}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json={}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    status = response.status
                    response_text = await response.text()
                    
                    print(f"    Status: {status}")
                    
                    if status in [200, 201]:
                        print(f"    ✅ SUCCESS!")
                        print(f"    Response: {response_text[:200]}")
                        return True
                    else:
                        print(f"    ❌ Failed: {response_text[:200]}")
                        
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:150]}")
    
    return False


async def test_with_api_key():
    """Test connection using API key for comparison"""
    
    print("\n" + "=" * 70)
    print("FALLBACK: Testing with API key")
    print("=" * 70)
    
    endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
    api_key = os.getenv("AZURE_AI_API_KEY")
    
    base_url = endpoint.rstrip('/')
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    url = f"{base_url}/threads?api-version=v1"
    print(f"\nEndpoint: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                status = response.status
                response_text = await response.text()
                
                print(f"Status: {status}")
                print(f"Response: {response_text[:300]}")
                
                if status in [200, 201]:
                    return True
                    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return False


async def main():
    # Try Azure AD authentication
    success = await test_with_bearer_token()
    
    if not success:
        # Fallback to API key
        success = await test_with_api_key()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Connection successful!")
    else:
        print("❌ All connection attempts failed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
