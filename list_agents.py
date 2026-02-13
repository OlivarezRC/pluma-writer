#!/usr/bin/env python3
"""
List available Azure AI agents in the project
"""
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
import msal

load_dotenv()


async def list_agents():
    """List all available agents in the Azure AI project"""
    
    print("=" * 70)
    print("AZURE AI AGENTS - LIST AVAILABLE AGENTS")
    print("=" * 70)
    
    # Load credentials
    endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
    client_id = os.getenv("AZURE_CLIENT_ID")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    print("\n📋 Configuration:")
    print(f"  Endpoint: {endpoint}")
    
    # Get Azure AD token
    print("\n" + "-" * 70)
    print("Getting Azure AD token...")
    print("-" * 70)
    
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = msal.ConfidentialClientApplication(
        client_id,
        authority=authority,
        client_credential=client_secret,
    )
    
    scope = "https://ai.azure.com/.default"
    result = app.acquire_token_for_client(scopes=[scope])
    
    if "access_token" not in result:
        print(f"❌ Failed to get token: {result.get('error_description', result.get('error'))}")
        return
    
    token = result["access_token"]
    print(f"✅ Got token")
    
    # List agents/assistants
    print("\n" + "-" * 70)
    print("Listing available agents...")
    print("-" * 70)
    
    base_url = endpoint.rstrip('/')
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Try different API endpoints to list agents
    endpoints_to_try = [
        f"{base_url}/assistants?api-version=v1",
        f"{base_url}/agents?api-version=v1",
        f"{base_url}?api-version=v1",
    ]
    
    for url in endpoints_to_try:
        print(f"\nTrying endpoint: {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    status = response.status
                    response_text = await response.text()
                    
                    print(f"  Status: {status}")
                    
                    if status == 200:
                        try:
                            data = await response.json()
                            print(f"\n✅ Success! Response:")
                            print(json.dumps(data, indent=2))
                            
                            # Try to extract agents/assistants from response
                            if isinstance(data, dict):
                                agents = data.get('data', data.get('value', []))
                                if agents:
                                    print(f"\n📋 Found {len(agents)} agent(s):")
                                    for i, agent in enumerate(agents, 1):
                                        agent_id = agent.get('id', 'N/A')
                                        name = agent.get('name', 'N/A')
                                        model = agent.get('model', 'N/A')
                                        print(f"  {i}. ID: {agent_id}")
                                        print(f"     Name: {name}")
                                        print(f"     Model: {model}")
                            
                        except Exception as e:
                            print(f"  ⚠️ Could not parse JSON: {str(e)}")
                            print(f"  Response: {response_text[:500]}")
                    else:
                        print(f"  ❌ Failed: {response_text[:300]}")
                        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # Also try to get info about the specific agent ID we have
    print("\n" + "-" * 70)
    print("Checking specific agent...")
    print("-" * 70)
    
    agent_id = os.getenv("AZURE_POLICY_AGENT_ID")
    print(f"Agent ID: {agent_id}")
    
    url = f"{base_url}/assistants/{agent_id}?api-version=v1"
    print(f"Endpoint: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                status = response.status
                response_text = await response.text()
                
                print(f"Status: {status}")
                
                if status == 200:
                    try:
                        data = await response.json()
                        print(f"\n✅ Agent Details:")
                        import json
                        print(json.dumps(data, indent=2))
                    except:
                        print(f"Response: {response_text}")
                else:
                    print(f"Response: {response_text[:500]}")
                    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import json
    asyncio.run(list_agents())
