"""
Test Azure AI Foundry Agents API (Threads format)
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Get credentials from .env
endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
api_key = os.getenv("AZURE_POLICY_KEY")
agent_id = os.getenv("AZURE_POLICY_AGENT_ID")
deployment = os.getenv("AZURE_POLICY_DEPLOYMENT")
api_version = os.getenv("AZURE_POLICY_API_VERSION")

print("\n" + "="*70)
print("AZURE AI FOUNDRY AGENTS API TEST (Threads)")
print("="*70)
print(f"Endpoint: {endpoint}")
print(f"Agent ID: {agent_id}")
print(f"Deployment: {deployment}")
print(f"API Version: {api_version}")
print("="*70 + "\n")

headers = {
    "api-key": api_key,
    "Content-Type": "application/json"
}

# Test different base URLs
base_urls = [
    endpoint.replace("/models", ""),  # Remove /models
    endpoint,  # Keep /models
    "https://bspchat.services.ai.azure.com/agents",
    "https://bspchat.services.ai.azure.com/api/agents",
]

for base_url in base_urls:
    print(f"\nTrying base URL: {base_url}")
    print("-" * 70)
    
    # Test 1: Create thread
    thread_url = f"{base_url}/threads?api-version={api_version}"
    print(f"Creating thread: {thread_url}")
    
    try:
        response = requests.post(thread_url, headers=headers, json={}, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"✅ SUCCESS! Thread created")
            print(json.dumps(result, indent=2))
            
            thread_id = result.get('id')
            if thread_id:
                print(f"\nThread ID: {thread_id}")
                
                # Test 2: Send message
                message_url = f"{base_url}/threads/{thread_id}/messages?api-version={api_version}"
                print(f"\nSending message: {message_url}")
                
                message_payload = {
                    "role": "user",
                    "content": "Hello, are you the BSP policy checker?"
                }
                
                msg_response = requests.post(message_url, headers=headers, json=message_payload, timeout=30)
                print(f"Status: {msg_response.status_code}")
                
                if msg_response.status_code in [200, 201]:
                    msg_result = msg_response.json()
                    print(f"✅ Message sent!")
                    print(json.dumps(msg_result, indent=2))
                    
                    # Test 3: Create run with agent
                    run_url = f"{base_url}/threads/{thread_id}/runs?api-version={api_version}"
                    print(f"\nCreating run: {run_url}")
                    
                    run_payload = {
                        "assistant_id": agent_id
                    }
                    
                    run_response = requests.post(run_url, headers=headers, json=run_payload, timeout=30)
                    print(f"Status: {run_response.status_code}")
                    
                    if run_response.status_code in [200, 201]:
                        run_result = run_response.json()
                        print(f"✅ Run created!")
                        print(json.dumps(run_result, indent=2))
                        print("\n🎉 FULL SUCCESS! Agents API is working!")
                        break
                    else:
                        print(f"❌ Run failed: {run_response.text}")
                else:
                    print(f"❌ Message failed: {msg_response.text}")
        else:
            print(f"❌ Thread creation failed: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
