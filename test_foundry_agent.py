"""
Test Azure AI Foundry Agent API connectivity
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
print("AZURE AI FOUNDRY AGENT TEST")
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

# Test 1: Try to invoke the agent directly with chat completions format
print("Test 1: Chat Completions Format")
print("-" * 70)

test_message = "Hello, are you the BSP policy checker agent?"

# Azure AI Foundry uses chat completions format
url = f"{endpoint}/{deployment}/chat/completions?api-version={api_version}"
print(f"URL: {url}\n")

payload = {
    "messages": [
        {"role": "user", "content": test_message}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ SUCCESS!\n")
        print("Response:")
        print(json.dumps(result, indent=2))
        
        # Extract the message if available
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0].get('message', {}).get('content', '')
            print(f"\nAgent Response:\n{message}")
    else:
        print(f"❌ FAILED")
        print(f"Response: {response.text}\n")
        
except Exception as e:
    print(f"❌ ERROR: {e}\n")

# Test 2: Try inference endpoint format
print("\n" + "="*70)
print("Test 2: Inference Endpoint Format")
print("-" * 70)

url2 = f"{endpoint}/chat/completions?api-version={api_version}"
print(f"URL: {url2}\n")

payload2 = {
    "model": deployment,
    "messages": [
        {"role": "user", "content": test_message}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

try:
    response2 = requests.post(url2, headers=headers, json=payload2, timeout=30)
    print(f"Status: {response2.status_code}")
    
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"✅ SUCCESS!\n")
        print("Response:")
        print(json.dumps(result2, indent=2))
        
        if 'choices' in result2 and len(result2['choices']) > 0:
            message = result2['choices'][0].get('message', {}).get('content', '')
            print(f"\nAgent Response:\n{message}")
    else:
        print(f"❌ FAILED")
        print(f"Response: {response2.text}\n")
        
except Exception as e:
    print(f"❌ ERROR: {e}\n")

# Test 3: Try without deployment in path
print("\n" + "="*70)
print("Test 3: Root Models Endpoint")
print("-" * 70)

url3 = endpoint.replace("/models", "") + "/chat/completions?api-version=" + api_version
print(f"URL: {url3}\n")

payload3 = {
    "model": deployment,
    "messages": [
        {"role": "user", "content": test_message}
    ],
    "max_tokens": 500
}

try:
    response3 = requests.post(url3, headers=headers, json=payload3, timeout=30)
    print(f"Status: {response3.status_code}")
    
    if response3.status_code == 200:
        result3 = response3.json()
        print(f"✅ SUCCESS!\n")
        print("Response:")
        print(json.dumps(result3, indent=2))
        
        if 'choices' in result3 and len(result3['choices']) > 0:
            message = result3['choices'][0].get('message', {}).get('content', '')
            print(f"\nAgent Response:\n{message}")
    else:
        print(f"❌ FAILED")
        print(f"Response: {response3.text}\n")
        
except Exception as e:
    print(f"❌ ERROR: {e}\n")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
