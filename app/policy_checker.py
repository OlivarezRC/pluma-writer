"""
BSP Policy Alignment Checker using Azure AI Agent

This module interfaces with a specialized Azure AI Agent that has been trained
on BSP memorandum circulars and policy guidelines to verify speech compliance.
"""
import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import msal


@dataclass
class PolicyViolation:
    """Represents a policy compliance issue identified by the agent"""
    severity: str  # critical, high, medium, low
    category: str  # e.g., "monetary_policy", "regulatory", "communication"
    location: str  # paragraph/section identifier
    issue: str  # description of the problem
    circular_reference: str  # which BSP circular is violated
    recommendation: str  # suggested fix
    original_text: str  # problematic text snippet


@dataclass
class PolicyCheckResult:
    """Complete policy alignment assessment"""
    overall_compliance: str  # compliant, minor_issues, major_issues, non_compliant
    compliance_score: float  # 0-1
    violations: List[PolicyViolation]
    commendations: List[Dict[str, str]]  # positive findings
    agent_analysis: str  # full agent response
    circular_references: List[str]  # BSP circulars referenced
    requires_revision: bool
    timestamp: str


class AzurePolicyAgentClient:
    """
    Client for interacting with Azure AI Policy Agent
    
    The agent has been pre-configured with BSP memorandum circulars and
    policy guidelines in its knowledge base.
    
    Uses Azure AD authentication for secure access.
    """
    
    def __init__(
        self,
        endpoint: str,
        agent_id: str,
        deployment: str,
        client_id: str,
        tenant_id: str,
        client_secret: str,
        timeout: int = 120,
        api_version: str = "v1"
    ):
        self.endpoint = endpoint
        self.agent_id = agent_id
        self.deployment = deployment
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.client_secret = client_secret
        self.timeout = timeout
        self.api_version = api_version
        
        # Construct base URL for agent API
        self.base_url = f"{endpoint.rstrip('/')}"
        
        # Initialize MSAL app for Azure AD authentication
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.msal_app = msal.ConfidentialClientApplication(
            client_id,
            authority=authority,
            client_credential=client_secret,
        )
        
        # Cache for access token
        self._access_token = None
        self._token_expiry = 0
    
    def _get_access_token(self) -> str:
        """Get Azure AD access token (with caching)"""
        import time
        
        # Check if we have a valid cached token
        if self._access_token and time.time() < self._token_expiry - 300:  # 5 min buffer
            return self._access_token
        
        # Acquire new token with correct scope for Azure AI
        scope = "https://ai.azure.com/.default"
        result = self.msal_app.acquire_token_for_client(scopes=[scope])
        
        if "access_token" not in result:
            error_msg = result.get('error_description', result.get('error', 'Unknown error'))
            raise Exception(f"Failed to acquire Azure AD token: {error_msg}")
        
        # Cache the token
        self._access_token = result["access_token"]
        self._token_expiry = time.time() + result.get("expires_in", 3600)
        
        return self._access_token
    
    def _get_headers(self) -> dict:
        """Get headers with current bearer token"""
        token = self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def create_thread(self) -> str:
        """Create a new conversation thread with the agent"""
        url = f"{self.base_url}/threads?api-version={self.api_version}"
        headers = self._get_headers()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={}) as response:
                if response.status != 201 and response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to create thread: {response.status} - {error_text}")
                
                data = await response.json()
                thread_id = data.get('id')
                print(f"  ✓ Created thread: {thread_id}")
                return thread_id
    
    async def send_message(
        self,
        thread_id: str,
        content: str,
        attachments: List[Dict] = None
    ) -> str:
        """Send a message to the agent in a thread"""
        url = f"{self.base_url}/threads/{thread_id}/messages?api-version={self.api_version}"
        
        payload = {
            "role": "user",
            "content": content
        }
        
        if attachments:
            payload["attachments"] = attachments
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._get_headers(), json=payload) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Failed to send message: {response.status} - {error_text}")
                
                data = await response.json()
                message_id = data.get('id')
                print(f"  ✓ Message sent: {message_id}")
                return message_id
    
    async def create_run(
        self,
        thread_id: str,
        additional_instructions: str = None
    ) -> str:
        """Start a run with the agent on a thread"""
        url = f"{self.base_url}/threads/{thread_id}/runs?api-version={self.api_version}"
        
        payload = {
            "assistant_id": self.agent_id,
            "model": self.deployment
        }
        
        if additional_instructions:
            payload["additional_instructions"] = additional_instructions
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._get_headers(), json=payload) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Failed to create run: {response.status} - {error_text}")
                
                data = await response.json()
                run_id = data.get('id')
                print(f"  ✓ Run started: {run_id}")
                return run_id
    
    async def wait_for_run_completion(
        self,
        thread_id: str,
        run_id: str,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Poll for run completion"""
        url = f"{self.base_url}/threads/{thread_id}/runs/{run_id}?api-version={self.api_version}"
        
        start_time = time.time()
        last_status = None
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Check timeout
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Agent run exceeded timeout of {self.timeout}s")
                
                async with session.get(url, headers=self._get_headers()) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Failed to get run status: {response.status} - {error_text}")
                    
                    data = await response.json()
                    status = data.get('status')
                    
                    # Show status updates
                    if status != last_status:
                        print(f"  ⟳ Agent status: {status}")
                        last_status = status
                    
                    # Terminal states
                    if status == "completed":
                        print(f"  ✓ Run completed in {time.time() - start_time:.1f}s")
                        return data
                    
                    elif status == "failed":
                        error_info = data.get('last_error', {})
                        raise Exception(f"Agent run failed: {error_info.get('message', 'Unknown error')}")
                    
                    elif status == "cancelled":
                        raise Exception("Agent run was cancelled")
                    
                    elif status == "expired":
                        raise Exception("Agent run expired")
                    
                    elif status in ["requires_action"]:
                        # Handle tool calls if needed
                        print(f"  ⚠️ Agent requires action - this shouldn't happen for policy checks")
                        raise Exception(f"Unexpected status: {status}")
                    
                    # Still running, wait and poll again
                    await asyncio.sleep(poll_interval)
    
    async def get_messages(
        self,
        thread_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve messages from a thread"""
        url = f"{self.base_url}/threads/{thread_id}/messages?api-version={self.api_version}"
        params = {"limit": limit}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._get_headers(), params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get messages: {response.status} - {error_text}")
                
                data = await response.json()
                messages = data.get('data', [])
                return messages
    
    async def check_speech_policy(
        self,
        speech_content: str,
        speech_metadata: Dict[str, Any]
    ) -> str:
        """
        Send speech to agent for policy compliance check
        
        Returns the agent's complete analysis as a string
        """
        # Create thread
        thread_id = await self.create_thread()
        
        # Format the policy check request
        request_message = self._format_policy_check_request(speech_content, speech_metadata)
        
        # Send message
        await self.send_message(thread_id, request_message)
        
        # Create run with specific instructions
        additional_instructions = """
        Please provide a comprehensive policy compliance assessment with:
        1. Overall compliance rating (Compliant/Minor Issues/Major Issues/Non-Compliant)
        2. Compliance score (0-100)
        3. List of violations with severity, category, location, issue description, circular reference, and recommendations
        4. Commendations for well-aligned sections
        5. List of BSP circulars referenced
        6. Final recommendation (Approve/Revise/Major Revision Required)
        
        Structure your response clearly with these sections.
        """
        
        run_id = await self.create_run(thread_id, additional_instructions)
        
        # Wait for completion
        await self.wait_for_run_completion(thread_id, run_id)
        
        # Retrieve agent's response
        messages = await self.get_messages(thread_id)
        
        # Extract the assistant's latest message
        assistant_messages = [
            msg for msg in messages
            if msg.get('role') == 'assistant'
        ]
        
        if not assistant_messages:
            raise Exception("No response from agent")
        
        # Get the content from the latest assistant message
        latest_message = assistant_messages[0]
        content_items = latest_message.get('content', [])
        
        # Combine all text content
        full_response = ""
        for item in content_items:
            if item.get('type') == 'text':
                full_response += item.get('text', {}).get('value', '') + "\n"
        
        return full_response.strip()
    
    def _format_policy_check_request(
        self,
        speech_content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Format the policy check request for the agent"""
        
        return f"""Please review the following speech for BSP policy compliance.

## SPEECH METADATA

**Topic**: {metadata.get('topic', 'Not specified')}
**Speaker**: {metadata.get('speaker', 'BSP Official')}
**Audience**: {metadata.get('audience', 'General audience')}
**Date**: {metadata.get('date', 'Not specified')}
**Query**: {metadata.get('query', 'Not specified')}

## SPEECH CONTENT

{speech_content}

## REVIEW REQUIREMENTS

Please conduct a thorough policy alignment review checking:

1. **Monetary Policy Consistency**: Does the speech align with current BSP monetary policy stance, interest rate decisions, and inflation targets?

2. **Regulatory Compliance**: Are all statements consistent with BSP memorandum circulars and regulatory guidelines?

3. **Communication Standards**: Does the speech follow BSP communication best practices (tone, forward guidance, conditional language)?

4. **Data Accuracy**: Are economic statistics and BSP data correctly represented?

5. **Market Sensitivity**: Are there any statements that could be misinterpreted or create unintended market reactions?

6. **Legal/Reputational Risk**: Any statements that could contradict BSP mandates or create legal/reputational issues?

Please provide your comprehensive assessment with specific violations, circular references, and actionable recommendations."""


class PolicyChecker:
    """
    High-level policy checker using Azure AI Agent
    Uses Azure AD authentication for secure access to the agent.
    """
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_POLICY_ENDPOINT")
        self.agent_id = os.getenv("AZURE_POLICY_AGENT_ID")
        self.deployment = os.getenv("AZURE_POLICY_DEPLOYMENT")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        self.api_version = os.getenv("AZURE_POLICY_API_VERSION", "v1")  # Default to v1
        
        # Validate required credentials
        missing = []
        if not self.endpoint:
            missing.append("AZURE_POLICY_ENDPOINT")
        if not self.agent_id:
            missing.append("AZURE_POLICY_AGENT_ID")
        if not self.deployment:
            missing.append("AZURE_POLICY_DEPLOYMENT")
        if not self.client_id:
            missing.append("AZURE_CLIENT_ID")
        if not self.tenant_id:
            missing.append("AZURE_TENANT_ID")
        if not self.client_secret:
            missing.append("AZURE_CLIENT_SECRET")
        
        if missing:
            raise ValueError(
                f"Missing Azure Policy Agent credentials: {', '.join(missing)}. "
                "Set these in .env file."
            )
        
        self.client = AzurePolicyAgentClient(
            endpoint=self.endpoint,
            agent_id=self.agent_id,
            deployment=self.deployment,
            client_id=self.client_id,
            tenant_id=self.tenant_id,
            client_secret=self.client_secret,
            api_version=self.api_version
        )
    
    async def check_policy_alignment(
        self,
        speech_content: str,
        speech_metadata: Dict[str, Any]
    ) -> PolicyCheckResult:
        """
        Check speech against BSP policies using the agent
        
        Args:
            speech_content: The full speech text
            speech_metadata: Dict with topic, speaker, audience, date, query
            
        Returns:
            PolicyCheckResult with structured assessment
        """
        print("\n" + "="*70)
        print("BSP POLICY ALIGNMENT CHECK")
        print("="*70)
        print(f"\n🏛️ Connecting to BSP Policy Agent...")
        print(f"  Agent ID: {self.agent_id}")
        print(f"  Deployment: {self.deployment}")
        
        try:
            # Get agent's analysis
            agent_response = await self.client.check_speech_policy(
                speech_content,
                speech_metadata
            )
            
            print(f"\n✓ Received analysis from agent ({len(agent_response)} characters)")
            
            # Parse agent response into structured format
            result = self._parse_agent_response(agent_response, speech_content)
            
            # Display summary
            self._display_summary(result)
            
            return result
            
        except Exception as e:
            print(f"\n✗ Policy check failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return error result
            return PolicyCheckResult(
                overall_compliance="error",
                compliance_score=0.0,
                violations=[],
                commendations=[],
                agent_analysis=f"Error: {str(e)}",
                circular_references=[],
                requires_revision=True,
                timestamp=datetime.now().isoformat()
            )
    
    def _parse_agent_response(
        self,
        response: str,
        speech_content: str
    ) -> PolicyCheckResult:
        """Parse agent's text response into structured PolicyCheckResult"""
        
        import re
        
        # Extract overall compliance
        compliance_patterns = [
            r'Overall[:\s]+(?:compliance[:\s]+)?([A-Za-z\s]+?)(?:\n|$)',
            r'Compliance[:\s]+([A-Za-z\s]+?)(?:\n|$)',
            r'Rating[:\s]+([A-Za-z\s]+?)(?:\n|$)'
        ]
        
        overall_compliance = "unknown"
        for pattern in compliance_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                comp_text = match.group(1).strip().lower()
                if "compliant" in comp_text or "approved" in comp_text:
                    overall_compliance = "compliant"
                elif "minor" in comp_text:
                    overall_compliance = "minor_issues"
                elif "major" in comp_text or "non" in comp_text:
                    overall_compliance = "major_issues"
                break
        
        # Extract compliance score
        score_patterns = [
            r'(?:Compliance\s+)?[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*[%/]?\s*(?:100|1\.0)?',
            r'(\d+(?:\.\d+)?)\s*[%/]\s*(?:100|1\.0)?'
        ]
        
        compliance_score = 0.8  # Default
        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                score_val = float(match.group(1))
                compliance_score = score_val / 100 if score_val > 1 else score_val
                break
        
        # Extract violations
        violations = self._extract_violations(response)
        
        # Extract commendations
        commendations = self._extract_commendations(response)
        
        # Extract circular references
        circular_refs = self._extract_circular_references(response)
        
        # Determine if revision required
        requires_revision = (
            overall_compliance in ["major_issues", "non_compliant"] or
            any(v.severity in ["critical", "high"] for v in violations) or
            compliance_score < 0.7 or
            "major revision" in response.lower()
        )
        
        return PolicyCheckResult(
            overall_compliance=overall_compliance,
            compliance_score=compliance_score,
            violations=violations,
            commendations=commendations,
            agent_analysis=response,
            circular_references=circular_refs,
            requires_revision=requires_revision,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_violations(self, response: str) -> List[PolicyViolation]:
        """Extract policy violations from agent response"""
        import re
        
        violations = []
        
        # Look for violation sections marked by severity
        violation_sections = re.split(
            r'(?:^|\n)(?:###?\s*)?(?:VIOLATION|ISSUE)[\s:]*\d*',
            response,
            flags=re.IGNORECASE | re.MULTILINE
        )
        
        for section in violation_sections[1:]:  # Skip first split (before any violations)
            violation = self._parse_violation_section(section)
            if violation:
                violations.append(violation)
        
        # Also look for bullet-pointed violations
        bullet_pattern = r'[-•*]\s*\*\*([A-Z\s]+)\*\*[:\s]+(.*?)(?=\n[-•*]|\n\n|$)'
        matches = re.finditer(bullet_pattern, response, re.DOTALL)
        
        for match in matches:
            severity_text = match.group(1).strip().lower()
            content = match.group(2).strip()
            
            if any(sev in severity_text for sev in ['critical', 'high', 'medium', 'low']):
                violation = PolicyViolation(
                    severity=self._extract_severity(severity_text),
                    category=self._extract_category(content),
                    location="See details",
                    issue=content[:200],
                    circular_reference=self._extract_circular_ref(content),
                    recommendation="See agent analysis",
                    original_text=""
                )
                violations.append(violation)
        
        return violations
    
    def _parse_violation_section(self, section: str) -> Optional[PolicyViolation]:
        """Parse a single violation section"""
        import re
        
        # Extract fields
        severity = self._extract_field(section, ['severity', 'level'])
        category = self._extract_field(section, ['category', 'type'])
        location = self._extract_field(section, ['location', 'section', 'paragraph'])
        issue = self._extract_field(section, ['issue', 'problem', 'concern'])
        circular = self._extract_field(section, ['circular', 'guideline', 'memorandum'])
        recommendation = self._extract_field(section, ['recommendation', 'fix', 'solution'])
        
        if not issue:
            return None
        
        return PolicyViolation(
            severity=severity or "medium",
            category=category or "general",
            location=location or "Not specified",
            issue=issue[:300],
            circular_reference=circular or "BSP guidelines",
            recommendation=recommendation or "See agent analysis",
            original_text=""
        )
    
    def _extract_field(self, text: str, field_names: List[str]) -> Optional[str]:
        """Extract a field value from text"""
        import re
        
        for field in field_names:
            pattern = rf'{field}[:\s]+([^\n]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity from text"""
        text_lower = text.lower()
        if "critical" in text_lower:
            return "critical"
        elif "high" in text_lower:
            return "high"
        elif "medium" in text_lower or "moderate" in text_lower:
            return "medium"
        elif "low" in text_lower or "minor" in text_lower:
            return "low"
        return "medium"
    
    def _extract_category(self, text: str) -> str:
        """Extract category from text"""
        text_lower = text.lower()
        if "monetary" in text_lower or "policy" in text_lower:
            return "monetary_policy"
        elif "regulat" in text_lower:
            return "regulatory"
        elif "communication" in text_lower or "tone" in text_lower:
            return "communication"
        elif "data" in text_lower or "statistic" in text_lower:
            return "data_accuracy"
        return "general"
    
    def _extract_circular_ref(self, text: str) -> str:
        """Extract BSP circular reference from text"""
        import re
        
        # Look for patterns like "M-2024-001" or "Circular No. 1234"
        patterns = [
            r'M-\d{4}-\d{3}',
            r'Circular\s+(?:No\.?\s*)?(\d+)',
            r'Memorandum\s+(?:No\.?\s*)?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "BSP guidelines"
    
    def _extract_commendations(self, response: str) -> List[Dict[str, str]]:
        """Extract positive findings from agent response"""
        import re
        
        commendations = []
        
        # Look for commendation section
        comm_section = re.search(
            r'(?:COMMENDATION|POSITIVE|STRENGTHS|WELL-ALIGNED)[:\s]+(.*?)(?:\n\n|###|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if comm_section:
            comm_text = comm_section.group(1)
            
            # Split by bullets or line breaks
            items = re.split(r'\n[-•*]\s*', comm_text)
            
            for item in items:
                item = item.strip()
                if item and len(item) > 20:
                    commendations.append({
                        "finding": item[:200],
                        "impact": "positive"
                    })
        
        return commendations
    
    def _extract_circular_references(self, response: str) -> List[str]:
        """Extract all BSP circular references mentioned"""
        import re
        
        circulars = set()
        
        # Pattern for M-YYYY-NNN format
        matches = re.finditer(r'M-\d{4}-\d{3}', response)
        for match in matches:
            circulars.add(match.group(0))
        
        # Pattern for "Circular No. NNN" format
        matches = re.finditer(r'Circular\s+(?:No\.?\s*)?(\d+)', response, re.IGNORECASE)
        for match in matches:
            circulars.add(f"Circular {match.group(1)}")
        
        return sorted(list(circulars))
    
    def _display_summary(self, result: PolicyCheckResult):
        """Display summary of policy check results"""
        print(f"\n{'─'*70}")
        print("POLICY ALIGNMENT SUMMARY")
        print(f"{'─'*70}")
        print(f"  Overall Compliance: {result.overall_compliance.upper()}")
        print(f"  Compliance Score: {result.compliance_score:.1%}")
        print(f"  Violations Found: {len(result.violations)}")
        
        if result.violations:
            severity_counts = {}
            for v in result.violations:
                severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1
            
            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    icon = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    print(f"    {icon} {severity.title()}: {count}")
        else:
            print(f"    ✓ No policy violations detected")
        
        print(f"  Commendations: {len(result.commendations)}")
        
        if result.circular_references:
            print(f"  BSP Circulars Referenced: {len(result.circular_references)}")
            for i, circ in enumerate(result.circular_references[:3], 1):
                print(f"    {i}. {circ}")
            if len(result.circular_references) > 3:
                print(f"    ... and {len(result.circular_references) - 3} more")
        
        print(f"  Requires Revision: {'Yes' if result.requires_revision else 'No'}")
        
        if result.requires_revision:
            print(f"\n  ⚠️ RECOMMENDATION: REVISION REQUIRED")
        else:
            print(f"\n  ✅ RECOMMENDATION: APPROVED FOR USE")


async def check_speech_policy_alignment(
    speech_content: str,
    speech_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to check policy alignment
    
    Args:
        speech_content: The full speech text to review
        speech_metadata: Dict with topic, speaker, audience, date, query
        
    Returns:
        Dict with structured results including violations, commendations, etc.
    """
    checker = PolicyChecker()
    
    try:
        result = await checker.check_policy_alignment(
            speech_content=speech_content,
            speech_metadata=speech_metadata
        )
        
        return {
            "success": True,
            "overall_compliance": result.overall_compliance,
            "compliance_score": result.compliance_score,
            "requires_revision": result.requires_revision,
            "violations_count": len(result.violations),
            "critical_violations": len([v for v in result.violations if v.severity == "critical"]),
            "high_violations": len([v for v in result.violations if v.severity == "high"]),
            "violations": [
                {
                    "severity": v.severity,
                    "category": v.category,
                    "location": v.location,
                    "issue": v.issue,
                    "circular_reference": v.circular_reference,
                    "recommendation": v.recommendation
                }
                for v in result.violations
            ],
            "commendations": result.commendations,
            "circular_references": result.circular_references,
            "agent_analysis": result.agent_analysis,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "overall_compliance": "error",
            "requires_revision": True,
            "timestamp": datetime.now().isoformat()
        }
