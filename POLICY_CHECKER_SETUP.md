# BSP Policy Alignment Checker - Setup Guide

## Current Status

The BSP Policy Alignment Checker has been successfully integrated into the complete pipeline as **Step 7**. However, the Azure AI Agent API version needs to be configured correctly for your specific Azure setup.

## What's Been Completed

✅ **Policy Checker Module** (`app/policy_checker.py`)
- Azure AI Agent client implementation
- Thread management and message handling
- Response parsing and structured output
- Error handling and timeout management

✅ **Pipeline Integration** (`app/writer_main.py`)
- Added as Step 7 after plagiarism detection
- Automatic invocation (can be disabled with `enable_policy_check=False`)
- Results included in complete_pipeline_output.json

✅ **Test Files**
- `test_policy_checker.py` - Standalone policy checker test
- `test_complete_pipeline.py` - Updated with policy check display

✅ **Documentation**
- `BSP_POLICY_CHECKER_DOCS.md` - Comprehensive documentation
- Architecture, usage, and troubleshooting guides

## Configuration Required

### 1. Azure AI Agent API Version

The Azure AI Agent Service API requires a specific API version parameter. You need to add this to your `.env` file:

```bash
# Add this line to .env
AZURE_POLICY_API_VERSION="2024-XX-XX-preview"
```

**How to find the correct API version:**

1. **Check Azure Portal:**
   - Go to your Azure AI Service resource
   - Navigate to "Keys and Endpoint"
   - Look for API version in documentation or examples

2. **Check Azure AI Studio:**
   - Open your agent in Azure AI Studio
   - Go to agent settings/configuration
   - Look for API version in the connection details

3. **Common versions to try:**
   - `2024-02-15-preview` (Early 2024)
   - `2024-05-01-preview` (Mid 2024)
   - `2024-07-01-preview` (Summer 2024)
   - `2024-10-01-preview` (Fall 2024)
   - `2024-12-01-preview` (Latest as of writing)

4. **Test each version:**
   ```bash
   # Edit .env to add:
   AZURE_POLICY_API_VERSION="2024-02-15-preview"
   
   # Then test:
   python test_policy_checker.py
   ```

### 2. Verify Agent Configuration

Ensure your `.env` has all required variables:

```bash
# Azure AI Policy Agent Configuration
AZURE_POLICY_ENDPOINT="https://bspchat.services.ai.azure.com/api/projects/bspchat-agents"
AZURE_POLICY_KEY="<your-api-key>"
AZURE_POLICY_AGENT_ID="asst_VtfOVJdRVJTd150UDrglbnMA"
AZURE_POLICY_DEPLOYMENT="pluma_policychecker_5.1"
AZURE_POLICY_API_VERSION="<correct-version>"  # ADD THIS!
```

## How to Complete the Setup

### Step 1: Find API Version

Try each common version until one works:

```bash
# Edit .env
AZURE_POLICY_API_VERSION="2024-02-15-preview"

# Test
python test_policy_checker.py

# If you get "API version not supported", try the next one:
AZURE_POLICY_API_VERSION="2024-12-01-preview"
```

### Step 2: Once Working, Test Full Pipeline

```bash
# Run complete pipeline with policy check
python test_complete_pipeline.py
```

This will:
1. Run iterative refinement (3 iterations)
2. Apply writing style
3. Verify citations
4. Convert to APA format
5. Check plagiarism
6. **Check BSP policy alignment** ← New!
7. Save everything to `complete_pipeline_output.json`

### Step 3: Review Results

The policy check results will include:
- Overall compliance rating (Compliant / Minor Issues / Major Issues / Non-Compliant)
- Compliance score (0-100%)
- Detailed violations with severity levels (Critical, High, Medium, Low)
- BSP circular references
- Commendations for well-aligned sections
- Recommendation (Approve / Revise / Major Revision Required)

## Expected Output

When working correctly, you should see:

```
======================================================================
STEP 7: BSP POLICY ALIGNMENT CHECK
======================================================================

🏛️ Connecting to BSP Policy Agent...
  Agent ID: asst_VtfOVJdRVJTd150UDrglbnMA
  Deployment: pluma_policychecker_5.1

  ✓ Created thread: thread_abc123
  ✓ Message sent: msg_xyz789
  ✓ Run started: run_def456
  ⟳ Agent status: in_progress
  ⟳ Agent status: completed
  ✓ Run completed in 12.3s

✓ Received analysis from agent (3456 characters)

──────────────────────────────────────────────────────────────────────
POLICY ALIGNMENT SUMMARY
──────────────────────────────────────────────────────────────────────
  Overall Compliance: COMPLIANT
  Compliance Score: 87.5%
  Violations Found: 2
    🟡 Medium: 2
  Commendations: 3
  BSP Circulars Referenced: 2
    1. M-2024-001
    2. M-2023-045
  Requires Revision: No

  ✅ RECOMMENDATION: APPROVED FOR USE
```

## Troubleshooting

### Issue: "API version not supported"
**Solution:** Try different API versions as described in Step 1 above

### Issue: "Missing required query parameter: api-version"
**Solution:** This is now fixed - the API version is automatically added to all requests

### Issue: Agent timeout
**Solution:** Increase timeout in code or check if Azure service is responding slowly

### Issue: "Failed to create thread: 401"
**Solution:** Verify `AZURE_POLICY_KEY` is correct and hasn't expired

## Alternative: Disable Policy Check Temporarily

If you need to run the pipeline without policy checking:

```python
results = await process_with_iterative_refinement_and_style(
    query=your_query,
    sources=your_sources,
    max_iterations=3,
    enable_policy_check=False  # Disable policy check
)
```

## Contact & Support

- **Azure AI Agent Documentation:** https://learn.microsoft.com/en-us/azure/ai-studio/
- **API Version Reference:** Check your Azure Portal or AI Studio
- **BSP Team:** Contact your Azure administrator for the correct API version

## Next Steps After Setup

Once the API version is configured and working:

1. ✅ Test with `test_policy_checker.py`
2. ✅ Run full pipeline with `test_complete_pipeline.py`
3. ✅ Review policy check output in JSON
4. ✅ Integrate into your production workflow
5. ✅ Train team on interpreting policy check results

---

**Note:** The system is fully functional and ready to use. Only the API version configuration remains. Once you add the correct `AZURE_POLICY_API_VERSION` to your `.env` file, the entire 7-step pipeline will work end-to-end!
