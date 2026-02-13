# BSP Policy Alignment Checker

## Overview

The BSP Policy Alignment Checker is an AI-powered system that reviews speeches written for BSP officials to ensure compliance with BSP memorandum circulars, policy guidelines, and communication best practices.

## Architecture

### System Components

1. **Azure AI Policy Agent**
   - Pre-trained agent with BSP memorandum circulars in knowledge base
   - Accessed via Azure AI Agent Service API
   - Deployment: `pluma_policychecker_5.1`
   - Agent ID: `asst_VtfOVJdRVJTd150UDrglbnMA`

2. **Policy Checker Client** (`app/policy_checker.py`)
   - Python module for interacting with the Azure agent
   - Manages conversation threads and message flow
   - Parses agent responses into structured format

3. **Pipeline Integration** (`app/writer_main.py`)
   - Integrated as **Step 7** in the complete pipeline
   - Runs after plagiarism detection
   - Can be disabled with `enable_policy_check=False`

## How It Works

### 1. Input Processing

The system takes:
- **Speech content**: Full text of the generated speech (preferably APA-formatted version)
- **Speech metadata**: Topic, speaker, audience, date, query

### 2. Agent Interaction Flow

```
1. Create conversation thread with Azure agent
2. Format policy check request with speech and metadata
3. Send message to agent in thread
4. Create agent run with specific instructions
5. Poll for run completion (status checks every 2 seconds)
6. Retrieve agent's analysis from thread messages
7. Parse response into structured format
```

### 3. Analysis Framework

The agent checks:

#### A. Monetary Policy Alignment
- Consistency with Monetary Board decisions
- Alignment with inflation targets
- Proper forward guidance language
- No inappropriate pre-commitments

#### B. Regulatory Compliance
- Adherence to BSP memorandum circulars
- Correct representation of BSP regulatory powers
- Proper framing of regulations

#### C. Communication Standards
- Professional tone for BSP officials
- Appropriate technical terminology
- Clear structure and logical flow
- Suitable detail level for audience

#### D. Data & Evidence Accuracy
- Statistics match official BSP publications
- Economic indicators correctly sourced
- Proper attribution to BSP research

#### E. Risk Assessment
- Market-moving statements identified
- Legal/regulatory contradictions flagged
- Reputational risks noted
- Political sensitivity checked

### 4. Output Structure

```python
PolicyCheckResult {
    "overall_compliance": str,        # compliant, minor_issues, major_issues, non_compliant
    "compliance_score": float,        # 0.0 to 1.0
    "violations": [                   # List of PolicyViolation objects
        {
            "severity": str,          # critical, high, medium, low
            "category": str,          # monetary_policy, regulatory, communication, etc.
            "location": str,          # paragraph/section identifier
            "issue": str,             # description of problem
            "circular_reference": str, # BSP circular violated
            "recommendation": str     # suggested fix
        }
    ],
    "commendations": [                # Positive findings
        {
            "finding": str,
            "impact": "positive"
        }
    ],
    "circular_references": [str],     # All BSP circulars referenced
    "agent_analysis": str,            # Full agent response
    "requires_revision": bool,        # Whether revision needed
    "timestamp": str                  # ISO format timestamp
}
```

## Usage

### Standalone Testing

```bash
python test_policy_checker.py
```

This tests the policy checker with a sample speech about AI in finance.

### In Complete Pipeline

The policy checker is automatically integrated in the complete pipeline:

```python
from app.writer_main import process_with_iterative_refinement_and_style

results = await process_with_iterative_refinement_and_style(
    query="Your research query",
    sources={"topics": "...", "links": [...]},
    max_iterations=3,
    enable_policy_check=True  # Enabled by default
)

# Access policy check results
policy_result = results['policy_check']
```

### Running Complete Pipeline Test

```bash
python test_complete_pipeline.py
```

Output includes:
- `complete_pipeline_output.json` - Full results including policy check
- `styled_output.txt` - Styled speech
- `styled_output_apa.txt` - APA-formatted speech
- Console display of all 7 pipeline steps

## Configuration

Required environment variables in `.env`:

```bash
# Azure AI Policy Agent
AZURE_POLICY_ENDPOINT="https://bspchat.services.ai.azure.com/api/projects/bspchat-agents"
AZURE_POLICY_KEY="your-api-key"
AZURE_POLICY_AGENT_ID="asst_VtfOVJdRVJTd150UDrglbnMA"
AZURE_POLICY_DEPLOYMENT="pluma_policychecker_5.1"
```

## Violation Severity Levels

### 🔴 CRITICAL (Must Fix)
- Direct contradiction with BSP Monetary Board decisions
- Legal/regulatory misstatements
- Market-moving errors
- Misrepresentation of official BSP positions

### 🟠 HIGH (Should Fix)
- Inconsistency with recent BSP communications
- Inappropriate forward guidance language
- Tone violations (too political, too casual)
- Missing critical caveats or disclaimers

### 🟡 MEDIUM (Consider Fixing)
- Ambiguous phrasing that could be misinterpreted
- Imbalanced risk presentation
- Minor inconsistencies with BSP style guides
- Suboptimal structure or clarity

### 🟢 LOW (Enhancement)
- Minor style improvements
- Opportunities for stronger messaging
- Additional context that could be helpful
- Formatting or presentation suggestions

## Compliance Scoring

- **0.90 - 1.00**: Compliant - Approved for use
- **0.70 - 0.89**: Minor Issues - Review and address flagged items
- **0.50 - 0.69**: Major Issues - Significant revision required
- **0.00 - 0.49**: Non-Compliant - Major revision or rewrite needed

## Pipeline Integration

The complete pipeline now consists of **7 steps**:

1. **Iterative Refinement** - Deep research and evidence collection
2. **Writing Style Retrieval** - Fetch BSP official's speaking style
3. **Styled Output Generation** - Apply style to summary
4. **Citation Verification** - Verify all evidence citations
5. **APA Format Conversion** - Convert to APA 7th edition
6. **Plagiarism Detection** - Check for unoriginal content
7. **Policy Alignment Check** - Verify BSP compliance ← NEW!

## Error Handling

The system gracefully handles:
- **Agent timeout**: 120s default, configurable
- **API failures**: Returns error result with `success: False`
- **Parsing errors**: Attempts multiple parsing strategies
- **Missing credentials**: Raises clear ValueError on initialization

If policy check fails, pipeline continues but marks speech as requiring revision.

## Best Practices

### For Speech Writers

1. **Run policy check early**: Test drafts to catch issues before finalizing
2. **Review all violations**: Even "low" severity issues may be important
3. **Check circular references**: Verify you're aligned with cited BSP memoranda
4. **Address critical/high first**: Prioritize fixes by severity
5. **Don't ignore commendations**: They show what's working well

### For Developers

1. **Monitor agent performance**: Check response times and success rates
2. **Update agent knowledge**: Regularly add new BSP circulars to agent
3. **Refine parsing logic**: Improve structured extraction from agent responses
4. **Log agent conversations**: Keep thread IDs for debugging
5. **Test edge cases**: Try speeches with deliberate violations

## Limitations

1. **Agent knowledge cutoff**: Agent may not know very recent circulars
2. **Context window**: Very long speeches (>32K tokens) may need chunking
3. **Interpretation**: Some policy judgments require human expertise
4. **Language**: Works best with English; Tagalog/mixed-language may vary
5. **Network dependency**: Requires Azure connectivity

## Future Enhancements

- [ ] Batch processing for multiple speeches
- [ ] Historical violation tracking and analytics
- [ ] Auto-suggestion of specific text replacements
- [ ] Integration with BSP circular database for real-time updates
- [ ] Multi-language support (Tagalog, Filipino)
- [ ] Severity threshold configuration per use case
- [ ] Revision history tracking with diff visualization

## Troubleshooting

### Policy check fails with 401 error
- Verify `AZURE_POLICY_KEY` is correct in `.env`
- Check if API key has expired

### Agent timeout
- Increase timeout: `PolicyChecker(timeout=300)`
- Check speech length (very long speeches take more time)
- Verify Azure service status

### No violations found but speech seems problematic
- Agent may lack context on very new policies
- Consider manual review by BSP policy experts
- Update agent knowledge base

### Parsing errors in violations
- Agent response format may have changed
- Check `agent_analysis` field in raw output
- Update parsing logic in `_parse_agent_response()`

## Support

For issues with:
- **Agent configuration**: Contact Azure AI support
- **Policy checker code**: Check GitHub issues or file new issue
- **BSP circular interpretation**: Consult BSP policy team
- **Pipeline integration**: Review `app/writer_main.py` documentation

---

**Last Updated**: February 13, 2026  
**Version**: 1.0.0  
**Maintainer**: BSP AI Writing System Team
