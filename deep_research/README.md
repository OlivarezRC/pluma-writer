# Deep Research Mode

## Overview

The Deep Research mode is an advanced research pipeline that uses Azure AI Inference (DeepSeek model) combined with Tavily web search to perform comprehensive, multi-iteration research on any topic. It's completely separate from the standard chat and Azure AI Foundry routes.

## Architecture

```
User Query → Deep Research Pipeline → LangGraph State Machine
    ↓
    ├── Generate Query (DeepSeek R1)
    ├── Web Research (Tavily API)
    ├── Summarize Sources (DeepSeek R1)
    ├── Reflect on Summary (DeepSeek R1)
    └── Finalize Report (DeepSeek R1)
```

## Key Components

### 1. `pipeline.py`
Main orchestration file containing:
- **State Machine**: LangGraph-based workflow with 5 nodes
- **Model Initialization**: Azure AI Inference with DeepSeek
- **Research Loop**: Iterates up to 3 times for comprehensive coverage
- **Latex Normalization**: Converts `\[...\]` → `$$...$$` and `\(...\)` → `$...$`
- **Thinking Token Extraction**: Strips `<think>...</think>` tags from DeepSeek responses

### 2. `prompts.py`
Contains system prompts for each research stage:
- Query generation with rationale
- Source summarization
- Reflection and knowledge gap identification

### 3. `formatting.py`
Utilities for:
- Deduplicating search results
- Formatting source citations
- Preparing web research data

### 4. `states.py`
Defines the LangGraph state schema:
- `SummaryState`: Full state with all fields
- `SummaryStateInput`: Input schema (research_topic)
- `SummaryStateOutput`: Output schema (running_summary)

## Configuration

### Required Environment Variables

Deep Research mode requires **separate credentials** from the standard chat modes:

```bash
# Azure AI Inference Endpoint (DeepSeek model)
AZURE_INFERENCE_ENDPOINT=https://your-inference-endpoint.inference.ai.azure.com
AZURE_DEEPSEEK_DEPLOYMENT=deepseek-r1  # or your deployment name
AZURE_AI_API_KEY=your_azure_ai_api_key

# Tavily API for web search
TAVILY_API_KEY=your_tavily_api_key
```

### Why Separate Credentials?

Deep Research uses:
- **Azure AI Inference API** (not Azure OpenAI or AI Foundry)
- **DeepSeek R1 model** for reasoning-aware research
- **Tavily search API** for high-quality web results

This is distinct from:
- Standard chat mode (uses LiteLLM with multiple providers)
- Foundry mode (uses Azure AI Agents)

## Usage Flow

1. **User switches to Deep Research mode** in the UI (3-mode toggle)
2. **User enters a research query**
3. **Pipeline executes**:
   - Iteration 1: Initial query → web search → summarize
   - Iteration 2: Reflect on gaps → refined query → web search → update summary
   - Iteration 3: Final refinement → web search → update summary
   - Finalize: Add images and format sources
4. **Result sent to user** with formatted markdown report

## Progress Notifications

The pipeline sends real-time updates via the `notify` callback:
- `generate_query`: Query and rationale
- `web_research`: Sources and images found
- `summarize`: Updated summary
- `reflection`: Knowledge gaps and follow-up query
- `routing`: Decision to continue or finalize
- `thinking`: Extracted reasoning from DeepSeek
- `finalize`: Final report with images

## Error Handling

### Configuration Errors
If environment variables are missing, the app will:
1. Catch the `ValueError` at module import
2. Display user-friendly error message
3. List missing variables
4. Prevent execution until configured

### Runtime Errors
- Web search failures: Logged but pipeline continues
- Model errors: Captured and reported to user
- JSON parsing errors: Fallback to simple queries

## Implementation Notes

### DeepSeek R1 Special Handling

DeepSeek R1 uses `<think>...</think>` tags for chain-of-thought reasoning:
```python
def _strip_thinking_tokens(text: str):
    """Extracts and removes thinking tokens from response"""
    # Returns: (thoughts, cleaned_text)
```

### Latex Math Rendering

Converts LaTeX for Chainlit/markdown compatibility:
```python
def _normalize_latex(text: str):
    """\\[formula\\] → $$formula$$"""
    """\\(inline\\) → $inline$"""
```

### Image Handling

- Tavily returns image URLs
- Stored in `state.images` list
- Top 2 images rendered in final report
- Responsive HTML with Tailwind CSS classes

## Extending the Pipeline

### Adding More Iterations
Edit `route_research()` in `pipeline.py`:
```python
if state.research_loop_count <= 5:  # Change from 3 to 5
    return "web_research"
```

### Custom Search Parameters
Modify `web_research()`:
```python
res = await tavily_client.search(
    state.search_query,
    max_results=3,  # Increase results
    max_tokens_per_source=2000,  # More content per source
    include_domains=["bsp.gov.ph"],  # Restrict to specific domains
)
```

### Alternative Models
Replace DeepSeek with another model:
```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # ... configuration
)
```

## Troubleshooting

### "Error generating response in chat_agent"
- **Cause**: Deep research mode is trying to use foundry credentials
- **Fix**: Ensure `AZURE_INFERENCE_ENDPOINT`, `AZURE_DEEPSEEK_DEPLOYMENT`, and `AZURE_AI_API_KEY` are set

### "Missing required environment variables"
- **Check**: `.env` file has all 4 required variables
- **Verify**: Variables are not empty or commented out

### "Tavily API error"
- **Check**: `TAVILY_API_KEY` is valid
- **Verify**: Tavily account has sufficient credits

### Poor Research Quality
- **Increase iterations**: Modify loop count in `route_research()`
- **Refine prompts**: Edit prompts in `prompts.py`
- **Adjust search parameters**: More results or sources in `web_research()`

## Testing

Run deep research tests:
```bash
pytest tests/test_app.py::test_deep_research_mode -v
```

Manual testing:
1. Set all required environment variables
2. Start the app: `chainlit run app.py`
3. Switch to "Deep Research" mode
4. Enter query: "Explain BSP monetary policy"
5. Observe progress updates and final report

## Performance

Typical execution time:
- 3 iterations: 2-4 minutes
- 5 iterations: 4-7 minutes

Factors affecting speed:
- Tavily API response time
- DeepSeek model latency
- Complexity of research topic

## Security Notes

- **API Keys**: Never commit `.env` files to version control
- **Rate Limits**: Tavily and Azure have rate limits; implement backoff if needed
- **Cost**: Each research query makes multiple API calls; monitor usage
