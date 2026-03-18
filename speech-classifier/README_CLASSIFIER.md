# BSP Speech Audience Classifier

Classifies scraped BSP speeches by audience setting using Azure OpenAI.

## Features

- Reads scraped speeches from `bsp_speeches.xlsx`
- Classifies each speech into one of 5 audience categories:
  - **INTERNAL_BSP**: Internal BSP staff/leadership
  - **GOV_OVERSIGHT**: Government oversight (Congress, DBCC)
  - **INDUSTRY_MARKET**: Banking industry and market stakeholders
  - **INTERNATIONAL_OFFICIAL**: International/multilateral forums
  - **PUBLIC_REGIONAL**: General public and regional outreach
- Generates structured metadata following the specified schema
- Outputs results in Excel and JSON formats

## Setup

1. **Configure Azure OpenAI credentials**:
   ```bash
   cp .env.template .env
   # Edit .env with your Azure OpenAI credentials
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install pandas openpyxl python-dotenv openai tqdm
   ```

## Usage

```bash
python classify_scraped_speeches.py
```

The script will:
1. Load speeches from `bsp_speeches.xlsx` (520 speeches)
2. Classify each speech using Azure OpenAI GPT-4
3. Generate metadata items with classification results
4. Save outputs:
   - `bsp_speeches_classified.xlsx` - Excel format with all metadata
   - `bsp_speeches_classified.json` - JSON format for Cosmos DB upload

## Output Schema

Each classified speech includes:
- `id`: Unique document ID
- `PartitionKey`: Partition key for Cosmos DB
- `title`, `date`, `place`, `Occasion`, `Speaker`: Extracted metadata
- `audience_setting_classification`: Classified audience category
- `classification_confidence`: Confidence score (0-1)
- `classification_reasons`: Explanation of classification
- `classification_evidence`: Text snippets supporting the classification
- `content`: Speech content (first 10,000 characters)
- `sourceType`, `sourcePath`, `language`, `tags`, `group_ids`, etc.

## Classification Labels

### Enhanced Cue Words (BSP-Specific)

**INTERNAL_BSP**:
- Core cues: fellow BSPers, turnover, anniversary, BSP family, oath-taking, promotion

**GOV_OVERSIGHT**:
- Core cues: Honorable, Senate, Congress, DBCC, legislative, fiscal policy, oversight

**INDUSTRY_MARKET**:
- Core cues: bankers, financial institutions, banking community, payment system, compliance, prudential

**INTERNATIONAL_OFFICIAL**:
- Core cues: IMF, BIS, World Bank, ASEAN, SEACEN, multilateral, Basel, G20

**PUBLIC_REGIONAL**:
- Core cues: financial literacy, consumer protection, Filipinos, regional, provincial, SME, OFWs

## Notes

- Processing 520 speeches takes approximately 15-20 minutes
- API rate limits: 0.2s delay between requests
- Content is truncated to 10,000 characters for storage efficiency
- Classification uses GPT-4 with temperature=0 for consistency
