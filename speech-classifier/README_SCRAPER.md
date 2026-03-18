# BSP Speech Scraper

Scrapes government speeches from the Bangko Sentral ng Pilipinas (BSP) website.

## Features

- ✅ Scrapes speeches from July 3, 2017 to February 3, 2026
- ✅ Handles JavaScript-rendered content using Playwright
- ✅ Extracts all metadata: Title, Date, Place, Occasion, Speaker, Content
- ✅ Outputs to both CSV and XLSX formats
- ✅ Resume capability - saves progress automatically
- ✅ Filters speeches by date range

## Requirements

```bash
pip install playwright pandas openpyxl beautifulsoup4 requests
python3 -m playwright install chromium
sudo python3 -m playwright install-deps chromium
```

## Usage

### Run the scraper:

```bash
cd /workspaces/pluma-writer/speech-classifier
python3 speech-scraper-final.py
```

### Test with one URL:

```bash
python3 test_playwright.py
```

## Output Files

- `bsp_speeches.csv` - CSV format with all scraped speeches
- `bsp_speeches.xlsx` - Excel format
- `scraper_progress.json` - Progress tracking (allows resuming)
- `scraper_log.txt` - Detailed execution log

## Output Structure

Each row contains:
- **Title**: Speech title
- **Date**: Speech date
- **Place**: Venue/location
- **Occasion**: Event/occasion
- **Speaker**: Speaker name (usually BSP Governor)
- **Content**: Full speech transcription
- **URL**: Source URL
- **ItemId**: Unique identifier

## Example URLs

- List page: https://www.bsp.gov.ph/SitePages/MediaAndResearch/SpeechesList.aspx
- Sample speech: https://www.bsp.gov.ph/SitePages/MediaAndResearch/SpeechesDisp.aspx?ItemId=1114

## Technical Details

The BSP website uses Angular.js to dynamically load content. Simple HTTP requests won't work because the data is rendered client-side via JavaScript. This scraper uses:

1. **Playwright** - Headless browser automation
2. **Chromium** - Browser engine
3. **BeautifulSoup** - HTML parsing (after Playwright renders the page)

### Why Playwright?

- Handles JavaScript rendering
- Waits for Angular to load data
- More reliable than Selenium in container environments
- Built-in auto-waiting mechanisms

## Performance

- ~3-5 seconds per speech page
- Processes ~450 ItemIds (1-650)
- Expected runtime: ~30-40 minutes for full scrape
- Saves progress every 20 speeches

## Troubleshooting

### Browser won't launch
```bash
# Install system dependencies
sudo python3 -m playwright install-deps chromium
```

### Data not extracting
- Check if the website structure changed
- Verify JavaScript is rendering (test with test_playwright.py)
- Check logs in scraper_log.txt

### Resume after interruption
The scraper automatically resumes from where it left off using `scraper_progress.json`. Simply run the script again.

## Date Range

Configured to scrape speeches from:
- **Start**: July 3, 2017
- **End**: February 3, 2026 (current date)

Speeches outside this range are automatically skipped.

## Notes

- The scraper is polite: adds delays between requests
- Respects the server by not overwhelming it
- Handles errors gracefully and continues
- All invalid/empty pages are skipped automatically
