# BSP DOCX Formatting Enhancements

## Overview
Enhanced the DOCX output generation to align with BSP examination report templates (FSS UKB MODEL, FSS UKB ROE, FSS TBRCB ROE) with proper table formatting for risk assessments.

## Key Improvements

### 1. **Table Detection and Creation**
- Automatic detection of tabular content in text (tab-separated or multi-space separated)
- Creates properly formatted tables with:
  - Gray shaded headers (hex color: D9D9D9)
  - Centered header text with bold formatting
  - Proper column width distribution
  - Cell vertical alignment
  - Border styling

### 2. **Risk Assessment Tables**
The system now creates professional tables for:
- **Overall Summary of Assessment**: Risk categories with net ratings
- **Institutional Level Support**: CAMEL-style ratings (Earnings, Capital, Liquidity, Governance)
- **Risk Categories**: Credit Risk, Liquidity Risk, IRRBB, IT Risk, Market Risk, etc.

### 3. **Rating Highlighting**
Automatically bolds rating keywords:
- STRONG
- MODERATE  
- LOW
- ACCEPTABLE
- WEAK
- HIGH

### 4. **Header Hierarchy**
Three levels of headers:
- **Major Headers** (12pt, bold, extra spacing): Roman numerals (I., II., III.) or assessment keywords
- **Minor Headers** (11pt, bold): Numbered items (1., 2., 3.) or ALL CAPS lines
- **Regular Content** (11pt, justified): Body text

### 5. **Table Formatting Standards**
Following BSP examination report conventions:
- 2-column tables: 3.5" + 2.0" = 5.5" total width
- Multi-column tables: Equal distribution across available width
- First row always formatted as header with gray background
- Centered header text, left-aligned body text
- Calibri font at 10pt for table cells
- Proper spacing before and after tables

## Usage Example

### Input Text with Table Structure:
```
Risk Category\tNet Rating
Credit Risk\tModerate
Liquidity Risk\tLow
Interest Rate Risk\tLow

Component\tRating
Earnings\tStrong
Capital\tStrong
Liquidity\tStrong
Governance\tAcceptable
```

### Generated Output:
Creates formatted tables with:
- Gray header row with "Risk Category | Net Rating"
- Bold "Moderate", "Low", "Strong", "Acceptable" ratings
- Professional BSP report styling

## Function Reference

### `make_docx_bytes(text: str, title: str | None = None) -> bytes`
Main function to generate DOCX with BSP formatting.

**Helper Functions:**
- `set_cell_shading(cell, color_hex)`: Applies background color to cell
- `set_cell_border(cell, **kwargs)`: Sets cell border properties
- `is_table_content(lines)`: Detects if text represents a table
- `create_assessment_table(lines)`: Creates formatted BSP-style table

### `set_cell_shading(cell, color_hex: str)`
- **Parameters**: 
  - `cell`: python-docx cell object
  - `color_hex`: Hex color code (e.g., 'D9D9D9' for light gray)
- **Usage**: `set_cell_shading(cell, 'D9D9D9')`

### `is_table_content(lines: list) -> bool`
Detects tabular content by analyzing:
- Tab character frequency
- Multiple consecutive spaces (3+)
- Returns True if >50% of lines have table indicators

### `create_assessment_table(lines: list) -> Table | None`
Creates a BSP-formatted table:
- Parses tab or space-separated values
- Determines optimal column count
- Applies BSP styling standards
- Returns python-docx Table object or None

## File Structure

### Modified Files:
- **app.py**: Main application with `make_docx_bytes()` function
  - Lines 340-800: DOCX generation functions
  - Lines 700-730: Table creation helpers
  
### Test Files:
- **test_docx_format.py**: Standalone test script demonstrating table formatting
- **sample_bsp_format.docx**: Generated sample output

## CosmosDB Integration

The system is designed to work with BSP templates stored in CosmosDB:
- **Database**: Azure Cosmos DB
- **Container**: `styles` (partition key: `/user_id`)
- **Shared Templates**: User ID `allbsp` contains organization-wide templates
- **Template Names**: FSS UKB MODEL, FSS UKB ROE, FSS TBRCB ROE

Query pattern (from `app/utils.py:151`):
```sql
WHERE c.user_id IN (@user_id, 'allbsp')
```

## Technical Details

### Dependencies:
- `python-docx`: Document creation and formatting
- `docx.oxml`: Low-level XML manipulation for advanced formatting
- `docx.enum.table`: Table alignment enums

### Key Imports:
```python
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
```

### Style Standards:
- **Font**: Calibri
- **Body Text**: 11pt
- **Table Text**: 10pt
- **Headers**: 12pt (major), 11pt (minor)
- **Margins**: 1" top/bottom, 1.25" left/right
- **Alignment**: Justified for body, centered for headers

## Testing

Run the test script to verify formatting:
```bash
python test_docx_format.py
```

This creates `sample_bsp_format.docx` with example tables demonstrating:
- Overall Summary of Assessment table
- Institutional Level Support table
- Proper header formatting
- Rating highlighting
- Narrative text formatting

## Next Steps

To further align with specific BSP templates:
1. Query CosmosDB templates during runtime (when environment variables available)
2. Cache template structures in session state
3. Apply template-specific formatting rules
4. Support template selection in UI

## Maintenance Notes

- Table detection threshold: 50% of lines must have table indicators
- Rating keywords are case-insensitive during detection
- Maximum 500MB file upload size (.streamlit/config.toml)
- Session state keys: `last_output`, `last_title`, `output_ready`
