# Markdown Output Feature

This document describes the new markdown output feature added to the DocumentReader project using Microsoft's markitdown utility.

## Overview

The markdown feature converts extracted OCR text and table data into clean, readable Markdown format. This provides a more user-friendly way to view and share document analysis results while maintaining proper formatting and structure.

## Features

### 1. Comprehensive Markdown Generation
- **Complete Analysis**: Combines OCR text and tables into a single comprehensive markdown document
- **Language-Specific Documents**: Separate markdown files for Ukrainian and English content
- **Individual Page/Table Files**: Individual markdown files for each page and table
- **Metadata Integration**: Includes document statistics, language breakdown, and processing information

### 2. Microsoft Markitdown Integration
- **Direct PDF Conversion**: Optional fallback using markitdown's native PDF conversion
- **Comparison Mode**: Compare your OCR results with markitdown's output
- **Advanced OCR**: Leverages markitdown's OCR capabilities when available

### 3. Smart Text Formatting
- **Paragraph Detection**: Automatically formats text into proper paragraphs
- **List Recognition**: Converts bullet points and numbered lists to markdown lists
- **Table Formatting**: Creates properly formatted markdown tables from extracted data
- **Language-Aware Formatting**: Handles multilingual content appropriately

## Installation

Install the required dependency:

```bash
pip install markitdown>=0.0.1
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Basic Usage (Markdown Enabled by Default)
```bash
# Process single file with markdown output
python -m src.cli process document.pdf

# Process batch with markdown output
python -m src.cli batch /path/to/pdfs/
```

#### Disable Markdown Output
```bash
# Disable markdown generation
python -m src.cli process document.pdf --disable-markdown

# Disable markdown for batch processing
python -m src.cli batch /path/to/pdfs/ --disable-markdown
```

#### Enable Markitdown Fallback
```bash
# Enable markitdown native PDF conversion for comparison
python -m src.cli process document.pdf --enable-markitdown-fallback

# With additional options
python -m src.cli process document.pdf \
    --enable-markitdown-fallback \
    --disable-language-splitting
```

### Output Structure

When markdown is enabled, the following files are generated:

```
output/
├── document_name/
│   ├── ocr_results/
│   │   ├── ocr_results_combined.md          # All OCR text in markdown
│   │   ├── ocr_results_combined_uk.md       # Ukrainian text only
│   │   ├── ocr_results_combined_en.md       # English text only
│   │   ├── ocr_results_page_001.md          # Individual page markdown
│   │   └── ...
│   ├── tables/
│   │   ├── tables_combined.md               # All tables in markdown
│   │   ├── tables_combined_uk.md            # Ukrainian tables only
│   │   ├── tables_combined_en.md            # English tables only
│   │   ├── table_page_001_table_001.md      # Individual table markdown
│   │   └── ...
│   ├── complete_analysis.md                 # Comprehensive document
│   ├── complete_analysis_uk.md              # Ukrainian comprehensive
│   ├── complete_analysis_en.md              # English comprehensive
│   └── markitdown_conversion.md             # (if --enable-markitdown-fallback)
```

## Markdown Document Structure

### Comprehensive Analysis Document

```markdown
# Complete Analysis - Document Name

## Document Information
- **Generated:** 2025-01-XX XX:XX:XX
- **Total Pages:** 5
- **Total Tables:** 3
- **Total Words:** 1,234
- **Ukrainian Words:** 567
- **English Words:** 667

---

## Extracted Text

### Page 1 Analysis - Page 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

- Bullet point one
- Bullet point two
- Bullet point three

### Page 2 Analysis - Page 2

More extracted text content...

## Extracted Tables

### Table - Page 1, Table 1

| Header 1 | Header 2 | Header 3 |
| --- | --- | --- |
| Data 1 | Data 2 | Data 3 |
| Data 4 | Data 5 | Data 6 |
```

## Python API Usage

### Basic Markdown Generation

```python
from src.utils.markdown_converter import create_markdown_converter

# Create converter
converter = create_markdown_converter()

# Convert text to markdown
markdown_text = converter.convert_text_to_markdown(
    "Your extracted text here",
    title="Document Title",
    page_num=1
)

# Convert table to markdown
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
markdown_table = converter.convert_table_to_markdown(
    df,
    title="Table Title",
    page_num=1,
    table_num=1
)
```

### Comprehensive Document Generation

```python
# Create comprehensive markdown document
markdown_content = converter.create_document_markdown(
    ocr_results=your_ocr_results,
    table_results=your_table_results,
    document_title="My Document Analysis",
    include_metadata=True
)

# Create language-specific document
ukrainian_markdown = converter.create_language_specific_markdown(
    ocr_results=your_ocr_results,
    table_results=your_table_results,
    language='uk',
    document_title="My Document Analysis"
)
```

### Direct PDF Conversion with Markitdown

```python
from pathlib import Path

# Convert PDF directly using markitdown
pdf_path = Path("document.pdf")
markitdown_result = converter.convert_pdf_with_markitdown(pdf_path)

if markitdown_result:
    print("Markitdown conversion successful")
    # Save or use the result
else:
    print("Markitdown conversion failed")
```

## Advanced Features

### 1. Custom Formatting

The markdown converter intelligently formats different types of content:

- **Lists**: Automatically detects and formats bullet points and numbered lists
- **Tables**: Creates properly aligned markdown tables with headers
- **Paragraphs**: Handles text wrapping and paragraph separation
- **Metadata**: Includes processing statistics and document information

### 2. Language-Aware Processing

When language splitting is enabled:

- Separate markdown files for Ukrainian and English content
- Language-specific metadata and statistics
- Proper handling of mixed-language documents
- Unicode support for Cyrillic characters

### 3. Markitdown Integration

Optional integration with Microsoft's markitdown:

- Native PDF processing for comparison
- Advanced OCR capabilities
- Support for multiple document formats
- Fallback option for difficult documents

## Configuration

### Environment Variables

```bash
# Optional: Set markitdown service endpoint (if using Azure Document Intelligence)
export MARKITDOWN_ENDPOINT="your-endpoint"
export MARKITDOWN_KEY="your-key"
```

### Python Configuration

```python
# Customize markdown converter
converter = create_markdown_converter(enable_ocr=True)

# Configure in DocumentProcessor
processor = DocumentProcessor(
    output_dir=Path("./output"),
    enable_markdown=True,
    enable_markitdown_fallback=False
)
```

## Troubleshooting

### Common Issues

1. **Markitdown Not Available**
   ```
   WARNING: MarkItDown not available. Install with: pip install markitdown
   ```
   **Solution**: Install markitdown: `pip install markitdown`

2. **Import Errors**
   ```
   ImportError: No module named 'markitdown'
   ```
   **Solution**: Ensure markitdown is installed and in your Python path

3. **Empty Markdown Output**
   - Check if input text/tables contain data
   - Verify OCR processing was successful
   - Check file permissions for output directory

4. **Formatting Issues**
   - Ensure input text encoding is UTF-8
   - Check for special characters in table data
   - Verify pandas DataFrame structure

### Performance Considerations

- Markdown generation adds minimal processing time
- Large documents may take longer for comprehensive analysis
- Individual file generation is parallelizable
- Memory usage scales with document size

## Examples

### Sample Output

See the `examples/` directory for sample markdown outputs from different document types:

- `sample_mixed_language.md` - Ukrainian/English mixed document
- `sample_tables_only.md` - Document with primarily tabular data
- `sample_comprehensive.md` - Full analysis with all features

### Integration Examples

```python
# Example: Process document and generate markdown
from src.cli import DocumentProcessor
from pathlib import Path

processor = DocumentProcessor(
    output_dir=Path("./results"),
    enable_markdown=True,
    enable_markitdown_fallback=True
)

result = processor.process_single_file(Path("document.pdf"))
print(f"Generated markdown files in: {result['output_dir']}")
```

## Future Enhancements

Planned improvements for the markdown feature:

1. **HTML Export**: Convert markdown to HTML for web viewing
2. **PDF Export**: Generate PDF from markdown using pandoc
3. **Template System**: Custom markdown templates for different document types
4. **Style Customization**: Configurable CSS for HTML output
5. **Batch Analysis**: Aggregate markdown reports for multiple documents

## Support

For issues related to markdown functionality:

1. Check this README for common solutions
2. Ensure all dependencies are installed correctly
3. Verify input document format and content
4. Check output directory permissions
5. Review log files for detailed error messages

The markdown feature integrates seamlessly with existing DocumentReader functionality while providing enhanced output formatting and readability. 