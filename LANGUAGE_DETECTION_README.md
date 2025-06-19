# Language Detection and Text Splitting Feature

## Overview

The DocumentReader now includes advanced language detection and text splitting capabilities to automatically separate Ukrainian and English text from scanned PDF documents. This feature is particularly useful for processing multilingual documents that contain both Ukrainian and English content.

## Features

### 🔍 Automatic Language Detection
- **Character-based detection**: Uses Ukrainian and English character patterns for accurate language identification
- **Sentence-level analysis**: Analyzes complete sentences for better context understanding
- **Word-level splitting**: Splits mixed sentences into language-specific components
- **Fallback mechanisms**: Uses multiple detection methods for improved accuracy

### 📄 Text Separation
- **Preserves original text**: Maintains the complete original extracted text
- **Creates language-specific versions**: Generates separate Ukrainian and English text files
- **Maintains structure**: Preserves document structure and formatting where possible
- **Handles mixed content**: Intelligently processes sentences containing both languages

### 📊 Table Language Processing
- **Cell-level splitting**: Processes each table cell for language-specific content
- **Maintains table structure**: Preserves table format while splitting content
- **Language-specific tables**: Creates separate Ukrainian and English versions of tables
- **Statistical analysis**: Provides detailed language statistics for each table

## Output Files

### Text Files
When language splitting is enabled, the following files are created:

#### Original Files (unchanged)
- `ocr_results_page_001.txt` - Original extracted text
- `ocr_results_combined.txt` - All pages combined
- `ocr_results_page_001.json` - Metadata with language information

#### Ukrainian Files
- `ocr_results_page_001_uk.txt` - Ukrainian text only
- `ocr_results_combined_uk.txt` - Combined Ukrainian text
- `ocr_results_combined_uk.json` - Ukrainian text with metadata

#### English Files
- `ocr_results_page_001_en.txt` - English text only
- `ocr_results_combined_en.txt` - Combined English text
- `ocr_results_combined_en.json` - English text with metadata

### Table Files
For each extracted table, the following files are created:

#### Original Table Files
- `table_page_001_table_001.csv` - Original table data
- `table_page_001_table_001.json` - Table metadata
- `table_page_001_table_001.xlsx` - Excel format

#### Ukrainian Table Files
- `table_page_001_table_001_uk.csv` - Ukrainian content only
- `table_page_001_table_001_uk.json` - Ukrainian table metadata
- `table_page_001_table_001_uk.xlsx` - Ukrainian Excel format

#### English Table Files
- `table_page_001_table_001_en.csv` - English content only
- `table_page_001_table_001_en.json` - English table metadata
- `table_page_001_table_001_en.xlsx` - English Excel format

## Usage

### Command Line Interface

#### Process Single File with Language Splitting (Default)
```bash
python -m src.cli process document.pdf --output ./output
```

#### Disable Language Splitting
```bash
python -m src.cli process document.pdf --output ./output --disable-language-splitting
```

#### Batch Processing with Language Splitting
```bash
python -m src.cli batch ./input_folder --output ./output
```

#### Batch Processing without Language Splitting
```bash
python -m src.cli batch ./input_folder --output ./output --disable-language-splitting
```

### Python API

```python
from src.ocr import UkrainianOCR
from src.tables import TableExtractor
from src.utils.language_detector import LanguageTextSplitter, split_text_by_language

# Initialize OCR with language splitting
ocr = UkrainianOCR(enable_language_splitting=True)

# Initialize table extractor with language splitting
table_extractor = TableExtractor(enable_language_splitting=True)

# Use the standalone language detector
splitter = LanguageTextSplitter()
ukrainian_text, english_text = splitter.split_text_by_language("Mixed text content")

# Or use the convenience function
uk_text, en_text = split_text_by_language("Hello world! Привіт світ!")
```

## Language Statistics

The system provides detailed statistics about language distribution:

```json
{
  "language_stats": {
    "total_words": 150,
    "ukrainian_words": 75,
    "english_words": 60,
    "uncertain_words": 15,
    "ukrainian_percentage": 50.0,
    "english_percentage": 40.0,
    "primary_language": "mixed"
  }
}
```

## Configuration

### Language Detection Settings
The language detection can be configured by modifying the `LanguageTextSplitter` class:

```python
# Initialize with custom confidence threshold
splitter = LanguageTextSplitter(confidence_threshold=0.8)

# Get detailed language statistics
stats = splitter.get_language_statistics(text)
```

### Supported Languages
Currently supported languages:
- **Ukrainian (uk)**: Full character set including і, ї, є, ґ
- **English (en)**: Standard Latin alphabet
- **Mixed content**: Intelligently handles mixed-language documents

## Dependencies

The language detection feature requires:
```
langdetect==1.0.9
```

Install with:
```bash
pip install -r requirements.txt
```

## Testing

Test the language detection functionality:
```bash
python test_language_detection.py
```

This will run comprehensive tests with sample mixed Ukrainian and English text to demonstrate the splitting capabilities.

## Advanced Features

### Word-Level Language Detection
- Uses character-based heuristics for high accuracy
- Handles mixed sentences intelligently
- Preserves word order where possible
- Filters out punctuation and numbers

### Sentence-Level Analysis
- Analyzes complete sentences for context
- Handles complex mixed-language sentences
- Maintains sentence structure and punctuation
- Provides confidence scores for language detection

### Table Cell Processing
- Processes each table cell individually
- Maintains table structure and formatting
- Creates language-specific table versions
- Provides detailed statistics per table

## Troubleshooting

### Common Issues

1. **Low Language Detection Accuracy**
   - Check that the text contains enough content for analysis
   - Verify that Ukrainian language data is properly installed
   - Consider adjusting confidence thresholds

2. **Missing Language-Specific Files**
   - Ensure language splitting is enabled
   - Check that the document contains text in both languages
   - Verify that language detection is working correctly

3. **Empty Language-Specific Output**
   - Some documents may contain only one language
   - Check the language statistics in the summary files
   - Verify that OCR quality is sufficient for language detection

### Performance Considerations
- Language detection adds minimal processing overhead
- Text splitting is performed during OCR processing
- Large documents may take slightly longer to process
- Consider disabling language splitting for single-language documents

## Example Output Structure

```
output/
├── document_name/
│   ├── ocr_results_page_001.txt           # Original text
│   ├── ocr_results_page_001_uk.txt        # Ukrainian text
│   ├── ocr_results_page_001_en.txt        # English text
│   ├── ocr_results_combined.txt           # All original text
│   ├── ocr_results_combined_uk.txt        # All Ukrainian text
│   ├── ocr_results_combined_en.txt        # All English text
│   ├── ocr_results_combined_uk.json       # Ukrainian metadata
│   ├── ocr_results_combined_en.json       # English metadata
│   ├── table_page_001_table_001.csv       # Original table
│   ├── table_page_001_table_001_uk.csv    # Ukrainian table
│   ├── table_page_001_table_001_en.csv    # English table
│   └── ocr_results_summary.json           # Processing summary
```

This feature ensures that multilingual documents are properly processed and separated, making it easier to work with specific language content while preserving the original document structure. 