# 🇺🇦 DocumentReader - Ukrainian PDF Document Processing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python application for extracting text and tables from scanned Ukrainian PDF documents using OCR technology and advanced table detection algorithms.

## ✨ Features

- **🔤 Ukrainian OCR Support**: Optimized for Ukrainian text extraction using Tesseract OCR
- **📊 Table Detection**: Advanced table extraction using Camelot and Tabula libraries
- **🖼️ Image Processing**: PDF to image conversion with enhancement for better OCR results
- **📁 Batch Processing**: Process multiple PDF files at once
- **🎯 Confidence Scoring**: OCR confidence metrics and quality assessment
- **💾 Multiple Output Formats**: Save results as TXT, JSON, CSV, and Excel files
- **🖥️ CLI Interface**: Easy-to-use command-line interface
- **🌐 GUI Interface**: Optional Streamlit-based web interface
- **📝 Detailed Logging**: Comprehensive logging with Unicode support
- **⚙️ Configurable**: Flexible configuration for different use cases

## 🎯 Use Cases

- Processing scanned Ukrainian government documents
- Extracting data from financial reports and invoices
- Converting legacy paper documents to digital format
- Automating data entry from tabular documents
- Academic research with Ukrainian historical documents

## 📋 Requirements

### System Dependencies

- **Python 3.9+**
- **Tesseract OCR** with Ukrainian language support
- **Poppler** (for PDF processing)
- **OpenCV** (for image processing)

### Operating System Support

- ✅ Linux (Ubuntu/Debian)
- ✅ macOS
- ✅ Windows

## 🚀 Installation

### 1. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-ukr tesseract-ocr-eng
sudo apt-get install poppler-utils
sudo apt-get install libgl1-mesa-glx  # For OpenCV
```

#### macOS
```bash
brew install tesseract tesseract-lang
brew install poppler
```

#### Windows
1. Download and install Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download Ukrainian language data from [tessdata](https://github.com/tesseract-ocr/tessdata)
3. Install Poppler from [poppler-windows](https://blog.alivate.com.au/poppler-windows/)

### 2. Clone Repository
```bash
git clone https://github.com/your-username/DocumentReader.git
cd DocumentReader
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python main.py check
```

## 💻 Usage

### Command Line Interface

#### Process Single PDF File
```bash
# Basic usage
python main.py process document.pdf

# With custom output directory
python main.py process document.pdf --output ./results

# With custom confidence threshold
python main.py process document.pdf --confidence-threshold 70

# With specific table extraction method
python main.py process document.pdf --table-method camelot

# Verbose output
python main.py process document.pdf --verbose
```

#### Batch Processing
```bash
# Process all PDFs in directory
python main.py batch ./pdf_documents/

# Recursive processing
python main.py batch ./pdf_documents/ --recursive

# Custom output directory
python main.py batch ./pdf_documents/ --output ./batch_results

# Custom file pattern
python main.py batch ./documents/ --pattern "*.pdf" --recursive
```

#### System Check
```bash
# Check all dependencies
python main.py check

# Check specific components
python main.py check --check-tesseract
python main.py check --check-languages
python main.py check --check-dependencies
```

### GUI Interface (Optional)

Launch the Streamlit web interface:
```bash
python main.py gui document.pdf
```

Or run Streamlit directly:
```bash
streamlit run src/gui.py
```

### Python API

```python
from pathlib import Path
from src.cli import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(output_dir=Path('./output'))

# Process single file
result = processor.process_single_file(Path('document.pdf'))

# Check results
if result['success']:
    print(f"Processed {result['pages_processed']} pages")
    print(f"OCR confidence: {result['ocr_results']['average_confidence']:.1f}%")
    print(f"Tables found: {result['table_results']['tables_found']}")
```

## 📊 Output Structure

The application creates the following output structure:

```
output/
├── document_name/
│   ├── ocr_results_page_001.txt          # Plain text for page 1
│   ├── ocr_results_page_001.json         # Detailed OCR data
│   ├── ocr_results_combined.txt          # All pages combined
│   ├── ocr_results_summary.json          # OCR processing summary
│   ├── tables_page_001_table_001.csv     # Table data in CSV
│   ├── tables_page_001_table_001.json    # Table data in JSON
│   ├── tables_page_001_table_001.xlsx    # Table data in Excel
│   └── tables_summary.json               # Table extraction summary
└── logs/
    └── document_reader.log                # Processing logs
```

## ⚙️ Configuration

### OCR Settings
- **Language**: Ukrainian + English (`ukr+eng`)
- **Confidence Threshold**: 60% (configurable)
- **DPI**: 300 (configurable)
- **Image Enhancement**: Enabled

### Table Detection Settings
- **Method**: Camelot (default) or Tabula
- **Camelot Flavor**: Stream (default) or Lattice
- **Output Formats**: CSV, JSON, Excel

### Advanced Configuration

Edit `src/config.py` to customize:
```python
# OCR Configuration
OCR_CONFIG = {
    'language': 'ukr+eng',
    'confidence_threshold': 60,
    'dpi': 300,
}

# Table Configuration
TABLE_CONFIG = {
    'detection_method': 'camelot',
    'camelot_flavor': 'stream',
}
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ocr.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src
```

## 🔧 Troubleshooting

### Common Issues

#### Tesseract Not Found
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Add Tesseract to PATH
```

#### Ukrainian Language Not Available
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-ukr

# macOS
brew install tesseract-lang

# Windows: Download ukr.traineddata to tessdata folder
```

#### PDF Conversion Issues
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

#### Memory Issues with Large PDFs
- Reduce DPI in configuration
- Process pages individually
- Increase system memory

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
python main.py process document.pdf --verbose
```

## 📈 Performance Tips

1. **Optimize DPI**: Use 300 DPI for good quality, 150 DPI for speed
2. **Batch Processing**: Process multiple files together for efficiency
3. **Table Method**: 
   - Use Camelot for complex tables
   - Use Tabula for simple tables
4. **Image Enhancement**: Disable for clean scans to save time
5. **Confidence Threshold**: Lower for difficult documents, higher for quality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 src/
black src/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for OCR capabilities
- [Camelot](https://github.com/camelot-dev/camelot) for table extraction
- [Tabula](https://github.com/tabulapdf/tabula-py) for table extraction
- [pdf2image](https://github.com/Belval/pdf2image) for PDF conversion
- [OpenCV](https://opencv.org/) for image processing
- [Streamlit](https://streamlit.io/) for GUI interface

## 📞 Support

- 📧 Email: support@documentreader.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/DocumentReader/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/DocumentReader/wiki)

## 🗺️ Roadmap

- [ ] Support for more languages
- [ ] Improved table detection accuracy
- [ ] Cloud processing support
- [ ] Docker containerization
- [ ] REST API interface
- [ ] Mobile app support

---

## Project Structure
DocumentReader/
├── src/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── ingest.py                # PDF processing and image conversion
│   ├── ocr.py                   # Ukrainian OCR processing
│   ├── tables.py                # Table detection and extraction
│   ├── cli.py                   # Command-line interface
│   ├── gui.py                   # Streamlit GUI interface
│   └── utils/                   # Utility modules
│       ├── __init__.py          # Utils package init
│       └── logger.py            # Unicode-aware logging
├── tests/                       # Test suite
│   ├── __init__.py              # Test package init
│   └── test_ocr.py              # OCR module tests
├── main.py                      # Main entry point
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md                    # Comprehensive documentation

Made with ❤️ for Ukrainian document processing 