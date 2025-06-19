"""
Configuration settings for DocumentReader application.
"""

import os
from pathlib import Path
from typing import Dict, List

# Application settings
APP_NAME = "DocumentReader"
VERSION = "1.0.0"

# OCR Configuration
OCR_CONFIG = {
    'language': 'ukr+eng',  # Ukrainian + English for better results
    'psm': 6,  # Assume a single uniform block of text
    'oem': 3,  # Use both legacy and LSTM engines
    'confidence_threshold': 60,  # Minimum confidence level for OCR
    'dpi': 300,  # DPI for PDF to image conversion
}

# Tesseract custom configuration
TESSERACT_CONFIG = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩьЮЯабвгдеєжзиіїйклмнопрстуфхцчшщьюя .,!?()-:;'

# Table Detection Configuration
TABLE_CONFIG = {
    'detection_method': 'tabula',  # or 'camelot' - using tabula as default for Python 3.13 compatibility
    'camelot_flavor': 'stream',  # 'lattice' or 'stream'
    'table_areas': None,  # Auto-detect
    'edge_tol': 50,
    'row_tol': 2,
    'column_tol': 0,
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'format': 'PNG',
    'dpi': 300,
    'grayscale': False,
    'enhance_contrast': True,
    'noise_reduction': True,
}

# Output Configuration
OUTPUT_CONFIG = {
    'text_format': 'txt',  # 'txt' or 'json'
    'table_formats': ['csv', 'json'],  # Available: 'csv', 'json', 'xlsx'
    'include_confidence': True,
    'create_summary': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'document_reader.log',
}

# File paths
DEFAULT_OUTPUT_DIR = Path('./output')
DEFAULT_TEMP_DIR = Path('./temp')
DEFAULT_LOG_DIR = Path('./logs')

# Supported file formats
SUPPORTED_PDF_EXTENSIONS = ['.pdf']
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

# Error messages
ERROR_MESSAGES = {
    'file_not_found': 'File not found: {}',
    'invalid_format': 'Unsupported file format: {}',
    'ocr_failed': 'OCR failed for page {}',
    'table_extraction_failed': 'Table extraction failed for page {}',
    'low_confidence': 'Low OCR confidence ({:.1f}%) for page {}',
}

def get_tesseract_path() -> str:
    """Get the path to tesseract executable."""
    # Common paths for different operating systems
    common_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/opt/homebrew/bin/tesseract',  # macOS with Homebrew
        'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  # Windows
        'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',  # Windows 32-bit
    ]
    
    # Check if tesseract is in PATH
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        return tesseract_path
    
    # Check common installation paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return 'tesseract'  # Default to PATH

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [DEFAULT_OUTPUT_DIR, DEFAULT_TEMP_DIR, DEFAULT_LOG_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True) 