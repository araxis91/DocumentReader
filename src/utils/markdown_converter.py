"""
Markdown conversion utilities using Microsoft's markitdown.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import logging
from datetime import datetime

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    MarkItDown = None

import pandas as pd

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """
    Converter for creating markdown output from OCR results and tables.
    Uses Microsoft's markitdown utility when available.
    """
    
    def __init__(self, enable_ocr: bool = True):
        """
        Initialize markdown converter.
        
        Args:
            enable_ocr: Whether to enable OCR in markitdown for image processing
        """
        self.enable_ocr = enable_ocr
        
        if MARKITDOWN_AVAILABLE:
            try:
                self.markitdown = MarkItDown()
                logger.info("MarkItDown initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MarkItDown: {e}")
                self.markitdown = None
        else:
            logger.warning("MarkItDown not available. Install with: pip install markitdown")
            self.markitdown = None
    
    def convert_text_to_markdown(self, text: str, title: str = "Extracted Text", 
                               page_num: Optional[int] = None) -> str:
        """
        Convert plain text to markdown format with proper formatting.
        
        Args:
            text: Plain text to convert
            title: Title for the markdown document
            page_num: Optional page number
            
        Returns:
            Formatted markdown string
        """
        if not text or not text.strip():
            return ""
        
        # Create markdown header
        markdown_lines = []
        
        # Add title
        if page_num:
            markdown_lines.append(f"# {title} - Page {page_num}")
        else:
            markdown_lines.append(f"# {title}")
        
        markdown_lines.append("")  # Empty line
        
        # Process text into paragraphs and lists
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Clean up the paragraph
            clean_paragraph = paragraph.strip()
            
            # Check if it looks like a list item
            if self._is_list_item(clean_paragraph):
                # Convert to markdown list
                list_items = clean_paragraph.split('\n')
                for item in list_items:
                    item = item.strip()
                    if item:
                        # Remove common bullet characters and add markdown bullets
                        item = re.sub(r'^[•·▪▫-]\s*', '', item)
                        item = re.sub(r'^\d+\.\s*', '', item)  # Remove numbered list markers
                        markdown_lines.append(f"- {item}")
                markdown_lines.append("")  # Empty line after list
            else:
                # Regular paragraph
                markdown_lines.append(clean_paragraph)
                markdown_lines.append("")  # Empty line
        
        return '\n'.join(markdown_lines)
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text appears to be a list."""
        lines = text.split('\n')
        if len(lines) < 2:
            return False
            
        list_indicators = 0
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if re.match(r'^[•·▪▫-]\s+', line) or re.match(r'^\d+\.\s+', line):
                list_indicators += 1
        
        return list_indicators >= 2
    
    def convert_table_to_markdown(self, df: pd.DataFrame, title: str = "Extracted Table",
                                page_num: Optional[int] = None, table_num: Optional[int] = None) -> str:
        """
        Convert DataFrame to markdown table format.
        
        Args:
            df: DataFrame to convert
            title: Title for the table
            page_num: Optional page number
            table_num: Optional table number
            
        Returns:
            Markdown formatted table
        """
        if df.empty:
            return ""
        
        markdown_lines = []
        
        # Add title
        if page_num and table_num:
            markdown_lines.append(f"## {title} - Page {page_num}, Table {table_num}")
        elif page_num:
            markdown_lines.append(f"## {title} - Page {page_num}")
        else:
            markdown_lines.append(f"## {title}")
        
        markdown_lines.append("")  # Empty line
        
        # Convert DataFrame to markdown table
        try:
            # Clean up DataFrame - replace NaN and empty strings
            clean_df = df.fillna('').astype(str)
            
            # Create markdown table
            headers = list(clean_df.columns)
            markdown_lines.append("| " + " | ".join(headers) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            for _, row in clean_df.iterrows():
                row_data = [str(cell).strip() for cell in row]
                markdown_lines.append("| " + " | ".join(row_data) + " |")
            
            markdown_lines.append("")  # Empty line after table
            
        except Exception as e:
            logger.warning(f"Failed to convert table to markdown: {e}")
            markdown_lines.append("*Error converting table to markdown*")
            markdown_lines.append("")
        
        return '\n'.join(markdown_lines)
    
    def create_document_markdown(self, ocr_results: List, table_results: List,
                               document_title: str = "Document Analysis Results",
                               include_metadata: bool = True) -> str:
        """
        Create a comprehensive markdown document from OCR and table results.
        
        Args:
            ocr_results: List of OCR results
            table_results: List of table results
            document_title: Title for the document
            include_metadata: Whether to include metadata section
            
        Returns:
            Complete markdown document
        """
        markdown_lines = []
        
        # Document header
        markdown_lines.append(f"# {document_title}")
        markdown_lines.append("")
        
        if include_metadata:
            # Add metadata section
            markdown_lines.append("## Document Information")
            markdown_lines.append("")
            markdown_lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            markdown_lines.append(f"- **Total Pages:** {len(ocr_results)}")
            markdown_lines.append(f"- **Total Tables:** {len(table_results)}")
            
            # Language statistics if available
            if ocr_results and hasattr(ocr_results[0], 'language_stats'):
                total_words = sum(r.language_stats.get('total_words', 0) for r in ocr_results if hasattr(r, 'language_stats'))
                uk_words = sum(r.language_stats.get('ukrainian_words', 0) for r in ocr_results if hasattr(r, 'language_stats'))
                en_words = sum(r.language_stats.get('english_words', 0) for r in ocr_results if hasattr(r, 'language_stats'))
                
                markdown_lines.append(f"- **Total Words:** {total_words}")
                markdown_lines.append(f"- **Ukrainian Words:** {uk_words}")
                markdown_lines.append(f"- **English Words:** {en_words}")
            
            markdown_lines.append("")
            markdown_lines.append("---")
            markdown_lines.append("")
        
        # Add extracted text by page
        if ocr_results:
            markdown_lines.append("## Extracted Text")
            markdown_lines.append("")
            
            for result in ocr_results:
                if hasattr(result, 'text') and result.text.strip():
                    page_markdown = self.convert_text_to_markdown(
                        result.text, 
                        f"Page {result.page_num}", 
                        result.page_num
                    )
                    markdown_lines.append(page_markdown)
                    markdown_lines.append("")
        
        # Add tables
        if table_results:
            markdown_lines.append("## Extracted Tables")
            markdown_lines.append("")
            
            for table in table_results:
                if hasattr(table, 'data') and not table.data.empty:
                    table_markdown = self.convert_table_to_markdown(
                        table.data,
                        "Table",
                        table.page_num,
                        table.table_num
                    )
                    markdown_lines.append(table_markdown)
                    markdown_lines.append("")
        
        return '\n'.join(markdown_lines)
    
    def create_language_specific_markdown(self, ocr_results: List, table_results: List,
                                        language: str, document_title: str = "Document Analysis Results") -> str:
        """
        Create markdown document for specific language (Ukrainian or English).
        
        Args:
            ocr_results: List of OCR results
            table_results: List of table results
            language: Language code ('uk' or 'en')
            document_title: Title for the document
            
        Returns:
            Language-specific markdown document
        """
        lang_name = "Ukrainian" if language == 'uk' else "English"
        title = f"{document_title} - {lang_name}"
        
        markdown_lines = []
        
        # Document header
        markdown_lines.append(f"# {title}")
        markdown_lines.append("")
        
        # Add metadata
        markdown_lines.append("## Document Information")
        markdown_lines.append("")
        markdown_lines.append(f"- **Language:** {lang_name}")
        markdown_lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Count pages and tables with content in this language
        pages_with_content = 0
        tables_with_content = 0
        
        for result in ocr_results:
            if language == 'uk' and hasattr(result, 'ukrainian_text') and result.ukrainian_text.strip():
                pages_with_content += 1
            elif language == 'en' and hasattr(result, 'english_text') and result.english_text.strip():
                pages_with_content += 1
        
        for table in table_results:
            if language == 'uk' and hasattr(table, 'ukrainian_data') and not table.ukrainian_data.empty:
                tables_with_content += 1
            elif language == 'en' and hasattr(table, 'english_data') and not table.english_data.empty:
                tables_with_content += 1
        
        markdown_lines.append(f"- **Pages with {lang_name} content:** {pages_with_content}")
        markdown_lines.append(f"- **Tables with {lang_name} content:** {tables_with_content}")
        markdown_lines.append("")
        markdown_lines.append("---")
        markdown_lines.append("")
        
        # Add extracted text by page
        if ocr_results:
            markdown_lines.append(f"## Extracted {lang_name} Text")
            markdown_lines.append("")
            
            for result in ocr_results:
                text = ""
                if language == 'uk' and hasattr(result, 'ukrainian_text'):
                    text = result.ukrainian_text
                elif language == 'en' and hasattr(result, 'english_text'):
                    text = result.english_text
                
                if text and text.strip():
                    page_markdown = self.convert_text_to_markdown(
                        text, 
                        f"Page {result.page_num}",
                        result.page_num
                    )
                    markdown_lines.append(page_markdown)
                    markdown_lines.append("")
        
        # Add tables
        if table_results:
            markdown_lines.append(f"## Extracted {lang_name} Tables")
            markdown_lines.append("")
            
            for table in table_results:
                table_data = None
                if language == 'uk' and hasattr(table, 'ukrainian_data'):
                    table_data = table.ukrainian_data
                elif language == 'en' and hasattr(table, 'english_data'):
                    table_data = table.english_data
                
                if table_data is not None and not table_data.empty:
                    table_markdown = self.convert_table_to_markdown(
                        table_data,
                        f"{lang_name} Table",
                        table.page_num,
                        table.table_num
                    )
                    markdown_lines.append(table_markdown)
                    markdown_lines.append("")
        
        return '\n'.join(markdown_lines)
    
    def convert_pdf_with_markitdown(self, pdf_path: Path) -> Optional[str]:
        """
        Convert PDF directly using markitdown for comparison/fallback.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Markdown content or None if conversion failed
        """
        if not self.markitdown:
            logger.warning("MarkItDown not available for PDF conversion")
            return None
        
        try:
            result = self.markitdown.convert(str(pdf_path))
            return result.text_content
        except Exception as e:
            logger.error(f"MarkItDown PDF conversion failed: {e}")
            return None


def create_markdown_converter(enable_ocr: bool = True) -> MarkdownConverter:
    """
    Factory function to create a markdown converter.
    
    Args:
        enable_ocr: Whether to enable OCR capabilities
        
    Returns:
        MarkdownConverter instance
    """
    return MarkdownConverter(enable_ocr=enable_ocr) 