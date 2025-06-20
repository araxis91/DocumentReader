"""
Table detection and extraction module.
"""

import camelot
import tabula
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import json
import tempfile
from PIL import Image
import cv2

from .config import TABLE_CONFIG, OUTPUT_CONFIG, ERROR_MESSAGES
from .utils.logger import get_logger
from .utils.language_detector import LanguageTextSplitter

logger = get_logger(__name__)


class TableResult:
    """Container for table extraction results with language separation."""
    
    def __init__(self, data: pd.DataFrame, page_num: int, table_num: int, confidence: float = 0.0,
                 enable_language_splitting: bool = True):
        """
        Initialize table result.
        
        Args:
            data: Extracted table data as DataFrame
            page_num: Page number (1-indexed)
            table_num: Table number on page (1-indexed)
            confidence: Extraction confidence score
            enable_language_splitting: Whether to enable language splitting
        """
        self.data = data
        self.page_num = page_num
        self.table_num = table_num
        self.confidence = confidence
        self.bbox = {}
        self.metadata = {}
        
        # Language-specific versions
        self.ukrainian_data = pd.DataFrame()
        self.english_data = pd.DataFrame()
        self.language_stats = {}
        
        # Initialize language splitting if data is present and enabled
        if not data.empty and enable_language_splitting:
            self._split_table_by_language()
    
    def _split_table_by_language(self):
        """Split table data by language."""
        try:
            splitter = LanguageTextSplitter()
            
            # Create copies of the original table structure
            self.ukrainian_data = self.data.copy()
            self.english_data = self.data.copy()
            
            # Process each cell in the table
            for col in self.data.columns:
                ukrainian_col = []
                english_col = []
                
                for idx, cell_value in self.data[col].items():
                    if pd.isna(cell_value) or str(cell_value).strip() == '':
                        ukrainian_col.append('')
                        english_col.append('')
                        continue
                    
                    # Split cell content by language
                    cell_text = str(cell_value)
                    uk_text, en_text = splitter.split_text_by_language(cell_text)
                    
                    ukrainian_col.append(uk_text)
                    english_col.append(en_text)
                
                # Update language-specific tables
                self.ukrainian_data[col] = ukrainian_col
                self.english_data[col] = english_col
            
            # Remove completely empty rows from language-specific tables
            self.ukrainian_data = self.ukrainian_data.loc[
                ~(self.ukrainian_data == '').all(axis=1)
            ]
            self.english_data = self.english_data.loc[
                ~(self.english_data == '').all(axis=1)
            ]
            
            # Calculate statistics
            total_cells = self.data.size
            uk_cells = sum(1 for col in self.ukrainian_data.columns 
                          for val in self.ukrainian_data[col] if str(val).strip())
            en_cells = sum(1 for col in self.english_data.columns 
                          for val in self.english_data[col] if str(val).strip())
            
            self.language_stats = {
                'total_cells': total_cells,
                'ukrainian_cells': uk_cells,
                'english_cells': en_cells,
                'ukrainian_rows': len(self.ukrainian_data),
                'english_rows': len(self.english_data)
            }
            
            logger.debug(f"Table {self.table_num} page {self.page_num}: Language split completed")
            
        except Exception as e:
            logger.warning(f"Language splitting failed for table {self.table_num} on page {self.page_num}: {e}")
            self.ukrainian_data = pd.DataFrame()
            self.english_data = pd.DataFrame()
            self.language_stats = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'page_num': self.page_num,
            'table_num': self.table_num,
            'confidence': self.confidence,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data': self.data.to_dict('records'),
            'bbox': self.bbox,
            'metadata': self.metadata,
            'ukrainian_data': self.ukrainian_data.to_dict('records') if not self.ukrainian_data.empty else [],
            'english_data': self.english_data.to_dict('records') if not self.english_data.empty else [],
            'language_stats': self.language_stats
        }
    
    def is_valid(self) -> bool:
        """Check if table contains valid data."""
        return not self.data.empty and self.data.shape[0] > 0 and self.data.shape[1] > 0


class TableExtractor:
    """Table detection and extraction processor."""
    
    def __init__(self, enable_language_splitting: bool = True):
        """Initialize table extractor."""
        self.method = TABLE_CONFIG['detection_method']
        self.camelot_flavor = TABLE_CONFIG['camelot_flavor']
        self.enable_language_splitting = enable_language_splitting
        
    def detect_tables_camelot(self, pdf_path: Path, pages: str = 'all') -> List[TableResult]:
        """
        Extract tables using Camelot.
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to process ('all' or page numbers)
            
        Returns:
            List of TableResult objects
        """
        results = []
        
        try:
            logger.info(f"Extracting tables using Camelot ({self.camelot_flavor} flavor)")
            
            # Extract tables
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor=self.camelot_flavor,
                edge_tol=TABLE_CONFIG['edge_tol'],
                row_tol=TABLE_CONFIG['row_tol'],
                column_tol=TABLE_CONFIG['column_tol']
            )
            
            logger.info(f"Found {len(tables)} tables using Camelot")
            
            # Process each table
            for i, table in enumerate(tables):
                try:
                    # Get table data
                    df = table.df
                    
                    # Clean empty rows and columns
                    df = self._clean_dataframe(df)
                    
                    if df.empty:
                        continue
                    
                    # Create result
                    result = TableResult(
                        data=df,
                        page_num=table.page,
                        table_num=i + 1,
                        confidence=table.accuracy if hasattr(table, 'accuracy') else 0.0,
                        enable_language_splitting=self.enable_language_splitting
                    )
                    
                    # Add metadata
                    if hasattr(table, '_bbox'):
                        result.bbox = {
                            'x1': table._bbox[0],
                            'y1': table._bbox[1],
                            'x2': table._bbox[2],
                            'y2': table._bbox[3]
                        }
                    
                    result.metadata = {
                        'extraction_method': 'camelot',
                        'flavor': self.camelot_flavor,
                        'parsing_report': table.parsing_report if hasattr(table, 'parsing_report') else {}
                    }
                    
                    results.append(result)
                    logger.debug(f"Extracted table {i+1} from page {table.page}")
                    
                except Exception as e:
                    logger.error(f"Failed to process table {i+1}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Camelot table extraction failed: {e}")
        
        return results
    
    def detect_tables_tabula(self, pdf_path: Path, pages: Union[str, List[int]] = 'all') -> List[TableResult]:
        """
        Extract tables using Tabula.
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to process ('all' or list of page numbers)
            
        Returns:
            List of TableResult objects
        """
        results = []
        
        try:
            logger.info("Extracting tables using Tabula")
            
            # Extract tables
            dfs = tabula.read_pdf(
                str(pdf_path),
                pages=pages,
                multiple_tables=True,
                pandas_options={'header': 0}
            )
            
            logger.info(f"Found {len(dfs)} tables using Tabula")
            
            # Process each table
            for i, df in enumerate(dfs):
                try:
                    # Clean dataframe
                    df = self._clean_dataframe(df)
                    
                    if df.empty:
                        continue
                    
                    # Create result (Tabula doesn't provide page numbers directly)
                    result = TableResult(
                        data=df,
                        page_num=1,  # Default, would need additional processing to get actual page
                        table_num=i + 1,
                        confidence=0.0,  # Tabula doesn't provide confidence scores
                        enable_language_splitting=self.enable_language_splitting
                    )
                    
                    result.metadata = {
                        'extraction_method': 'tabula',
                        'shape': df.shape
                    }
                    
                    results.append(result)
                    logger.debug(f"Extracted table {i+1}")
                    
                except Exception as e:
                    logger.error(f"Failed to process table {i+1}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Tabula table extraction failed: {e}")
        
        return results
    
    def detect_tables_from_images(self, images: List[np.ndarray], pdf_path: Optional[Path] = None) -> List[TableResult]:
        """
        Detect tables from images using image processing techniques.
        
        Args:
            images: List of images as numpy arrays
            pdf_path: Optional PDF path for additional processing
            
        Returns:
            List of TableResult objects
        """
        results = []
        
        # First try PDF-based extraction if PDF path is provided
        if pdf_path:
            if self.method == 'camelot':
                results.extend(self.detect_tables_camelot(pdf_path))
            elif self.method == 'tabula':
                results.extend(self.detect_tables_tabula(pdf_path))
        
        # If no tables found or no PDF, try image-based detection
        if not results:
            logger.info("Attempting image-based table detection")
            results.extend(self._detect_tables_opencv(images))
        
        return results
    
    def _detect_tables_opencv(self, images: List[np.ndarray]) -> List[TableResult]:
        """
        Detect tables using OpenCV image processing.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of TableResult objects
        """
        results = []
        
        for page_num, image in enumerate(images, 1):
            try:
                # Convert to grayscale
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                # Apply threshold
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Detect horizontal lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
                
                # Detect vertical lines
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
                
                # Combine lines
                table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
                
                # Find contours
                contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area
                min_area = 1000  # Minimum area threshold
                table_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                
                if table_contours:
                    logger.info(f"Found {len(table_contours)} potential tables on page {page_num}")
                    
                    # For each detected table region, try to extract structured data
                    for i, contour in enumerate(table_contours):
                        try:
                            # Get bounding rectangle
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Extract table region
                            table_region = image[y:y+h, x:x+w]
                            
                            # Try to parse as table (this is a simplified approach)
                            # In practice, you'd need more sophisticated table parsing
                            df = self._parse_table_region(table_region)
                            
                            if not df.empty:
                                result = TableResult(
                                    data=df,
                                    page_num=page_num,
                                    table_num=i + 1,
                                    confidence=0.5,  # Default confidence for image-based detection
                                    enable_language_splitting=self.enable_language_splitting
                                )
                                
                                result.bbox = {'x': x, 'y': y, 'width': w, 'height': h}
                                result.metadata = {
                                    'extraction_method': 'opencv',
                                    'contour_area': cv2.contourArea(contour)
                                }
                                
                                results.append(result)
                                
                        except Exception as e:
                            logger.warning(f"Failed to parse table region {i+1} on page {page_num}: {e}")
                            continue
                
            except Exception as e:
                logger.error(f"OpenCV table detection failed for page {page_num}: {e}")
                continue
        
        return results
    
    def _parse_table_region(self, image: np.ndarray) -> pd.DataFrame:
        """
        Parse table region to extract structured data.
        This is a simplified implementation - in practice you'd need more sophisticated parsing.
        
        Args:
            image: Table region image
            
        Returns:
            DataFrame with extracted data
        """
        # This is a placeholder implementation
        # In practice, you'd use OCR on the table cells and parse the structure
        
        # Create a simple dummy table for demonstration
        dummy_data = {
            'Column1': ['Cell1', 'Cell3'],
            'Column2': ['Cell2', 'Cell4']
        }
        
        return pd.DataFrame(dummy_data)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Replace NaN with empty strings
        df = df.fillna('')
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def save_tables(self, tables: List[TableResult], output_dir: Path, filename_prefix: str = 'table',
                  enable_markdown: bool = True):
        """
        Save extracted tables to files with language separation and markdown output.
        
        Args:
            tables: List of table results
            output_dir: Output directory
            filename_prefix: Prefix for output files
            enable_markdown: Whether to generate markdown output
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        ukrainian_tables_count = 0
        english_tables_count = 0
        
        for table in tables:
            if not table.is_valid():
                continue
            
            base_name = f"{filename_prefix}_page_{table.page_num:03d}_table_{table.table_num:03d}"
            
            # Save original table
            if 'csv' in OUTPUT_CONFIG['table_formats']:
                csv_file = output_dir / f"{base_name}.csv"
                table.data.to_csv(csv_file, index=False, encoding='utf-8')
                saved_files.append(csv_file)
                logger.debug(f"Saved original table as CSV: {csv_file}")
            
            if 'json' in OUTPUT_CONFIG['table_formats']:
                json_file = output_dir / f"{base_name}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(table.to_dict(), f, ensure_ascii=False, indent=2)
                saved_files.append(json_file)
                logger.debug(f"Saved original table as JSON: {json_file}")
            
            if 'xlsx' in OUTPUT_CONFIG['table_formats']:
                xlsx_file = output_dir / f"{base_name}.xlsx"
                table.data.to_excel(xlsx_file, index=False, engine='openpyxl')
                saved_files.append(xlsx_file)
                logger.debug(f"Saved original table as Excel: {xlsx_file}")
            
            # Save Ukrainian version if present
            if not table.ukrainian_data.empty:
                ukrainian_tables_count += 1
                
                if 'csv' in OUTPUT_CONFIG['table_formats']:
                    uk_csv_file = output_dir / f"{base_name}_uk.csv"
                    table.ukrainian_data.to_csv(uk_csv_file, index=False, encoding='utf-8')
                    saved_files.append(uk_csv_file)
                    logger.debug(f"Saved Ukrainian table as CSV: {uk_csv_file}")
                
                if 'json' in OUTPUT_CONFIG['table_formats']:
                    uk_json_file = output_dir / f"{base_name}_uk.json"
                    uk_data = {
                        'page_num': table.page_num,
                        'table_num': table.table_num,
                        'confidence': table.confidence,
                        'shape': table.ukrainian_data.shape,
                        'columns': list(table.ukrainian_data.columns),
                        'data': table.ukrainian_data.to_dict('records'),
                        'language_stats': table.language_stats,
                        'metadata': {**table.metadata, 'language': 'ukrainian'}
                    }
                    with open(uk_json_file, 'w', encoding='utf-8') as f:
                        json.dump(uk_data, f, ensure_ascii=False, indent=2)
                    saved_files.append(uk_json_file)
                    logger.debug(f"Saved Ukrainian table as JSON: {uk_json_file}")
                
                if 'xlsx' in OUTPUT_CONFIG['table_formats']:
                    uk_xlsx_file = output_dir / f"{base_name}_uk.xlsx"
                    table.ukrainian_data.to_excel(uk_xlsx_file, index=False, engine='openpyxl')
                    saved_files.append(uk_xlsx_file)
                    logger.debug(f"Saved Ukrainian table as Excel: {uk_xlsx_file}")
            
            # Save English version if present
            if not table.english_data.empty:
                english_tables_count += 1
                
                if 'csv' in OUTPUT_CONFIG['table_formats']:
                    en_csv_file = output_dir / f"{base_name}_en.csv"
                    table.english_data.to_csv(en_csv_file, index=False, encoding='utf-8')
                    saved_files.append(en_csv_file)
                    logger.debug(f"Saved English table as CSV: {en_csv_file}")
                
                if 'json' in OUTPUT_CONFIG['table_formats']:
                    en_json_file = output_dir / f"{base_name}_en.json"
                    en_data = {
                        'page_num': table.page_num,
                        'table_num': table.table_num,
                        'confidence': table.confidence,
                        'shape': table.english_data.shape,
                        'columns': list(table.english_data.columns),
                        'data': table.english_data.to_dict('records'),
                        'language_stats': table.language_stats,
                        'metadata': {**table.metadata, 'language': 'english'}
                    }
                    with open(en_json_file, 'w', encoding='utf-8') as f:
                        json.dump(en_data, f, ensure_ascii=False, indent=2)
                    saved_files.append(en_json_file)
                    logger.debug(f"Saved English table as JSON: {en_json_file}")
                
                if 'xlsx' in OUTPUT_CONFIG['table_formats']:
                    en_xlsx_file = output_dir / f"{base_name}_en.xlsx"
                    table.english_data.to_excel(en_xlsx_file, index=False, engine='openpyxl')
                    saved_files.append(en_xlsx_file)
                    logger.debug(f"Saved English table as Excel: {en_xlsx_file}")
        
        # Generate markdown output if enabled
        if enable_markdown and tables:
            self._save_markdown_tables(tables, output_dir, filename_prefix)
        
        # Save summary with language statistics
        language_stats = {
            'ukrainian_cells': sum(t.language_stats.get('ukrainian_cells', 0) for t in tables),
            'english_cells': sum(t.language_stats.get('english_cells', 0) for t in tables),
            'ukrainian_tables': ukrainian_tables_count,
            'english_tables': english_tables_count
        }
        
        summary = {
            'total_tables': len(tables),
            'valid_tables': len([t for t in tables if t.is_valid()]),
            'tables_by_page': {},
            'extraction_methods': list(set([t.metadata.get('extraction_method', 'unknown') for t in tables])),
            'saved_files': [str(f) for f in saved_files],
            'language_breakdown': language_stats
        }
        
        # Group tables by page
        for table in tables:
            page_key = f"page_{table.page_num}"
            if page_key not in summary['tables_by_page']:
                summary['tables_by_page'][page_key] = 0
            summary['tables_by_page'][page_key] += 1
        
        summary_file = output_dir / f"{filename_prefix}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(saved_files)} table files to {output_dir}")
        logger.info(f"Summary: {summary['valid_tables']}/{summary['total_tables']} tables extracted successfully")
        logger.info(f"Language breakdown: {ukrainian_tables_count} Ukrainian tables, {english_tables_count} English tables")
    
    def _save_markdown_tables(self, tables: List[TableResult], output_dir: Path, filename_prefix: str):
        """
        Save table results in markdown format.
        
        Args:
            tables: List of table results
            output_dir: Output directory
            filename_prefix: Prefix for output files
        """
        try:
            from src.utils.markdown_converter import create_markdown_converter
            
            converter = create_markdown_converter()
            
            # Generate comprehensive markdown document with all tables
            document_title = f"Table Analysis Results - {output_dir.name}"
            markdown_content = converter.create_document_markdown(
                [], tables, document_title, include_metadata=True
            )
            
            if markdown_content:
                markdown_file = output_dir / f"{filename_prefix}_combined.md"
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Saved combined table markdown: {markdown_file}")
            
            # Generate language-specific markdown documents
            if any(hasattr(t, 'ukrainian_data') and not t.ukrainian_data.empty for t in tables):
                uk_markdown = converter.create_language_specific_markdown(
                    [], tables, 'uk', document_title
                )
                if uk_markdown:
                    uk_markdown_file = output_dir / f"{filename_prefix}_combined_uk.md"
                    with open(uk_markdown_file, 'w', encoding='utf-8') as f:
                        f.write(uk_markdown)
                    logger.info(f"Saved Ukrainian table markdown: {uk_markdown_file}")
            
            if any(hasattr(t, 'english_data') and not t.english_data.empty for t in tables):
                en_markdown = converter.create_language_specific_markdown(
                    [], tables, 'en', document_title
                )
                if en_markdown:
                    en_markdown_file = output_dir / f"{filename_prefix}_combined_en.md"
                    with open(en_markdown_file, 'w', encoding='utf-8') as f:
                        f.write(en_markdown)
                    logger.info(f"Saved English table markdown: {en_markdown_file}")
            
            # Generate individual table markdown files
            for table in tables:
                if table.is_valid():
                    table_markdown = converter.convert_table_to_markdown(
                        table.data, f"Table Analysis", table.page_num, table.table_num
                    )
                    if table_markdown:
                        table_md_file = output_dir / f"{filename_prefix}_page_{table.page_num:03d}_table_{table.table_num:03d}.md"
                        with open(table_md_file, 'w', encoding='utf-8') as f:
                            f.write(table_markdown)
                        logger.debug(f"Saved table markdown: {table_md_file}")
            
        except ImportError:
            logger.warning("Markdown converter not available. Install markitdown: pip install markitdown")
        except Exception as e:
            logger.error(f"Failed to generate table markdown output: {e}")
    
    def extract_tables(self, pdf_path: Optional[Path] = None, images: Optional[List[np.ndarray]] = None) -> List[TableResult]:
        """
        Main method to extract tables from PDF or images.
        
        Args:
            pdf_path: Optional path to PDF file
            images: Optional list of images
            
        Returns:
            List of extracted tables
        """
        if pdf_path is None and images is None:
            raise ValueError("Either pdf_path or images must be provided")
        
        results = []
        
        try:
            if pdf_path and pdf_path.exists():
                # Try PDF-based extraction first
                if self.method == 'camelot':
                    results = self.detect_tables_camelot(pdf_path)
                elif self.method == 'tabula':
                    results = self.detect_tables_tabula(pdf_path)
                
                # If no results and we have images, try image-based extraction
                if not results and images:
                    logger.info("PDF-based extraction found no tables, trying image-based extraction")
                    results = self._detect_tables_opencv(images)
            
            elif images:
                # Image-based extraction only
                results = self._detect_tables_opencv(images)
            
            logger.info(f"Table extraction completed: {len(results)} tables found")
            
            # Filter out invalid tables
            valid_results = [r for r in results if r.is_valid()]
            
            if len(valid_results) < len(results):
                logger.warning(f"Filtered out {len(results) - len(valid_results)} invalid tables")
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return [] 