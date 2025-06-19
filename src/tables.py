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

logger = get_logger(__name__)


class TableResult:
    """Container for table extraction results."""
    
    def __init__(self, data: pd.DataFrame, page_num: int, table_num: int, confidence: float = 0.0):
        """
        Initialize table result.
        
        Args:
            data: Extracted table data as DataFrame
            page_num: Page number (1-indexed)
            table_num: Table number on page (1-indexed)
            confidence: Extraction confidence score
        """
        self.data = data
        self.page_num = page_num
        self.table_num = table_num
        self.confidence = confidence
        self.bbox = {}
        self.metadata = {}
    
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
            'metadata': self.metadata
        }
    
    def is_valid(self) -> bool:
        """Check if table contains valid data."""
        return not self.data.empty and self.data.shape[0] > 0 and self.data.shape[1] > 0


class TableExtractor:
    """Table detection and extraction processor."""
    
    def __init__(self):
        """Initialize table extractor."""
        self.method = TABLE_CONFIG['detection_method']
        self.camelot_flavor = TABLE_CONFIG['camelot_flavor']
        
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
                        confidence=table.accuracy if hasattr(table, 'accuracy') else 0.0
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
                        confidence=0.0  # Tabula doesn't provide confidence scores
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
                                    confidence=0.5  # Default confidence for image-based detection
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
    
    def save_tables(self, tables: List[TableResult], output_dir: Path, filename_prefix: str = 'table'):
        """
        Save extracted tables to files.
        
        Args:
            tables: List of table results
            output_dir: Output directory
            filename_prefix: Prefix for output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for table in tables:
            if not table.is_valid():
                continue
            
            base_name = f"{filename_prefix}_page_{table.page_num:03d}_table_{table.table_num:03d}"
            
            # Save as CSV
            if 'csv' in OUTPUT_CONFIG['table_formats']:
                csv_file = output_dir / f"{base_name}.csv"
                table.data.to_csv(csv_file, index=False, encoding='utf-8')
                saved_files.append(csv_file)
                logger.debug(f"Saved table as CSV: {csv_file}")
            
            # Save as JSON
            if 'json' in OUTPUT_CONFIG['table_formats']:
                json_file = output_dir / f"{base_name}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(table.to_dict(), f, ensure_ascii=False, indent=2)
                saved_files.append(json_file)
                logger.debug(f"Saved table as JSON: {json_file}")
            
            # Save as Excel
            if 'xlsx' in OUTPUT_CONFIG['table_formats']:
                xlsx_file = output_dir / f"{base_name}.xlsx"
                table.data.to_excel(xlsx_file, index=False, engine='openpyxl')
                saved_files.append(xlsx_file)
                logger.debug(f"Saved table as Excel: {xlsx_file}")
        
        # Save summary
        summary = {
            'total_tables': len(tables),
            'valid_tables': len([t for t in tables if t.is_valid()]),
            'tables_by_page': {},
            'extraction_methods': list(set([t.metadata.get('extraction_method', 'unknown') for t in tables])),
            'saved_files': [str(f) for f in saved_files]
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