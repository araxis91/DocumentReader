"""
OCR module for Ukrainian text extraction.
"""

import pytesseract
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import json
import cv2

from .config import OCR_CONFIG, TESSERACT_CONFIG, get_tesseract_path, ERROR_MESSAGES
from .utils.logger import get_logger
from .utils.language_detector import LanguageTextSplitter

logger = get_logger(__name__)


class OCRResult:
    """Container for OCR results with confidence metrics and language splitting."""
    
    def __init__(self, text: str, confidence: float, page_num: int, bbox: Optional[Dict] = None, 
                 enable_language_splitting: bool = True):
        """
        Initialize OCR result.
        
        Args:
            text: Extracted text
            confidence: OCR confidence score (0-100)
            page_num: Page number (1-indexed)
            bbox: Bounding box coordinates
            enable_language_splitting: Whether to enable language splitting
        """
        self.text = text
        self.confidence = confidence
        self.page_num = page_num
        self.bbox = bbox or {}
        self.word_data = []
        
        # Language-specific text versions
        self.ukrainian_text = ""
        self.english_text = ""
        self.language_stats = {}
        
        # Initialize language splitting if text is present and enabled
        if text and enable_language_splitting:
            self._split_text_by_language()
    
    def _split_text_by_language(self):
        """Split text into Ukrainian and English versions."""
        try:
            splitter = LanguageTextSplitter()
            self.ukrainian_text, self.english_text = splitter.split_text_by_language(self.text)
            self.language_stats = splitter.get_language_statistics(self.text)
            logger.debug(f"Page {self.page_num}: Split text into UK({len(self.ukrainian_text)} chars) and EN({len(self.english_text)} chars)")
        except Exception as e:
            logger.warning(f"Language splitting failed for page {self.page_num}: {e}")
            self.ukrainian_text = ""
            self.english_text = ""
            self.language_stats = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'page_num': self.page_num,
            'bbox': self.bbox,
            'word_count': len(self.text.split()),
            'char_count': len(self.text),
            'word_data': self.word_data,
            'ukrainian_text': self.ukrainian_text,
            'english_text': self.english_text,
            'language_stats': self.language_stats
        }


class UkrainianOCR:
    """OCR processor with Ukrainian language support."""
    
    def __init__(self, enable_language_splitting: bool = True):
        """Initialize OCR processor."""
        self.setup_tesseract()
        self.confidence_threshold = OCR_CONFIG['confidence_threshold']
        self.enable_language_splitting = enable_language_splitting
        
    def setup_tesseract(self):
        """Set up Tesseract configuration."""
        try:
            # Set tesseract path
            tesseract_path = get_tesseract_path()
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Test tesseract installation and Ukrainian language support
            available_langs = pytesseract.get_languages(config='')
            logger.info(f"Available Tesseract languages: {available_langs}")
            
            if 'ukr' not in available_langs:
                logger.warning("Ukrainian language data not found in Tesseract!")
                logger.warning("Please install Ukrainian language data: apt-get install tesseract-ocr-ukr")
            
            if 'eng' not in available_langs:
                logger.warning("English language data not found in Tesseract!")
                
            logger.info("Tesseract setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup Tesseract: {e}")
            raise RuntimeError("Tesseract is not properly installed or configured")
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply threshold to create binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise with morphological operations
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_with_confidence(self, image: np.ndarray, page_num: int = 1) -> OCRResult:
        """
        Extract text from image with confidence scores.
        
        Args:
            image: Input image as numpy array
            page_num: Page number (1-indexed)
            
        Returns:
            OCRResult object
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image_for_ocr(image)
            
            # Configure Tesseract
            config = f"--oem {OCR_CONFIG['oem']} --psm {OCR_CONFIG['psm']} -l {OCR_CONFIG['language']}"
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=config)
            
            # Get detailed data with confidence scores
            data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Extract word-level data
            word_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    word_info = {
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    }
                    word_data.append(word_info)
            
            # Create result
            result = OCRResult(text.strip(), avg_confidence, page_num, enable_language_splitting=self.enable_language_splitting)
            result.word_data = word_data
            
            # Log confidence information
            if avg_confidence < self.confidence_threshold:
                logger.warning(ERROR_MESSAGES['low_confidence'].format(avg_confidence, page_num))
            else:
                logger.info(f"OCR completed for page {page_num} with {avg_confidence:.1f}% confidence")
            
            return result
            
        except Exception as e:
            logger.error(ERROR_MESSAGES['ocr_failed'].format(page_num))
            logger.error(f"OCR error details: {e}")
            return OCRResult("", 0, page_num, enable_language_splitting=self.enable_language_splitting)
    
    def extract_text_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Extract text from multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of OCRResult objects
        """
        results = []
        logger.info(f"Starting OCR processing for {len(images)} pages")
        
        for i, image in enumerate(images):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            try:
                result = self.extract_text_with_confidence(image, page_num)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                results.append(OCRResult("", 0, page_num, enable_language_splitting=self.enable_language_splitting))
        
        # Log summary
        successful_pages = len([r for r in results if r.confidence > 0])
        avg_confidence = np.mean([r.confidence for r in results if r.confidence > 0])
        
        logger.info(f"OCR completed: {successful_pages}/{len(images)} pages processed")
        if successful_pages > 0:
            logger.info(f"Average confidence: {avg_confidence:.1f}%")
        
        return results
    
    def detect_text_orientation(self, image: np.ndarray) -> Dict:
        """
        Detect text orientation in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with orientation information
        """
        try:
            # Get orientation and script detection
            osd = pytesseract.image_to_osd(image)
            
            # Parse OSD output
            orientation_info = {}
            for line in osd.split('\n'):
                if 'Orientation in degrees:' in line:
                    orientation_info['degrees'] = int(line.split(':')[1].strip())
                elif 'Rotate:' in line:
                    orientation_info['rotate'] = int(line.split(':')[1].strip())
                elif 'Orientation confidence:' in line:
                    orientation_info['confidence'] = float(line.split(':')[1].strip())
            
            return orientation_info
            
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return {'degrees': 0, 'rotate': 0, 'confidence': 0.0}
    
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Correct image orientation if needed.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Orientation-corrected image
        """
        try:
            orientation_info = self.detect_text_orientation(image)
            
            if orientation_info.get('confidence', 0) > 1.0:
                degrees = orientation_info.get('degrees', 0)
                
                if degrees != 0:
                    logger.info(f"Correcting orientation: rotating by {degrees} degrees")
                    
                    # Rotate image
                    if degrees == 90:
                        corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif degrees == 180:
                        corrected = cv2.rotate(image, cv2.ROTATE_180)
                    elif degrees == 270:
                        corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        # For other angles, use general rotation
                        height, width = image.shape[:2]
                        center = (width // 2, height // 2)
                        matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)
                        corrected = cv2.warpAffine(image, matrix, (width, height))
                    
                    return corrected
            
            return image
            
        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")
            return image
    
    def save_results(self, results: List[OCRResult], output_dir: Path, filename_prefix: str = 'ocr_result',
                   enable_markdown: bool = True):
        """
        Save OCR results to files with language separation and markdown output.
        
        Args:
            results: List of OCR results
            output_dir: Output directory
            filename_prefix: Prefix for output files
            enable_markdown: Whether to generate markdown output
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect Ukrainian and English text separately
        ukrainian_pages = []
        english_pages = []
        combined_ukrainian_text = []
        combined_english_text = []
        
        # Save individual page results and collect combined text
        for result in results:
            if result.text:
                # Save original text file
                text_file = output_dir / f"{filename_prefix}_page_{result.page_num:03d}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result.text)
                
                # Save Ukrainian version if present
                if result.ukrainian_text:
                    uk_file = output_dir / f"{filename_prefix}_page_{result.page_num:03d}_uk.txt"
                    with open(uk_file, 'w', encoding='utf-8') as f:
                        f.write(result.ukrainian_text)
                    ukrainian_pages.append(result.page_num)
                    combined_ukrainian_text.append(f"--- Page {result.page_num} ---\n\n{result.ukrainian_text}")
                
                # Save English version if present
                if result.english_text:
                    en_file = output_dir / f"{filename_prefix}_page_{result.page_num:03d}_en.txt"
                    with open(en_file, 'w', encoding='utf-8') as f:
                        f.write(result.english_text)
                    english_pages.append(result.page_num)
                    combined_english_text.append(f"--- Page {result.page_num} ---\n\n{result.english_text}")
                
                # Save as JSON with metadata including language data
                json_file = output_dir / f"{filename_prefix}_page_{result.page_num:03d}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Saved OCR results for page {result.page_num}")
        
        # Save combined results - Original
        combined_text = "\n\n--- Page {} ---\n\n".join([
            result.text for result in results if result.text
        ])
        
        if combined_text:
            combined_file = output_dir / f"{filename_prefix}_combined.txt"
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
        
        # Save combined Ukrainian results
        if combined_ukrainian_text:
            uk_combined_file = output_dir / f"{filename_prefix}_combined_uk.txt"
            with open(uk_combined_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(combined_ukrainian_text))
            logger.info(f"Saved Ukrainian text from {len(ukrainian_pages)} pages")
        
        # Save combined English results
        if combined_english_text:
            en_combined_file = output_dir / f"{filename_prefix}_combined_en.txt"
            with open(en_combined_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(combined_english_text))
            logger.info(f"Saved English text from {len(english_pages)} pages")
        
        # Create language-specific JSON files
        self._save_language_specific_json(results, output_dir, filename_prefix, 'uk')
        self._save_language_specific_json(results, output_dir, filename_prefix, 'en')
        
        # Generate markdown output if enabled
        if enable_markdown:
            self._save_markdown_results(results, output_dir, filename_prefix)
        
        # Save summary report with language statistics
        total_uk_chars = sum([len(r.ukrainian_text) for r in results])
        total_en_chars = sum([len(r.english_text) for r in results])
        total_uk_words = sum([len(r.ukrainian_text.split()) for r in results if r.ukrainian_text])
        total_en_words = sum([len(r.english_text.split()) for r in results if r.english_text])
        
        summary = {
            'total_pages': len(results),
            'successful_pages': len([r for r in results if r.confidence > 0]),
            'average_confidence': np.mean([r.confidence for r in results if r.confidence > 0]) if results else 0,
            'low_confidence_pages': [r.page_num for r in results if 0 < r.confidence < self.confidence_threshold],
            'failed_pages': [r.page_num for r in results if r.confidence == 0],
            'total_characters': sum([len(r.text) for r in results]),
            'total_words': sum([len(r.text.split()) for r in results]),
            'language_breakdown': {
                'ukrainian_pages': ukrainian_pages,
                'english_pages': english_pages,
                'ukrainian_characters': total_uk_chars,
                'english_characters': total_en_chars,
                'ukrainian_words': total_uk_words,
                'english_words': total_en_words,
                'language_stats_per_page': [r.language_stats for r in results if r.language_stats]
            }
        }
        
        summary_file = output_dir / f"{filename_prefix}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved OCR results to {output_dir}")
        logger.info(f"Summary: {summary['successful_pages']}/{summary['total_pages']} pages processed successfully")
        logger.info(f"Language breakdown: {len(ukrainian_pages)} Ukrainian pages, {len(english_pages)} English pages")
    
    def _save_language_specific_json(self, results: List[OCRResult], output_dir: Path, 
                                   filename_prefix: str, language: str):
        """
        Save language-specific JSON file with only text from specified language.
        
        Args:
            results: List of OCR results
            output_dir: Output directory
            filename_prefix: Prefix for output files
            language: Language code ('uk' or 'en')
        """
        language_results = []
        
        for result in results:
            if language == 'uk' and result.ukrainian_text:
                lang_result = {
                    'page_num': result.page_num,
                    'text': result.ukrainian_text,
                    'confidence': result.confidence,
                    'word_count': len(result.ukrainian_text.split()),
                    'char_count': len(result.ukrainian_text),
                    'language_stats': result.language_stats
                }
                language_results.append(lang_result)
            elif language == 'en' and result.english_text:
                lang_result = {
                    'page_num': result.page_num,
                    'text': result.english_text,
                    'confidence': result.confidence,
                    'word_count': len(result.english_text.split()),
                    'char_count': len(result.english_text),
                    'language_stats': result.language_stats
                }
                language_results.append(lang_result)
        
        if language_results:
            lang_file = output_dir / f"{filename_prefix}_combined_{language}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(language_results, f, ensure_ascii=False, indent=2)
    
    def _save_markdown_results(self, results: List[OCRResult], output_dir: Path, filename_prefix: str):
        """
        Save OCR results in markdown format.
        
        Args:
            results: List of OCR results
            output_dir: Output directory
            filename_prefix: Prefix for output files
        """
        try:
            from src.utils.markdown_converter import create_markdown_converter
            
            converter = create_markdown_converter()
            
            # Generate comprehensive markdown document
            document_title = f"OCR Analysis Results - {output_dir.name}"
            markdown_content = converter.create_document_markdown(
                results, [], document_title, include_metadata=True
            )
            
            if markdown_content:
                markdown_file = output_dir / f"{filename_prefix}_combined.md"
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Saved combined markdown: {markdown_file}")
            
            # Generate language-specific markdown documents
            if any(hasattr(r, 'ukrainian_text') and r.ukrainian_text for r in results):
                uk_markdown = converter.create_language_specific_markdown(
                    results, [], 'uk', document_title
                )
                if uk_markdown:
                    uk_markdown_file = output_dir / f"{filename_prefix}_combined_uk.md"
                    with open(uk_markdown_file, 'w', encoding='utf-8') as f:
                        f.write(uk_markdown)
                    logger.info(f"Saved Ukrainian markdown: {uk_markdown_file}")
            
            if any(hasattr(r, 'english_text') and r.english_text for r in results):
                en_markdown = converter.create_language_specific_markdown(
                    results, [], 'en', document_title
                )
                if en_markdown:
                    en_markdown_file = output_dir / f"{filename_prefix}_combined_en.md"
                    with open(en_markdown_file, 'w', encoding='utf-8') as f:
                        f.write(en_markdown)
                    logger.info(f"Saved English markdown: {en_markdown_file}")
            
            # Generate individual page markdown files
            for result in results:
                if result.text:
                    page_markdown = converter.convert_text_to_markdown(
                        result.text, f"Page {result.page_num} Analysis", result.page_num
                    )
                    if page_markdown:
                        page_md_file = output_dir / f"{filename_prefix}_page_{result.page_num:03d}.md"
                        with open(page_md_file, 'w', encoding='utf-8') as f:
                            f.write(page_markdown)
                        logger.debug(f"Saved page {result.page_num} markdown: {page_md_file}")
            
        except ImportError:
            logger.warning("Markdown converter not available. Install markitdown: pip install markitdown")
        except Exception as e:
            logger.error(f"Failed to generate markdown output: {e}") 