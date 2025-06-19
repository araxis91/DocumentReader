"""
PDF ingestion and image conversion module.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
import pdf2image
import numpy as np
import cv2

from .config import IMAGE_CONFIG, SUPPORTED_PDF_EXTENSIONS, DEFAULT_TEMP_DIR
from .utils.logger import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    """Handles PDF ingestion and conversion to images."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize PDF processor.
        
        Args:
            temp_dir: Temporary directory for intermediate files
        """
        self.temp_dir = temp_dir or DEFAULT_TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate input file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() not in SUPPORTED_PDF_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return file_path
    
    def get_pdf_info(self, pdf_path: Path) -> dict:
        """
        Get PDF document information.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_count': doc.page_count,
                'encrypted': doc.is_encrypted,
            }
            doc.close()
            logger.info(f"PDF info extracted: {info['page_count']} pages")
            return info
        except Exception as e:
            logger.error(f"Failed to extract PDF info: {e}")
            return {}
    
    def convert_pdf_to_images_pymupdf(self, pdf_path: Path, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF to images using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of numpy arrays representing images
        """
        images = []
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Converting {doc.page_count} pages to images using PyMuPDF")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Convert to image
                mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_data))
                
                # Convert to numpy array
                img_array = np.array(pil_img)
                images.append(img_array)
                
                logger.debug(f"Converted page {page_num + 1} to image")
                
            doc.close()
            logger.info(f"Successfully converted {len(images)} pages")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF using PyMuPDF: {e}")
            return []
    
    def convert_pdf_to_images_pdf2image(self, pdf_path: Path, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF to images using pdf2image.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of numpy arrays representing images
        """
        try:
            logger.info(f"Converting PDF to images using pdf2image at {dpi} DPI")
            
            # Convert PDF to PIL images
            pil_images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt=IMAGE_CONFIG['format']
            )
            
            # Convert PIL images to numpy arrays
            images = []
            for i, pil_img in enumerate(pil_images):
                img_array = np.array(pil_img)
                images.append(img_array)
                logger.debug(f"Converted page {i + 1} to image")
            
            logger.info(f"Successfully converted {len(images)} pages")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF using pdf2image: {e}")
            return []
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale if specified
            if IMAGE_CONFIG.get('grayscale', False):
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Noise reduction
            if IMAGE_CONFIG.get('noise_reduction', True):
                image = cv2.medianBlur(image, 3)
            
            # Enhance contrast
            if IMAGE_CONFIG.get('enhance_contrast', True):
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                if len(image.shape) == 2:  # Grayscale
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    image = clahe.apply(image)
                else:  # Color image
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def process_pdf(self, pdf_path: Union[str, Path], method: str = 'pdf2image') -> Tuple[List[np.ndarray], dict]:
        """
        Process PDF file and convert to enhanced images.
        
        Args:
            pdf_path: Path to PDF file
            method: Conversion method ('pdf2image' or 'pymupdf')
            
        Returns:
            Tuple of (list of images, pdf info)
        """
        pdf_path = self.validate_file(pdf_path)
        
        # Get PDF information
        pdf_info = self.get_pdf_info(pdf_path)
        
        # Convert to images
        if method == 'pymupdf':
            images = self.convert_pdf_to_images_pymupdf(pdf_path, IMAGE_CONFIG['dpi'])
        else:
            images = self.convert_pdf_to_images_pdf2image(pdf_path, IMAGE_CONFIG['dpi'])
        
        if not images:
            logger.error(f"Failed to convert PDF to images: {pdf_path}")
            return [], pdf_info
        
        # Enhance images
        enhanced_images = []
        for i, image in enumerate(images):
            try:
                enhanced_image = self.enhance_image(image)
                enhanced_images.append(enhanced_image)
                logger.debug(f"Enhanced image {i + 1}")
            except Exception as e:
                logger.warning(f"Failed to enhance image {i + 1}: {e}")
                enhanced_images.append(image)
        
        logger.info(f"Successfully processed PDF: {len(enhanced_images)} pages")
        return enhanced_images, pdf_info
    
    def save_images(self, images: List[np.ndarray], output_dir: Path, prefix: str = 'page') -> List[Path]:
        """
        Save images to files.
        
        Args:
            images: List of images as numpy arrays
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for i, image in enumerate(images):
            try:
                filename = f"{prefix}_{i+1:03d}.png"
                filepath = output_dir / filename
                
                # Convert numpy array to PIL Image and save
                pil_image = Image.fromarray(image)
                pil_image.save(filepath)
                
                saved_paths.append(filepath)
                logger.debug(f"Saved image: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save image {i+1}: {e}")
        
        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths


# Add missing import
import io 