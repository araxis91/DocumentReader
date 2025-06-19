"""
Unit tests for OCR module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

from src.ocr import UkrainianOCR, OCRResult


class TestOCRResult:
    """Test OCRResult class."""
    
    def test_init(self):
        """Test OCRResult initialization."""
        result = OCRResult("Test text", 85.5, 1)
        assert result.text == "Test text"
        assert result.confidence == 85.5
        assert result.page_num == 1
        assert result.bbox == {}
        assert result.word_data == []
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = OCRResult("Hello world", 90.0, 2)
        result_dict = result.to_dict()
        
        expected_keys = ['text', 'confidence', 'page_num', 'bbox', 'word_count', 'char_count', 'word_data']
        assert all(key in result_dict for key in expected_keys)
        assert result_dict['text'] == "Hello world"
        assert result_dict['confidence'] == 90.0
        assert result_dict['page_num'] == 2
        assert result_dict['word_count'] == 2
        assert result_dict['char_count'] == 11


class TestUkrainianOCR:
    """Test UkrainianOCR class."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor instance."""
        try:
            return UkrainianOCR()
        except RuntimeError:
            pytest.skip("Tesseract not available for testing")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple white image with black text
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Add some black text-like shapes (simplified)
        image[40:60, 50:150] = 0  # Horizontal bar
        return image
    
    def test_init(self, ocr_processor):
        """Test OCR processor initialization."""
        assert ocr_processor.confidence_threshold == 60
    
    def test_preprocess_image_for_ocr(self, ocr_processor, sample_image):
        """Test image preprocessing."""
        processed = ocr_processor.preprocess_image_for_ocr(sample_image)
        assert processed is not None
        assert len(processed.shape) == 2  # Should be grayscale
    
    def test_extract_text_with_confidence(self, ocr_processor, sample_image):
        """Test text extraction with confidence."""
        result = ocr_processor.extract_text_with_confidence(sample_image, 1)
        assert isinstance(result, OCRResult)
        assert result.page_num == 1
        assert isinstance(result.confidence, (int, float))
        assert isinstance(result.text, str)
    
    def test_extract_text_batch(self, ocr_processor):
        """Test batch text extraction."""
        images = [np.ones((50, 100, 3), dtype=np.uint8) * 255 for _ in range(3)]
        results = ocr_processor.extract_text_batch(images)
        
        assert len(results) == 3
        assert all(isinstance(r, OCRResult) for r in results)
        assert all(r.page_num == i + 1 for i, r in enumerate(results))
    
    def test_save_results(self, ocr_processor, tmp_path):
        """Test saving OCR results."""
        results = [
            OCRResult("Page 1 text", 85.0, 1),
            OCRResult("Page 2 text", 90.0, 2)
        ]
        
        ocr_processor.save_results(results, tmp_path, "test_ocr")
        
        # Check if files were created
        assert (tmp_path / "test_ocr_page_001.txt").exists()
        assert (tmp_path / "test_ocr_page_002.txt").exists()
        assert (tmp_path / "test_ocr_combined.txt").exists()
        assert (tmp_path / "test_ocr_summary.json").exists()
        
        # Check content
        with open(tmp_path / "test_ocr_page_001.txt", 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == "Page 1 text"


if __name__ == '__main__':
    pytest.main([__file__]) 