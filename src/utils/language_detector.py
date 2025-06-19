"""
Language detection and text splitting utilities.
"""

import re
from typing import Tuple, List, Dict, Optional
from langdetect import detect, DetectorFactory, LangDetectException
from langdetect.lang_detect_exception import ErrorCode
import logging

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class LanguageTextSplitter:
    """
    Utility class for detecting and splitting text by language.
    Specialized for Ukrainian and English text separation.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize language detector.
        
        Args:
            confidence_threshold: Minimum confidence for language classification
        """
        self.confidence_threshold = confidence_threshold
        
        # Patterns for better language detection
        self.ukrainian_chars = re.compile(r'[а-яА-ЯіІїЇєЄґҐ]')
        self.english_chars = re.compile(r'[a-zA-Z]')
        self.punctuation = re.compile(r'[^\w\s]')
        self.numbers = re.compile(r'\d+')
        
    def detect_word_language(self, word: str) -> Optional[str]:
        """
        Detect language of a single word using character-based heuristics.
        
        Args:
            word: Single word to analyze
            
        Returns:
            'uk' for Ukrainian, 'en' for English, None for uncertain
        """
        # Clean word from punctuation and numbers
        clean_word = re.sub(r'[^\w]', '', word)
        
        if len(clean_word) < 2:
            return None
            
        # Count character types
        uk_chars = len(self.ukrainian_chars.findall(clean_word))
        en_chars = len(self.english_chars.findall(clean_word))
        
        # Determine language based on character majority
        if uk_chars > 0 and en_chars == 0:
            return 'uk'
        elif en_chars > 0 and uk_chars == 0:
            return 'en'
        else:
            # Mixed or ambiguous - try langdetect for longer words
            if len(clean_word) >= 4:
                try:
                    detected = detect(clean_word)
                    if detected in ['uk', 'en']:
                        return detected
                except LangDetectException:
                    pass
            return None
    
    def detect_text_language(self, text: str) -> str:
        """
        Detect primary language of text using langdetect.
        
        Args:
            text: Text to analyze
            
        Returns:
            'uk' for Ukrainian, 'en' for English, 'mixed' for mixed content
        """
        if not text or len(text.strip()) < 3:
            return 'mixed'
            
        try:
            detected = detect(text)
            if detected == 'uk':
                return 'uk'
            elif detected == 'en':
                return 'en'
            else:
                return 'mixed'
        except LangDetectException:
            # Fallback to character-based detection
            uk_chars = len(self.ukrainian_chars.findall(text))
            en_chars = len(self.english_chars.findall(text))
            
            if uk_chars > en_chars * 2:
                return 'uk'
            elif en_chars > uk_chars * 2:
                return 'en'
            else:
                return 'mixed'
    
    def split_text_by_language(self, text: str, preserve_order: bool = True) -> Tuple[str, str]:
        """
        Split text into Ukrainian and English versions.
        
        Args:
            text: Input text to split
            preserve_order: Whether to preserve original word order
            
        Returns:
            Tuple of (ukrainian_text, english_text)
        """
        if not text or not text.strip():
            return ("", "")
        
        # Split text into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        ukrainian_parts = []
        english_parts = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Always check if sentence contains both languages by word-level analysis
            uk_words, en_words = self._split_sentence_by_words(sentence, preserve_order)
            
            # If sentence has words in both languages, split by words
            if uk_words.strip() and en_words.strip():
                if uk_words.strip():
                    ukrainian_parts.append(uk_words.strip())
                if en_words.strip():
                    english_parts.append(en_words.strip())
            else:
                # If sentence is mostly one language, use sentence-level detection
                sentence_lang = self.detect_text_language(sentence)
                
                if sentence_lang == 'uk':
                    ukrainian_parts.append(sentence.strip())
                elif sentence_lang == 'en':
                    english_parts.append(sentence.strip())
                else:
                    # Mixed sentence - fall back to word splitting
                    if uk_words.strip():
                        ukrainian_parts.append(uk_words.strip())
                    if en_words.strip():
                        english_parts.append(en_words.strip())
        
        # Join results
        ukrainian_text = '. '.join(ukrainian_parts)
        english_text = '. '.join(english_parts)
        
        # Add final punctuation if needed
        if ukrainian_text and not ukrainian_text.endswith('.'):
            ukrainian_text += '.'
        if english_text and not english_text.endswith('.'):
            english_text += '.'
            
        return (ukrainian_text.strip(), english_text.strip())
    
    def _split_sentence_by_words(self, sentence: str, preserve_order: bool = True) -> Tuple[str, str]:
        """
        Split a sentence into Ukrainian and English words.
        
        Args:
            sentence: Sentence to split
            preserve_order: Whether to preserve original word order
            
        Returns:
            Tuple of (ukrainian_words, english_words)
        """
        words = sentence.split()
        ukrainian_words = []
        english_words = []
        
        for word in words:
            # Clean word from punctuation for language detection but keep original for output
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Skip words that are mostly punctuation or numbers
            if len(clean_word) < 2 or self.numbers.match(clean_word):
                continue
                
            word_lang = self.detect_word_language(clean_word)
            
            if word_lang == 'uk':
                ukrainian_words.append(word)
            elif word_lang == 'en':
                english_words.append(word)
            # Skip uncertain words instead of ignoring them completely
        
        return (' '.join(ukrainian_words), ' '.join(english_words))
    
    def get_language_statistics(self, text: str) -> Dict[str, any]:
        """
        Get detailed language statistics for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language statistics
        """
        if not text:
            return {
                'total_words': 0,
                'ukrainian_words': 0,
                'english_words': 0,
                'uncertain_words': 0,
                'ukrainian_percentage': 0.0,
                'english_percentage': 0.0,
                'primary_language': 'unknown'
            }
        
        words = text.split()
        uk_count = 0
        en_count = 0
        uncertain_count = 0
        
        for word in words:
            if self.punctuation.match(word) or self.numbers.match(word):
                continue
                
            word_lang = self.detect_word_language(word)
            if word_lang == 'uk':
                uk_count += 1
            elif word_lang == 'en':
                en_count += 1
            else:
                uncertain_count += 1
        
        total_lang_words = uk_count + en_count + uncertain_count
        
        if total_lang_words == 0:
            return {
                'total_words': len(words),
                'ukrainian_words': 0,
                'english_words': 0,
                'uncertain_words': 0,
                'ukrainian_percentage': 0.0,
                'english_percentage': 0.0,
                'primary_language': 'unknown'
            }
        
        uk_percentage = (uk_count / total_lang_words) * 100
        en_percentage = (en_count / total_lang_words) * 100
        
        # Determine primary language
        if uk_percentage > en_percentage * 1.5:
            primary_language = 'ukrainian'
        elif en_percentage > uk_percentage * 1.5:
            primary_language = 'english'
        else:
            primary_language = 'mixed'
        
        return {
            'total_words': len(words),
            'ukrainian_words': uk_count,
            'english_words': en_count,
            'uncertain_words': uncertain_count,
            'ukrainian_percentage': uk_percentage,
            'english_percentage': en_percentage,
            'primary_language': primary_language
        }


def split_text_by_language(text: str, preserve_order: bool = True) -> Tuple[str, str]:
    """
    Convenience function to split text by language.
    
    Args:
        text: Input text to split
        preserve_order: Whether to preserve original word order
        
    Returns:
        Tuple of (ukrainian_text, english_text)
    """
    splitter = LanguageTextSplitter()
    return splitter.split_text_by_language(text, preserve_order) 