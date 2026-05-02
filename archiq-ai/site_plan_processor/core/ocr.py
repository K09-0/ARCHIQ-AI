"""OCR processing for site plans."""

import pytesseract
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import re

from ..models.objects import OCRResult


class OCRProcessor:
    """Extract text from site plans using Tesseract OCR."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OCR processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.language = config.get('ocr_lang', 'eng')
    
    def extract(self, image: np.ndarray) -> OCRResult:
        """Extract text and dimensions from image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            OCRResult with extracted text and dimensions
        """
        # Preprocess for OCR
        processed = self._preprocess_for_ocr(image)
        
        # Extract full text
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(
            processed, 
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        text = pytesseract.image_to_string(
            processed, 
            config=custom_config
        )
        
        # Extract dimensions
        dimensions = self._extract_dimensions(text, data)
        
        # Extract labels with positions
        labels = self._extract_labels(data)
        
        # Calculate confidence
        confidences = [c for c in data.get('conf', []) if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_confidence = max(0, min(100, avg_confidence)) / 100  # Normalize to 0-1
        
        return OCRResult(
            text=text.strip(),
            dimensions=dimensions,
            labels=labels,
            confidence=avg_confidence
        )
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Threshold
        _, binary = cv2.threshold(
            denoised, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Dilate to connect broken text
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        return dilated
    
    def _extract_dimensions(self, text: str, 
                           data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract dimension values from text.
        
        Args:
            text: Full OCR text
            data: Tesseract output data
            
        Returns:
            List of extracted dimensions
        """
        dimensions = []
        
        # Patterns for dimension extraction
        patterns = [
            r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)',  # Width x Height
            r'(\d+(?:\.\d+)?)\s*[×*xX]\s*(\d+(?:\.\d+)?)',  # Various multiplication symbols
            r'(\d+(?:\.\d+)?)\s*[\']\s*(\d+(?:\.\d+)?)\s*["\']',  # Feet and inches
            r'(\d+(?:\.\d+)?)\s*[mM]\b',  # Meters
            r'(\d+(?:\.\d+)?)\s*[fF]t\b',  # Feet
            r'(\d+(?:\.\d+)?)\s*[iI]n\b',  # Inches
            r'(\d+(?:\.\d+)?)\s*[cC]m\b',  # Centimeters
            r'(\d+(?:\.\d+)?)\s*[kK]m\b',  # Kilometers
        ]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    dimensions.append({
                        'text': match.group(0),
                        'value1': float(match.group(1)),
                        'value2': float(match.group(2)) if match.lastindex >= 2 else None,
                        'unit': self._detect_unit(match.group(0)),
                        'line': i,
                        'position': match.start()
                    })
        
        return dimensions
    
    def _detect_unit(self, text: str) -> str:
        """Detect unit from dimension text.
        
        Args:
            text: Dimension text
            
        Returns:
            Unit string
        """
        text_lower = text.lower()
        
        if 'm\'' in text_lower or 'm\"' in text_lower or re.search(r'\d+m\b', text_lower):
            return 'meters'
        elif 'ft' in text_lower or 'f\'' in text_lower or 'f"' in text_lower:
            return 'feet'
        elif 'in' in text_lower or '"' in text_lower:
            return 'inches'
        elif 'cm' in text_lower:
            return 'centimeters'
        elif 'km' in text_lower:
            return 'kilometers'
        elif '×' in text or 'x' in text_lower or '*' in text:
            return 'dimension'
        
        return 'unknown'
    
    def _extract_labels(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text labels with position information.
        
        Args:
            data: Tesseract output data
            
        Returns:
            List of labels with positions
        """
        labels = []
        
        n_boxes = len(data.get('text', []))
        
        for i in range(n_boxes):
            text = data.get('text', [])[i].strip()
            confidence = int(data.get('conf', [])[i])
            
            if not text or confidence <= 0:
                continue
            
            x = data.get('left', [])[i]
            y = data.get('top', [])[i]
            w = data.get('width', [])[i]
            h = data.get('height', [])[i]
            
            # Skip very small text (likely noise)
            if w < 5 or h < 5:
                continue
            
            labels.append({
                'text': text,
                'confidence': confidence / 100,  # Normalize to 0-1
                'bbox': (x, y, x + w, y + h),
                'center': (x + w // 2, y + h // 2)
            })
        
        return labels
    
    def extract_scale_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract scale information from text.
        
        Args:
            text: OCR text
            
        Returns:
            Scale information or None
        """
        # Scale patterns
        patterns = [
            r'1\s*[:：]\s*(\d+(?:\.\d+)?)',  # 1:500
            r'Scale\s*[:：]\s*1\s*[:：]\s*(\d+(?:\.\d+)?)',
            r'比例\s*[:：]\s*1\s*[:：]\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'ratio': f"1:{match.group(1)}",
                    'value': float(match.group(1))
                }
        
        return None
    
    def extract_coordinates(self, text: str) -> List[Dict[str, Any]]:
        """Extract coordinate information from text.
        
        Args:
            text: OCR text
            
        Returns:
            List of coordinate extractions
        """
        coordinates = []
        
        # Coordinate patterns
        patterns = [
            r'[XY]\s*[:：]\s*(\d+(?:\.\d+)?)\s*,?\s*[XY]\s*[:：]\s*(\d+(?:\.\d+)?)',
            r'\((\d+(?:\.\d+)?)[,，]\s*(\d+(?:\.\d+)?)\)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                coordinates.append({
                    'text': match.group(0),
                    'x': float(match.group(1)),
                    'y': float(match.group(2))
                })
        
        return coordinates