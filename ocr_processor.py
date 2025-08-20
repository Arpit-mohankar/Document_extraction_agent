import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from typing import List, Dict
import io
import warnings
import os

# Import your cloud OCR processor instead
from cloud_ocr_processor import CloudOCRProcessor

# Suppress any warnings
warnings.filterwarnings("ignore")

class OCRProcessor:
    def __init__(self):
        """Initialize with cloud OCR services"""
        
        # Use cloud OCR instead of local processing
        self.cloud_ocr = CloudOCRProcessor()
        print("✅ Cloud OCR processor initialized")
    
    def process_document(self, file_bytes: bytes, file_type: str) -> Dict:
        """Process document with cloud OCR"""
        
        # Use cloud OCR directly
        result = self.cloud_ocr.process_document(file_bytes, file_type)
        
        if result.get('full_text', '').strip():
            print("✅ Used cloud OCR successfully")
            return result
        else:
            return {
                'full_text': 'OCR processing failed',
                'text_blocks': [],
                'error': 'No text extracted'
            }
