from cloud_ocr_processor import CloudOCRProcessor
from typing import Dict
import warnings
import os

# Suppress any warnings
warnings.filterwarnings("ignore")

class OCRProcessor:
    def __init__(self):
        """Initialize with cloud OCR services"""
        
        # Primary: OCR.Space (25,000 free requests/month)
        self.primary_ocr = CloudOCRProcessor()
        
        # Backup: Try to initialize EasyOCR as fallback
        self.fallback_ocr = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import easyocr
                self.fallback_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        except:
            print("EasyOCR not available as fallback")
        
        print("✅ Cloud OCR processor initialized (OCR.Space + EasyOCR fallback)")
    
    def process_document(self, file_bytes: bytes, file_type: str) -> Dict:
        """Process document with cloud OCR (primary) and EasyOCR (fallback)"""
        
        # Try cloud OCR first
        result = self.primary_ocr.process_document(file_bytes, file_type)
        
        # Check if cloud OCR was successful
        if result.get('full_text', '').strip() and not result.get('error'):
            print("✅ Used cloud OCR (OCR.Space)")
            return result
        
        # Fallback to EasyOCR if available
        if self.fallback_ocr:
            try:
                print("🔄 Falling back to EasyOCR...")
                return self._process_with_easyocr(file_bytes, file_type)
            except Exception as e:
                print(f"EasyOCR fallback failed: {e}")
        
        # If both fail, return the cloud OCR result with error info
        return result
    
    def _process_with_easyocr(self, file_bytes: bytes, file_type: str) -> Dict:
        """Fallback processing with EasyOCR"""
        
        from PIL import Image
        import io
        import numpy as np
        from utils import create_bbox
        
        try:
            # Process image
            image_stream = io.BytesIO(file_bytes)
            image = Image.open(image_stream)
            
            # EasyOCR processing
            img_array = np.array(image)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.fallback_ocr.readtext(img_array)
            
            # Convert EasyOCR results
            text_blocks = []
            full_text = []
            
            for (bbox, text, confidence) in results:
                if text.strip() and confidence > 0.3:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[12] for point in bbox]
                    
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                    
                    bbox_obj = create_bbox(x1, y1, x2-x1, y2-y1)
                    
                    text_blocks.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox_obj,
                        'page': 1
                    })
                    full_text.append(text.strip())
            
            return {
                'full_text': ' '.join(full_text),
                'text_blocks': text_blocks,
                'page': 1,
                'ocr_engine': 'easyocr_fallback'
            }
            
        except Exception as e:
            return {
                'full_text': f'EasyOCR processing failed: {str(e)}',
                'text_blocks': [],
                'error': str(e)
            }
