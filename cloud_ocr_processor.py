import requests
import base64
import io
import json
from PIL import Image
from pdf2image import convert_from_bytes
from typing import Dict, List, Optional
from utils import create_bbox

class CloudOCRProcessor:
    def __init__(self, ocr_space_api_key: Optional[str] = None):
        # OCR.Space offers 25,000 free requests/month
        self.ocr_space_key = ocr_space_api_key or "helloworld"  # Free demo key
        self.ocr_space_url = "https://api.ocr.space/parse/image"
        
    def process_document(self, file_bytes: bytes, file_type: str) -> Dict:
        """Process document using OCR.Space cloud API"""
        
        try:
            if file_type.startswith('image') or file_type in ['image/png', 'image/jpeg', 'image/jpg']:
                return self._process_image(file_bytes)
            elif file_type == 'application/pdf' or file_type == 'pdf':
                return self._process_pdf(file_bytes)
            else:
                # Try to process as image
                return self._process_image(file_bytes)
                
        except Exception as e:
            return {
                'full_text': f'Cloud OCR processing failed: {str(e)}',
                'text_blocks': [],
                'error': str(e)
            }
    
    def _process_image(self, image_bytes: bytes) -> Dict:
        """Process single image using OCR.Space API"""
        
        # Convert bytes to base64
        image_b64 = base64.b64encode(image_bytes).decode()
        
        payload = {
            'apikey': self.ocr_space_key,
            'language': 'eng',
            'isOverlayRequired': True,  # Get bounding boxes
            'base64Image': f"data:image/png;base64,{image_b64}",
            'scale': True,  # Improve quality for small text
            'OCREngine': 2,  # Use the better OCR engine
            'isTable': True  # Better for structured documents
        }
        
        try:
            response = requests.post(
                self.ocr_space_url, 
                data=payload,
                timeout=30
            )
            
            result = response.json()
            
            # Check for errors
            if result.get('IsErroredOnProcessing'):
                error_messages = result.get('ErrorMessage', ['Unknown error'])
                return {
                    'full_text': '',
                    'text_blocks': [],
                    'error': '; '.join(error_messages)
                }
            
            # Extract results
            parsed_result = result['ParsedResults'][0]
            full_text = parsed_result['ParsedText']
            
            # Process text overlay for bounding boxes
            text_blocks = self._extract_text_blocks(parsed_result)
            
            return {
                'full_text': full_text,
                'text_blocks': text_blocks,
                'page': 1,
                'ocr_engine': 'ocr.space',
                'confidence': self._calculate_overall_confidence(text_blocks)
            }
            
        except requests.exceptions.Timeout:
            return {
                'full_text': 'OCR request timed out. Please try again.',
                'text_blocks': [],
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'full_text': f'OCR.Space API error: {str(e)}',
                'text_blocks': [],
                'error': str(e)
            }
    
    def _process_pdf(self, pdf_bytes: bytes) -> Dict:
        """Process PDF by converting to images first"""
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            results = []
            
            for i, image in enumerate(images):
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                image_bytes = img_buffer.getvalue()
                
                # Process each page
                page_result = self._process_image(image_bytes)
                page_result['page'] = i + 1
                results.append(page_result)
            
            return self._combine_pages(results)
            
        except Exception as e:
            return {
                'full_text': f'PDF processing failed: {str(e)}',
                'text_blocks': [],
                'error': str(e)
            }
    
    def _extract_text_blocks(self, parsed_result: Dict) -> List[Dict]:
        """Extract text blocks with bounding boxes from OCR.Space result"""
        
        text_blocks = []
        
        if 'TextOverlay' not in parsed_result:
            return text_blocks
        
        overlay = parsed_result['TextOverlay']
        
        if 'Lines' not in overlay:
            return text_blocks
        
        for line in overlay['Lines']:
            for word in line.get('Words', []):
                # Create text block with bounding box
                bbox = create_bbox(
                    word['Left'],
                    word['Top'], 
                    word['Width'],
                    word['Height']
                )
                
                text_blocks.append({
                    'text': word['WordText'],
                    'confidence': 0.9,  # OCR.Space doesn't provide word-level confidence
                    'bbox': bbox,
                    'page': 1
                })
        
        return text_blocks
    
    def _calculate_overall_confidence(self, text_blocks: List[Dict]) -> float:
        """Calculate overall confidence from text blocks"""
        if not text_blocks:
            return 0.0
        
        # OCR.Space generally has good confidence, so we estimate based on text quality
        total_chars = sum(len(block['text']) for block in text_blocks)
        
        if total_chars > 100:
            return 0.9  # High confidence for substantial text
        elif total_chars > 20:
            return 0.8  # Good confidence
        else:
            return 0.7  # Moderate confidence for short text
    
    def _combine_pages(self, page_results: List[Dict]) -> Dict:
        """Combine results from multiple pages"""
        
        full_text = []
        all_blocks = []
        errors = []
        
        for result in page_results:
            if result.get('error'):
                errors.append(f"Page {result.get('page', '?')}: {result['error']}")
            
            if result.get('full_text'):
                full_text.append(result['full_text'])
            
            all_blocks.extend(result.get('text_blocks', []))
        
        combined_text = '\n'.join(full_text)
        
        return {
            'full_text': combined_text,
            'text_blocks': all_blocks,
            'pages': len(page_results),
            'ocr_engine': 'ocr.space',
            'confidence': self._calculate_overall_confidence(all_blocks),
            'errors': errors if errors else None
        }

# Alternative: Google Cloud Vision implementation
class GoogleVisionOCRProcessor:
    def __init__(self, credentials_path: str):
        """
        Initialize Google Cloud Vision OCR
        Requires: pip install google-cloud-vision
        """
        try:
            from google.cloud import vision
            from google.oauth2 import service_account
            
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
            self.available = True
        except Exception as e:
            print(f"Google Vision OCR not available: {e}")
            self.available = False
    
    def process_document(self, file_bytes: bytes, file_type: str) -> Dict:
        """Process using Google Cloud Vision"""
        
        if not self.available:
            return {
                'full_text': 'Google Vision OCR not configured',
                'text_blocks': [],
                'error': 'Service unavailable'
            }
        
        try:
            from google.cloud import vision
            
            image = vision.Image(content=file_bytes)
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                return {
                    'full_text': '',
                    'text_blocks': [],
                    'error': response.error.message
                }
            
            texts = response.text_annotations
            
            if not texts:
                return {
                    'full_text': '',
                    'text_blocks': [],
                    'error': 'No text detected'
                }
            
            # First annotation contains the full text
            full_text = texts[0].description
            
            # Process individual text blocks
            text_blocks = []
            for text in texts[1:]:  # Skip first which is full text
                vertices = text.bounding_poly.vertices
                
                # Get bounding box coordinates
                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                
                bbox = create_bbox(
                    min(xs), min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys)
                )
                
                text_blocks.append({
                    'text': text.description,
                    'confidence': 0.95,  # Google Vision typically has high confidence
                    'bbox': bbox,
                    'page': 1
                })
            
            return {
                'full_text': full_text,
                'text_blocks': text_blocks,
                'page': 1,
                'ocr_engine': 'google_vision',
                'confidence': 0.95
            }
            
        except Exception as e:
            return {
                'full_text': f'Google Vision error: {str(e)}',
                'text_blocks': [],
                'error': str(e)
            }
