# document_classifier.py
from openai import OpenAI
import base64
from PIL import Image
from schemas import DocumentClassification
from config import Config
from utils import image_to_base64

class DocumentClassifier:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
    def classify(self, image: Image.Image) -> DocumentClassification:
        """Classify document type from image"""
        
        # Convert image to base64
        image_base64 = image_to_base64(image)
        
        prompt = """
        Analyze this document image and classify it as one of these types:
        - invoice: Business invoices, receipts, bills for services/products
        - medical_bill: Hospital bills, medical invoices, healthcare statements  
        - prescription: Medical prescriptions, pharmacy receipts
        
        Look at headers, layout, terminology, and visual cues.
        
        Respond with JSON in this exact format:
        {
            "doc_type": "invoice|medical_bill|prescription",
            "confidence": 0.95,
            "reasoning": "Contains invoice number, vendor details, and line items typical of business invoices"
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model=Config.CLASSIFICATION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return DocumentClassification(**result)
            
        except Exception as e:
            # Fallback classification
            return DocumentClassification(
                doc_type="invoice",
                confidence=0.5,
                reasoning=f"Classification failed: {str(e)}, defaulting to invoice"
            )
