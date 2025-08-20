# extraction_chain.py
from openai import OpenAI
import json
from typing import Dict, List, Optional
from collections import Counter
from schemas import ExtractedField, FieldSource
from config import Config


class ExtractionChain:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
    def extract_fields(self, doc_type: str, ocr_result: Dict, custom_fields: Optional[List[str]] = None) -> List[ExtractedField]:
        """Extract structured fields using self-consistency"""
        
        # Multiple runs for consistency
        all_results = []
        for run in range(Config.CONSISTENCY_RUNS):
            result = self._single_extraction(doc_type, ocr_result, custom_fields)
            all_results.append(result)
        
        # Aggregate results with majority voting
        return self._aggregate_results(all_results)
    
    def _single_extraction(self, doc_type: str, ocr_result: Dict, custom_fields: Optional[List[str]] = None) -> List[Dict]:
        """Single extraction run"""
        
        # Get field definitions based on doc type
        field_definitions = self._get_field_definitions(doc_type, custom_fields)
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(doc_type, ocr_result['full_text'], field_definitions)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert document extraction system. Extract information accurately and assign confidence scores. Never return null or empty values for field values."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            extracted_fields = result.get('fields', [])
            
            # ✅ Validate and filter out invalid fields
            valid_fields = []
            for field in extracted_fields:
                # Check if field has required properties and valid values
                if (isinstance(field, dict) and 
                    field.get('name') and 
                    field.get('value') is not None and 
                    str(field.get('value', '')).strip()):
                    
                    # Ensure all required fields are present with proper types
                    valid_fields.append({
                        'name': str(field.get('name', '')).strip(),
                        'value': str(field.get('value', '')).strip(),
                        'confidence': float(field.get('confidence', 0.5))
                    })
            
            return valid_fields
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Extraction error: {e}")
            return []
    
    def _get_field_definitions(self, doc_type: str, custom_fields: Optional[List[str]] = None) -> List[str]:
        """Get field definitions for document type"""
        
        field_defs = {
            'invoice': [
                'invoice_number', 'date', 'vendor_name', 'customer_name', 
                'total_amount', 'subtotal', 'tax_amount', 'due_date'
            ],
            'medical_bill': [
                'patient_name', 'date_of_service', 'provider_name', 
                'total_amount', 'insurance_amount', 'patient_responsibility'
            ],
            'prescription': [
                'patient_name', 'doctor_name', 'medication', 'dosage', 
                'quantity', 'date_prescribed', 'pharmacy_name'
            ]
        }
        
        if custom_fields:
            return custom_fields
        
        return field_defs.get(doc_type, field_defs['invoice'])
    
    def _create_extraction_prompt(self, doc_type: str, text: str, fields: List[str]) -> str:
        """Create extraction prompt with few-shot examples"""
        
        example = self._get_few_shot_example(doc_type)
        
        prompt = f"""
Extract the following fields from this {doc_type} document:

Fields to extract: {', '.join(fields)}

Document text:
{text}

Example output format:
{example}

Extract the fields and return a JSON object with this structure:
{{
    "fields": [
        {{
            "name": "field_name",
            "value": "extracted_value",
            "confidence": 0.95
        }}
    ]
}}

IMPORTANT RULES:
1. NEVER use null, None, or empty strings for field values
2. Only extract fields that are clearly present in the text
3. If a field is not found, do NOT include it in the response
4. Assign confidence based on how clearly the value is stated
5. Use confidence 0.9+ for clearly stated values
6. Use confidence 0.7-0.9 for inferred values  
7. Use confidence <0.7 for uncertain values
8. All field values must be non-empty strings
9. Confidence must be a number between 0 and 1

Return only valid JSON with non-empty field values.
        """
        
        return prompt
    
    def _get_few_shot_example(self, doc_type: str) -> str:
        """Get few-shot example for document type"""
        
        examples = {
            'invoice': '''
{
    "fields": [
        {"name": "invoice_number", "value": "INV-2024-001", "confidence": 0.95},
        {"name": "total_amount", "value": "$1,250.00", "confidence": 0.92}
    ]
}''',
            'medical_bill': '''
{
    "fields": [
        {"name": "patient_name", "value": "John Smith", "confidence": 0.98},
        {"name": "total_amount", "value": "$450.00", "confidence": 0.90}
    ]
}''',
            'prescription': '''
{
    "fields": [
        {"name": "patient_name", "value": "Jane Doe", "confidence": 0.95},
        {"name": "medication", "value": "Lisinopril 10mg", "confidence": 0.93}
    ]
}'''
        }
        
        return examples.get(doc_type, examples['invoice'])
    
    def _aggregate_results(self, all_results: List[List[Dict]]) -> List[ExtractedField]:
        """Aggregate multiple extraction runs using majority voting"""
        
        # Collect all field values by name
        field_votes = {}
        
        for result in all_results:
            for field in result:
                # ✅ Validate field data before processing
                field_name = field.get('name')
                field_value = field.get('value')
                field_conf = field.get('confidence', 0.5)
                
                # Skip invalid fields
                if not field_name or not field_value or field_value is None:
                    continue
                
                # Convert to string and strip whitespace
                field_name = str(field_name).strip()
                field_value = str(field_value).strip()
                
                # Skip empty values
                if not field_name or not field_value:
                    continue
                
                if field_name not in field_votes:
                    field_votes[field_name] = []
                
                field_votes[field_name].append({
                    'value': field_value,
                    'confidence': float(field_conf)
                })
        
        # Apply majority voting
        final_fields = []
        
        for field_name, votes in field_votes.items():
            if not votes:  # Skip if no valid votes
                continue
                
            # Get most common value
            value_counts = Counter(vote['value'] for vote in votes)
            most_common_value = value_counts.most_common(1)[0][0]
            
            # Calculate average confidence for this value
            matching_votes = [vote for vote in votes if vote['value'] == most_common_value]
            avg_confidence = sum(vote['confidence'] for vote in matching_votes) / len(matching_votes)
            
            # Boost confidence if multiple runs agree
            consistency_boost = len(matching_votes) / len(votes)
            final_confidence = min(avg_confidence * (0.5 + 0.5 * consistency_boost), 1.0)
            
            # ✅ Only add fields with valid, non-empty values
            if most_common_value and most_common_value.strip():
                try:
                    final_fields.append(ExtractedField(
                        name=field_name,
                        value=most_common_value,
                        confidence=final_confidence
                    ))
                except Exception as e:
                    print(f"Error creating ExtractedField for {field_name}: {e}")
                    continue
        
        return final_fields
