# validation_engine.py
import re
from typing import Dict, List, Tuple

class ValidationEngine:
    def __init__(self):
        self.rules = {
            'invoice': [
                self._validate_invoice_number,
                self._validate_amount_format,
                self._validate_date_format
            ],
            'medical_bill': [
                self._validate_patient_name,
                self._validate_amount_format,
                self._validate_date_format
            ],
            'prescription': [
                self._validate_patient_name,
                self._validate_medication_format,
                self._validate_doctor_name
            ]
        }
    
    def validate_extraction(self, doc_type: str, fields: List[Dict]) -> Tuple[List[str], List[str], str]:
        """
        Validate extracted fields
        Returns: (passed_rules, failed_rules, notes)
        """
        
        passed_rules = []
        failed_rules = []
        notes = []
        
        # Convert fields list to dict for easier access
        field_dict = {field['name']: field['value'] for field in fields}
        
        # Run document-specific validation rules
        rules_to_run = self.rules.get(doc_type, [])
        
        for rule_func in rules_to_run:
            try:
                rule_name = rule_func.__name__.replace('_validate_', '')
                if rule_func(field_dict):
                    passed_rules.append(rule_name)
                else:
                    failed_rules.append(rule_name)
            except Exception as e:
                failed_rules.append(f"{rule_func.__name__}_error")
                notes.append(f"Validation error in {rule_func.__name__}: {str(e)}")
        
        # Check confidence levels
        low_confidence_fields = [f for f in fields if f.get('confidence', 1.0) < 0.6]
        if low_confidence_fields:
            notes.append(f"{len(low_confidence_fields)} low-confidence fields")
        
        return passed_rules, failed_rules, "; ".join(notes)
    
    def _validate_invoice_number(self, fields: Dict) -> bool:
        """Validate invoice number format"""
        invoice_num = fields.get('invoice_number', '')
        if not invoice_num:
            return True  # Optional field
        
        # Basic invoice number pattern
        return bool(re.match(r'^[A-Z]*[-]?\d+', invoice_num))
    
    def _validate_amount_format(self, fields: Dict) -> bool:
        """Validate amount fields have proper format"""
        amount_fields = ['total_amount', 'subtotal', 'tax_amount']
        
        for field_name in amount_fields:
            if field_name in fields:
                amount = fields[field_name]
                # Check if it looks like a currency amount
                if not re.match(r'[\$]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', amount):
                    return False
        
        return True
    
    def _validate_date_format(self, fields: Dict) -> bool:
        """Validate date fields"""
        date_fields = ['date', 'due_date', 'date_of_service', 'date_prescribed']
        
        for field_name in date_fields:
            if field_name in fields:
                date_val = fields[field_name]
                # Basic date pattern check
                if not re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}', date_val):
                    return False
        
        return True
    
    def _validate_patient_name(self, fields: Dict) -> bool:
        """Validate patient name exists and looks reasonable"""
        patient_name = fields.get('patient_name', '')
        if not patient_name:
            return False
        
        # Check if it looks like a name (letters and spaces)
        return bool(re.match(r'^[A-Za-z\s.]+$', patient_name))
    
    def _validate_medication_format(self, fields: Dict) -> bool:
        """Validate medication format"""
        medication = fields.get('medication', '')
        if not medication:
            return True  # Optional
        
        # Basic medication pattern (letters, numbers for dosage)
        return bool(re.match(r'^[A-Za-z\s]+(?:\d+mg)?', medication))
    
    def _validate_doctor_name(self, fields: Dict) -> bool:
        """Validate doctor name format"""
        doctor_name = fields.get('doctor_name', '')
        if not doctor_name:
            return True  # Optional
        
        # Check if it looks like a name
        return bool(re.match(r'^[A-Za-z\s.,]+$', doctor_name))
