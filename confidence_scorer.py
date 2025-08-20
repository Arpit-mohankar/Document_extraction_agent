import re
from typing import Dict, List
from datetime import datetime

class ConfidenceScorer:
    def __init__(self):
        self.weights = {
            'extraction_confidence': 0.4,  # From LLM extraction
            'pattern_match': 0.3,          # Regex/format validation
            'context_consistency': 0.2,     # Cross-field validation
            'ocr_confidence': 0.1           # OCR quality
        }
    
    def calculate_field_confidence(self, field_name: str, value: str, base_confidence: float, context: Dict) -> float:
        """Calculate multi-factor confidence score"""
        
        scores = {
            'extraction_confidence': base_confidence,
            'pattern_match': self._pattern_confidence(field_name, value),
            'context_consistency': self._context_confidence(field_name, value, context),
            'ocr_confidence': context.get('avg_ocr_confidence', 0.8)
        }
        
        # Weighted average
        weighted_score = sum(scores[key] * self.weights[key] for key in scores)
        
        return max(0.0, min(1.0, weighted_score))
    
    def _pattern_confidence(self, field_name: str, value: str) -> float:
        """Validate field value against expected patterns"""
        
        if not value or not value.strip():
            return 0.0
        
        patterns = {
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}',
            'amount': r'[\$]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            'phone': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'invoice_number': r'[A-Z]*[-]?\d+',
            'medication': r'[A-Za-z]+(?:\s+\d+mg)?',
            'name': r'^[A-Za-z\s.,-]+$',
            'number': r'^\d+$'
        }
        
        field_lower = field_name.lower()
        
        # Check field name for pattern keywords
        for pattern_type, regex in patterns.items():
            if pattern_type in field_lower:
                try:
                    if re.search(regex, value.strip(), re.IGNORECASE):
                        return 0.9
                    else:
                        return 0.3
                except:
                    return 0.5
        
        # Special checks for common field types
        if 'name' in field_lower:
            return self._validate_name_pattern(value)
        elif 'total' in field_lower or 'amount' in field_lower or 'price' in field_lower:
            return self._validate_amount_pattern(value)
        elif 'date' in field_lower:
            return self._validate_date_pattern(value)
        
        # Default confidence for fields without specific patterns
        return 0.7
    
    def _context_confidence(self, field_name: str, value: str, context: Dict) -> float:
        """Check consistency with other extracted fields"""
        
        if not value or not value.strip():
            return 0.0
        
        # Basic consistency checks
        field_lower = field_name.lower()
        
        if 'date' in field_lower:
            return self._validate_date_consistency(value)
        elif 'amount' in field_lower or 'total' in field_lower or 'price' in field_lower:
            return self._validate_amount_consistency(value)
        elif 'name' in field_lower:
            return self._validate_name_consistency(value)
        
        return 0.7  # Default consistency score
    
    def _validate_name_pattern(self, name_value: str) -> float:
        """Validate name format"""
        if not name_value or len(name_value.strip()) < 2:
            return 0.1
        
        # Check if it looks like a name (letters, spaces, common punctuation)
        if re.match(r'^[A-Za-z\s.,-]+$', name_value.strip()):
            # Check for reasonable length
            if 2 <= len(name_value.strip()) <= 50:
                return 0.9
            else:
                return 0.6
        else:
            return 0.3
    
    def _validate_amount_pattern(self, amount_value: str) -> float:
        """Validate amount format"""
        if not amount_value:
            return 0.0
        
        # Remove common currency symbols and whitespace
        clean_amount = re.sub(r'[^\d.,]', '', amount_value.strip())
        
        # Check if it matches amount patterns
        if re.match(r'^\d{1,3}(?:,\d{3})*(?:\.\d{2})?$', clean_amount):
            return 0.95
        elif re.match(r'^\d+\.?\d*$', clean_amount):
            return 0.8
        else:
            return 0.3
    
    def _validate_date_pattern(self, date_value: str) -> float:
        """Validate date format"""
        if not date_value:
            return 0.0
        
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, date_value, re.IGNORECASE):
                return 0.9
        
        return 0.2
    
    def _validate_date_consistency(self, date_value: str) -> float:
        """Validate date makes sense"""
        try:
            # Try to parse date
            parsed_date = self._parse_date(date_value)
            if parsed_date:
                current_year = datetime.now().year
                if 1900 <= parsed_date.year <= current_year + 5:  # Allow future dates up to 5 years
                    return 0.9
                else:
                    return 0.4
        except:
            pass
        
        return 0.3
    
    def _validate_amount_consistency(self, amount_value: str) -> float:
        """Validate amount format and range"""
        # Remove currency symbols and commas
        clean_amount = re.sub(r'[^\d.]', '', amount_value)
        
        try:
            amount = float(clean_amount)
            # Reasonable range check
            if 0 <= amount <= 1000000:  # $0 to $1M
                return 0.9
            elif amount > 1000000:  # Very large amounts
                return 0.6
            else:  # Negative amounts
                return 0.3
        except:
            return 0.2
    
    def _validate_name_consistency(self, name_value: str) -> float:
        """Validate name consistency"""
        if not name_value or len(name_value.strip()) < 2:
            return 0.2
        
        # Check for reasonable name characteristics
        words = name_value.strip().split()
        
        # Single word names are less confident
        if len(words) == 1:
            return 0.6
        # Two or more words are more likely to be full names
        elif len(words) >= 2:
            return 0.8
        else:
            return 0.5
    
    def _parse_date(self, date_string: str):
        """Parse date string with multiple formats"""
        if not date_string:
            return None
        
        formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d',
            '%m/%d/%y', '%m-%d-%y', '%d/%m/%Y',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y',
            '%d-%m-%Y', '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except:
                continue
        
        return None
    
    def calculate_overall_confidence(self, field_confidences: List[float]) -> float:
        """Calculate overall document confidence"""
        if not field_confidences:
            return 0.0
        
        # Filter out zero confidences
        valid_scores = [score for score in field_confidences if score > 0]
        
        if not valid_scores:
            return 0.0
        
        # Remove extreme outliers if we have enough scores
        if len(valid_scores) > 4:
            sorted_scores = sorted(valid_scores)
            # Remove lowest and highest 10%
            trim_count = max(1, len(sorted_scores) // 10)
            trimmed_scores = sorted_scores[trim_count:-trim_count]
        else:
            trimmed_scores = valid_scores
        
        if not trimmed_scores:
            return sum(valid_scores) / len(valid_scores)
        
        # Use harmonic mean (more conservative than arithmetic mean)
        try:
            harmonic_mean = len(trimmed_scores) / sum(1/max(score, 0.01) for score in trimmed_scores)
            return min(harmonic_mean, 1.0)
        except:
            # Fallback to arithmetic mean
            return sum(trimmed_scores) / len(trimmed_scores)
