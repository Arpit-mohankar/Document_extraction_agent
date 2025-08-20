from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime

class BoundingBox(BaseModel):
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

class FieldSource(BaseModel):
    page: int = 1
    bbox: Optional[BoundingBox] = None

class ExtractedField(BaseModel):
    name: str
    value: str = ""  # ✅ Required string with default empty string
    confidence: float = Field(ge=0, le=1, default=0.0)
    source: Optional[FieldSource] = None

class QualityAssurance(BaseModel):
    passed_rules: List[str] = []
    failed_rules: List[str] = []
    notes: str = ""

class ExtractionResult(BaseModel):
    doc_type: Literal["invoice", "medical_bill", "prescription"] = "invoice"
    fields: List[ExtractedField] = []
    overall_confidence: float = Field(ge=0, le=1, default=0.0)
    qa: QualityAssurance = Field(default_factory=QualityAssurance)

class DocumentClassification(BaseModel):
    doc_type: Literal["invoice", "medical_bill", "prescription"] = "invoice"
    confidence: float = Field(ge=0, le=1, default=0.0)
    reasoning: str = ""
