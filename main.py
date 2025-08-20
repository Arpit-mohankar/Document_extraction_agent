import streamlit as st
import plotly.express as px
import json
from PIL import Image
import io
import time

# Import all modules
from document_classifier import DocumentClassifier
from ocr_processor import OCRProcessor
from extraction_chain import ExtractionChain
from validation_engine import ValidationEngine
from confidence_scorer import ConfidenceScorer
from schemas import ExtractionResult, QualityAssurance
from config import Config
import warnings
import os

# Suppress the specific PyTorch warning
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but no accelerator is found.*")

st.set_page_config(
    page_title="Agentic Document Extraction",
    page_icon="📄",
    layout="wide"
)

class DocumentExtractionApp:
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.ocr_processor = OCRProcessor()
        self.extraction_chain = ExtractionChain()
        self.validator = ValidationEngine()
        self.confidence_scorer = ConfidenceScorer()
    
    def run(self):
        st.title("🤖 Agentic Document Extraction System")
        st.markdown("*AI-powered document processing with confidence scoring*")
        
        # Compact sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            show_debug = st.checkbox("Show Debug Info", False)
            
            st.subheader("Processing Options")
            enable_classification = st.checkbox("Enable AI Classification", True)
            enable_validation = st.checkbox("Enable Validation Rules", True)
            
            st.subheader("Custom Fields")
            custom_fields_text = st.text_area(
                "Specify fields to extract (one per line):",
                placeholder="patient_name\ndoctor_name\nmedication",
                height=120,
                help="Leave empty for automatic field detection"
            )
        
        # Main interface - single column layout
        st.header("📤 Upload Document")
        
        # Compact file uploader
        uploaded_file = st.file_uploader(
            "Choose a document (PDF, PNG, JPG, JPEG):",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload invoices, medical bills, or prescriptions"
        )
        
        if uploaded_file:
            # Compact file info in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File", uploaded_file.name)
            with col2:
                st.metric("Size", f"{uploaded_file.size:,} bytes")
            with col3:
                st.metric("Type", uploaded_file.type)
            
            # Compact preview
            try:
                if uploaded_file.type.startswith('image'):
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    
                    # Show preview in expander to save space
                    with st.expander("🔍 Preview Document", expanded=False):
                        st.image(image, caption="Document Preview", use_container_width=True)
                    
                    uploaded_file.seek(0)
                else:
                    st.info("📄 PDF uploaded successfully")
            except Exception as e:
                st.warning(f"Preview unavailable: {str(e)}")
            
            # Process button - full width
            custom_fields = [f.strip() for f in custom_fields_text.split('\n') if f.strip()] if custom_fields_text else None
            
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                self.process_document(uploaded_file, custom_fields, show_debug, enable_classification, enable_validation)
        
        else:
            # Compact instructions when no file uploaded
            st.info("""
            **Quick Start:**
            1. 📁 Upload a document above
            2. ⚙️ Adjust settings in sidebar (optional)
            3. 🚀 Click "Process Document"
            
            **Supported:** Invoices, Medical Bills, Prescriptions (PDF/Image)
            """)
    
    def process_document(self, uploaded_file, custom_fields, show_debug, enable_classification, enable_validation):
        """Process uploaded document through the full pipeline"""
        
        # Clear any previous results
        if 'results' in st.session_state:
            del st.session_state.results
        
        with st.spinner("🔄 Processing document..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Read file bytes properly
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                
                if not file_bytes:
                    st.error("❌ Could not read file. Please try uploading again.")
                    return
                
                # Step 1: Classification
                status_text.text("🔍 Classifying document type...")
                progress_bar.progress(20)
                
                doc_type = "invoice"  # Default
                classification_confidence = 0.5
                
                if enable_classification and uploaded_file.type.startswith('image'):
                    try:
                        image_stream = io.BytesIO(file_bytes)
                        image = Image.open(image_stream)
                        classification = self.classifier.classify(image)
                        doc_type = classification.doc_type
                        classification_confidence = classification.confidence
                        
                        if show_debug:
                            st.info(f"🔍 Classification: {doc_type} (confidence: {classification_confidence:.2f})")
                            
                    except Exception as e:
                        st.warning(f"Classification failed: {str(e)}, using default type")
                
                # Step 2: OCR Processing
                status_text.text("📝 Extracting text with OCR...")
                progress_bar.progress(40)
                
                ocr_result = self.ocr_processor.process_document(file_bytes, uploaded_file.type)
                
                if not ocr_result.get('full_text', '').strip():
                    st.warning("⚠️ No text could be extracted from the document. Please check if the image is clear and contains readable text.")
                    return
                
                if show_debug:
                    st.info(f"📝 OCR extracted {len(ocr_result['full_text'])} characters")
                
                # Step 3: Document Classification from text (if not done above)
                if not enable_classification or not uploaded_file.type.startswith('image'):
                    doc_type = self._classify_from_text(ocr_result['full_text'])
                    if show_debug:
                        st.info(f"🔍 Text-based classification: {doc_type}")
                
                # Step 4: Field Extraction
                status_text.text("🎯 Extracting structured fields...")
                progress_bar.progress(60)
                
                extracted_fields = self.extraction_chain.extract_fields(
                    doc_type, ocr_result, custom_fields
                )
                
                if not extracted_fields:
                    st.warning("⚠️ No fields could be extracted. The document might not contain clear structured information.")
                
                if show_debug:
                    st.info(f"🎯 Extracted {len(extracted_fields)} fields")
                
                # Step 5: Confidence Scoring
                status_text.text("📊 Calculating confidence scores...")
                progress_bar.progress(80)
                
                for field in extracted_fields:
                    field.confidence = self.confidence_scorer.calculate_field_confidence(
                        field.name, 
                        field.value, 
                        field.confidence,
                        {
                            'avg_ocr_confidence': sum(block.get('confidence', 0.8) for block in ocr_result.get('text_blocks', [])) / max(len(ocr_result.get('text_blocks', [])), 1),
                            'classification_confidence': classification_confidence
                        }
                    )
                
                # Step 6: Validation
                status_text.text("✅ Running validation...")
                progress_bar.progress(90)
                
                passed_rules = []
                failed_rules = []
                notes = ""
                
                if enable_validation and extracted_fields:
                    field_dicts = [{'name': f.name, 'value': f.value, 'confidence': f.confidence} for f in extracted_fields]
                    passed_rules, failed_rules, notes = self.validator.validate_extraction(doc_type, field_dicts)
                
                # Calculate overall confidence
                if extracted_fields:
                    field_confidences = [f.confidence for f in extracted_fields]
                    overall_confidence = self.confidence_scorer.calculate_overall_confidence(field_confidences)
                else:
                    overall_confidence = 0.0
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Create result
                result = ExtractionResult(
                    doc_type=doc_type,
                    fields=extracted_fields,
                    overall_confidence=overall_confidence,
                    qa=QualityAssurance(
                        passed_rules=passed_rules,
                        failed_rules=failed_rules,
                        notes=notes
                    )
                )
                
                # Display results
                self.display_results(result, show_debug, ocr_result if show_debug else None)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                if show_debug:
                    st.exception(e)
    
    def _classify_from_text(self, text: str) -> str:
        """Simple keyword-based classification for documents"""
        text_lower = text.lower()
        
        # Medical bill keywords
        medical_keywords = ['patient', 'medical', 'hospital', 'doctor', 'physician', 'clinic', 'healthcare', 'insurance', 'copay', 'deductible']
        medical_score = sum(1 for keyword in medical_keywords if keyword in text_lower)
        
        # Prescription keywords  
        prescription_keywords = ['prescription', 'medication', 'pharmacy', 'rx', 'dosage', 'pills', 'tablets', 'mg', 'refill']
        prescription_score = sum(1 for keyword in prescription_keywords if keyword in text_lower)
        
        # Invoice keywords
        invoice_keywords = ['invoice', 'bill', 'receipt', 'payment', 'total', 'subtotal', 'tax', 'amount', 'due', 'vendor']
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        
        # Determine document type based on keyword scores
        scores = {
            'medical_bill': medical_score,
            'prescription': prescription_score,
            'invoice': invoice_score
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def display_results(self, result: ExtractionResult, show_debug: bool, ocr_result=None):
        """Display extraction results"""
        
        st.header("📊 Extraction Results")
        
        # Compact metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Document Type", result.doc_type.replace('_', ' ').title())
        with col2:
            confidence_color = "🟢" if result.overall_confidence > 0.8 else "🟡" if result.overall_confidence > 0.6 else "🔴"
            st.metric("Overall Confidence", f"{result.overall_confidence:.1%}")
        with col3:
            if result.fields:
                high_conf_count = sum(1 for f in result.fields if f.confidence > 0.8)
                st.metric("High Confidence Fields", f"{high_conf_count}/{len(result.fields)}")
            else:
                st.metric("Extracted Fields", "0")
        with col4:
            validation_status = "✅" if not result.qa.failed_rules else "⚠️"
            st.metric("Validation", f"{len(result.qa.passed_rules)} passed")
        
        # Confidence chart in expander to save space
        if result.fields:
            with st.expander("📈 Field Confidence Chart", expanded=True):
                field_names = [f.name.replace('_', ' ').title() for f in result.fields]
                confidences = [f.confidence for f in result.fields]
                
                fig = px.bar(
                    x=field_names,
                    y=confidences,
                    color=confidences,
                    color_continuous_scale='RdYlGn',
                    title="Confidence by Field",
                    labels={'x': 'Field Name', 'y': 'Confidence Score'}
                )
                
                fig.update_layout(showlegend=False, height=350, xaxis_tickangle=-45)
                fig.update_yaxes(range=[0, 1])
                fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="High")
                fig.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Medium")
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Extracted fields - more compact
        st.subheader("📋 Extracted Fields")
        
        if result.fields:
            for field in result.fields:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{field.name.replace('_', ' ').title()}:** {field.value}")
                with col2:
                    confidence_color = "🟢" if field.confidence > 0.8 else "🟡" if field.confidence > 0.6 else "🔴"
                    st.write(f"Confidence: {confidence_color} {field.confidence:.1%}")
                with col3:
                    if field.source:
                        st.write(f"Page {field.source.page}")
        else:
            st.info("No structured fields were extracted from the document.")
            if ocr_result and ocr_result.get('full_text'):
                with st.expander("📝 View OCR Text"):
                    st.text_area("Extracted text:", ocr_result['full_text'], height=150)
        
        # Quality assurance - compact
        with st.expander("✅ Quality Assurance", expanded=len(result.qa.failed_rules) > 0):
            col1, col2 = st.columns(2)
            with col1:
                if result.qa.passed_rules:
                    st.success(f"**Passed ({len(result.qa.passed_rules)}):**")
                    for rule in result.qa.passed_rules:
                        st.write(f"• {rule.replace('_', ' ').title()}")
                else:
                    st.info("No validation rules run")
            
            with col2:
                if result.qa.failed_rules:
                    st.error(f"**Failed ({len(result.qa.failed_rules)}):**")
                    for rule in result.qa.failed_rules:
                        st.write(f"• {rule.replace('_', ' ').title()}")
                else:
                    st.success("All validation rules passed!")
            
            if result.qa.notes:
                st.info(f"**Notes:** {result.qa.notes}")
        
        # JSON output - compact
        with st.expander("💻 JSON Output & Download", expanded=False):
            result_dict = result.model_dump()
            # Clean up the JSON for better readability
            for field in result_dict.get('fields', []):
                if 'source' in field and field['source'] is None:
                    del field['source']
            
            json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.code(json_str, language='json', line_numbers=True)
            with col2:
                st.download_button(
                    "📥 Download",
                    data=json_str,
                    file_name=f"extraction_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Debug information - collapsed by default
        if show_debug and ocr_result:
            with st.expander("🐛 Debug Information", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**OCR Text:**")
                    st.text_area("", ocr_result.get('full_text', ''), height=200, key="debug_text")
                
                with col2:
                    st.write("**Statistics:**")
                    stats = {
                        "Characters": len(ocr_result.get('full_text', '')),
                        "Text Blocks": len(ocr_result.get('text_blocks', [])),
                        "Fields": len(result.fields),
                        "Avg Confidence": f"{sum(f.confidence for f in result.fields) / len(result.fields):.1%}" if result.fields else "N/A"
                    }
                    for key, value in stats.items():
                        st.metric(key, value)

if __name__ == "__main__":
    # Check for API key
    if not Config.OPENAI_API_KEY:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.info("Create a .env file in the project root with: OPENAI_API_KEY=your_key_here")
        st.stop()
    
    try:
        app = DocumentExtractionApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application failed to start: {str(e)}")
        st.exception(e)

# ——————————————————————————————————————————————————————————————————————————————
# 👨‍💻 Created by Arpit Mohankar
# 🚀 Agentic Document Extraction System
# ——————————————————————————————————————————————————————————————————————————————
st.markdown("""
---
**  💡Created by Arpit Mohankar**  
*Agentic Document Extraction System - AI Challenge Solution*
""")
