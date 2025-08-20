# Agentic Document Extraction System

## 📌 Description
The **Agentic Document Extraction System** is an AI-powered solution for extracting structured information from various document types, such as invoices, medical bills, and prescriptions.  
It leverages:
- Cloud-based OCR for text extraction
- LLMs for document classification and data extraction
- Validation rules and confidence scoring for reliability  

This makes it a robust system for automating document understanding and data entry workflows.

---

## 🚀 Features
- **Document Classification** – Automatically identifies document types (invoice, medical bill, prescription).
- **OCR Text Extraction** – Extracts text content from PDFs and images using cloud OCR services.
- **Self-Consistent Field Extraction** – Extracts structured fields using LLMs with multiple runs for consistency.
- **Custom Field Extraction** – Users can specify custom fields for extraction.
- **Validation Rules** – Ensures extracted data follows rules defined per document type.
- **Confidence Scoring** – Assigns per-field and overall confidence scores.
- **User Interface** – A Streamlit-based web app for uploading documents and viewing results.

---

## 🛠️ Technologies Used
- **Python**
- **Streamlit**
- **OpenAI API**
- **OCR.Space API** (or Google Cloud Vision API, if configured)
- **Pydantic**
- **Requests**
- **pdf2image**
- **Pillow (PIL)**
- **python-dotenv**
- **Plotly**

---
``` .
├── .env                   # Environment variables (API keys)
├── LICENSE                # License file
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
├── config.py              # Configuration settings
├── confidence_scorer.py   # Confidence score logic
├── document_classifier.py # Document classification
├── extraction_chain.py    # Field extraction pipeline
├── main.py                # Streamlit app entry point
├── ocr_processor.py       # OCR orchestration
├── cloud_ocr_processor.py # Cloud OCR implementation (e.g., OCR.Space)
├── schemas.py             # Pydantic data models
├── utils.py               # Utility functions
└── validation_engine.py   # Validation logic
```

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.7+**
- OpenAI API Key 
- OCR API Key 

### Installation
Clone the repository:
```bash
git clone <repository_url>
cd <project_directory>
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate

pip install -r requirements.txt
OPENAI_API_KEY=your_openai_api_key_here
OCR_SPACE_API_KEY=your_ocr_space_api_key_here
streamlit run main.py
```
🤝 Contributing

Contributions are welcome!
Please follow the standard GitHub pull request workflow and ensure
Code adheres to the project style
Tests are included for new features

📧 Contact

Maintainer: [Arpit Mohankar]
Email: [arpitmohankar24@gmail.com]
