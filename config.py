import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Cloud OCR settings
    OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "helloworld")  # Free demo key
    GOOGLE_VISION_CREDENTIALS = os.getenv("GOOGLE_VISION_CREDENTIALS_PATH")
    
    # Model configurations
    CLASSIFICATION_MODEL = "gpt-4-vision-preview"
    EXTRACTION_MODEL = "gpt-4-turbo"
    
    # Processing settings
    CONSISTENCY_RUNS = 3
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.5
