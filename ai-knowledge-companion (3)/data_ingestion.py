import os
import io
import tempfile
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

# PDF processing
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text

# OCR for images
import easyocr
from PIL import Image
import numpy as np

# Audio transcription
import whisper

# Utilities
from utils import TextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.ocr_reader = None
        self.whisper_model = None
        self._init_models()
    
    def _init_models(self):
        """Initialize OCR and Whisper models lazily"""
        try:
            # Initialize EasyOCR (supports multiple languages)
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
        
        try:
            # Initialize Whisper model (base model for balance of speed/accuracy)
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
    
    def extract_from_pdf(self, file_path: Union[str, io.BytesIO]) -> Dict[str, any]:
        """Extract text from PDF using multiple methods for robustness"""
        extracted_text = ""
        metadata = {
            'source_type': 'pdf',
            'extraction_method': None,
            'page_count': 0,
            'success': False
        }
        
        try:
            # Method 1: Try PyPDF2 first (faster)
            if isinstance(file_path, str):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata['page_count'] = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text.strip():
                                extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                            continue
            else:
                # Handle BytesIO object
                pdf_reader = PyPDF2.PdfReader(file_path)
                metadata['page_count'] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                        continue
            
            if extracted_text.strip():
                metadata['extraction_method'] = 'PyPDF2'
                metadata['success'] = True
            else:
                # Method 2: Fallback to pdfminer for complex PDFs
                logger.info("PyPDF2 failed, trying pdfminer...")
                if isinstance(file_path, str):
                    extracted_text = pdfminer_extract_text(file_path)
                else:
                    # Save BytesIO to temp file for pdfminer
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_file.write(file_path.getvalue())
                        temp_file_path = temp_file.name
                    
                    try:
                        extracted_text = pdfminer_extract_text(temp_file_path)
                    finally:
                        os.unlink(temp_file_path)
                
                if extracted_text.strip():
                    metadata['extraction_method'] = 'pdfminer'
                    metadata['success'] = True
        
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            metadata['error'] = str(e)
        
        # Clean and process the extracted text
        if extracted_text.strip():
            extracted_text = self.text_processor.clean_text(extracted_text)
        
        return {
            'text': extracted_text,
            'metadata': metadata
        }
    
    def extract_from_image(self, file_path: Union[str, io.BytesIO, np.ndarray]) -> Dict[str, any]:
        """Extract text from images using OCR"""
        extracted_text = ""
        metadata = {
            'source_type': 'image',
            'extraction_method': 'EasyOCR',
            'success': False
        }
        
        if not self.ocr_reader:
            metadata['error'] = "OCR reader not initialized"
            return {'text': extracted_text, 'metadata': metadata}
        
        try:
            # Handle different input types
            if isinstance(file_path, str):
                # File path
                image = Image.open(file_path)
            elif isinstance(file_path, io.BytesIO):
                # BytesIO object
                image = Image.open(file_path)
            else:
                # Numpy array
                image = Image.fromarray(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL image to numpy array for EasyOCR
            image_array = np.array(image)
            
            # Extract text using EasyOCR
            results = self.ocr_reader.readtext(image_array)
            
            # Combine all detected text
            text_blocks = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low-confidence detections
                    text_blocks.append(text)
            
            extracted_text = ' '.join(text_blocks)
            
            if extracted_text.strip():
                metadata['success'] = True
                metadata['text_blocks_count'] = len(text_blocks)
                metadata['avg_confidence'] = np.mean([conf for _, _, conf in results])
            
        except Exception as e:
            logger.error(f"Image OCR extraction failed: {e}")
            metadata['error'] = str(e)
        
        # Clean the extracted text
        if extracted_text.strip():
            extracted_text = self.text_processor.clean_text(extracted_text)
        
        return {
            'text': extracted_text,
            'metadata': metadata
        }
    
    def extract_from_audio(self, file_path: Union[str, io.BytesIO]) -> Dict[str, any]:
        """Extract text from audio using Whisper"""
        extracted_text = ""
        metadata = {
            'source_type': 'audio',
            'extraction_method': 'Whisper',
            'success': False
        }
        
        if not self.whisper_model:
            metadata['error'] = "Whisper model not loaded"
            return {'text': extracted_text, 'metadata': metadata}
        
        try:
            # Handle BytesIO by saving to temporary file
            if isinstance(file_path, io.BytesIO):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(file_path.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    result = self.whisper_model.transcribe(temp_file_path)
                    extracted_text = result["text"]
                    metadata['language'] = result.get("language", "unknown")
                finally:
                    os.unlink(temp_file_path)
            else:
                # Direct file path
                result = self.whisper_model.transcribe(file_path)
                extracted_text = result["text"]
                metadata['language'] = result.get("language", "unknown")
            
            if extracted_text.strip():
                metadata['success'] = True
                # Add segments information if available
                if 'segments' in result:
                    metadata['segments_count'] = len(result['segments'])
                    metadata['duration'] = max([seg['end'] for seg in result['segments']], default=0)
        
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            metadata['error'] = str(e)
        
        # Clean the extracted text
        if extracted_text.strip():
            extracted_text = self.text_processor.clean_text(extracted_text)
        
        return {
            'text': extracted_text,
            'metadata': metadata
        }
    
    def process_file(self, file_path: Union[str, io.BytesIO], file_type: Optional[str] = None) -> Dict[str, any]:
        """Process any supported file type and extract text"""
        
        # Determine file type if not provided
        if file_type is None and isinstance(file_path, str):
            file_extension = Path(file_path).suffix.lower()
            if file_extension == '.pdf':
                file_type = 'pdf'
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                file_type = 'image'
            elif file_extension in ['.wav', '.mp3', '.m4a', '.flac']:
                file_type = 'audio'
            else:
                return {
                    'text': '',
                    'metadata': {
                        'success': False,
                        'error': f'Unsupported file type: {file_extension}'
                    }
                }
        
        # Process based on file type
        if file_type == 'pdf':
            return self.extract_from_pdf(file_path)
        elif file_type == 'image':
            return self.extract_from_image(file_path)
        elif file_type == 'audio':
            return self.extract_from_audio(file_path)
        else:
            return {
                'text': '',
                'metadata': {
                    'success': False,
                    'error': f'Unknown file type: {file_type}'
                }
            }
    
    def process_text_input(self, text: str) -> Dict[str, any]:
        """Process direct text input"""
        cleaned_text = self.text_processor.clean_text(text)
        
        return {
            'text': cleaned_text,
            'metadata': {
                'source_type': 'text',
                'success': True,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }
        }
    
    def batch_process_files(self, file_paths: List[str]) -> List[Dict[str, any]]:
        """Process multiple files in batch"""
        results = []
        
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            try:
                result = self.process_file(file_path)
                result['file_path'] = file_path
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'text': '',
                    'metadata': {
                        'success': False,
                        'error': str(e)
                    }
                })
        
        return results
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported file formats"""
        return {
            'pdf': ['.pdf'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'audio': ['.wav', '.mp3', '.m4a', '.flac'],
            'text': ['direct_input']
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = DataIngestionPipeline()
    
    # Test with sample text
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from and make predictions on data. It has applications in various 
    fields including natural language processing, computer vision, and robotics.
    """
    
    result = pipeline.process_text_input(sample_text)
    print("Text processing result:")
    print(f"Success: {result['metadata']['success']}")
    print(f"Word count: {result['metadata']['word_count']}")
    print(f"Extracted text preview: {result['text'][:100]}...")
    
    print(f"\nSupported formats: {pipeline.get_supported_formats()}")
