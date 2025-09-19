import os
import shutil
from typing import List, Dict, Optional, Union
from pathlib import Path
import hashlib
import json
from datetime import datetime

from data_ingestion import DataIngestionPipeline
from utils import TextProcessor

class FileHandler:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.pipeline = DataIngestionPipeline()
        self.text_processor = TextProcessor()
        
        # Create subdirectories for different file types
        for subdir in ['pdfs', 'images', 'audio', 'processed']:
            (self.upload_dir / subdir).mkdir(exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str, file_type: str) -> str:
        """Save uploaded file and return the file path"""
        # Generate unique filename to avoid conflicts
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        file_extension = Path(filename).suffix
        unique_filename = f"{Path(filename).stem}_{file_hash}{file_extension}"
        
        # Determine subdirectory based on file type
        if file_type == 'pdf':
            subdir = 'pdfs'
        elif file_type == 'image':
            subdir = 'images'
        elif file_type == 'audio':
            subdir = 'audio'
        else:
            subdir = 'processed'
        
        file_path = self.upload_dir / subdir / unique_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path)
    
    def process_uploaded_file(self, file_path: str, document_name: Optional[str] = None) -> Dict[str, any]:
        """Process uploaded file and extract structured information"""
        
        if document_name is None:
            document_name = Path(file_path).stem
        
        # Extract text using the ingestion pipeline
        extraction_result = self.pipeline.process_file(file_path)
        
        if not extraction_result['metadata']['success']:
            return {
                'success': False,
                'error': extraction_result['metadata'].get('error', 'Unknown extraction error'),
                'document_name': document_name
            }
        
        extracted_text = extraction_result['text']
        
        # Process the extracted text
        processed_data = self._process_extracted_text(extracted_text, document_name)
        processed_data.update({
            'file_path': file_path,
            'extraction_metadata': extraction_result['metadata'],
            'success': True
        })
        
        # Save processed data
        self._save_processed_data(processed_data, document_name)
        
        return processed_data
    
    def _process_extracted_text(self, text: str, document_name: str) -> Dict[str, any]:
        """Process extracted text into structured format"""
        
        # Basic text analysis
        sentences = self.text_processor.segment_sentences(text)
        keywords = self.text_processor.extract_keywords(text)
        readability_features = self.text_processor.calculate_readability_features(text)
        
        # Chunk text for better processing
        chunks = self.text_processor.chunk_text(text, chunk_size=500, overlap=50)
        
        # Calculate difficulty level
        from utils import calculate_difficulty_level
        difficulty_level = calculate_difficulty_level(readability_features)
        
        # Extract topics (simplified - could be enhanced with topic modeling)
        topics = self._extract_topics(text, keywords)
        
        return {
            'document_name': document_name,
            'full_text': text,
            'sentences': sentences,
            'chunks': chunks,
            'keywords': keywords,
            'topics': topics,
            'readability_features': readability_features,
            'difficulty_level': difficulty_level,
            'word_count': len(text.split()),
            'sentence_count': len(sentences),
            'chunk_count': len(chunks),
            'processed_at': datetime.now().isoformat()
        }
    
    def _extract_topics(self, text: str, keywords: List[str]) -> List[str]:
        """Extract main topics from text (simplified approach)"""
        # This is a simplified topic extraction
        # In a real implementation, you might use topic modeling (LDA, BERTopic)
        
        topics = []
        
        # Academic subjects detection
        academic_keywords = {
            'mathematics': ['math', 'equation', 'formula', 'theorem', 'proof', 'algebra', 'calculus'],
            'science': ['experiment', 'hypothesis', 'theory', 'research', 'study', 'analysis'],
            'history': ['century', 'war', 'empire', 'revolution', 'ancient', 'medieval'],
            'literature': ['author', 'novel', 'poem', 'character', 'plot', 'theme'],
            'technology': ['computer', 'software', 'algorithm', 'data', 'system', 'programming'],
            'business': ['market', 'company', 'profit', 'strategy', 'management', 'finance']
        }
        
        text_lower = text.lower()
        for topic, topic_keywords in academic_keywords.items():
            if any(keyword in text_lower for keyword in topic_keywords):
                topics.append(topic)
        
        # Add keywords as potential topics
        topics.extend(keywords[:5])  # Top 5 keywords as topics
        
        return list(set(topics))  # Remove duplicates
    
    def _save_processed_data(self, processed_data: Dict[str, any], document_name: str):
        """Save processed data to JSON file"""
        processed_file_path = self.upload_dir / 'processed' / f"{document_name}_processed.json"
        
        # Convert any non-serializable objects
        serializable_data = {}
        for key, value in processed_data.items():
            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)
        
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def load_processed_data(self, document_name: str) -> Optional[Dict[str, any]]:
        """Load previously processed data"""
        processed_file_path = self.upload_dir / 'processed' / f"{document_name}_processed.json"
        
        if processed_file_path.exists():
            with open(processed_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def list_processed_documents(self) -> List[Dict[str, any]]:
        """List all processed documents with metadata"""
        processed_dir = self.upload_dir / 'processed'
        documents = []
        
        for file_path in processed_dir.glob("*_processed.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    documents.append({
                        'document_name': data.get('document_name', file_path.stem),
                        'word_count': data.get('word_count', 0),
                        'difficulty_level': data.get('difficulty_level', 'Unknown'),
                        'topics': data.get('topics', []),
                        'processed_at': data.get('processed_at', 'Unknown'),
                        'file_path': str(file_path)
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return sorted(documents, key=lambda x: x['processed_at'], reverse=True)
    
    def delete_document(self, document_name: str) -> bool:
        """Delete a processed document and its files"""
        try:
            # Delete processed JSON file
            processed_file_path = self.upload_dir / 'processed' / f"{document_name}_processed.json"
            if processed_file_path.exists():
                processed_file_path.unlink()
            
            # Find and delete original file (search in all subdirectories)
            for subdir in ['pdfs', 'images', 'audio']:
                subdir_path = self.upload_dir / subdir
                for file_path in subdir_path.glob(f"{document_name}*"):
                    file_path.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting document {document_name}: {e}")
            return False
    
    def get_file_stats(self) -> Dict[str, any]:
        """Get statistics about uploaded and processed files"""
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'file_types': {'pdf': 0, 'image': 0, 'audio': 0},
            'total_size_mb': 0
        }
        
        # Count files in each subdirectory
        for subdir in ['pdfs', 'images', 'audio']:
            subdir_path = self.upload_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                stats['file_types'][subdir.rstrip('s')] = len(files)
                stats['total_files'] += len(files)
                
                # Calculate total size
                for file_path in files:
                    try:
                        stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                    except:
                        continue
        
        # Count processed files
        processed_dir = self.upload_dir / 'processed'
        if processed_dir.exists():
            stats['processed_files'] = len(list(processed_dir.glob("*_processed.json")))
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats

# Example usage
if __name__ == "__main__":
    handler = FileHandler()
    
    # Example: Process a text input
    sample_text = """
    Artificial Intelligence (AI) is revolutionizing various industries by enabling machines 
    to perform tasks that typically require human intelligence. Machine learning, a subset 
    of AI, allows systems to automatically learn and improve from experience without being 
    explicitly programmed. Deep learning, which uses neural networks with multiple layers, 
    has shown remarkable success in image recognition, natural language processing, and 
    game playing. The applications of AI span across healthcare, finance, transportation, 
    and entertainment, promising to transform how we live and work.
    """
    
    # Save as a text file for testing
    test_file_path = handler.upload_dir / "test_ai_document.txt"
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # Process the file
    result = handler.pipeline.process_text_input(sample_text)
    processed_data = handler._process_extracted_text(result['text'], "test_ai_document")
    
    print("File processing completed!")
    print(f"Document: {processed_data['document_name']}")
    print(f"Word count: {processed_data['word_count']}")
    print(f"Difficulty: {processed_data['difficulty_level']}")
    print(f"Topics: {processed_data['topics']}")
    print(f"Keywords: {processed_data['keywords'][:5]}")
