import re
import string
from typing import List, Dict, Tuple
import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

class TextProcessor:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-$$$$]', '', text)
        return text.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        if nlp:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            # Fallback to NLTK
            sentences = nltk.sent_tokenize(text)
            return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords using TF-IDF and named entities"""
        keywords = []
        
        if nlp:
            doc = nlp(text)
            # Extract named entities
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
            keywords.extend(entities)
            
            # Extract important nouns and adjectives
            important_words = [token.lemma_.lower() for token in doc 
                             if (token.pos_ in ['NOUN', 'ADJ'] and 
                                 token.lemma_.lower() not in self.stop_words and
                                 len(token.lemma_) > 3)]
            keywords.extend(important_words)
        
        # Remove duplicates and return top_k
        unique_keywords = list(set(keywords))
        return unique_keywords[:top_k]
    
    def calculate_readability_features(self, text: str) -> Dict[str, float]:
        """Calculate readability and complexity features"""
        features = {}
        
        # Basic readability scores
        features['flesch_reading_ease'] = flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        
        # Text statistics
        sentences = self.segment_sentences(text)
        words = text.split()
        
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['sentence_count'] = len(sentences)
        features['word_count'] = len(words)
        
        # Vocabulary richness (unique words / total words)
        unique_words = set(word.lower() for word in words)
        features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
        
        return features
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = self.segment_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk with overlap
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_length = len(sent.split())
                    if overlap_length + sent_length <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_length
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

def calculate_difficulty_level(features: Dict[str, float]) -> str:
    """Determine difficulty level based on readability features"""
    flesch_score = features.get('flesch_reading_ease', 50)
    grade_level = features.get('flesch_kincaid_grade', 8)
    avg_sentence_length = features.get('avg_sentence_length', 15)
    
    # Scoring system
    difficulty_score = 0
    
    # Flesch Reading Ease (higher = easier)
    if flesch_score >= 70:
        difficulty_score += 1  # Easy
    elif flesch_score >= 50:
        difficulty_score += 2  # Medium
    else:
        difficulty_score += 3  # Hard
    
    # Grade level
    if grade_level <= 8:
        difficulty_score += 1
    elif grade_level <= 12:
        difficulty_score += 2
    else:
        difficulty_score += 3
    
    # Sentence length
    if avg_sentence_length <= 15:
        difficulty_score += 1
    elif avg_sentence_length <= 25:
        difficulty_score += 2
    else:
        difficulty_score += 3
    
    # Final classification
    if difficulty_score <= 4:
        return "Easy"
    elif difficulty_score <= 7:
        return "Medium"
    else:
        return "Hard"

def format_time_spent(seconds: int) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
