import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import json
from pathlib import Path
import logging
from datetime import datetime

# Vector database and embeddings
import faiss
from sentence_transformers import SentenceTransformer

# Database integration
from database import DatabaseManager
from utils import TextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeStorage:
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "faiss_index",
                 dimension: int = 384):
        
        self.embedding_model_name = embedding_model_name
        self.index_path = Path(index_path)
        self.dimension = dimension
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.document_chunks = []  # Store chunk metadata
        self.chunk_embeddings = []  # Store embeddings
        
        # Initialize database and text processor
        self.db_manager = DatabaseManager()
        self.text_processor = TextProcessor()
        
        # Create index directory
        self.index_path.mkdir(exist_ok=True)
        
        # Initialize models and load existing index
        self._initialize_embedding_model()
        self._load_or_create_index()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "chunk_metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                # Load existing index
                self.faiss_index = faiss.read_index(str(index_file))
                
                # Load chunk metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.document_chunks = json.load(f)
                
                logger.info(f"Loaded existing index with {len(self.document_chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.document_chunks = []
        logger.info("Created new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            metadata_file = self.index_path / "chunk_metadata.json"
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(index_file))
            
            # Save chunk metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_chunks, f, indent=2, ensure_ascii=False)
            
            logger.info("Saved FAISS index and metadata")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def add_document(self, document_data: Dict[str, Any]) -> bool:
        """Add a document to the knowledge base"""
        try:
            document_name = document_data['document_name']
            chunks = document_data['chunks']
            difficulty_level = document_data['difficulty_level']
            topics = document_data['topics']
            
            logger.info(f"Adding document '{document_name}' with {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk for chunk in chunks if len(chunk.strip()) > 20]
            if not chunk_texts:
                logger.warning(f"No valid chunks found for document {document_name}")
                return False
            
            embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=False)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to FAISS index
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store chunk metadata
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                chunk_metadata = {
                    'id': len(self.document_chunks),
                    'document_name': document_name,
                    'chunk_index': i,
                    'text': chunk_text,
                    'difficulty_level': difficulty_level,
                    'topics': topics,
                    'word_count': len(chunk_text.split()),
                    'added_at': datetime.now().isoformat()
                }
                self.document_chunks.append(chunk_metadata)
            
            # Save to database
            self._save_chunks_to_db(document_name, chunk_texts, embeddings, difficulty_level, topics)
            
            # Save index
            self._save_index()
            
            logger.info(f"Successfully added {len(chunk_texts)} chunks from '{document_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def _save_chunks_to_db(self, document_name: str, chunks: List[str], 
                          embeddings: np.ndarray, difficulty_level: str, topics: List[str]):
        """Save chunks to database"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        for chunk_text, embedding in zip(chunks, embeddings):
            # Convert embedding to bytes for storage
            embedding_bytes = embedding.tobytes()
            
            cursor.execute('''
                INSERT INTO knowledge_chunks (document_name, chunk_text, chunk_embedding, 
                                            difficulty_level, topics)
                VALUES (?, ?, ?, ?, ?)
            ''', (document_name, chunk_text, embedding_bytes, difficulty_level, json.dumps(topics)))
        
        conn.commit()
        conn.close()
    
    def search_similar_chunks(self, query: str, top_k: int = 5, 
                            filter_difficulty: Optional[str] = None,
                            filter_topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity"""
        try:
            if not self.document_chunks:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), 
                                                    min(top_k * 2, len(self.document_chunks)))
            
            # Filter and rank results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.document_chunks):
                    continue
                
                chunk_metadata = self.document_chunks[idx].copy()
                chunk_metadata['similarity_score'] = float(score)
                
                # Apply filters
                if filter_difficulty and chunk_metadata['difficulty_level'] != filter_difficulty:
                    continue
                
                if filter_topics:
                    chunk_topics = chunk_metadata.get('topics', [])
                    if not any(topic in chunk_topics for topic in filter_topics):
                        continue
                
                results.append(chunk_metadata)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_document_chunks(self, document_name: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        return [chunk for chunk in self.document_chunks 
                if chunk['document_name'] == document_name]
    
    def get_random_chunks(self, count: int = 5, difficulty_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get random chunks for review"""
        filtered_chunks = self.document_chunks
        
        if difficulty_level:
            filtered_chunks = [chunk for chunk in filtered_chunks 
                             if chunk['difficulty_level'] == difficulty_level]
        
        if not filtered_chunks:
            return []
        
        # Randomly sample chunks
        import random
        sample_size = min(count, len(filtered_chunks))
        return random.sample(filtered_chunks, sample_size)
    
    def delete_document(self, document_name: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # Find chunks to delete
            chunks_to_delete = [i for i, chunk in enumerate(self.document_chunks) 
                              if chunk['document_name'] == document_name]
            
            if not chunks_to_delete:
                logger.warning(f"No chunks found for document: {document_name}")
                return False
            
            # Remove from metadata (in reverse order to maintain indices)
            for idx in reversed(chunks_to_delete):
                del self.document_chunks[idx]
            
            # Rebuild FAISS index (this is expensive but necessary)
            self._rebuild_index()
            
            # Delete from database
            self._delete_chunks_from_db(document_name)
            
            logger.info(f"Deleted {len(chunks_to_delete)} chunks for document: {document_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            return False
    
    def _rebuild_index(self):
        """Rebuild FAISS index from remaining chunks"""
        if not self.document_chunks:
            self._create_new_index()
            return
        
        # Extract all chunk texts
        chunk_texts = [chunk['text'] for chunk in self.document_chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create new index
        self._create_new_index()
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Update chunk IDs
        for i, chunk in enumerate(self.document_chunks):
            chunk['id'] = i
        
        # Save updated index
        self._save_index()
    
    def _delete_chunks_from_db(self, document_name: str):
        """Delete chunks from database"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM knowledge_chunks WHERE document_name = ?', (document_name,))
        
        conn.commit()
        conn.close()
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        if not self.document_chunks:
            return {
                'total_chunks': 0,
                'total_documents': 0,
                'difficulty_distribution': {},
                'topic_distribution': {},
                'avg_chunk_length': 0
            }
        
        # Calculate statistics
        documents = set(chunk['document_name'] for chunk in self.document_chunks)
        
        difficulty_dist = {}
        topic_dist = {}
        total_words = 0
        
        for chunk in self.document_chunks:
            # Difficulty distribution
            difficulty = chunk['difficulty_level']
            difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
            
            # Topic distribution
            for topic in chunk.get('topics', []):
                topic_dist[topic] = topic_dist.get(topic, 0) + 1
            
            # Word count
            total_words += chunk.get('word_count', 0)
        
        return {
            'total_chunks': len(self.document_chunks),
            'total_documents': len(documents),
            'difficulty_distribution': difficulty_dist,
            'topic_distribution': dict(sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_chunk_length': total_words / len(self.document_chunks) if self.document_chunks else 0,
            'documents': list(documents)
        }

class RetrievalAugmentedGeneration:
    """RAG system for question answering using the knowledge base"""
    
    def __init__(self, knowledge_storage: KnowledgeStorage):
        self.knowledge_storage = knowledge_storage
        self.text_processor = TextProcessor()
    
    def retrieve_context(self, query: str, top_k: int = 3, 
                        min_similarity: float = 0.3) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant context for a query"""
        
        # Search for similar chunks
        similar_chunks = self.knowledge_storage.search_similar_chunks(query, top_k=top_k)
        
        # Filter by minimum similarity
        relevant_chunks = [chunk for chunk in similar_chunks 
                          if chunk['similarity_score'] >= min_similarity]
        
        if not relevant_chunks:
            return [], []
        
        # Extract context texts
        context_texts = [chunk['text'] for chunk in relevant_chunks]
        
        return context_texts, relevant_chunks
    
    def generate_context_summary(self, contexts: List[str], max_length: int = 500) -> str:
        """Generate a summary of retrieved contexts"""
        if not contexts:
            return ""
        
        # Combine contexts
        combined_text = " ".join(contexts)
        
        # If combined text is short enough, return as is
        if len(combined_text.split()) <= max_length:
            return combined_text
        
        # Otherwise, take the most relevant sentences
        sentences = self.text_processor.segment_sentences(combined_text)
        
        # Simple extractive summarization - take first sentences up to max_length
        summary_sentences = []
        word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= max_length:
                summary_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        return " ".join(summary_sentences)
    
    def prepare_rag_context(self, query: str, max_context_length: int = 500) -> Dict[str, Any]:
        """Prepare context for RAG-based question answering"""
        
        # Retrieve relevant contexts
        context_texts, chunk_metadata = self.retrieve_context(query, top_k=5)
        
        if not context_texts:
            return {
                'has_context': False,
                'context': "",
                'sources': [],
                'query': query
            }
        
        # Generate context summary
        context_summary = self.generate_context_summary(context_texts, max_context_length)
        
        # Prepare source information
        sources = []
        for chunk in chunk_metadata:
            sources.append({
                'document': chunk['document_name'],
                'similarity': round(chunk['similarity_score'], 3),
                'difficulty': chunk['difficulty_level'],
                'topics': chunk.get('topics', [])
            })
        
        return {
            'has_context': True,
            'context': context_summary,
            'sources': sources,
            'query': query,
            'context_length': len(context_summary.split())
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize knowledge storage
    knowledge_storage = KnowledgeStorage()
    
    # Example document data
    sample_document = {
        'document_name': 'machine_learning_basics',
        'chunks': [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks.",
            "Unsupervised learning finds hidden patterns in data without labeled examples. Clustering and dimensionality reduction are common unsupervised techniques.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns. It has revolutionized fields like computer vision and natural language processing."
        ],
        'difficulty_level': 'Medium',
        'topics': ['machine learning', 'artificial intelligence', 'data science']
    }
    
    # Add document to knowledge base
    success = knowledge_storage.add_document(sample_document)
    print(f"Document added successfully: {success}")
    
    # Test search functionality
    query = "What is supervised learning?"
    results = knowledge_storage.search_similar_chunks(query, top_k=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['similarity_score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Document: {result['document_name']}")
        print()
    
    # Test RAG system
    rag = RetrievalAugmentedGeneration(knowledge_storage)
    rag_context = rag.prepare_rag_context(query)
    
    print("RAG Context:")
    print(f"Has context: {rag_context['has_context']}")
    print(f"Context: {rag_context['context'][:200]}...")
    print(f"Sources: {[s['document'] for s in rag_context['sources']]}")
    
    # Get knowledge base statistics
    stats = knowledge_storage.get_knowledge_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Difficulty distribution: {stats['difficulty_distribution']}")
