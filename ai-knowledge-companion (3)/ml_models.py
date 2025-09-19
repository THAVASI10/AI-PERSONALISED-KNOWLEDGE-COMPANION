import os
import re
import random
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime

# Transformer models
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, T5Tokenizer, T5ForConditionalGeneration
)
import torch

# Knowledge retrieval
from knowledge_storage import KnowledgeStorage, RetrievalAugmentedGeneration
from utils import TextProcessor
from database import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationModel:
    """Text summarization using BART/T5 models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self.text_processor = TextProcessor()
        
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model"""
        try:
            logger.info(f"Loading summarization model: {self.model_name}")
            
            # Use pipeline for easier inference
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            # Fallback to a smaller model
            try:
                logger.info("Trying fallback model: facebook/bart-base")
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-base",
                    tokenizer="facebook/bart-base",
                    device=-1  # CPU only for fallback
                )
                self.model_name = "facebook/bart-base"
                logger.info("Fallback summarization model loaded")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                self.summarizer = None
    
    def summarize_text(self, text: str, max_length: int = 150, 
                      min_length: int = 50, summary_type: str = "bullet") -> Dict[str, Any]:
        """Generate summary of the input text"""
        
        if not self.summarizer:
            return {
                'success': False,
                'error': 'Summarization model not available',
                'summary': '',
                'summary_type': summary_type
            }
        
        try:
            # Clean and prepare text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Check text length
            if len(cleaned_text.split()) < 50:
                return {
                    'success': False,
                    'error': 'Text too short for summarization (minimum 50 words)',
                    'summary': cleaned_text,
                    'summary_type': summary_type
                }
            
            # Handle long texts by chunking
            max_input_length = 1024  # BART's max input length
            text_chunks = self._chunk_text_for_summarization(cleaned_text, max_input_length)
            
            summaries = []
            for chunk in text_chunks:
                try:
                    # Generate summary for chunk
                    result = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )
                    
                    if result and len(result) > 0:
                        summaries.append(result[0]['summary_text'])
                
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk: {e}")
                    continue
            
            if not summaries:
                return {
                    'success': False,
                    'error': 'Failed to generate summary',
                    'summary': '',
                    'summary_type': summary_type
                }
            
            # Combine summaries
            combined_summary = " ".join(summaries)
            
            # Format summary based on type
            formatted_summary = self._format_summary(combined_summary, summary_type)
            
            return {
                'success': True,
                'summary': formatted_summary,
                'summary_type': summary_type,
                'original_length': len(text.split()),
                'summary_length': len(formatted_summary.split()),
                'compression_ratio': len(formatted_summary.split()) / len(text.split()),
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': '',
                'summary_type': summary_type
            }
    
    def _chunk_text_for_summarization(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks suitable for summarization"""
        sentences = self.text_processor.segment_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _format_summary(self, summary: str, summary_type: str) -> str:
        """Format summary based on requested type"""
        
        if summary_type == "bullet":
            # Convert to bullet points
            sentences = self.text_processor.segment_sentences(summary)
            bullet_points = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    bullet_points.append(f"• {sentence.strip()}")
            
            return "\n".join(bullet_points)
        
        elif summary_type == "paragraph":
            return summary
        
        elif summary_type == "key_points":
            # Extract key points (simplified approach)
            sentences = self.text_processor.segment_sentences(summary)
            key_points = []
            
            for i, sentence in enumerate(sentences, 1):
                if len(sentence.strip()) > 10:
                    key_points.append(f"{i}. {sentence.strip()}")
            
            return "\n".join(key_points)
        
        else:
            return summary

class QuestionGenerationModel:
    """Question generation using T5 model"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.text_processor = TextProcessor()
        
        self._load_model()
    
    def _load_model(self):
        """Load the question generation model"""
        try:
            logger.info(f"Loading question generation model: {self.model_name}")
            
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info("Question generation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load question generation model: {e}")
            # Try smaller model as fallback
            try:
                logger.info("Trying fallback model: google/flan-t5-small")
                self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
                self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
                self.model_name = "google/flan-t5-small"
                logger.info("Fallback question generation model loaded")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                self.model = None
                self.tokenizer = None
    
    def generate_questions(self, text: str, num_questions: int = 5, 
                          question_types: List[str] = None) -> List[Dict[str, Any]]:
        """Generate questions from text"""
        
        if not self.model or not self.tokenizer:
            return []
        
        if question_types is None:
            question_types = ["what", "how", "why", "when", "where"]
        
        try:
            # Clean text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Extract key information for question generation
            keywords = self.text_processor.extract_keywords(cleaned_text, top_k=10)
            sentences = self.text_processor.segment_sentences(cleaned_text)
            
            questions = []
            
            # Method 1: Template-based question generation
            template_questions = self._generate_template_questions(sentences, keywords, question_types)
            questions.extend(template_questions)
            
            # Method 2: T5-based question generation
            if len(questions) < num_questions:
                t5_questions = self._generate_t5_questions(cleaned_text, num_questions - len(questions))
                questions.extend(t5_questions)
            
            # Method 3: Heuristic questions from important sentences
            if len(questions) < num_questions:
                heuristic_questions = self._generate_heuristic_questions(sentences, keywords)
                questions.extend(heuristic_questions)
            
            # Limit to requested number and add metadata
            final_questions = []
            for i, q in enumerate(questions[:num_questions]):
                final_questions.append({
                    'question': q,
                    'question_id': i + 1,
                    'generated_by': 'ml_model',
                    'difficulty': self._estimate_question_difficulty(q, text),
                    'topics': keywords[:3]  # Top 3 keywords as topics
                })
            
            return final_questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return []
    
    def _generate_template_questions(self, sentences: List[str], keywords: List[str], 
                                   question_types: List[str]) -> List[str]:
        """Generate questions using templates"""
        questions = []
        
        # Question templates
        templates = {
            "what": [
                f"What is {keyword}?" for keyword in keywords[:3]
            ],
            "how": [
                f"How does {keyword} work?" for keyword in keywords[:2]
            ],
            "why": [
                f"Why is {keyword} important?" for keyword in keywords[:2]
            ],
            "when": [
                f"When is {keyword} used?" for keyword in keywords[:1]
            ],
            "where": [
                f"Where can {keyword} be found?" for keyword in keywords[:1]
            ]
        }
        
        for q_type in question_types:
            if q_type in templates:
                questions.extend(templates[q_type])
        
        return questions[:5]  # Limit template questions
    
    def _generate_t5_questions(self, text: str, num_questions: int) -> List[str]:
        """Generate questions using T5 model"""
        questions = []
        
        try:
            # Prepare input for T5
            input_text = f"generate questions: {text[:500]}"  # Limit input length
            
            # Tokenize
            inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=100,
                    num_return_sequences=min(num_questions, 3),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode questions
            for output in outputs:
                question = self.tokenizer.decode(output, skip_special_tokens=True)
                if question and len(question) > 10:
                    questions.append(question)
        
        except Exception as e:
            logger.warning(f"T5 question generation failed: {e}")
        
        return questions
    
    def _generate_heuristic_questions(self, sentences: List[str], keywords: List[str]) -> List[str]:
        """Generate questions using heuristic rules"""
        questions = []
        
        # Look for sentences with important keywords
        important_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in sentence_lower)
            if keyword_count >= 2:
                important_sentences.append(sentence)
        
        # Generate questions from important sentences
        for sentence in important_sentences[:3]:
            # Simple transformation: statement to question
            if "is" in sentence.lower():
                question = sentence.replace(".", "?")
                questions.append(f"What {question.lower()}")
            elif "can" in sentence.lower():
                question = sentence.replace(".", "?")
                questions.append(question)
        
        return questions
    
    def _estimate_question_difficulty(self, question: str, context: str) -> str:
        """Estimate question difficulty based on complexity"""
        
        # Simple heuristics for difficulty estimation
        question_lower = question.lower()
        
        # Easy questions
        if any(word in question_lower for word in ["what is", "define", "list"]):
            return "Easy"
        
        # Hard questions
        elif any(word in question_lower for word in ["analyze", "evaluate", "compare", "why"]):
            return "Hard"
        
        # Medium questions (default)
        else:
            return "Medium"

class FlashcardGenerator:
    """Generate flashcards from text content"""
    
    def __init__(self, question_model: QuestionGenerationModel):
        self.question_model = question_model
        self.text_processor = TextProcessor()
        self.db_manager = DatabaseManager()
    
    def generate_flashcards(self, text: str, document_name: str, 
                          num_flashcards: int = 5) -> List[Dict[str, Any]]:
        """Generate flashcards from text"""
        
        try:
            # Generate questions
            questions = self.question_model.generate_questions(text, num_flashcards * 2)
            
            if not questions:
                return []
            
            # Create flashcards by finding answers in text
            flashcards = []
            sentences = self.text_processor.segment_sentences(text)
            
            for question_data in questions[:num_flashcards]:
                question = question_data['question']
                
                # Find potential answer in text (simplified approach)
                answer = self._find_answer_in_text(question, sentences)
                
                if answer:
                    flashcard = {
                        'question': question,
                        'answer': answer,
                        'difficulty_level': question_data['difficulty'],
                        'topics': question_data['topics'],
                        'document_name': document_name,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    flashcards.append(flashcard)
                    
                    # Save to database
                    self.db_manager.save_flashcard(
                        document_name, question, answer, question_data['difficulty']
                    )
            
            return flashcards
            
        except Exception as e:
            logger.error(f"Flashcard generation failed: {e}")
            return []
    
    def _find_answer_in_text(self, question: str, sentences: List[str]) -> str:
        """Find answer to question in text (simplified approach)"""
        
        # Extract keywords from question
        question_keywords = self.text_processor.extract_keywords(question, top_k=3)
        
        # Find sentences with highest keyword overlap
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            overlap = sum(1 for keyword in question_keywords 
                         if keyword.lower() in sentence_lower)
            
            if overlap > max_overlap and len(sentence.split()) > 5:
                max_overlap = overlap
                best_sentence = sentence
        
        # Clean and return answer
        if best_sentence:
            return self.text_processor.clean_text(best_sentence)
        
        return "Answer not found in the provided text."

class ConversationalQA:
    """Conversational Q&A system using RAG"""
    
    def __init__(self, knowledge_storage: KnowledgeStorage):
        self.knowledge_storage = knowledge_storage
        self.rag = RetrievalAugmentedGeneration(knowledge_storage)
        self.text_processor = TextProcessor()
        
        # Initialize conversation history
        self.conversation_history = []
    
    def answer_question(self, question: str, use_context: bool = True, 
                       conversation_context: bool = True) -> Dict[str, Any]:
        """Answer a question using RAG and conversation context"""
        
        try:
            # Prepare context
            rag_context = None
            if use_context:
                rag_context = self.rag.prepare_rag_context(question)
            
            # Generate answer
            if rag_context and rag_context['has_context']:
                answer = self._generate_contextual_answer(question, rag_context)
            else:
                answer = self._generate_fallback_answer(question)
            
            # Add to conversation history
            conversation_entry = {
                'question': question,
                'answer': answer['answer'],
                'timestamp': datetime.now().isoformat(),
                'has_context': rag_context['has_context'] if rag_context else False,
                'sources': rag_context['sources'] if rag_context and rag_context['has_context'] else []
            }
            
            if conversation_context:
                self.conversation_history.append(conversation_entry)
                # Keep only last 10 exchanges
                self.conversation_history = self.conversation_history[-10:]
            
            return {
                'success': True,
                'answer': answer['answer'],
                'confidence': answer.get('confidence', 0.5),
                'sources': conversation_entry['sources'],
                'has_context': conversation_entry['has_context'],
                'conversation_id': len(self.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': "I'm sorry, I couldn't process your question at the moment.",
                'confidence': 0.0,
                'sources': [],
                'has_context': False
            }
    
    def _generate_contextual_answer(self, question: str, rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using retrieved context"""
        
        context = rag_context['context']
        sources = rag_context['sources']
        
        # Simple rule-based answer generation (in a real system, you'd use a language model)
        answer_parts = []
        
        # Extract relevant sentences from context
        context_sentences = self.text_processor.segment_sentences(context)
        question_keywords = self.text_processor.extract_keywords(question, top_k=5)
        
        # Find most relevant sentences
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            keyword_matches = sum(1 for keyword in question_keywords 
                                if keyword.lower() in sentence_lower)
            
            if keyword_matches >= 1:
                relevant_sentences.append((sentence, keyword_matches))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in relevant_sentences[:3]]
        
        if top_sentences:
            answer = " ".join(top_sentences)
            confidence = min(0.9, 0.5 + len(top_sentences) * 0.1)
        else:
            answer = "Based on the available information: " + context[:200] + "..."
            confidence = 0.6
        
        return {
            'answer': answer,
            'confidence': confidence,
            'method': 'contextual_rag'
        }
    
    def _generate_fallback_answer(self, question: str) -> Dict[str, Any]:
        """Generate fallback answer when no context is available"""
        
        # Simple fallback responses
        fallback_responses = [
            "I don't have specific information about that topic in my knowledge base. Could you provide more context or ask about a different topic?",
            "I couldn't find relevant information to answer your question. Please try rephrasing your question or ask about topics covered in your study materials.",
            "That's an interesting question, but I don't have enough information in my current knowledge base to provide a comprehensive answer."
        ]
        
        return {
            'answer': random.choice(fallback_responses),
            'confidence': 0.3,
            'method': 'fallback'
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Example usage and testing
if __name__ == "__main__":
    # Test text
    sample_text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention. Machine learning algorithms 
    build a model based on training data in order to make predictions or decisions without being 
    explicitly programmed to do so. Machine learning algorithms are used in a wide variety of 
    applications, such as in medicine, email filtering, speech recognition, and computer vision, 
    where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
    """
    
    print("Testing ML Models...")
    
    # Test summarization
    print("\n1. Testing Summarization:")
    summarizer = SummarizationModel()
    summary_result = summarizer.summarize_text(sample_text, summary_type="bullet")
    
    if summary_result['success']:
        print(f"Summary ({summary_result['summary_type']}):")
        print(summary_result['summary'])
        print(f"Compression ratio: {summary_result['compression_ratio']:.2f}")
    else:
        print(f"Summarization failed: {summary_result['error']}")
    
    # Test question generation
    print("\n2. Testing Question Generation:")
    question_generator = QuestionGenerationModel()
    questions = question_generator.generate_questions(sample_text, num_questions=3)
    
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q['question']} (Difficulty: {q['difficulty']})")
    
    # Test flashcard generation
    print("\n3. Testing Flashcard Generation:")
    flashcard_gen = FlashcardGenerator(question_generator)
    flashcards = flashcard_gen.generate_flashcards(sample_text, "ml_basics", num_flashcards=2)
    
    for i, card in enumerate(flashcards, 1):
        print(f"Flashcard {i}:")
        print(f"Q: {card['question']}")
        print(f"A: {card['answer'][:100]}...")
        print(f"Difficulty: {card['difficulty_level']}")
        print()
