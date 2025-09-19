import os
import io
import tempfile
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
from config import settings
from data_ingestion import DataIngestionPipeline
from file_handler import FileHandler
from knowledge_storage import KnowledgeStorage
from ml_models import SummarizationModel, QuestionGenerationModel, FlashcardGenerator, ConversationalQA
from adaptive_learning import DifficultyPredictor, AdaptiveLearningEngine
from database import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Personalized Knowledge Companion",
    description="A comprehensive AI-powered learning assistant that transforms study materials into interactive learning experiences",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
data_pipeline = None
file_handler = None
knowledge_storage = None
summarization_model = None
question_model = None
flashcard_generator = None
conversational_qa = None
difficulty_predictor = None
adaptive_engine = None
db_manager = None

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="Input text to process")
    document_name: Optional[str] = Field(None, description="Name for the document")

class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(150, description="Maximum summary length")
    min_length: int = Field(50, description="Minimum summary length")
    summary_type: str = Field("bullet", description="Type of summary: bullet, paragraph, key_points")

class QuestionGenerationRequest(BaseModel):
    text: str = Field(..., description="Text to generate questions from")
    num_questions: int = Field(5, description="Number of questions to generate")
    question_types: Optional[List[str]] = Field(None, description="Types of questions: what, how, why, when, where")

class QARequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    use_context: bool = Field(True, description="Whether to use knowledge base context")

class FlashcardRequest(BaseModel):
    text: str = Field(..., description="Text to generate flashcards from")
    document_name: str = Field(..., description="Document name")
    num_flashcards: int = Field(5, description="Number of flashcards to generate")

class StudySessionData(BaseModel):
    document_name: str
    session_type: str
    performance_score: float = Field(..., ge=0.0, le=1.0)
    time_spent: int = Field(..., description="Time spent in minutes")
    topics: List[str] = Field(default_factory=list)

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return")
    filter_difficulty: Optional[str] = Field(None, description="Filter by difficulty: Easy, Medium, Hard")
    filter_topics: Optional[List[str]] = Field(None, description="Filter by topics")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global data_pipeline, file_handler, knowledge_storage, summarization_model
    global question_model, flashcard_generator, conversational_qa, difficulty_predictor
    global adaptive_engine, db_manager
    
    logger.info("Initializing AI Knowledge Companion...")
    
    try:
        # Initialize core components
        db_manager = DatabaseManager()
        data_pipeline = DataIngestionPipeline()
        file_handler = FileHandler()
        knowledge_storage = KnowledgeStorage()
        
        # Initialize ML models
        summarization_model = SummarizationModel()
        question_model = QuestionGenerationModel()
        flashcard_generator = FlashcardGenerator(question_model)
        conversational_qa = ConversationalQA(knowledge_storage)
        
        # Initialize adaptive learning
        difficulty_predictor = DifficultyPredictor()
        adaptive_engine = AdaptiveLearningEngine(knowledge_storage)
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# File upload and processing endpoints
@app.post("/upload/file")
async def upload_file(
    file: UploadFile = File(...),
    document_name: Optional[str] = None
):
    """Upload and process a file (PDF, image, or audio)"""
    try:
        # Validate file type
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.wav', '.mp3', '.m4a', '.flac']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {supported_extensions}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Determine file type
        if file_extension == '.pdf':
            file_type = 'pdf'
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            file_type = 'image'
        elif file_extension in ['.wav', '.mp3', '.m4a', '.flac']:
            file_type = 'audio'
        else:
            file_type = 'unknown'
        
        # Save file
        file_path = file_handler.save_uploaded_file(file_content, file.filename, file_type)
        
        # Process file
        if document_name is None:
            document_name = os.path.splitext(file.filename)[0]
        
        processed_data = file_handler.process_uploaded_file(file_path, document_name)
        
        if not processed_data['success']:
            raise HTTPException(status_code=400, detail=processed_data['error'])
        
        # Add to knowledge base
        knowledge_storage.add_document(processed_data)
        
        return {
            "success": True,
            "message": "File processed successfully",
            "document_name": document_name,
            "file_type": file_type,
            "word_count": processed_data.get('word_count', 0),
            "difficulty_level": processed_data.get('difficulty_level', 'Unknown'),
            "topics": processed_data.get('topics', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/text")
async def upload_text(text_input: TextInput):
    """Process direct text input"""
    try:
        document_name = text_input.document_name or f"text_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process text
        processed_data = file_handler._process_extracted_text(text_input.text, document_name)
        processed_data['success'] = True
        
        # Add to knowledge base
        knowledge_storage.add_document(processed_data)
        
        return {
            "success": True,
            "message": "Text processed successfully",
            "document_name": document_name,
            "word_count": processed_data.get('word_count', 0),
            "difficulty_level": processed_data.get('difficulty_level', 'Unknown'),
            "topics": processed_data.get('topics', [])
        }
        
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML model endpoints
@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    """Generate summary of text"""
    try:
        result = summarization_model.summarize_text(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            summary_type=request.summary_type
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-questions")
async def generate_questions(request: QuestionGenerationRequest):
    """Generate questions from text"""
    try:
        questions = question_model.generate_questions(
            text=request.text,
            num_questions=request.num_questions,
            question_types=request.question_types
        )
        
        return {
            "success": True,
            "questions": questions,
            "count": len(questions)
        }
        
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-flashcards")
async def generate_flashcards(request: FlashcardRequest):
    """Generate flashcards from text"""
    try:
        flashcards = flashcard_generator.generate_flashcards(
            text=request.text,
            document_name=request.document_name,
            num_flashcards=request.num_flashcards
        )
        
        return {
            "success": True,
            "flashcards": flashcards,
            "count": len(flashcards)
        }
        
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QARequest):
    """Ask a question using conversational QA"""
    try:
        result = conversational_qa.answer_question(
            question=request.question,
            use_context=request.use_context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-difficulty")
async def predict_difficulty(text_input: TextInput):
    """Predict difficulty level of text"""
    try:
        result = difficulty_predictor.predict_difficulty(text_input.text)
        return result
        
    except Exception as e:
        logger.error(f"Difficulty prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge base endpoints
@app.post("/search")
async def search_knowledge_base(request: SearchRequest):
    """Search the knowledge base"""
    try:
        results = knowledge_storage.search_similar_chunks(
            query=request.query,
            top_k=request.top_k,
            filter_difficulty=request.filter_difficulty,
            filter_topics=request.filter_topics
        )
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = file_handler.list_processed_documents()
        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document from the knowledge base"""
    try:
        # Delete from knowledge storage
        kb_success = knowledge_storage.delete_document(document_name)
        
        # Delete from file handler
        fh_success = file_handler.delete_document(document_name)
        
        return {
            "success": kb_success and fh_success,
            "message": f"Document '{document_name}' deleted successfully" if (kb_success and fh_success) else "Partial deletion occurred"
        }
        
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Adaptive learning endpoints
@app.get("/user/progress")
async def get_user_progress(user_id: str = "default_user"):
    """Get user progress and statistics"""
    try:
        progress = db_manager.get_user_progress()
        performance_analysis = adaptive_engine.analyze_user_performance(user_id)
        
        return {
            "success": True,
            "progress": progress,
            "performance_analysis": performance_analysis
        }
        
    except Exception as e:
        logger.error(f"Progress retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/session")
async def record_study_session(session_data: StudySessionData, user_id: str = "default_user"):
    """Record a study session"""
    try:
        adaptive_engine.update_performance_tracking(session_data.dict(), user_id)
        
        return {
            "success": True,
            "message": "Study session recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Session recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/recommendations")
async def get_recommendations(user_id: str = "default_user"):
    """Get personalized learning recommendations"""
    try:
        recommendations = adaptive_engine.generate_personalized_recommendations(user_id)
        
        return {
            "success": True,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/adaptive-content")
async def get_adaptive_content(
    user_id: str = "default_user",
    content_type: str = "mixed"
):
    """Get adaptive content based on user performance"""
    try:
        content = adaptive_engine.get_adaptive_content(user_id, content_type)
        
        return {
            "success": True,
            "content": content
        }
        
    except Exception as e:
        logger.error(f"Adaptive content generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics and analytics endpoints
@app.get("/stats/knowledge-base")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = knowledge_storage.get_knowledge_stats()
        file_stats = file_handler.get_file_stats()
        
        return {
            "success": True,
            "knowledge_base": stats,
            "files": file_stats
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flashcards")
async def get_flashcards(document_name: Optional[str] = None):
    """Get flashcards for review"""
    try:
        flashcards = db_manager.get_flashcards(document_name)
        
        return {
            "success": True,
            "flashcards": flashcards,
            "count": len(flashcards)
        }
        
    except Exception as e:
        logger.error(f"Flashcard retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversation history endpoints
@app.get("/conversation/history")
async def get_conversation_history():
    """Get conversation history"""
    try:
        history = conversational_qa.get_conversation_history()
        
        return {
            "success": True,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Conversation history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/history")
async def clear_conversation_history():
    """Clear conversation history"""
    try:
        conversational_qa.clear_conversation_history()
        
        return {
            "success": True,
            "message": "Conversation history cleared"
        }
        
    except Exception as e:
        logger.error(f"Conversation history clearing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
