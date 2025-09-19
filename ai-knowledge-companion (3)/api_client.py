import requests
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AIKnowledgeCompanionClient:
    """Client for interacting with the AI Knowledge Companion API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._make_request("GET", "/health")
    
    # File upload methods
    def upload_file(self, file_path: str, document_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload and process a file"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            params = {}
            if document_name:
                params['document_name'] = document_name
            
            return self._make_request("POST", "/upload/file", files=files, params=params)
    
    def upload_text(self, text: str, document_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload and process text"""
        data = {"text": text}
        if document_name:
            data["document_name"] = document_name
        
        return self._make_request("POST", "/upload/text", json=data)
    
    # ML model methods
    def summarize_text(self, text: str, max_length: int = 150, 
                      min_length: int = 50, summary_type: str = "bullet") -> Dict[str, Any]:
        """Generate text summary"""
        data = {
            "text": text,
            "max_length": max_length,
            "min_length": min_length,
            "summary_type": summary_type
        }
        return self._make_request("POST", "/summarize", json=data)
    
    def generate_questions(self, text: str, num_questions: int = 5, 
                          question_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate questions from text"""
        data = {
            "text": text,
            "num_questions": num_questions
        }
        if question_types:
            data["question_types"] = question_types
        
        return self._make_request("POST", "/generate-questions", json=data)
    
    def generate_flashcards(self, text: str, document_name: str, 
                           num_flashcards: int = 5) -> Dict[str, Any]:
        """Generate flashcards from text"""
        data = {
            "text": text,
            "document_name": document_name,
            "num_flashcards": num_flashcards
        }
        return self._make_request("POST", "/generate-flashcards", json=data)
    
    def ask_question(self, question: str, use_context: bool = True) -> Dict[str, Any]:
        """Ask a question using conversational QA"""
        data = {
            "question": question,
            "use_context": use_context
        }
        return self._make_request("POST", "/ask-question", json=data)
    
    def predict_difficulty(self, text: str) -> Dict[str, Any]:
        """Predict difficulty level of text"""
        data = {"text": text}
        return self._make_request("POST", "/predict-difficulty", json=data)
    
    # Knowledge base methods
    def search_knowledge_base(self, query: str, top_k: int = 5, 
                             filter_difficulty: Optional[str] = None,
                             filter_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search the knowledge base"""
        data = {
            "query": query,
            "top_k": top_k
        }
        if filter_difficulty:
            data["filter_difficulty"] = filter_difficulty
        if filter_topics:
            data["filter_topics"] = filter_topics
        
        return self._make_request("POST", "/search", json=data)
    
    def list_documents(self) -> Dict[str, Any]:
        """List all processed documents"""
        return self._make_request("GET", "/documents")
    
    def delete_document(self, document_name: str) -> Dict[str, Any]:
        """Delete a document"""
        return self._make_request("DELETE", f"/documents/{document_name}")
    
    # User progress methods
    def get_user_progress(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Get user progress"""
        return self._make_request("GET", f"/user/progress?user_id={user_id}")
    
    def record_study_session(self, session_data: Dict[str, Any], 
                           user_id: str = "default_user") -> Dict[str, Any]:
        """Record a study session"""
        return self._make_request("POST", f"/user/session?user_id={user_id}", json=session_data)
    
    def get_recommendations(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Get personalized recommendations"""
        return self._make_request("GET", f"/user/recommendations?user_id={user_id}")
    
    def get_adaptive_content(self, user_id: str = "default_user", 
                           content_type: str = "mixed") -> Dict[str, Any]:
        """Get adaptive content"""
        return self._make_request("GET", f"/user/adaptive-content?user_id={user_id}&content_type={content_type}")
    
    # Statistics methods
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self._make_request("GET", "/stats/knowledge-base")
    
    def get_flashcards(self, document_name: Optional[str] = None) -> Dict[str, Any]:
        """Get flashcards"""
        endpoint = "/flashcards"
        if document_name:
            endpoint += f"?document_name={document_name}"
        return self._make_request("GET", endpoint)
    
    # Conversation methods
    def get_conversation_history(self) -> Dict[str, Any]:
        """Get conversation history"""
        return self._make_request("GET", "/conversation/history")
    
    def clear_conversation_history(self) -> Dict[str, Any]:
        """Clear conversation history"""
        return self._make_request("DELETE", "/conversation/history")

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = AIKnowledgeCompanionClient()
    
    # Test health check
    print("Testing API client...")
    health = client.health_check()
    print(f"Health check: {health}")
    
    # Test text upload and processing
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """
    
    print("\nUploading sample text...")
    upload_result = client.upload_text(sample_text, "ai_basics")
    print(f"Upload result: {upload_result}")
    
    if upload_result.get('success'):
        # Test summarization
        print("\nTesting summarization...")
        summary = client.summarize_text(sample_text)
        print(f"Summary: {summary}")
        
        # Test question generation
        print("\nTesting question generation...")
        questions = client.generate_questions(sample_text, num_questions=3)
        print(f"Questions: {questions}")
        
        # Test search
        print("\nTesting search...")
        search_results = client.search_knowledge_base("What is artificial intelligence?")
        print(f"Search results: {search_results}")
        
        # Test difficulty prediction
        print("\nTesting difficulty prediction...")
        difficulty = client.predict_difficulty(sample_text)
        print(f"Difficulty: {difficulty}")
