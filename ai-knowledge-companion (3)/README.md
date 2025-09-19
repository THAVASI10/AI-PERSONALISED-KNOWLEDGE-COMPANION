# 🧠 AI Personalized Knowledge Companion

A comprehensive Python-based AI system that transforms raw study materials into structured, intelligent, and interactive learning experiences using Machine Learning, Natural Language Processing, and Data Science methods.

## 🌟 Features

### 📤 Multi-Format Content Ingestion
- **PDF Processing**: Extract text from academic papers, textbooks, and documents
- **Image OCR**: Convert handwritten notes and printed materials to text
- **Audio Transcription**: Transform lectures and recordings into searchable text
- **Direct Text Input**: Process copy-pasted content instantly

### 🤖 AI-Powered Learning Tools
- **Smart Summarization**: Generate bullet points, paragraphs, or key points using BART/T5 models
- **Question Generation**: Create practice questions automatically using transformer models
- **Flashcard Creation**: Generate Q&A pairs with difficulty assessment
- **Conversational AI Tutor**: RAG-powered chatbot for interactive learning

### 🧠 Intelligent Analysis
- **Difficulty Prediction**: ML classifier determines content complexity
- **Topic Extraction**: Automatic identification of key subjects and themes
- **Semantic Search**: FAISS-powered vector search across your knowledge base
- **Performance Analytics**: Track learning progress and identify weak areas

### 🎯 Adaptive Learning System
- **Personalized Recommendations**: AI suggests next topics based on performance
- **Adaptive Content**: Difficulty adjusts to your learning pace
- **Progress Tracking**: XP points, badges, and streak counters
- **Performance Analytics**: Detailed insights into learning patterns

## 🛠 Technology Stack

### Core Framework
- **Backend**: FastAPI with async support
- **Frontend**: Streamlit for interactive UI
- **Database**: SQLite for user progress and content storage

### Machine Learning & NLP
- **Transformers**: HuggingFace BART, T5, Flan-T5 for text generation
- **Embeddings**: Sentence-BERT for semantic understanding
- **Classification**: Random Forest and Logistic Regression for difficulty prediction
- **Clustering**: K-Means for adaptive learning recommendations

### Data Processing
- **Text Processing**: spaCy and NLTK for linguistic analysis
- **PDF Extraction**: PyPDF2 and pdfminer.six
- **OCR**: EasyOCR for image text extraction
- **Audio**: OpenAI Whisper for speech-to-text

### Vector Database & Search
- **FAISS**: Facebook AI Similarity Search for semantic retrieval
- **RAG**: Retrieval-Augmented Generation for contextual Q&A

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for ML models)
- 2GB+ disk space

### Installation

1. **Clone and Setup**
\`\`\`bash
git clone <repository-url>
cd ai-knowledge-companion
chmod +x start.sh
./start.sh
\`\`\`

2. **Manual Setup** (if start.sh fails)
\`\`\`bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download ML models
python -m spacy download en_core_web_sm

# Launch application
python run_app.py
\`\`\`

### Access Points
- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 Usage Guide

### 1. Upload Content
- **Files**: Upload PDFs, images (JPG, PNG), or audio files (WAV, MP3)
- **Text**: Paste content directly into the text area
- **Batch Processing**: Upload multiple files simultaneously

### 2. Generate Learning Materials
- **Summaries**: Choose bullet points, paragraphs, or key points format
- **Questions**: Generate practice questions with customizable difficulty
- **Flashcards**: Create Q&A pairs for spaced repetition learning

### 3. Interactive Learning
- **AI Tutor**: Ask questions and get contextual answers from your materials
- **Adaptive Study**: Receive personalized recommendations based on performance
- **Progress Tracking**: Monitor XP, streaks, and learning analytics

### 4. Analytics & Insights
- **Performance Metrics**: Track accuracy and improvement over time
- **Topic Analysis**: Identify strong and weak subject areas
- **Study Patterns**: Visualize learning habits and trends

## 🏗 Architecture

### Data Flow
\`\`\`
Input (PDF/Image/Audio/Text) 
    ↓
Data Ingestion Pipeline
    ↓
Text Processing & Analysis
    ↓
Vector Embeddings (FAISS)
    ↓
ML Models (Summarization/QA/Classification)
    ↓
Adaptive Learning Engine
    ↓
User Interface (Streamlit)
\`\`\`

### Key Components

1. **Data Ingestion** (`data_ingestion.py`)
   - Multi-format file processing
   - OCR and speech recognition
   - Text cleaning and normalization

2. **Knowledge Storage** (`knowledge_storage.py`)
   - Vector embeddings with Sentence-BERT
   - FAISS indexing for semantic search
   - RAG system for contextual retrieval

3. **ML Models** (`ml_models.py`)
   - BART/T5 for summarization
   - T5 for question generation
   - Custom flashcard generation

4. **Adaptive Learning** (`adaptive_learning.py`)
   - Difficulty prediction with Random Forest
   - Performance analysis and clustering
   - Personalized recommendation engine

5. **API Backend** (`main.py`)
   - FastAPI with comprehensive endpoints
   - Async processing for file uploads
   - RESTful API design

6. **Frontend** (`streamlit_app.py`)
   - Interactive dashboard
   - Real-time chat interface
   - Analytics visualizations

## 📊 API Endpoints

### Content Management
- `POST /upload/file` - Upload and process files
- `POST /upload/text` - Process text input
- `GET /documents` - List processed documents
- `DELETE /documents/{name}` - Delete document

### AI Features
- `POST /summarize` - Generate text summaries
- `POST /generate-questions` - Create practice questions
- `POST /generate-flashcards` - Generate flashcard sets
- `POST /ask-question` - Conversational Q&A
- `POST /predict-difficulty` - Assess content difficulty

### Learning Analytics
- `GET /user/progress` - User progress and stats
- `POST /user/session` - Record study session
- `GET /user/recommendations` - Personalized suggestions
- `GET /stats/knowledge-base` - Knowledge base analytics

## 🎯 Use Cases

### Students
- Convert lecture notes into study materials
- Generate practice questions for exams
- Create flashcards from textbooks
- Get personalized study recommendations

### Researchers
- Summarize academic papers
- Extract key concepts from literature
- Organize research materials
- Generate research questions

### Professionals
- Process training materials
- Create knowledge bases from documents
- Generate summaries for reports
- Build interactive learning systems

## 🔧 Configuration

### Model Settings (`config.py`)
\`\`\`python
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
\`\`\`

### Performance Tuning
- Adjust chunk sizes for different document types
- Configure model batch sizes based on available memory
- Tune similarity thresholds for search accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers for pre-trained models
- Facebook AI for FAISS vector search
- OpenAI for Whisper speech recognition
- Streamlit for the interactive frontend
- FastAPI for the high-performance backend

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API documentation at `/redoc`

---

**Transform your learning experience with AI! 🚀**
