#!/bin/bash

# AI Knowledge Companion Startup Script
echo "🧠 AI Knowledge Companion - Setup and Launch"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download required models and data
echo "🤖 Setting up ML models..."
python -c "
import nltk
import spacy
from sentence_transformers import SentenceTransformer

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded')
except:
    print('⚠️ NLTK download failed')

# Download spaCy model
try:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True, capture_output=True)
    print('✅ spaCy model downloaded')
except:
    print('⚠️ spaCy model download failed')

# Load sentence transformer model
try:
    SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Sentence transformer model loaded')
except:
    print('⚠️ Sentence transformer model failed')
"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads/{pdfs,images,audio,processed}
mkdir -p models
mkdir -p data

# Initialize database
echo "🗄️ Initializing database..."
python -c "
from database import DatabaseManager
db = DatabaseManager()
print('✅ Database initialized')
"

# Launch the application
echo "🚀 Launching AI Knowledge Companion..."
python run_app.py
