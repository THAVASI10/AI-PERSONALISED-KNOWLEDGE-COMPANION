#!/usr/bin/env python3
"""
Environment setup script for AI Knowledge Companion
Downloads required models and initializes the system
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logging
from database import init_db

logger = setup_logging()

def install_requirements():
    """Install Python requirements"""
    logger.info("Installing Python requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e.stderr}")
        return False

def download_models():
    """Download and cache required ML models"""
    logger.info("Downloading and caching ML models...")
    
    try:
        # Import and initialize models to trigger downloads
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
        
        # Download summarization model
        logger.info("Downloading summarization model...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Download sentence transformer
        logger.info("Downloading sentence transformer...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Download question generation model
        logger.info("Downloading question generation model...")
        qg_model = pipeline("text2text-generation", model="t5-small")
        
        logger.info("All models downloaded and cached successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {str(e)}")
        return False

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "data",
        "uploads", 
        "models",
        "logs",
        "knowledge_base"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def initialize_database():
    """Initialize the database"""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False

def main():
    """Main setup process"""
    logger.info("Starting environment setup for AI Knowledge Companion...")
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Setting up directories", setup_directories),
        ("Initializing database", initialize_database),
        ("Downloading ML models", download_models),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Step: {step_name}")
        if not step_func():
            logger.error(f"Setup failed at step: {step_name}")
            return 1
        logger.info(f"Completed: {step_name}")
    
    logger.info("Environment setup completed successfully!")
    logger.info("You can now run the application using: python run_app.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
