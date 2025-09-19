#!/usr/bin/env python3
"""
Comprehensive system test for AI Knowledge Companion
Tests all components with real data processing
"""

import asyncio
import os
import sys
import requests
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api_client import APIClient
from database import init_db
from utils import setup_logging

logger = setup_logging()

class SystemTester:
    def __init__(self):
        self.api_client = APIClient()
        self.base_url = "http://localhost:8000"
        
    def wait_for_server(self, max_attempts=30):
        """Wait for FastAPI server to be ready"""
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                logger.info(f"Waiting for server... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
        return False
    
    async def test_file_upload(self):
        """Test file upload and processing"""
        logger.info("Testing file upload...")
        
        # Create a test text file
        test_content = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. There are three main types of machine learning:
        
        1. Supervised Learning: Uses labeled training data to learn a mapping function from inputs to outputs.
        2. Unsupervised Learning: Finds hidden patterns in data without labeled examples.
        3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties.
        
        Key concepts include:
        - Feature engineering: Selecting and transforming variables for models
        - Model validation: Techniques like cross-validation to assess performance
        - Overfitting: When a model performs well on training data but poorly on new data
        - Bias-variance tradeoff: Balancing model complexity and generalization
        """
        
        test_file = Path("test_ml_content.txt")
        test_file.write_text(test_content)
        
        try:
            result = await self.api_client.upload_file(str(test_file))
            logger.info(f"File upload result: {result}")
            return result.get('success', False)
        finally:
            if test_file.exists():
                test_file.unlink()
    
    async def test_summarization(self):
        """Test text summarization"""
        logger.info("Testing summarization...")
        
        text = """
        Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain, consisting of interconnected nodes (neurons) organized in layers. The "deep" in deep learning refers to the number of layers in the network, typically ranging from three to hundreds of layers.
        
        The key advantage of deep learning is its ability to automatically learn hierarchical representations of data. Lower layers learn simple features like edges in images, while higher layers combine these to recognize complex patterns like faces or objects. This automatic feature learning eliminates the need for manual feature engineering, which was a major bottleneck in traditional machine learning approaches.
        """
        
        result = await self.api_client.summarize_text(text)
        logger.info(f"Summarization result: {result}")
        return 'summary' in result
    
    async def test_question_generation(self):
        """Test question generation"""
        logger.info("Testing question generation...")
        
        text = """
        Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        """
        
        result = await self.api_client.generate_questions(text)
        logger.info(f"Question generation result: {result}")
        return 'questions' in result and len(result['questions']) > 0
    
    async def test_flashcard_generation(self):
        """Test flashcard generation"""
        logger.info("Testing flashcard generation...")
        
        text = """
        Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts and involves two main stages: light-dependent reactions and light-independent reactions (Calvin cycle). The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
        """
        
        result = await self.api_client.generate_flashcards(text)
        logger.info(f"Flashcard generation result: {result}")
        return 'flashcards' in result and len(result['flashcards']) > 0
    
    async def test_conversational_qa(self):
        """Test conversational QA"""
        logger.info("Testing conversational QA...")
        
        # First add some content to the knowledge base
        await self.test_file_upload()
        
        result = await self.api_client.ask_question("What is machine learning?")
        logger.info(f"QA result: {result}")
        return 'answer' in result
    
    async def test_search(self):
        """Test knowledge base search"""
        logger.info("Testing search...")
        
        result = await self.api_client.search_knowledge("machine learning types")
        logger.info(f"Search result: {result}")
        return 'results' in result
    
    async def run_all_tests(self):
        """Run all system tests"""
        logger.info("Starting comprehensive system tests...")
        
        if not self.wait_for_server():
            logger.error("Server not available. Please start the server first.")
            return False
        
        tests = [
            ("File Upload", self.test_file_upload),
            ("Summarization", self.test_summarization),
            ("Question Generation", self.test_question_generation),
            ("Flashcard Generation", self.test_flashcard_generation),
            ("Conversational QA", self.test_conversational_qa),
            ("Search", self.test_search),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
                status = "PASS" if result else "FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                results[test_name] = False
                logger.error(f"{test_name}: FAIL - {str(e)}")
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        logger.info(f"\nTest Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("All tests passed! System is working correctly.")
        else:
            logger.warning("Some tests failed. Check logs for details.")
        
        return passed == total

async def main():
    """Main test execution"""
    # Initialize database
    init_db()
    
    # Run tests
    tester = SystemTester()
    success = await tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
