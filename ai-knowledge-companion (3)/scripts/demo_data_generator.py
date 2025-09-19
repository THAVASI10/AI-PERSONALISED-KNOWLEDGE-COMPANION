#!/usr/bin/env python3
"""
Demo data generator for AI Knowledge Companion
Creates sample study materials for testing and demonstration
"""

import asyncio
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api_client import APIClient
from utils import setup_logging

logger = setup_logging()

class DemoDataGenerator:
    def __init__(self):
        self.api_client = APIClient()
        
    async def generate_sample_content(self):
        """Generate sample educational content"""
        
        sample_contents = [
            {
                "filename": "machine_learning_basics.txt",
                "content": """
                Machine Learning Fundamentals
                
                Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
                
                Types of Machine Learning:
                1. Supervised Learning: Uses labeled training data
                2. Unsupervised Learning: Finds hidden patterns in unlabeled data  
                3. Reinforcement Learning: Learns through trial and error
                
                Key Algorithms:
                - Linear Regression: Predicts continuous values
                - Decision Trees: Creates decision rules
                - Neural Networks: Mimics brain structure
                - Support Vector Machines: Finds optimal boundaries
                
                Applications:
                - Image recognition
                - Natural language processing
                - Recommendation systems
                - Fraud detection
                """
            },
            {
                "filename": "data_structures.txt", 
                "content": """
                Data Structures and Algorithms
                
                Data structures are ways of organizing and storing data so that they can be accessed and worked with efficiently. They define the relationship between the data, and the operations that can be performed on the data.
                
                Common Data Structures:
                
                1. Arrays: Fixed-size sequential collection
                   - Time Complexity: O(1) access, O(n) search
                   - Use cases: When size is known, need fast access
                
                2. Linked Lists: Dynamic size, sequential access
                   - Time Complexity: O(n) access, O(1) insertion/deletion
                   - Use cases: Frequent insertions/deletions
                
                3. Stacks: Last In, First Out (LIFO)
                   - Operations: push, pop, peek
                   - Use cases: Function calls, undo operations
                
                4. Queues: First In, First Out (FIFO)
                   - Operations: enqueue, dequeue
                   - Use cases: Task scheduling, breadth-first search
                
                5. Trees: Hierarchical structure
                   - Binary trees, AVL trees, B-trees
                   - Use cases: Databases, file systems
                
                6. Hash Tables: Key-value pairs
                   - Average O(1) access time
                   - Use cases: Caches, databases
                """
            },
            {
                "filename": "python_programming.txt",
                "content": """
                Python Programming Essentials
                
                Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991.
                
                Key Features:
                - Easy to learn and use
                - Interpreted language
                - Object-oriented
                - Large standard library
                - Cross-platform compatibility
                
                Basic Syntax:
                
                Variables and Data Types:
                - Numbers: int, float, complex
                - Strings: text data
                - Lists: ordered, mutable collections
                - Tuples: ordered, immutable collections
                - Dictionaries: key-value pairs
                - Sets: unordered collections of unique elements
                
                Control Structures:
                - if/elif/else statements
                - for loops
                - while loops
                - try/except for error handling
                
                Functions:
                - def keyword to define functions
                - Parameters and arguments
                - Return statements
                - Lambda functions for simple operations
                
                Object-Oriented Programming:
                - Classes and objects
                - Inheritance
                - Encapsulation
                - Polymorphism
                
                Popular Libraries:
                - NumPy: Numerical computing
                - Pandas: Data manipulation
                - Matplotlib: Data visualization
                - Scikit-learn: Machine learning
                - Django/Flask: Web development
                """
            }
        ]
        
        logger.info("Generating demo content...")
        
        for content_data in sample_contents:
            # Create temporary file
            temp_file = Path(f"temp_{content_data['filename']}")
            temp_file.write_text(content_data['content'])
            
            try:
                # Upload to system
                result = await self.api_client.upload_file(str(temp_file))
                if result.get('success'):
                    logger.info(f"Successfully uploaded: {content_data['filename']}")
                else:
                    logger.error(f"Failed to upload: {content_data['filename']}")
            except Exception as e:
                logger.error(f"Error uploading {content_data['filename']}: {str(e)}")
            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
        
        logger.info("Demo content generation completed!")
    
    async def demonstrate_features(self):
        """Demonstrate key system features"""
        logger.info("Demonstrating system features...")
        
        # Test summarization
        sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
        """
        
        try:
            summary_result = await self.api_client.summarize_text(sample_text)
            logger.info(f"Summary generated: {summary_result.get('summary', 'N/A')}")
        except Exception as e:
            logger.error(f"Summarization demo failed: {str(e)}")
        
        # Test question generation
        try:
            questions_result = await self.api_client.generate_questions(sample_text)
            logger.info(f"Questions generated: {len(questions_result.get('questions', []))}")
        except Exception as e:
            logger.error(f"Question generation demo failed: {str(e)}")
        
        # Test conversational QA
        try:
            qa_result = await self.api_client.ask_question("What is artificial intelligence?")
            logger.info(f"QA response: {qa_result.get('answer', 'N/A')[:100]}...")
        except Exception as e:
            logger.error(f"QA demo failed: {str(e)}")

async def main():
    """Main demo execution"""
    generator = DemoDataGenerator()
    
    # Generate sample content
    await generator.generate_sample_content()
    
    # Demonstrate features
    await generator.demonstrate_features()
    
    logger.info("Demo completed! You can now explore the system with sample data.")

if __name__ == "__main__":
    asyncio.run(main())
