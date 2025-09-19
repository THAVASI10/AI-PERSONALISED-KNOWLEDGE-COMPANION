import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import json
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

from database import DatabaseManager
from utils import TextProcessor, calculate_difficulty_level
from knowledge_storage import KnowledgeStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifficultyPredictor:
    """ML-based difficulty prediction for text content"""
    
    def __init__(self, model_path: str = "models/difficulty_model.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)
        
        self.text_processor = TextProcessor()
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_names = []
        
        # Try to load existing model
        self._load_model()
        
        # If no model exists, create and train a new one
        if self.classifier is None:
            self._create_and_train_model()
    
    def _extract_difficulty_features(self, text: str) -> Dict[str, float]:
        """Extract features for difficulty prediction"""
        
        # Get readability features
        readability_features = self.text_processor.calculate_readability_features(text)
        
        # Additional linguistic features
        sentences = self.text_processor.segment_sentences(text)
        words = text.split()
        
        # Vocabulary complexity
        unique_words = set(word.lower() for word in words)
        long_words = [word for word in words if len(word) > 6]
        complex_words = [word for word in words if len(word) > 8]
        
        # Sentence complexity
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        max_sentence_length = max([len(sent.split()) for sent in sentences], default=0)
        
        # Punctuation and structure
        punctuation_density = sum(1 for char in text if char in '.,;:!?') / len(text) if text else 0
        
        # Technical terms (simplified heuristic)
        technical_indicators = ['algorithm', 'analysis', 'theory', 'methodology', 'hypothesis', 
                              'implementation', 'optimization', 'framework', 'paradigm', 'synthesis']
        technical_word_count = sum(1 for word in words 
                                 if word.lower() in technical_indicators)
        
        features = {
            # Readability features
            'flesch_reading_ease': readability_features.get('flesch_reading_ease', 50),
            'flesch_kincaid_grade': readability_features.get('flesch_kincaid_grade', 8),
            'avg_sentence_length': readability_features.get('avg_sentence_length', 15),
            'avg_word_length': readability_features.get('avg_word_length', 5),
            'vocabulary_richness': readability_features.get('vocabulary_richness', 0.5),
            
            # Additional complexity features
            'long_word_ratio': len(long_words) / len(words) if words else 0,
            'complex_word_ratio': len(complex_words) / len(words) if words else 0,
            'max_sentence_length': max_sentence_length,
            'punctuation_density': punctuation_density,
            'technical_word_ratio': technical_word_count / len(words) if words else 0,
            
            # Text statistics
            'word_count': len(words),
            'sentence_count': len(sentences),
            'unique_word_ratio': len(unique_words) / len(words) if words else 0
        }
        
        return features
    
    def _create_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data for difficulty prediction"""
        
        # Sample texts with known difficulty levels
        training_samples = [
            # Easy texts
            ("The cat sat on the mat. It was a sunny day. The cat was happy.", "Easy"),
            ("Dogs are pets. They like to play. People love dogs very much.", "Easy"),
            ("Water is wet. Fire is hot. Ice is cold and hard to touch.", "Easy"),
            
            # Medium texts
            ("Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.", "Medium"),
            ("The scientific method involves forming hypotheses, conducting experiments, and analyzing results to draw conclusions.", "Medium"),
            ("Economic theories attempt to explain market behavior through mathematical models and statistical analysis.", "Medium"),
            
            # Hard texts
            ("The epistemological foundations of quantum mechanics necessitate a paradigmatic shift in our understanding of ontological reality.", "Hard"),
            ("Neuroplasticity research demonstrates that synaptic connections undergo morphological adaptations in response to environmental stimuli.", "Hard"),
            ("The phenomenological approach to consciousness studies emphasizes the subjective experiential aspects of cognitive processes.", "Hard")
        ]
        
        # Add more synthetic examples
        easy_templates = [
            "The {noun} is {adjective}. It {verb} every day. People {verb2} it.",
            "{Animal} live in {place}. They eat {food}. {Animal} are {adjective}.",
            "Today is {day}. The weather is {weather}. I feel {emotion}."
        ]
        
        medium_templates = [
            "The concept of {concept} involves {process} that {result}. This {method} is used in {field}.",
            "Research shows that {subject} can {action} through {mechanism}. Scientists {verb} this {phenomenon}.",
            "The relationship between {var1} and {var2} demonstrates {pattern} in {context}."
        ]
        
        hard_templates = [
            "The {complex_concept} paradigm necessitates {complex_process} through {methodology} that {complex_result}.",
            "Epistemological considerations regarding {abstract_concept} require {complex_analysis} of {theoretical_framework}.",
            "The phenomenological {approach} to {domain} emphasizes {complex_aspect} within {theoretical_context}."
        ]
        
        # Generate more examples (simplified for demo)
        for _ in range(20):
            training_samples.append(("This is a simple sentence. Easy to read. Very basic.", "Easy"))
        
        for _ in range(20):
            training_samples.append(("The analysis of complex systems requires understanding of multiple variables and their interactions.", "Medium"))
        
        for _ in range(20):
            training_samples.append(("The phenomenological hermeneutics of existential ontology necessitates epistemological considerations.", "Hard"))
        
        # Extract features and labels
        X = []
        y = []
        
        for text, difficulty in training_samples:
            features = self._extract_difficulty_features(text)
            feature_vector = list(features.values())
            X.append(feature_vector)
            y.append(difficulty)
        
        # Store feature names
        if training_samples:
            sample_features = self._extract_difficulty_features(training_samples[0][0])
            self.feature_names = list(sample_features.keys())
        
        return np.array(X), np.array(y)
    
    def _create_and_train_model(self):
        """Create and train the difficulty prediction model"""
        
        logger.info("Creating and training difficulty prediction model...")
        
        try:
            # Generate training data
            X, y = self._create_training_data()
            
            if len(X) == 0:
                logger.error("No training data available")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest classifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            
            self.classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Failed to create and train model: {e}")
            # Fallback to rule-based approach
            self.classifier = None
    
    def _save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load existing model"""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.classifier = model_data['classifier']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.classifier = None
    
    def predict_difficulty(self, text: str) -> Dict[str, Any]:
        """Predict difficulty level of text"""
        
        try:
            # Extract features
            features = self._extract_difficulty_features(text)
            
            if self.classifier is not None:
                # Use ML model
                feature_vector = np.array([list(features.values())])
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                # Predict
                prediction = self.classifier.predict(feature_vector_scaled)[0]
                probabilities = self.classifier.predict_proba(feature_vector_scaled)[0]
                
                # Get class probabilities
                classes = self.classifier.classes_
                prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
                
                return {
                    'predicted_difficulty': prediction,
                    'confidence': max(probabilities),
                    'probabilities': prob_dict,
                    'method': 'ml_model',
                    'features': features
                }
            else:
                # Fallback to rule-based approach
                rule_based_difficulty = calculate_difficulty_level(features)
                
                return {
                    'predicted_difficulty': rule_based_difficulty,
                    'confidence': 0.7,  # Default confidence for rule-based
                    'probabilities': {rule_based_difficulty: 0.7},
                    'method': 'rule_based',
                    'features': features
                }
                
        except Exception as e:
            logger.error(f"Difficulty prediction failed: {e}")
            return {
                'predicted_difficulty': 'Medium',
                'confidence': 0.5,
                'probabilities': {'Medium': 0.5},
                'method': 'fallback',
                'error': str(e)
            }

class AdaptiveLearningEngine:
    """Adaptive learning system that personalizes content based on user performance"""
    
    def __init__(self, knowledge_storage: KnowledgeStorage):
        self.knowledge_storage = knowledge_storage
        self.db_manager = DatabaseManager()
        self.difficulty_predictor = DifficultyPredictor()
        
        # Learning parameters
        self.performance_threshold = 0.7  # 70% performance threshold
        self.weak_area_threshold = 0.5   # 50% performance threshold for weak areas
        
    def analyze_user_performance(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Analyze user performance across different topics and difficulty levels"""
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Get study sessions
            cursor.execute('''
                SELECT session_type, performance_score, topics_covered, created_at
                FROM study_sessions 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
            ''', (user_id,))
            
            sessions = cursor.fetchall()
            conn.close()
            
            if not sessions:
                return {
                    'total_sessions': 0,
                    'avg_performance': 0.0,
                    'strong_areas': [],
                    'weak_areas': [],
                    'difficulty_performance': {},
                    'recent_trend': 'no_data'
                }
            
            # Analyze performance by topic
            topic_performance = {}
            difficulty_performance = {'Easy': [], 'Medium': [], 'Hard': []}
            recent_scores = []
            
            for session in sessions:
                session_type, score, topics_json, created_at = session
                
                # Parse topics
                try:
                    topics = json.loads(topics_json) if topics_json else []
                except:
                    topics = []
                
                # Track recent performance
                recent_scores.append(score)
                
                # Track topic performance
                for topic in topics:
                    if topic not in topic_performance:
                        topic_performance[topic] = []
                    topic_performance[topic].append(score)
            
            # Calculate averages
            avg_performance = np.mean(recent_scores) if recent_scores else 0.0
            
            # Identify strong and weak areas
            strong_areas = []
            weak_areas = []
            
            for topic, scores in topic_performance.items():
                avg_score = np.mean(scores)
                if avg_score >= self.performance_threshold:
                    strong_areas.append({'topic': topic, 'performance': avg_score})
                elif avg_score <= self.weak_area_threshold:
                    weak_areas.append({'topic': topic, 'performance': avg_score})
            
            # Sort by performance
            strong_areas.sort(key=lambda x: x['performance'], reverse=True)
            weak_areas.sort(key=lambda x: x['performance'])
            
            # Analyze recent trend
            if len(recent_scores) >= 5:
                recent_trend = self._analyze_performance_trend(recent_scores[-5:])
            else:
                recent_trend = 'insufficient_data'
            
            return {
                'total_sessions': len(sessions),
                'avg_performance': avg_performance,
                'strong_areas': strong_areas[:5],  # Top 5 strong areas
                'weak_areas': weak_areas[:5],     # Top 5 weak areas
                'difficulty_performance': difficulty_performance,
                'recent_trend': recent_trend,
                'topic_performance': topic_performance
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {
                'total_sessions': 0,
                'avg_performance': 0.0,
                'strong_areas': [],
                'weak_areas': [],
                'difficulty_performance': {},
                'recent_trend': 'error',
                'error': str(e)
            }
    
    def _analyze_performance_trend(self, recent_scores: List[float]) -> str:
        """Analyze if performance is improving, declining, or stable"""
        
        if len(recent_scores) < 3:
            return 'insufficient_data'
        
        # Simple trend analysis
        first_half = np.mean(recent_scores[:len(recent_scores)//2])
        second_half = np.mean(recent_scores[len(recent_scores)//2:])
        
        difference = second_half - first_half
        
        if difference > 0.1:
            return 'improving'
        elif difference < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def generate_personalized_recommendations(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Generate personalized learning recommendations"""
        
        try:
            # Analyze current performance
            performance_analysis = self.analyze_user_performance(user_id)
            
            # Get knowledge base statistics
            kb_stats = self.knowledge_storage.get_knowledge_stats()
            
            recommendations = {
                'next_topics': [],
                'difficulty_adjustment': 'maintain',
                'study_focus': 'balanced',
                'recommended_activities': [],
                'estimated_study_time': 30  # minutes
            }
            
            # Recommend topics based on weak areas
            weak_areas = performance_analysis['weak_areas']
            if weak_areas:
                recommendations['study_focus'] = 'remediation'
                recommendations['next_topics'] = [area['topic'] for area in weak_areas[:3]]
                recommendations['difficulty_adjustment'] = 'decrease'
                recommendations['recommended_activities'] = [
                    'Review flashcards for weak topics',
                    'Practice questions on challenging areas',
                    'Read summaries of difficult concepts'
                ]
                recommendations['estimated_study_time'] = 45
            
            # If no weak areas, focus on new topics or advanced content
            elif performance_analysis['avg_performance'] > self.performance_threshold:
                available_topics = list(kb_stats.get('topic_distribution', {}).keys())
                studied_topics = set(area['topic'] for area in performance_analysis['strong_areas'])
                new_topics = [topic for topic in available_topics if topic not in studied_topics]
                
                recommendations['study_focus'] = 'exploration'
                recommendations['next_topics'] = new_topics[:3]
                recommendations['difficulty_adjustment'] = 'increase'
                recommendations['recommended_activities'] = [
                    'Explore new topics',
                    'Try advanced questions',
                    'Generate comprehensive summaries'
                ]
                recommendations['estimated_study_time'] = 35
            
            # Balanced approach for average performance
            else:
                recommendations['study_focus'] = 'balanced'
                recommendations['recommended_activities'] = [
                    'Mix of review and new content',
                    'Practice flashcards',
                    'Interactive Q&A sessions'
                ]
            
            # Add specific recommendations based on trend
            trend = performance_analysis['recent_trend']
            if trend == 'declining':
                recommendations['recommended_activities'].insert(0, 'Take a break and review fundamentals')
                recommendations['difficulty_adjustment'] = 'decrease'
            elif trend == 'improving':
                recommendations['recommended_activities'].append('Challenge yourself with harder content')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                'next_topics': [],
                'difficulty_adjustment': 'maintain',
                'study_focus': 'balanced',
                'recommended_activities': ['Continue regular study routine'],
                'estimated_study_time': 30,
                'error': str(e)
            }
    
    def update_performance_tracking(self, session_data: Dict[str, Any], user_id: str = "default_user"):
        """Update performance tracking with new session data"""
        
        try:
            # Save session to database
            self.db_manager.save_study_session(
                document_name=session_data.get('document_name', 'unknown'),
                session_type=session_data.get('session_type', 'general'),
                performance_score=session_data.get('performance_score', 0.0),
                time_spent=session_data.get('time_spent', 0),
                topics=session_data.get('topics', [])
            )
            
            # Update user progress (XP and badges)
            xp_gained = self._calculate_xp_gain(session_data)
            badge = self._check_for_badges(session_data, user_id)
            
            self.db_manager.update_user_progress(xp_gained, badge)
            
            logger.info(f"Performance tracking updated for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update performance tracking: {e}")
    
    def _calculate_xp_gain(self, session_data: Dict[str, Any]) -> int:
        """Calculate XP gain based on session performance"""
        
        base_xp = 10
        performance_score = session_data.get('performance_score', 0.0)
        time_spent = session_data.get('time_spent', 0)  # in minutes
        session_type = session_data.get('session_type', 'general')
        
        # Base XP calculation
        xp = base_xp
        
        # Performance bonus
        if performance_score >= 0.9:
            xp += 15  # Excellent performance
        elif performance_score >= 0.7:
            xp += 10  # Good performance
        elif performance_score >= 0.5:
            xp += 5   # Average performance
        
        # Time bonus (encourage longer study sessions)
        if time_spent >= 30:
            xp += 10
        elif time_spent >= 15:
            xp += 5
        
        # Session type bonus
        if session_type == 'flashcard_review':
            xp += 5
        elif session_type == 'question_practice':
            xp += 8
        elif session_type == 'comprehensive_study':
            xp += 12
        
        return max(xp, 5)  # Minimum 5 XP per session
    
    def _check_for_badges(self, session_data: Dict[str, Any], user_id: str) -> Optional[str]:
        """Check if user earned any badges"""
        
        # Get current user progress
        progress = self.db_manager.get_user_progress()
        current_badges = progress.get('badges', [])
        
        # Badge criteria
        performance_score = session_data.get('performance_score', 0.0)
        
        # Perfect Score Badge
        if performance_score >= 1.0 and 'perfect_score' not in current_badges:
            return 'perfect_score'
        
        # High Performer Badge
        if performance_score >= 0.9 and 'high_performer' not in current_badges:
            return 'high_performer'
        
        # Streak badges (based on current streak)
        current_streak = progress.get('current_streak', 0)
        if current_streak >= 7 and 'week_streak' not in current_badges:
            return 'week_streak'
        elif current_streak >= 30 and 'month_streak' not in current_badges:
            return 'month_streak'
        
        # XP milestone badges
        total_xp = progress.get('total_xp', 0)
        if total_xp >= 1000 and 'xp_master' not in current_badges:
            return 'xp_master'
        elif total_xp >= 500 and 'xp_expert' not in current_badges:
            return 'xp_expert'
        elif total_xp >= 100 and 'xp_novice' not in current_badges:
            return 'xp_novice'
        
        return None
    
    def get_adaptive_content(self, user_id: str = "default_user", 
                           content_type: str = "mixed") -> Dict[str, Any]:
        """Get adaptive content based on user performance"""
        
        try:
            # Get recommendations
            recommendations = self.generate_personalized_recommendations(user_id)
            
            # Get content based on recommendations
            recommended_topics = recommendations['next_topics']
            difficulty_adjustment = recommendations['difficulty_adjustment']
            
            # Determine target difficulty
            if difficulty_adjustment == 'increase':
                target_difficulties = ['Medium', 'Hard']
            elif difficulty_adjustment == 'decrease':
                target_difficulties = ['Easy', 'Medium']
            else:
                target_difficulties = ['Easy', 'Medium', 'Hard']
            
            # Get content from knowledge base
            adaptive_content = {
                'flashcards': [],
                'questions': [],
                'summaries': [],
                'recommendations': recommendations
            }
            
            # Get flashcards
            if content_type in ['mixed', 'flashcards']:
                flashcards = self.db_manager.get_flashcards()
                
                # Filter by topics and difficulty
                filtered_flashcards = []
                for card in flashcards:
                    if (not recommended_topics or 
                        any(topic in str(card.get('topics', [])) for topic in recommended_topics)):
                        if card.get('difficulty_level') in target_difficulties:
                            filtered_flashcards.append(card)
                
                adaptive_content['flashcards'] = filtered_flashcards[:10]
            
            # Get random chunks for questions/summaries
            if content_type in ['mixed', 'questions', 'summaries']:
                for difficulty in target_difficulties:
                    chunks = self.knowledge_storage.get_random_chunks(
                        count=3, difficulty_level=difficulty
                    )
                    
                    for chunk in chunks:
                        if recommended_topics:
                            chunk_topics = chunk.get('topics', [])
                            if any(topic in chunk_topics for topic in recommended_topics):
                                if content_type in ['mixed', 'summaries']:
                                    adaptive_content['summaries'].append({
                                        'text': chunk['text'][:200] + '...',
                                        'difficulty': chunk['difficulty_level'],
                                        'topics': chunk['topics']
                                    })
            
            return adaptive_content
            
        except Exception as e:
            logger.error(f"Adaptive content generation failed: {e}")
            return {
                'flashcards': [],
                'questions': [],
                'summaries': [],
                'recommendations': {},
                'error': str(e)
            }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Adaptive Learning System...")
    
    # Test difficulty prediction
    print("\n1. Testing Difficulty Prediction:")
    predictor = DifficultyPredictor()
    
    test_texts = [
        "The cat is happy. It plays all day.",
        "Machine learning algorithms can automatically learn patterns from data to make predictions.",
        "The phenomenological hermeneutics of existential ontology necessitates epistemological considerations of consciousness."
    ]
    
    for i, text in enumerate(test_texts, 1):
        result = predictor.predict_difficulty(text)
        print(f"Text {i}: {result['predicted_difficulty']} (confidence: {result['confidence']:.2f})")
    
    # Test adaptive learning (requires knowledge storage)
    print("\n2. Testing Adaptive Learning:")
    try:
        from knowledge_storage import KnowledgeStorage
        knowledge_storage = KnowledgeStorage()
        adaptive_engine = AdaptiveLearningEngine(knowledge_storage)
        
        # Simulate some performance data
        session_data = {
            'document_name': 'test_document',
            'session_type': 'flashcard_review',
            'performance_score': 0.85,
            'time_spent': 25,
            'topics': ['machine learning', 'data science']
        }
        
        adaptive_engine.update_performance_tracking(session_data)
        
        # Get recommendations
        recommendations = adaptive_engine.generate_personalized_recommendations()
        print(f"Study focus: {recommendations['study_focus']}")
        print(f"Recommended activities: {recommendations['recommended_activities'][:2]}")
        print(f"Estimated study time: {recommendations['estimated_study_time']} minutes")
        
    except Exception as e:
        print(f"Adaptive learning test failed: {e}")
