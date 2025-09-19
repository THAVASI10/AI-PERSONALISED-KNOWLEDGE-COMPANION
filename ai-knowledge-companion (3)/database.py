import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json

class DatabaseManager:
    def __init__(self, db_path: str = "knowledge_companion.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default_user',
                total_xp INTEGER DEFAULT 0,
                current_streak INTEGER DEFAULT 0,
                badges TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Study sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default_user',
                document_name TEXT,
                session_type TEXT,
                performance_score REAL,
                time_spent INTEGER,
                topics_covered TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Knowledge chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT,
                chunk_text TEXT,
                chunk_embedding BLOB,
                difficulty_level TEXT,
                topics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Flashcards table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT,
                question TEXT,
                answer TEXT,
                difficulty_level TEXT,
                user_performance REAL DEFAULT 0.0,
                times_reviewed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Initialize default user if not exists
        self.init_default_user()
    
    def init_default_user(self):
        """Initialize default user progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM user_progress WHERE user_id = ?', ('default_user',))
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO user_progress (user_id, total_xp, current_streak, badges)
                VALUES (?, ?, ?, ?)
            ''', ('default_user', 0, 0, '[]'))
            conn.commit()
        
        conn.close()
    
    def update_user_progress(self, xp_gained: int, badge: Optional[str] = None):
        """Update user progress with XP and badges"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current progress
        cursor.execute('SELECT total_xp, current_streak, badges FROM user_progress WHERE user_id = ?', 
                      ('default_user',))
        result = cursor.fetchone()
        
        if result:
            current_xp, current_streak, badges_json = result
            badges = json.loads(badges_json)
            
            # Update XP and streak
            new_xp = current_xp + xp_gained
            new_streak = current_streak + 1
            
            # Add badge if provided
            if badge and badge not in badges:
                badges.append(badge)
            
            cursor.execute('''
                UPDATE user_progress 
                SET total_xp = ?, current_streak = ?, badges = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (new_xp, new_streak, json.dumps(badges), 'default_user'))
            
            conn.commit()
        
        conn.close()
    
    def get_user_progress(self) -> Dict:
        """Get current user progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT total_xp, current_streak, badges FROM user_progress WHERE user_id = ?', 
                      ('default_user',))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {
                'total_xp': result[0],
                'current_streak': result[1],
                'badges': json.loads(result[2])
            }
        return {'total_xp': 0, 'current_streak': 0, 'badges': []}
    
    def save_study_session(self, document_name: str, session_type: str, 
                          performance_score: float, time_spent: int, topics: List[str]):
        """Save study session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO study_sessions (user_id, document_name, session_type, 
                                      performance_score, time_spent, topics_covered)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('default_user', document_name, session_type, performance_score, 
              time_spent, json.dumps(topics)))
        
        conn.commit()
        conn.close()
    
    def save_flashcard(self, document_name: str, question: str, answer: str, 
                      difficulty_level: str):
        """Save generated flashcard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO flashcards (document_name, question, answer, difficulty_level)
            VALUES (?, ?, ?, ?)
        ''', (document_name, question, answer, difficulty_level))
        
        conn.commit()
        conn.close()
    
    def get_flashcards(self, document_name: Optional[str] = None) -> List[Dict]:
        """Get flashcards for review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if document_name:
            cursor.execute('''
                SELECT question, answer, difficulty_level, user_performance, times_reviewed
                FROM flashcards WHERE document_name = ?
                ORDER BY user_performance ASC, times_reviewed ASC
            ''', (document_name,))
        else:
            cursor.execute('''
                SELECT question, answer, difficulty_level, user_performance, times_reviewed
                FROM flashcards
                ORDER BY user_performance ASC, times_reviewed ASC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'question': row[0],
                'answer': row[1],
                'difficulty_level': row[2],
                'user_performance': row[3],
                'times_reviewed': row[4]
            }
            for row in results
        ]
