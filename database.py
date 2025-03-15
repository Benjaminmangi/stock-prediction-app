import sqlite3
import json
from datetime import datetime
import os
from pathlib import Path
import hashlib
import secrets

class Database:
    def __init__(self):
        try:
            # Create data directory if it doesn't exist
            self.data_dir = Path('data')
            self.data_dir.mkdir(exist_ok=True)
            print(f"Data directory path: {self.data_dir.absolute()}")
            
            # Database file path
            self.db_file = self.data_dir / 'stock_predictions.db'
            print(f"Database file path: {self.db_file.absolute()}")
            
            # Check if database exists
            if not self.db_file.exists():
                print("Database file does not exist. Creating new database...")
            
            # Initialize database
            self.init_database()
            print("Database initialization completed successfully")
            
        except Exception as e:
            print(f"Error in Database initialization: {str(e)}")
            raise

    def _hash_password(self, password, salt=None):
        """Hash a password using SHA-256 with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Combine salt and password
        salted_password = f"{salt}{password}"
        
        # Create SHA-256 hash
        hash_obj = hashlib.sha256()
        hash_obj.update(salted_password.encode('utf-8'))
        password_hash = hash_obj.hexdigest()
        
        # Return both salt and hash
        return salt, password_hash

    def _verify_password(self, password, salt, stored_hash):
        """Verify a password against its stored hash"""
        _, computed_hash = self._hash_password(password, salt)
        return computed_hash == stored_hash

    def init_database(self):
        """Initialize database tables"""
        conn = None
        try:
            print(f"Connecting to database at: {self.db_file.absolute()}")
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()

            # Create users table with salt column
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create user preferences table
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER,
                    watched_stocks TEXT,
                    preferred_sectors TEXT,
                    prediction_timeframe INTEGER DEFAULT 7,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Create prediction history table
            c.execute('''
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    stock_symbol TEXT,
                    prediction_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()
            print("Database tables created successfully")
            
            # Verify tables were created
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = c.fetchall()
            print(f"Created tables: {[table[0] for table in tables]}")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
                print("Database connection closed")

    def create_user(self, username, password, preferences=None):
        """Create a new user"""
        if not username or not password:
            return False, "Username and password are required"
            
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            # Generate salt and hash password
            salt, password_hash = self._hash_password(password)
            
            # Insert user with salt and hash
            c.execute('''
                INSERT INTO users (username, password_hash, salt) 
                VALUES (?, ?, ?)
            ''', (username, password_hash, salt))
            
            user_id = c.lastrowid
            
            # Insert preferences
            if preferences:
                c.execute('''
                    INSERT INTO user_preferences 
                    (user_id, watched_stocks, preferred_sectors, prediction_timeframe)
                    VALUES (?, ?, ?, ?)
                ''', (
                    user_id,
                    json.dumps(preferences.get('watched_stocks', [])),
                    json.dumps(preferences.get('preferred_sectors', [])),
                    preferences.get('prediction_timeframe', 7)
                ))
            
            conn.commit()
            return True, "User created successfully"
            
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except Exception as e:
            return False, f"Error creating user: {str(e)}"
        finally:
            conn.close()

    def authenticate_user(self, username, password):
        """Authenticate user"""
        if not username or not password:
            return False, "Username and password are required"
            
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            # Get user's salt and hash
            c.execute('''
                SELECT id, password_hash, salt 
                FROM users 
                WHERE username = ?
            ''', (username,))
            
            result = c.fetchone()
            
            if result and self._verify_password(password, result[2], result[1]):
                user_id = result[0]
                
                # Get user preferences
                c.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
                pref_result = c.fetchone()
                
                preferences = {
                    'watched_stocks': json.loads(pref_result[1]) if pref_result else [],
                    'preferred_sectors': json.loads(pref_result[2]) if pref_result else [],
                    'prediction_timeframe': pref_result[3] if pref_result else 7
                }
                
                return True, {'user_id': user_id, 'preferences': preferences}
            
            return False, "Invalid username or password"
            
        except Exception as e:
            return False, f"Error during authentication: {str(e)}"
        finally:
            conn.close()

    def update_preferences(self, user_id, preferences):
        """Update user preferences"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            c.execute('''
                UPDATE user_preferences 
                SET watched_stocks = ?,
                    preferred_sectors = ?,
                    prediction_timeframe = ?
                WHERE user_id = ?
            ''', (
                json.dumps(preferences.get('watched_stocks', [])),
                json.dumps(preferences.get('preferred_sectors', [])),
                preferences.get('prediction_timeframe', 7),
                user_id
            ))
            
            conn.commit()
            return True, "Preferences updated successfully"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def save_prediction(self, user_id, stock_symbol, prediction_data):
        """Save prediction results"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO prediction_history 
                (user_id, stock_symbol, prediction_data)
                VALUES (?, ?, ?)
            ''', (user_id, stock_symbol, json.dumps(prediction_data)))
            
            conn.commit()
            return True, "Prediction saved successfully"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def get_user_predictions(self, user_id, limit=10):
        """Get user's prediction history"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            c.execute('''
                SELECT stock_symbol, prediction_data, created_at 
                FROM prediction_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            predictions = [{
                'stock_symbol': row[0],
                'prediction_data': json.loads(row[1]),
                'created_at': row[2]
            } for row in c.fetchall()]
            
            return predictions
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            return []
        finally:
            conn.close()
