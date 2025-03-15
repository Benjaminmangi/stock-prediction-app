import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime

class Database:
    def __init__(self):
        self.db_name = 'stock_predictions.db'
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        # Create users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
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
        conn.close()

    def create_user(self, username, password, preferences=None):
        """Create a new user"""
        if not username or not password:
            return False, "Username and password are required"
            
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        try:
            # Hash password
            password_hash = generate_password_hash(password)
            
            # Insert user
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                     (username, password_hash))
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
            print(f"User created successfully: {username} with ID {user_id}")  # Debug log
            return True, "User created successfully"
            
        except sqlite3.IntegrityError as e:
            print(f"Error creating user: {str(e)}")  # Debug log
            return False, "Username already exists"
        except Exception as e:
            print(f"Unexpected error creating user: {str(e)}")  # Debug log
            return False, f"Error creating user: {str(e)}"
        finally:
            conn.close()

    def authenticate_user(self, username, password):
        """Authenticate user"""
        if not username or not password:
            return False, "Username and password are required"
            
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        try:
            c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
            result = c.fetchone()
            
            if result and check_password_hash(result[1], password):
                user_id = result[0]
                
                # Get user preferences
                c.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
                pref_result = c.fetchone()
                
                preferences = {
                    'watched_stocks': json.loads(pref_result[1]) if pref_result else [],
                    'preferred_sectors': json.loads(pref_result[2]) if pref_result else [],
                    'prediction_timeframe': pref_result[3] if pref_result else 7
                }
                
                print(f"User authenticated successfully: {username}")  # Debug log
                return True, {'user_id': user_id, 'preferences': preferences}
            
            print(f"Authentication failed for user: {username}")  # Debug log
            return False, "Invalid username or password"
            
        except Exception as e:
            print(f"Error during authentication: {str(e)}")  # Debug log
            return False, f"Error during authentication: {str(e)}"
        finally:
            conn.close()

    def update_preferences(self, user_id, preferences):
        """Update user preferences"""
        conn = sqlite3.connect(self.db_name)
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
        conn = sqlite3.connect(self.db_name)
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
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
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
        
        conn.close()
        return predictions
