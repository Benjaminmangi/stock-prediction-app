import json
from pathlib import Path
import pandas as pd
from datetime import datetime

class UserProfile:
    def __init__(self):
        self.users_file = 'data/users.json'
        self.predictions_dir = Path('data/predictions')
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        Path('data').mkdir(exist_ok=True)
        
        # Initialize users file if it doesn't exist
        if not Path(self.users_file).exists():
            with open(self.users_file, 'w') as f:
                json.dump({}, f)

    def create_user(self, username, password, preferences=None):
        """Create a new user profile"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if username in users:
            return False, "Username already exists"
        
        users[username] = {
            'password': password,  # In production, use proper password hashing
            'preferences': preferences or {
                'watched_stocks': [],
                'preferred_sectors': [],
                'prediction_timeframe': 7  # default 7-day predictions
            },
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f)
        
        return True, "User created successfully"

    def authenticate_user(self, username, password):
        """Authenticate user login"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if username in users and users[username]['password'] == password:
            return True, users[username]
        return False, "Invalid username or password"

    def update_preferences(self, username, preferences):
        """Update user preferences"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if username not in users:
            return False, "User not found"
        
        users[username]['preferences'].update(preferences)
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f)
        
        return True, "Preferences updated successfully"

    def save_user_prediction(self, username, stock_symbol, prediction_data):
        """Save prediction results for a user"""
        prediction_file = self.predictions_dir / f"{username}_{stock_symbol}.json"
        
        prediction_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': prediction_data
        }
        
        with open(prediction_file, 'w') as f:
            json.dump(prediction_info, f)
