import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import streamlit as st
from requests.exceptions import RequestException
import time
import os
import random

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.models = {}  # Initialize models dictionary
        self.look_back = 60
        self.stock_categories = {
            'Technology': ['MSFT', 'GOOGL', 'AAPL', 'IBM'],
            'Software': ['NVDA', 'META', 'ADBE'],
            'Healthcare': ['JNJ', 'UNH'],
            'Telecommunications': ['T', 'VZ'],
            'Services': ['UPS']
        }
        self.max_retries = 5  # Increased max retries
        self.timeout = 10
        self.base_delay = 2  # Base delay in seconds
        self.max_delay = 60  # Maximum delay in seconds
        self.request_timestamps = {}  # Track request timestamps
        self.min_request_interval = 2  # Minimum seconds between requests
        st.write("Stock predictor ready")  # Debug message

    def get_stock_data(self, symbol, days=100):
        """Get recent stock data using yfinance with rate limiting and exponential backoff"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Check if we need to wait before making a new request
        current_time = time.time()
        if symbol in self.request_timestamps:
            last_request_time = self.request_timestamps[symbol]
            time_since_last_request = current_time - last_request_time
            if time_since_last_request < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last_request
                time.sleep(wait_time)
        
        for attempt in range(self.max_retries):
            try:
                # Calculate exponential backoff delay
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                if attempt > 0:
                    # Add jitter to prevent thundering herd
                    delay += random.uniform(0, 1)
                    st.warning(f"Waiting {delay:.1f} seconds before retry {attempt + 1}...")
                    time.sleep(delay)
                
                stock = yf.Ticker(symbol)
                df = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    timeout=self.timeout
                )
                
                if not df.empty:
                    # Update request timestamp
                    self.request_timestamps[symbol] = time.time()
                    return df
                
                st.warning(f"Attempt {attempt + 1}: No data found for {symbol}")
                
            except Exception as e:
                error_message = str(e)
                if "Too Many Requests" in error_message:
                    st.warning(f"Rate limited. Waiting longer before retry {attempt + 1}...")
                    time.sleep(delay * 2)  # Double the delay for rate limits
                else:
                    st.warning(f"Attempt {attempt + 1} failed: {error_message}")
                continue
        
        # If all attempts fail, try using cached data if available
        try:
            cache_file = f'cache/{symbol}_data.csv'
            if os.path.exists(cache_file):
                st.info("Using cached data...")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                df = df[start_date:end_date]
                if not df.empty:
                    return df
        except Exception as e:
            st.error(f"Cache failed: {str(e)}")
        
        return pd.DataFrame()  # Return empty DataFrame if all attempts fail

    def prepare_data(self, df, look_back=60):
        """Prepare data for LSTM model"""
        # Use only the 'Close' prices
        data = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train_model(self, X, y):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X, y, epochs=25, batch_size=32, verbose=0)
    
    def predict_stock(self, symbol, days_ahead=7):
        """Generate predictions for a specific stock"""
        try:
            # Get recent data
            df = self.get_stock_data(symbol)
            if df.empty:
                return None, f"No data available for {symbol}"

            # Ensure we have enough data
            if len(df) < self.look_back:
                return None, f"Insufficient historical data for {symbol}"

            # Prepare data for prediction
            X, y = self.prepare_data(df, self.look_back)
            if len(X) == 0:
                return None, "Insufficient data for prediction"

            # Create and train model if not exists
            if symbol not in self.models:
                self.build_model()
                self.train_model(X, y)
                self.models[symbol] = self.model

            # Get the model for this symbol
            model = self.models[symbol]

            # Prepare last sequence for prediction
            last_sequence = X[-1:]

            # Generate predictions
            predictions = []
            for _ in range(days_ahead):
                pred = model.predict(last_sequence)
                predictions.append(pred[0][0])
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[0][-1] = pred[0][0]

            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            return {
                'current_price': float(df['Close'].iloc[-1]),
                'predicted_prices': [float(p[0]) for p in predictions],
                'dates': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                         for i in range(days_ahead)]
            }, "Success"

        except Exception as e:
            return None, f"Error in prediction: {str(e)}"
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        return df
    
    def get_category_performance(self, category):
        """Get performance metrics for stocks in a category"""
        if category not in self.stock_categories:
            return {}
        
        performance = {}
        for symbol in self.stock_categories[category]:
            try:
                df = self.get_stock_data(symbol, days=30)
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    monthly_return = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                    volatility = df['Close'].pct_change().std() * 100
                    
                    performance[symbol] = {
                        'current_price': current_price,
                        'monthly_return': monthly_return,
                        'volatility': volatility
                    }
            except Exception as e:
                print(f"Error calculating performance for {symbol}: {str(e)}")
        
        return performance
    
    def search_stock(self, symbol):
        """Search for stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                return None, "Stock not found"
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('regularMarketPrice', 0),
                'historical_data': self.get_stock_data(symbol, days=30)
            }, None
            
        except Exception as e:
            return None, f"Error searching for stock: {str(e)}"

    def load_model(self, symbol):
        """Load model only when needed"""
        if symbol not in self.models:
            try:
                model_path = f'models/{symbol}_model.h5'
                if os.path.exists(model_path):
                    self.models[symbol] = load_model(model_path)
                    st.write(f"Loaded model for {symbol}")
                    return True
                return False
            except Exception as e:
                st.error(f"Error loading model for {symbol}: {str(e)}")
                return False
        return True

    def basic_prediction(self, df, days_ahead):
        """Basic prediction when no model is available"""
        try:
            # Calculate simple moving average
            ma = df['Close'].rolling(window=5).mean()
            current_price = float(df['Close'].iloc[-1])
            last_ma = float(ma.iloc[-1])
            
            # Calculate daily return
            daily_return = (current_price - float(df['Close'].iloc[-2])) / float(df['Close'].iloc[-2])
            
            # Generate simple predictions
            predictions = []
            price = current_price
            for _ in range(days_ahead):
                price = price * (1 + daily_return)
                predictions.append(price)

            return {
                'current_price': current_price,
                'predicted_prices': predictions,
                'dates': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                         for i in range(days_ahead)]
            }, "Basic prediction (no trained model available)"

        except Exception as e:
            return None, f"Error in basic prediction: {str(e)}"

    def get_personalized_predictions(self, username, password):
        """Get predictions for user's watched stocks"""
        # Authenticate user
        auth_success, user_data = self.user_profile.authenticate_user(username, password)
        if not auth_success:
            return False, "Authentication failed"

        predictions = {}
        watched_stocks = user_data['preferences']['watched_stocks']
        timeframe = user_data['preferences']['prediction_timeframe']

        for stock in watched_stocks:
            if stock in self.models:
                pred = self.predict_stock(stock, timeframe)
                predictions[stock] = pred

        return True, predictions
