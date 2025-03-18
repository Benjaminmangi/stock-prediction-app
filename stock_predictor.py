import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import streamlit as st
from requests.exceptions import RequestException
import time

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.look_back = 60
        self.stock_categories = {
            'Technology': ['MSFT', 'GOOGL', 'AAPL', 'IBM'],
            'Software': ['NVDA', 'META', 'ADBE'],
            'Healthcare': ['JNJ', 'UNH'],
            'Telecommunications': ['T', 'VZ'],
            'Services': ['UPS']
        }
        self.max_retries = 3
        self.timeout = 10  # reduced timeout
        # Load models lazily (only when needed) instead of all at once
        st.write("Stock predictor ready")  # Debug message

    def get_stock_data(self, symbol, days=365):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=f"{days}d")
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
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
    
    def predict_stock(self, symbol, days=7):
        """Predict stock prices for the next n days"""
        try:
            # Get historical data
            df = self.get_stock_data(symbol)
            if df.empty:
                return None, "Could not fetch historical data"
            
            # Prepare data
            X, y = self.prepare_data(df)
            if len(X) == 0:
                return None, "Insufficient data for prediction"
            
            # Train model
            self.train_model(X, y)
            
            # Prepare last 60 days of data for prediction
            last_60_days = df['Close'].values[-60:]
            last_60_days_scaled = self.scaler.transform(last_60_days.reshape(-1, 1))
            
            # Generate predictions
            predictions = []
            current_batch = last_60_days_scaled.reshape((1, 60, 1))
            
            for _ in range(days):
                current_pred = self.model.predict(current_batch)[0]
                predictions.append(current_pred[0])
                current_batch = np.append(current_batch[:, 1:, :], 
                                        [[current_pred]], axis=1)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Generate dates for predictions
            last_date = df.index[-1]
            prediction_dates = [last_date + timedelta(days=x+1) for x in range(days)]
            
            return {
                'current_price': df['Close'].iloc[-1],
                'predicted_prices': predictions.flatten(),
                'dates': prediction_dates
            }, None
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"
    
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
        # Define category stocks (you can expand this)
        category_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Software': ['ADBE', 'META', 'IBM'],
            'Healthcare': ['JNJ', 'UNH'],
            'Telecommunications': ['T', 'VZ'],
            'Services': ['UPS']
        }
        
        if category not in category_stocks:
            return {}
        
        performance = {}
        for symbol in category_stocks[category]:
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

    def get_stock_data(self, symbol, days=100):
        """Get recent stock data using yfinance with retries"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    timeout=self.timeout
                )
                
                if not df.empty:
                    return df
                
                st.warning(f"Attempt {attempt + 1}: No data found for {symbol}, retrying...")
                time.sleep(2)  # Wait 2 seconds between attempts
                
            except Exception as e:
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                continue
        
        # If all attempts fail, try alternative data source
        try:
            st.info("Trying alternative data source...")
            # Using pandas_datareader as backup
            import pandas_datareader as pdr
            df = pdr.get_data_yahoo(
                symbol,
                start=start_date,
                end=end_date
            )
            if not df.empty:
                return df
        except Exception as e:
            st.error(f"Alternative source failed: {str(e)}")
        
        return pd.DataFrame()  # Return empty DataFrame if all attempts fail

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
            X, scaler = self.prepare_data(df, self.look_back)
            if len(X) == 0:
                return None, "Insufficient data for prediction"

            # Make prediction
            model = self.models.get(symbol)
            if model is None:
                # If no pre-trained model exists, use a similar sector model or basic prediction
                return self.basic_prediction(df, days_ahead)

            predictions = []
            last_sequence = X[-1:]

            # Generate predictions for specified number of days
            for _ in range(days_ahead):
                pred = model.predict(last_sequence)
                predictions.append(pred[0][0])
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[0][-1] = pred[0][0]

            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            return {
                'current_price': float(df['Close'].iloc[-1]),
                'predicted_prices': [float(p[0]) for p in predictions],
                'dates': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                         for i in range(days_ahead)]
            }, "Success"

        except Exception as e:
            return None, f"Error in prediction: {str(e)}"

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
