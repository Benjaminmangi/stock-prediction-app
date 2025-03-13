from user_management import UserProfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import os
import streamlit as st
from requests.exceptions import RequestException
import time

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
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

    def prepare_data(self, df, look_back=60):
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        X = []
        for i in range(len(data_scaled) - look_back):
            X.append(data_scaled[i:(i + look_back), 0])
        
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, scaler

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

    def get_category_performance(self, category):
        """Get performance metrics for all stocks in a category"""
        performance = {}
        for symbol in self.stock_categories.get(category, []):
            df = self.get_stock_data(symbol, days=30)  # Get last 30 days
            if not df.empty:
                performance[symbol] = {
                    'current_price': float(df['Close'].iloc[-1]),
                    'monthly_return': float(df['Close'].pct_change(30).iloc[-1] * 100),
                    'volatility': float(df['Close'].pct_change().std() * 100)
                }
        return performance

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

    def search_stock(self, ticker_symbol):
        """Search and analyze any stock by ticker symbol"""
        try:
            st.info(f"Fetching data for {ticker_symbol}...")
            
            # Get stock info using yfinance with retry
            for attempt in range(self.max_retries):
                try:
                    stock = yf.Ticker(ticker_symbol)
                    info = stock.info
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        st.error(f"Could not fetch stock info: {str(e)}")
                        info = {}
                    time.sleep(2)
            
            # Get historical data
            df = self.get_stock_data(ticker_symbol)
            
            if df.empty:
                return None, "No data available for this stock"

            # Basic stock information
            stock_info = {
                'name': info.get('longName', ticker_symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', df['Close'].iloc[-1]),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'historical_data': df
            }

            return stock_info, "Success"
            
        except Exception as e:
            st.error(f"Error in stock search: {str(e)}")
            return None, f"Error searching stock: {str(e)}"
