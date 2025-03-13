from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import datetime
import yfinance as yf
import os

app = Flask(__name__)

# Dictionary to map stock names to their ticker symbols
STOCK_SYMBOLS = {
    'Apple': 'AAPL',
    'Google': 'GOOGL',
    'Microsoft': 'MSFT',
    'NVIDIA': 'NVDA',
    'META Platforms': 'META',
    'IBM': 'IBM',
    'Adobe': 'ADBE',
    'Johnson&Johnson': 'JNJ',
    'Verizon': 'VZ',
    'United Health Group': 'UNH',
    'AT&T': 'T',
    'UPS': 'UPS'
}

def prepare_data(df, look_back=60):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back), 0])
    
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, scaler

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

@app.route('/')
def home():
    return render_template('index.html', stocks=list(STOCK_SYMBOLS.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        stock_name = request.form['stock']
        prediction_date = request.form['date']
        
        # Convert prediction date to datetime
        pred_date = datetime.datetime.strptime(prediction_date, '%Y-%m-%d')
        
        # Get historical data up to prediction date
        symbol = STOCK_SYMBOLS[stock_name]
        start_date = pred_date - datetime.timedelta(days=100)
        df = get_stock_data(symbol, start_date, pred_date)
        
        if df.empty:
            return jsonify({'error': 'No data available for the selected date range'})
        
        # Check if model exists
        model_path = f'models/{symbol}_model.h5'
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model for {stock_name} not found!'})
            
        # Prepare data for prediction
        X, scaler = prepare_data(df, look_back=60)
        
        # Load the corresponding model
        model = load_model(model_path, custom_objects={'loss': MeanSquaredError()})
        
        # Make prediction
        prediction = model.predict(X[-1:])
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        current_price = df['Close'].iloc[-1]
        
        # Convert numpy types to Python native types
        return jsonify({
            'stock': stock_name,
            'date': prediction_date,
            'predicted_price': float(predicted_price),  # Convert to Python float
            'current_price': float(current_price)      # Convert to Python float
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/test')
def test():
    return "Flask is working!"

if __name__ == '__main__':
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Templates directory: {os.path.join(os.getcwd(), 'templates')}")
    
    # Check if required directories and files exist
    if not os.path.exists('templates'):
        print("WARNING: 'templates' directory not found!")
    if not os.path.exists('models'):
        print("WARNING: 'models' directory not found!")
    
    app.run(debug=True)