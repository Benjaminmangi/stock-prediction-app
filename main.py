import streamlit as st
from database import Database
from stock_predictor import StockPredictor
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
import plotly.express as px
import requests
import pandas as pd
from io import BytesIO

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

def main():
    st.title("Stock Market Prediction ")
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'watched_stocks': [],
            'preferred_sectors': [],
            'prediction_timeframe': 7
        }
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # Initialize components with progress
    with st.spinner('Initializing system...'):
        try:
            # Initialize database
            db = Database()
            st.success("Database connected")
            
            # Initialize predictor
            predictor = StockPredictor()
            st.success("Stock predictor ready")
            
            # Basic test functionality
            st.subheader("Quick Stock Check")
            symbol = st.selectbox(
                "Select a stock to test",
                ["AAPL", "MSFT", "GOOGL"]
            )
            
            if st.button("Check Stock"):
                with st.spinner('Fetching data...'):
                    data = predictor.get_stock_data(symbol, days=5)
                    if not data.empty:
                        st.write("Recent stock data:")
                        st.dataframe(data.tail())
                    else:
                        st.error("No data available")
                        
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    # Sidebar for user management
    with st.sidebar:
        st.subheader("User Management")
        
        if not st.session_state.logged_in:
            action = st.radio("Choose action", ["Login", "Create Account"])
            
            if action == "Create Account":
                with st.form("signup_form"):
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    
                    # Stock preferences
                    st.subheader("Preferred Sectors")
                    sectors = {
                        "Technology": st.checkbox("Technology"),
                        "Software": st.checkbox("Software"),
                        "Healthcare": st.checkbox("Healthcare"),
                        "Telecommunications": st.checkbox("Telecommunications"),
                        "Services": st.checkbox("Services")
                    }
                    
                    stocks_to_watch = st.multiselect(
                        "Select stocks to track",
                        ["MSFT", "GOOGL", "AAPL", "NVDA", "META", "IBM", "ADBE", 
                         "JNJ", "UNH", "T", "VZ", "UPS"]
                    )
                    
                    prediction_days = st.slider("Prediction timeframe (days)", 1, 30, 7)
                    
                    if st.form_submit_button("Create Account"):
                        preferences = {
                            'watched_stocks': stocks_to_watch,
                            'preferred_sectors': [k for k, v in sectors.items() if v],
                            'prediction_timeframe': prediction_days
                        }
                        
                        success, message = db.create_user(new_username, new_password, preferences)
                        if success:
                            st.success(message)
                            # Set session state variables directly
                            success, user_data = db.authenticate_user(new_username, new_password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.user_id = user_data['user_id']
                                st.session_state.preferences = user_data['preferences']
                                st.success("Account created and logged in successfully!")
                        else:
                            st.error(message)
            
            else:  # Login
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("Login"):
                        success, user_data = db.authenticate_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user_id = user_data['user_id']
                            st.session_state.preferences = user_data['preferences']
                            st.success("Logged in successfully!")
                        else:
                            st.error("Invalid username or password")
        
        else:  # Logged in
            st.write(f"Welcome back!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.preferences = {
                    'watched_stocks': [],
                    'preferred_sectors': [],
                    'prediction_timeframe': 7
                }
                st.rerun()

    # Add a modern sidebar with categories
    st.sidebar.title("📈 Stock Categories")
    category = st.sidebar.selectbox(
        "Select Market Sector",
        ["Technology", "Software", "Healthcare", "Telecommunications", "Services"]
    )

    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Analysis", "📱 Portfolio"])

    with tab1:
        st.header("Stock Predictions")
        
        # Add stock and date selection
        col1, col2 = st.columns(2)
        with col1:
            selected_stock = st.selectbox(
                "Select Stock",
                list(STOCK_SYMBOLS.keys()),
                key="prediction_stock_select"
            )
        with col2:
            prediction_date = st.date_input(
                "Select Prediction Date",
                min_value=datetime.now().date(),
                value=datetime.now().date() + timedelta(days=7),
                key="prediction_date_select"
            )
        
        # Calculate days ahead for prediction
        days_ahead = (prediction_date - datetime.now().date()).days
        
        if selected_stock and prediction_date:
            symbol = STOCK_SYMBOLS[selected_stock]
            st.write(f"### {selected_stock} ({symbol})")
            
            # Generate predictions
            predictions, message = predictor.predict_stock(symbol, days_ahead)
            
            if predictions:
                # Create price prediction chart
                fig = go.Figure()
                
                # Add actual price line
                fig.add_trace(go.Scatter(
                    x=[datetime.now().strftime('%Y-%m-%d')],
                    y=[predictions['current_price']],
                    name="Current Price",
                    mode="markers+lines"
                ))
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=predictions['dates'],
                    y=predictions['predicted_prices'],
                    name="Predicted Price",
                    mode="lines",
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{selected_stock} Price Prediction until {prediction_date}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)"
                )
                
                st.plotly_chart(fig)
                
                # Display prediction metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Price", f"${predictions['current_price']:.2f}")
                with col2:
                    st.metric(
                        f"Predicted Price ({prediction_date})",
                        f"${predictions['predicted_prices'][-1]:.2f}"
                    )
                
                # Calculate and display price change
                price_change = predictions['predicted_prices'][-1] - predictions['current_price']
                price_change_pct = (price_change / predictions['current_price']) * 100
                
                st.metric(
                    "Predicted Price Change",
                    f"${price_change:.2f} ({price_change_pct:.2f}%)",
                    delta=f"{price_change_pct:.2f}%"
                )
            else:
                st.error(f"Could not generate predictions for {selected_stock}: {message}")
        
        # Display user's watched stocks
        st.subheader("Your Watched Stocks")
        
        if st.session_state.preferences['watched_stocks']:
            for stock in st.session_state.preferences['watched_stocks']:
                st.write(f"### {stock}")
                # Generate predictions
                predictions, message = predictor.predict_stock(
                    stock, 
                    st.session_state.preferences['prediction_timeframe']
                )
                
                if predictions:
                    # Save prediction to database
                    db.save_prediction(
                        st.session_state.user_id,
                        stock,
                        predictions
                    )
                    
                    # Create price prediction chart
                    fig = go.Figure()
                    
                    # Add actual price line
                    fig.add_trace(go.Scatter(
                        x=[datetime.now().strftime('%Y-%m-%d')],
                        y=[predictions['current_price']],
                        name="Current Price",
                        mode="markers+lines"
                    ))
                    
                    # Add prediction line
                    fig.add_trace(go.Scatter(
                        x=predictions['dates'],
                        y=predictions['predicted_prices'],
                        name="Predicted Price",
                        mode="lines",
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{stock} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price ($)"
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display prediction metrics
                    st.write(f"Current Price: ${predictions['current_price']:.2f}")
                    st.write(f"Predicted Price (7 days): ${predictions['predicted_prices'][-1]:.2f}")
                else:
                    st.error(f"Could not generate predictions for {stock}: {message}")
        else:
            st.info("You haven't added any stocks to watch yet.")
            
        # Display sector performance
        if st.session_state.preferences['preferred_sectors']:
            st.subheader("Sector Performance")
            for sector in st.session_state.preferences['preferred_sectors']:
                st.write(f"### {sector}")
                performance = predictor.get_category_performance(sector)
                
                for symbol, metrics in performance.items():
                    st.write(f"**{symbol}**")
                    st.write(f"Current Price: ${metrics['current_price']:.2f}")
                    st.write(f"Monthly Return: {metrics['monthly_return']:.2f}%")
                    st.write(f"Volatility: {metrics['volatility']:.2f}%")

        # Add a stock search section
        st.subheader("Search Stocks")
        search_symbol = st.text_input("Enter stock symbol (e.g., AAPL)")
        search_button = st.button("Search")

        if search_button and search_symbol:
            with st.spinner('Searching...'):
                try:
                    stock_info, message = predictor.search_stock(search_symbol.upper())
                    if stock_info:
                        st.write(f"### {stock_info['name']} ({search_symbol.upper()})")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${stock_info['current_price']:.2f}")
                        with col2:
                            st.metric("Sector", stock_info['sector'])
                        with col3:
                            st.metric("Industry", stock_info['industry'])
                        if 'historical_data' in stock_info:
                            st.line_chart(stock_info['historical_data']['Close'])
                            
                        # Add news section
                        st.subheader("Latest News")
                        news_items = fetch_stock_news(search_symbol.upper())
                        for news in news_items:
                            with st.expander(news['title']):
                                st.write(news['description'])
                                st.write(f"[Read more]({news['url']})")
                    else:
                        st.error(f"Could not find data for {search_symbol}: {message}")
                except Exception as e:
                    st.error(f"Error searching for stock: {str(e)}")

    with tab2:
        st.header("Technical Analysis")
        # Add technical indicators
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume Analysis", "ATR"],
            key="analysis_type_select"
        )
        
        # Stock selection for technical analysis
        selected_stock = st.selectbox(
            "Select Stock",
            list(STOCK_SYMBOLS.keys()),
            key="analysis_stock_select"
        )
        
        if selected_stock:
            symbol = STOCK_SYMBOLS[selected_stock]
            df = predictor.get_stock_data(symbol)
            
            if not df.empty:
                # Calculate technical indicators
                df = predictor.calculate_technical_indicators(df)
                
                if analysis_type == "Moving Averages":
                    st.subheader("Moving Averages Analysis")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-day MA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA'))
                    fig.update_layout(title=f"{selected_stock} Moving Averages",
                                   xaxis_title="Date",
                                   yaxis_title="Price")
                    st.plotly_chart(fig)
                    
                    # Display current MA values
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("20-day MA", f"${df['MA20'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("50-day MA", f"${df['MA50'].iloc[-1]:.2f}")
                
                elif analysis_type == "RSI":
                    st.subheader("Relative Strength Index (RSI) Analysis")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                    fig.add_hline(y=70, line_dash="dash", line_color="red")
                    fig.add_hline(y=30, line_dash="dash", line_color="green")
                    fig.update_layout(title=f"{selected_stock} RSI",
                                   xaxis_title="Date",
                                   yaxis_title="RSI")
                    st.plotly_chart(fig)
                    
                    # Display current RSI value and interpretation
                    current_rsi = df['RSI'].iloc[-1]
                    st.metric("Current RSI", f"{current_rsi:.2f}")
                    if current_rsi > 70:
                        st.warning("Overbought condition - Consider selling")
                    elif current_rsi < 30:
                        st.success("Oversold condition - Consider buying")
                    else:
                        st.info("Neutral condition")
                
                elif analysis_type == "MACD":
                    st.subheader("MACD Analysis")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'))
                    fig.update_layout(title=f"{selected_stock} MACD",
                                   xaxis_title="Date",
                                   yaxis_title="MACD")
                    st.plotly_chart(fig)
                    
                    # Display current MACD values
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Signal Line", f"{df['Signal_Line'].iloc[-1]:.2f}")
                    
                    # MACD interpretation
                    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
                        st.success("Bullish signal - MACD above Signal Line")
                    else:
                        st.warning("Bearish signal - MACD below Signal Line")
                
                elif analysis_type == "Bollinger Bands":
                    st.subheader("Bollinger Bands Analysis")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='Middle Band'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band'))
                    fig.update_layout(title=f"{selected_stock} Bollinger Bands",
                                   xaxis_title="Date",
                                   yaxis_title="Price")
                    st.plotly_chart(fig)
                    
                    # Display current Bollinger Bands values
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Upper Band", f"${df['BB_upper'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Middle Band", f"${df['BB_middle'].iloc[-1]:.2f}")
                    with col3:
                        st.metric("Lower Band", f"${df['BB_lower'].iloc[-1]:.2f}")
                    
                    # Bollinger Bands interpretation
                    current_price = df['Close'].iloc[-1]
                    if current_price > df['BB_upper'].iloc[-1]:
                        st.warning("Overbought condition - Price above upper band")
                    elif current_price < df['BB_lower'].iloc[-1]:
                        st.success("Oversold condition - Price below lower band")
                    else:
                        st.info("Price within normal range")
                
                elif analysis_type == "Volume Analysis":
                    st.subheader("Volume Analysis")
                    
                    # Create volume chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
                    fig.update_layout(
                        title=f"{selected_stock} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume"
                    )
                    st.plotly_chart(fig)
                    
                    # Calculate volume metrics
                    avg_volume = df['Volume'].mean()
                    current_volume = df['Volume'].iloc[-1]
                    volume_ma20 = df['Volume'].rolling(window=20).mean().iloc[-1]
                    
                    # Display volume metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Volume", f"{current_volume:,.0f}")
                    with col2:
                        st.metric("20-day Avg Volume", f"{volume_ma20:,.0f}")
                    with col3:
                        st.metric("Average Volume", f"{avg_volume:,.0f}")
                    
                    # Volume analysis interpretation
                    volume_ratio = current_volume / volume_ma20
                    if volume_ratio > 1.5:
                        st.success("High volume - Strong trading activity")
                    elif volume_ratio < 0.5:
                        st.warning("Low volume - Weak trading activity")
                    else:
                        st.info("Normal volume - Average trading activity")
                    
                    # Add volume trend chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(window=20).mean(), 
                                           name='20-day Volume MA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(window=50).mean(), 
                                           name='50-day Volume MA'))
                    fig.update_layout(
                        title=f"{selected_stock} Volume Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Volume"
                    )
                    st.plotly_chart(fig)
                
                elif analysis_type == "ATR":
                    st.subheader("Average True Range (ATR) Analysis")
                    
                    # Calculate ATR
                    high = df['High']
                    low = df['Low']
                    close = df['Close']
                    
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    
                    # Create ATR chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=atr, name='ATR'))
                    fig.update_layout(
                        title=f"{selected_stock} Average True Range",
                        xaxis_title="Date",
                        yaxis_title="ATR"
                    )
                    st.plotly_chart(fig)
                    
                    # Display ATR metrics
                    current_atr = atr.iloc[-1]
                    current_price = df['Close'].iloc[-1]
                    atr_percentage = (current_atr / current_price) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current ATR", f"${current_atr:.2f}")
                    with col2:
                        st.metric("ATR % of Price", f"{atr_percentage:.2f}%")
                    
                    # ATR interpretation
                    if atr_percentage > 3:
                        st.warning("High volatility - Consider reducing position size")
                    elif atr_percentage < 1:
                        st.info("Low volatility - Market is relatively stable")
                    else:
                        st.success("Moderate volatility - Normal market conditions")
                    
                    # Add ATR bands to price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'] + atr, name='Upper ATR Band',
                                           line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'] - atr, name='Lower ATR Band',
                                           line=dict(dash='dash')))
                    fig.update_layout(
                        title=f"{selected_stock} Price with ATR Bands",
                        xaxis_title="Date",
                        yaxis_title="Price"
                    )
                    st.plotly_chart(fig)
            else:
                st.error(f"Could not fetch data for {selected_stock}")

    with tab3:
        st.header("My Portfolio")
        # Add portfolio tracking
        show_portfolio_section()

    # Add to your settings section
    def show_user_settings():
        st.subheader("⚙️ Settings")
        
        # Theme preference
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"]
        )
        
        # Notification preferences
        st.subheader("Notifications")
        notify_price_alert = st.checkbox("Price Alerts")
        notify_news = st.checkbox("News Updates")
        
        if notify_price_alert:
            alert_threshold = st.slider(
                "Alert Threshold (%)",
                min_value=1,
                max_value=10,
                value=5
            )

def show_portfolio_section():
    st.subheader("Portfolio Management")
    
    # Portfolio Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Value", "$10,000", "+5.2%")
    with col2:
        st.metric("Daily Change", "+$120", "+1.2%")
    with col3:
        st.metric("Total Stocks", "12", "+2")
    
    # Portfolio Composition
    st.subheader("Portfolio Composition")
    portfolio_data = {
        'Technology': 40,
        'Healthcare': 25,
        'Software': 20,
        'Telecom': 10,
        'Services': 5
    }
    
    # Create pie chart
    fig = px.pie(values=list(portfolio_data.values()), 
                 names=list(portfolio_data.keys()),
                 title='Portfolio Distribution')
    st.plotly_chart(fig)

def fetch_stock_news(symbol):
    """Fetch news for a specific stock"""
    # You can use various APIs like Alpha Vantage, NewsAPI, etc.
    news_api_key = "YOUR_NEWS_API_KEY"
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles'][:5]  # Return top 5 news
    return []

def show_stock_comparison():
    st.subheader("Stock Comparison")
    
    # Select stocks to compare
    stocks_to_compare = st.multiselect(
        "Select stocks to compare",
        list(STOCK_SYMBOLS.keys()),
        default=["Microsoft", "Apple"]
    )
    
    if len(stocks_to_compare) > 0:
        # Create comparison chart
        fig = go.Figure()
        for stock in stocks_to_compare:
            symbol = STOCK_SYMBOLS[stock]
            df = get_stock_data(symbol)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close']/df['Close'].iloc[0],  # Normalize prices
                name=stock
            ))
        
        fig.update_layout(
            title="Price Comparison (Normalized)",
            xaxis_title="Date",
            yaxis_title="Normalized Price"
        )
        st.plotly_chart(fig)

def export_data(df, format='csv'):
    """Export data in various formats"""
    if format == 'csv':
        return df.to_csv().encode('utf-8')
    elif format == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data')
        return output.getvalue()

if __name__ == "__main__":
    main()
