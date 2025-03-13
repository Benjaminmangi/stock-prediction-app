import streamlit as st
from database import Database
from stock_predictor import StockPredictor
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time

def main():
    st.title("Stock Market Prediction System")
    
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
        
        if 'user_id' not in st.session_state:
            action = st.radio("Choose action", ["Login", "Create Account"])
            
            if action == "Create Account":
                with st.form("signup_form"):
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    
                    # Stock category preferences
                    st.subheader("Preferred Sectors")
                    categories = {
                        "Technology": st.checkbox("Technology"),
                        "Software": st.checkbox("Software"),
                        "Healthcare": st.checkbox("Healthcare"),
                        "Telecommunications": st.checkbox("Telecommunications"),
                        "Services": st.checkbox("Services")
                    }
                    
                    # Stock preferences
                    selected_categories = [k for k, v in categories.items() if v]
                    available_stocks = []
                    for category in selected_categories:
                        available_stocks.extend(predictor.stock_categories.get(category, []))
                    
                    stocks_to_watch = st.multiselect(
                        "Select stocks to track",
                        available_stocks
                    )
                    
                    prediction_days = st.slider("Prediction timeframe (days)", 1, 30, 7)
                    
                    if st.form_submit_button("Create Account"):
                        preferences = {
                            'watched_stocks': stocks_to_watch,
                            'preferred_sectors': selected_categories,
                            'prediction_timeframe': prediction_days
                        }
                        
                        success, message = db.create_user(new_username, new_password, preferences)
                        st.write(message)
            
            else:  # Login
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("Login"):
                        success, user_data = db.authenticate_user(username, password)
                        if success:
                            st.session_state.user_id = user_data['user_id']
                            st.session_state.preferences = user_data['preferences']
                            st.success("Logged in successfully!")
                        else:
                            st.error("Invalid username or password")
        
        else:  # Logged in
            if st.button("Logout"):
                st.session_state.clear()
                st.experimental_rerun()

    # Main content area
    if 'user_id' in st.session_state:
        # Display user's watched stocks
        st.subheader("Your Watched Stocks")
        
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
        st.subheader("Search Other Stocks")
        with st.expander("Search and Analyze Any Stock"):
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., TSLA, NFLX)", "")
            
            with search_col2:
                search_button = st.button("Search")
            
            if search_button and search_symbol:
                stock_info, message = predictor.search_stock(search_symbol.upper())
                
                if stock_info:
                    # Display stock information
                    st.write(f"### {stock_info['name']} ({search_symbol.upper()})")
                    
                    # Create three columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${stock_info['current_price']:.2f}")
                    
                    with col2:
                        if isinstance(stock_info['pe_ratio'], (int, float)):
                            st.metric("P/E Ratio", f"{stock_info['pe_ratio']:.2f}")
                        else:
                            st.metric("P/E Ratio", "N/A")
                    
                    with col3:
                        if isinstance(stock_info['dividend_yield'], (int, float)):
                            st.metric("Dividend Yield", f"{stock_info['dividend_yield']:.2%}")
                        else:
                            st.metric("Dividend Yield", "N/A")
                    
                    # Display sector and industry
                    st.write(f"**Sector:** {stock_info['sector']}")
                    st.write(f"**Industry:** {stock_info['industry']}")
                    
                    # Plot historical data
                    df = stock_info['historical_data']
                    
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))
                    
                    fig.update_layout(
                        title=f"{search_symbol} Stock Price History",
                        yaxis_title="Price ($)",
                        xaxis_title="Date"
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Volume chart
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name='Volume'
                    ))
                    
                    volume_fig.update_layout(
                        title=f"{search_symbol} Trading Volume",
                        yaxis_title="Volume",
                        xaxis_title="Date"
                    )
                    
                    st.plotly_chart(volume_fig)
                    
                    # Add prediction option
                    if st.button(f"Generate Prediction for {search_symbol}"):
                        predictions, pred_message = predictor.predict_stock(
                            search_symbol,
                            st.session_state.preferences['prediction_timeframe']
                        )
                        
                        if predictions:
                            # Create prediction chart
                            pred_fig = go.Figure()
                            
                            # Add actual price line
                            pred_fig.add_trace(go.Scatter(
                                x=[datetime.now().strftime('%Y-%m-%d')],
                                y=[predictions['current_price']],
                                name="Current Price",
                                mode="markers+lines"
                            ))
                            
                            # Add prediction line
                            pred_fig.add_trace(go.Scatter(
                                x=predictions['dates'],
                                y=predictions['predicted_prices'],
                                name="Predicted Price",
                                mode="lines",
                                line=dict(dash='dash')
                            ))
                            
                            pred_fig.update_layout(
                                title=f"{search_symbol} Price Prediction",
                                xaxis_title="Date",
                                yaxis_title="Price ($)"
                            )
                            
                            st.plotly_chart(pred_fig)
                            
                            # Display prediction metrics
                            st.write(f"Current Price: ${predictions['current_price']:.2f}")
                            st.write(f"Predicted Price (7 days): ${predictions['predicted_prices'][-1]:.2f}")
                            
                            # Option to add to watched stocks
                            if search_symbol not in st.session_state.preferences['watched_stocks']:
                                if st.button("Add to Watched Stocks"):
                                    new_preferences = st.session_state.preferences
                                    new_preferences['watched_stocks'].append(search_symbol)
                                    success, message = db.update_preferences(
                                        st.session_state.user_id,
                                        new_preferences
                                    )
                                    if success:
                                        st.success(f"Added {search_symbol} to your watched stocks!")
                                        st.session_state.preferences = new_preferences
                                    else:
                                        st.error(f"Failed to add stock: {message}")
                        else:
                            st.error(f"Could not generate predictions: {pred_message}")
                else:
                    st.error(message)

if __name__ == "__main__":
    main()
