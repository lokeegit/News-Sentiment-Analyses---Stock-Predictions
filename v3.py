import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from textblob import TextBlob
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# App title and configuration
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("Stock Prediction Dashboard")

# Company to ticker symbol map
company_ticker_map = {
    'apple': 'AAPL',
    'nvidia': 'NVDA',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'amazon': 'AMZN',
    'meta': 'META',
    'tesla': 'TSLA'
}

# Initialize News API
api_key = st.sidebar.text_input("Enter News API Key (or use default)", value='19671c9a98d54cb5a545c6fbaa7984bd', type="password")
newsapi = NewsApiClient(api_key=api_key)

# Sidebar for user inputs
st.sidebar.header("Settings")
selected_company = st.sidebar.selectbox(
    "Select Company",
    options=list(company_ticker_map.keys()),
    format_func=lambda x: x.capitalize()
)

days_to_fetch = st.sidebar.slider("Days of Historical Data", min_value=30, max_value=365, value=60)
prediction_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=7, value=1)

# Functions from the original code with enhancements
def fetch_stock_data(ticker, days):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)
    return df

def get_latest_news_sentiment(company_name):
    try:
        articles = newsapi.get_everything(q=company_name, language='en', sort_by='publishedAt', page_size=5)
        sentiments = []
        news_data = []
        
        for article in articles['articles']:
            content = article['title'] + '. ' + article.get('description', '')
            sentiment = TextBlob(content).sentiment.polarity
            sentiments.append(sentiment)
            
            news_data.append({
                'title': article['title'],
                'description': article.get('description', 'No description'),
                'url': article['url'],
                'source': article['source']['name'],
                'sentiment': sentiment,
                'published': article['publishedAt'][:10]
            })
            
        return sentiments, news_data
    except Exception as e:
        st.error(f"News API error: {e}")
        return [0], []  # Neutral sentiment fallback

def prepare_data(df, sentiments):
    sentiment_avg = np.mean(sentiments) if sentiments else 0
    df = df.copy()  # Create a copy to avoid modifying the cached data
    df['Sentiment'] = sentiment_avg
    df['Return'] = df['Close'].pct_change().shift(-1)  # Predict next day % return
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Price_Diff'] = df['Close'] - df['Open']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Add technical indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    df.dropna(inplace=True)
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_prediction_model(df):
    features = ['High_Low_Spread', 'Price_Diff', 'Volume_Change', 'Sentiment']
    X = df[features]
    y = df['Return']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Calculate model metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Get feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    
    return model, X_test, y_test, predictions, mae, r2, feature_importance

def create_price_chart(df):
    # Create candlestick chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_5'],
        mode='lines',
        name='5-day MA',
        line=dict(color='orange')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        mode='lines',
        name='20-day MA',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(color='rgba(0, 0, 255, 0.3)')
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{selected_company.capitalize()} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    # Sort feature importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, importance_values, color='skyblue')
    
    # Add values to the end of each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance_values[i]:.4f}', va='center')
    
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance for Prediction Model')
    plt.tight_layout()
    
    return fig

def create_sentiment_gauge(sentiment_avg):
    # Create a gauge chart for sentiment
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_avg,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "News Sentiment"},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0], 'color': "salmon"},
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_avg
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def prediction_confidence_gauge(prediction):
    # Scale the prediction to a confidence score
    confidence = min(abs(prediction) * 10, 1.0)  # Scale to 0-1
    direction = "Bullish" if prediction > 0 else "Bearish"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Prediction Confidence ({direction})"},
        delta={'reference': 0.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': "darkblue" if prediction > 0 else "darkred"},
            'steps': [
                {'range': [0, 0.33], 'color': "lightgray"},
                {'range': [0.33, 0.67], 'color': "gray"},
                {'range': [0.67, 1], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

# Main app execution
ticker = company_ticker_map[selected_company]

# Display app overview
st.write(f"**Selected Company:** {selected_company.capitalize()} ({ticker})")

# Add a button to run the analysis
if st.button("Run Analysis"):
    # Show a progress message
    progress_text = st.empty()
    progress_text.text(f"Fetching stock data for {selected_company.capitalize()} ({ticker})...")
    
    try:
        # Get stock data
        df_stock = fetch_stock_data(ticker, days_to_fetch)
        
        progress_text.text("Analyzing news sentiment...")
        # Get news sentiment
        sentiments, news_data = get_latest_news_sentiment(selected_company)
        sentiment_avg = np.mean(sentiments) if sentiments else 0
        
        progress_text.text("Preparing data for analysis...")
        # Prepare data for modeling
        df_prepared = prepare_data(df_stock, sentiments)
        
        # Clear progress message
        progress_text.empty()
        
        # Display basic company info
        st.subheader("Market Overview")
        
        # Get price metrics safely
        if not df_stock.empty and len(df_stock) >= 2:
            current_price = float(df_stock['Close'].iloc[-1])
            previous_price = float(df_stock['Close'].iloc[-2])
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            latest_volume = int(df_stock['Volume'].iloc[-1])
            
            # Display info in a more basic way with multiple text elements
            st.write(f"**Current Price:** ${current_price:.2f}")
            change_sign = "+" if price_change >= 0 else ""
            st.write(f"**Day Change:** {change_sign}${price_change:.2f} ({change_sign}{price_change_pct:.2f}%)")
            st.write(f"**Trading Volume:** {latest_volume:,}")
            st.write(f"**News Sentiment Score:** {sentiment_avg:.2f} (scale: -1 to +1)")
        else:
            st.error("Could not retrieve enough price data. Please try again or select a different company.")
            st.stop()
            
        # Display stock chart
        st.subheader("Stock Price Analysis")
        price_chart = create_price_chart(df_prepared)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Train model and make prediction
        st.text("Training prediction model...")
        model, X_test, y_test, predictions, mae, r2, feature_importance = train_prediction_model(df_prepared)
        
        # Get prediction for next day
        latest_features = df_prepared[['High_Low_Spread', 'Price_Diff', 'Volume_Change', 'Sentiment']].iloc[-1].values.reshape(1, -1)
        predicted_change = model.predict(latest_features)[0]
        direction = "UP" if predicted_change >= 0 else "DOWN"
        
        # Display prediction
        st.subheader("Prediction Results")
        st.info(f"### Prediction: Stock may go {direction} by {predicted_change * 100:.2f}% tomorrow.")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        # Display predicted price
        predicted_price = current_price * (1 + predicted_change)
        st.write(f"Current price: ${current_price:.2f}")
        st.write(f"Predicted price (next day): ${predicted_price:.2f}")
            
        # Display prediction confidence gauge
        st.subheader("Prediction Confidence")
        confidence_gauge = prediction_confidence_gauge(predicted_change)
        st.plotly_chart(confidence_gauge, use_container_width=True)
        
        # Display feature importance
        st.subheader("Feature Importance")
        importance_chart = create_feature_importance_chart(feature_importance)
        st.pyplot(importance_chart)
        
        # Display news sentiment
        st.subheader("News Sentiment Analysis")
        sentiment_gauge = create_sentiment_gauge(sentiment_avg)
        st.plotly_chart(sentiment_gauge, use_container_width=True)
        
        # Display news articles
        st.subheader("Recent News Articles")
        for i, news in enumerate(news_data[:5]):
            st.write(f"**{news['title']}**")
            st.write(f"Source: {news['source']} | Published: {news['published']} | "
                     f"Sentiment: {news['sentiment']:.2f}")
            st.write(f"{news['description']}")
            st.write(f"[Read more]({news['url']})")
            if i < len(news_data) - 1:
                st.divider()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your inputs and try again.")
    
else:
    st.info("Click 'Run Analysis' to fetch data and generate predictions")

# Footer
st.markdown("---")
st.write("Note: This app is for educational purposes only. Do not use for financial decisions.")
