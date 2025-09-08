# Stock Prediction Dashboard

An interactive web application built with **Python** and **Streamlit** that predicts short-term stock price movements using a combination of **historical stock data**, **technical indicators**, and **real-time news sentiment analysis**. The app leverages machine learning to forecast price changes and provides visualizations for better decision-making.

---

## Features

- **Interactive Web Dashboard:** Powered by Streamlit with a clean, responsive UI.
- **Stock Market Data:** Fetches real-time and historical stock data via `yfinance`.
- **News Sentiment Analysis:**
  - Integrates with the **News API** to gather recent articles.
  - Uses **TextBlob** to calculate sentiment polarity (-1 to +1).
  - Displays sentiment on an intuitive gauge chart.
- **Machine Learning Predictions:**
  - Trains a **Random Forest Regressor** on historical + sentiment features.
  - Predicts next-day percentage return and stock direction (UP/DOWN).
  - Shows feature importance for model interpretability.
- **Technical Indicators:**
  - Moving Averages (5-day and 20-day SMA).
  - RSI (Relative Strength Index).
  - Price spreads, differences, and volume changes.
- **Data Visualization:**
  - Interactive candlestick chart with overlays (Plotly).
  - Volume bars and moving averages.
  - Prediction confidence gauge and sentiment gauge.
  - Feature importance bar chart (Matplotlib).
- **Configurable Settings:**
  - Select company (Apple, Tesla, Microsoft, etc.).
  - Choose days of historical data (30–365).
  - Select number of prediction days (1–7).
  - Enter custom or default News API key.
- **Error Handling:** Displays safe fallbacks if API calls fail or insufficient data is available.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – for interactive UI
- **yfinance** – for stock market data
- **NewsAPI** – for financial news headlines
- **scikit-learn** – for machine learning models
- **TextBlob** – for sentiment analysis
- **Pandas / NumPy** – for data manipulation
- **Plotly** – for interactive charts
- **Matplotlib** – for static feature importance chart

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is **not intended as financial advice**. Predictions are experimental and should not be used for making investment decisions. Always consult with a certified financial advisor before making trades.

---

## Setup and Installation

### Prerequisites

- Python 3.10 or newer
- `pip` (Python package installer)
- A [NewsAPI](https://newsapi.org/) key (free tier available)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/stock-prediction-dashboard.git](https://github.com/your-username/stock-prediction-dashboard.git)
    cd stock-prediction-dashboard
    ```

2.  **Create and activate a virtual environment (recommended):**
    -   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file in the project directory with the following content (pinning versions is recommended):
    ```
    streamlit
    yfinance
    newsapi-python
    scikit-learn
    textblob
    pandas
    numpy
    plotly
    matplotlib
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download TextBlob corpora (one-time setup):**
    ```bash
    python -m textblob.download_corpora
    ```

---

## How to Use

1.  **Run the Streamlit app:**
    ```bash
    streamlit run v3.py
    ```

2.  **Enter NewsAPI Key:**
    Use the sidebar to enter your API key.

3.  **Configure Settings:**
    -   Select a company (e.g., Apple, Tesla, Microsoft).
    -   Choose the number of historical days to fetch.
    -   Select the number of days to predict.

4.  **Run the Analysis:**
    -   Click the **Run Analysis** button.
    -   The app will fetch stock price data, analyze news sentiment, train a machine learning model, and display the results.

5.  **View Results:**
    -   **Market Overview:** Current price, day change, volume, and sentiment.
    -   **Stock Price Analysis:** Candlestick chart with SMAs and volume.
    -   **Prediction Results:** Next-day direction & predicted price change.
    -   **Prediction Confidence:** Gauge showing bullish/bearish confidence.
    -   **Feature Importance:** See which factors drive predictions.
    -   **News Sentiment:** Gauge + list of recent articles with sentiment scores.

---

## Example Output

Prediction: Stock may go UP by 1.25% tomorrow.
Confidence: 0.75 (Bullish)

Feature Importance:

Price Difference: 0.40

Sentiment: 0.25

Volume Change: 0.20

High-Low Spread: 0.15
