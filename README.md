
# Stock Market Prediction Model in Real-Time

This repository contains the code and resources for the **Stock Market Prediction Model in Real-Time**, a project that uses machine learning techniques and sentiment analysis to forecast stock prices and provide investment recommendations.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Model Architecture](#model-architecture)
- [Prediction](#prediction)
- [Recommendation](#recommendation)
- [Web Scraping](#web-scraping)
- [Sentiment Analysis](#sentiment-analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Stock market prediction involves projecting a company's future developments and estimating stock price directions. This project leverages LSTM neural networks to forecast stock prices and integrates sentiment analysis of news headlines for recommendation generation.

## Installation
To run this project, clone the repository and install the necessary Python libraries.

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

## Data Collection
The dataset is sourced from Yahoo Finance using the `yfinance` library. The following script can be used to download the historical stock data:

```python
import yfinance as yf

def download_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data.to_csv(f'{symbol}.csv')
        print(f'Data for {symbol} downloaded successfully.')
    except Exception as e:
        print(f"Error downloading data: {e}")

# Example usage
download_stock_data('GOOG', '2000-01-01', '2023-01-01')
```

## Model Architecture
The LSTM architecture is used to predict stock prices. Here is an example implementation:

```python
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('GOOG.csv')

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split data into training and testing
train_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[:train_data_len, :]

# Create training dataset
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, batch_size=1, epochs=1)
```

## Prediction
Once the model is trained, you can predict the stock prices for the test dataset:

```python
# Create testing dataset
test_data = scaled_data[train_data_len - 60:, :]
x_test = []
y_test = data['Close'][train_data_len:].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot results
import matplotlib.pyplot as plt

train = data[:train_data_len]
valid = data[train_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
```

## Recommendation
The project uses technical analysis tools such as SMA and EMA to generate buy/hold recommendations:

```python
# Simple Moving Average (SMA)
data['SMA_20'] = data['Close'].rolling(window=20).mean()

# Exponential Moving Average (EMA)
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Generate signals
data['Signal'] = 0
data.loc[data['Close'] > data['SMA_20'], 'Signal'] = 1
data.loc[data['Close'] > data['EMA_20'], 'Signal'] += 1

data['Recommendation'] = 'Hold'
data.loc[data['Signal'] == 2, 'Recommendation'] = 'Buy'
```

## Web Scraping
Automate news scraping from Yahoo News for sentiment analysis:

```python
import requests
from bs4 import BeautifulSoup

def get_news(symbol):
    url = f"https://news.search.yahoo.com/search?p={symbol}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    headlines = []
    for item in soup.find_all('h4', class_='s-title'):
        headlines.append(item.text)
    
    return headlines

# Example usage
news = get_news('GOOG')
print(news)
```

## Sentiment Analysis
Perform sentiment analysis on the scraped news headlines:

```python
from textblob import TextBlob

def analyze_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        analysis = TextBlob(headline)
        sentiments.append(analysis.sentiment.polarity)
    
    return sentiments

# Example usage
sentiments = analyze_sentiment(news)
print(sentiments)
```

## Conclusion
This project demonstrates a comprehensive approach to stock market prediction by combining machine learning and sentiment analysis. Future enhancements may include real-time data integration and more sophisticated sentiment analysis techniques.

## References
- [Stock Price Prediction Using LSTM](#)
- [Sentiment Analysis Techniques](#)
