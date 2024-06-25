#author: @shahnawazsyed

import pandas as pd
import os 
import numpy as np
import yfinance as yf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)
def get_sentiment(text, stock_mentions): #sentiment analysis using TextBlob library
    if stock_mentions: #some days may not have mentions of the stock in WSB
        return TextBlob(text).sentiment.polarity
    else: #thus we assign those days a sentiment of 0 to avoid gaps in graph
        return 0
def contains_stock_keyword(text, stock):
    return stock.lower() in text.lower()

date_formatter = mdates.DateFormatter('%m-%d')

#start and end dates pertain to dataset's limitations
start_date = '2021-1-1'
end_date = '2021-2-28'

stock = input("Enter stock symbol: ")
print(f"Generating analysis for {stock} from {start_date} to {end_date}")

df = pd.read_csv('reddit_data.csv')
df.dropna(subset=['title', 'body'], inplace=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

df['cleaned_text'] = df['title'] + ' ' + df['body'].apply(preprocess_text) #cleaning text data for easier analysis
df['stock_mentions'] = df['cleaned_text'].apply(lambda x: contains_stock_keyword(x, stock)) #grabbing instances of mentions of the stock
df['sentiment'] = df.apply(lambda row: get_sentiment(row['cleaned_text'], row['stock_mentions']), axis=1) #calculating sentiment using get_sentiment
df['date'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')
daily_sentiment = df.groupby(df['date'].dt.date)['sentiment'].mean() #computing the daily sentiment (mean sentiment of all mentions on that day)

#yahoo finance used to pull stock price data
stock_data = yf.download(tickers = stock, start = start_date, end = end_date)
stock_data.reset_index(inplace=True)
stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

#dual y-axes plotted (right side: sentiment, left side: stock price)
fig, ax1 = plt.subplots()
color = 'blue'
ax1.set_ylabel('Average Sentiment (solid)', color=color, size=15) 
ax1.plot(daily_sentiment.index, daily_sentiment.values, color=color, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_formatter(date_formatter)

ax2 = ax1.twinx()

color = 'firebrick'
ax2.set_xlabel('Date')
ax2.set_ylabel('Closing price (dashed)', color=color, size=15)
ax2.plot(stock_data['Date'], stock_data['Close'], color=color, linewidth=2.5, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=color)

fig.set_figheight(6)
fig.set_figwidth(11)
plt.title(f"{stock} Average Daily Sentiment in R/WallStreetBets and Closing Price (Jan-Feb 2021)", weight='bold', size=15)
plt.show()


