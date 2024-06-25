# WSB-Sentiment-Analysis
Author: @shahnawazsyed (linkedin.com/in/shahnawazsyed1/)

Using a dataset provided by ASU's Unit for Data Science (01/2021-02/2021), this program analyzes the sentiment of a certain inputted stock on r/WallStreetBets (textblob), computes the average daily sentiment, and plots (matplotlib) it against the stock's actual price (yfinance).

As not all stocks were mentioned on each day, a sentiment value of 0 was assigned in such cases.

Suggested usage: "memefied" stocks that were popular on r/WallStreetBets in January and February of 2021 (e.g. GME, AMC, BB, TSLA).

Potential future improvements: including text data from other online forums (e.g. Yahoo Finance, Stockopedia).

Libraries used: pandas, os, numpy, yfinance, nltk, re, textblob, matplotlib, seaborn
