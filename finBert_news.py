#import finBERT model
#use a pipeline as a high-level helper
from transformers import pipeline
import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd
from rapidfuzz import fuzz

#news api key: d99e2ff45573499aab19b385224cf018


midcap_df = pd.read_csv("midcaps.csv")

newsAPI = NewsApiClient(api_key='d99e2ff45573499aab19b385224cf018')
pipe = pipeline("text-classification", model="ProsusAI/finbert")


def finbert_analysis(ticker, name):
  try:
    ticker = yf.Ticker(ticker)
    news = ticker.news

    total_score = 0
    total_articles = 0

    print(f"\n----Finding news for {ticker} ----")

    #run all pulled articles through finBERT analysis
    for i, article in enumerate(news):

      #get article info
      content = article.get('content')
      title = content.get('title')
      summary = content.get('summary')
      #url = content.get('previewUrl') or 'No URL available'

      #skip for irrelevant articles
      score = fuzz.partial_ratio(name, summary)
      if score < 80:
          continue

      #pull finBERT sentiment from article summary
      analysis = pipe(summary)[0]

      #gather total sentiment
      if analysis['label'] == 'positive':
        total_score += analysis['score']
      elif analysis['label'] == 'negative':
        total_score -= analysis['score']

      total_articles += 1

    if total_articles != 0:
      final_score = total_score / total_articles
      print(f"Final Score: {final_score}")
    else:
      print("No articles found.")
  except:
    print(f"--- news get failed for {ticker}")

def check_stocks():
  for ticker, name in zip(midcap_df["Ticker"], midcap_df["Name"]):
    finbert_analysis(ticker, name)

if __name__ == "__main__":
  finbert_analysis("VKTX", "Viking Therapeutics, Inc")
  #check_stocks()