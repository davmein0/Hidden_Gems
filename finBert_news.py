#import finBERT model
#use a pipeline as a high-level helper
from transformers import pipeline
import yfinance as yf
from newsapi import NewsApiClient

#news api key: d99e2ff45573499aab19b385224cf018

newsAPI = NewsApiClient(api_key='d99e2ff45573499aab19b385224cf018')
pipe = pipeline("text-classification", model="ProsusAI/finbert")

ticker = yf.Ticker("AAPL")
keyword = "apple"
news = ticker.news

total_score = 0
total_articles = 0

#run all pulled articles through finBERT analysis
for i, article in enumerate(news):

  #get article info
  #print(article)
  content = article.get('content')
  title = content.get('title')
  summary = content.get('summary')
  url = content.get('previewUrl') or 'No URL available'
  #print(url)
  #print(content.get('url') or 'No URL available')

  #skip for irrelevant articles
  if keyword not in summary.lower():
    continue

  #pull finBERT sentiment from article summary
  analysis = pipe(summary)[0]
      #later integrate to fully utilize the historical 10k data
      #parse large docs into small summaries

  #gather total sentiment
  if analysis['label'] == 'positive':
    total_score += analysis['score']
  elif analysis['label'] == 'negative':
    total_score -= analysis['score']

  total_articles += 1


  print(f"Title: {title}")
  print(f"Summary: {summary}")
  print(url)
  print(f"Analysis: {analysis}")
  print("-" * 40)


if total_articles != 0:
  final_score = total_score / total_articles
  print(f"Final Score: {final_score}")
else:
  print("No articles found.")