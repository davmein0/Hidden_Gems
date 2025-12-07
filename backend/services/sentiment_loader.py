import os
import logging
from typing import Dict, Optional, List
from transformers import pipeline
from newsapi import NewsApiClient
from rapidfuzz import fuzz
import yfinance as yf

logger = logging.getLogger(__name__)

_PIPELINE = None
_NEWSAPI = None

newsAPI = NewsApiClient(api_key='d99e2ff45573499aab19b385224cf018')
pipe = pipeline("text-classification", model="ProsusAI/finbert")

def _get_pipeline():
    if pipe:
        return pipe
    global _PIPELINE
    if _PIPELINE is None:
        model_name = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")
        logger.info("Loading finBERT model: %s", model_name)
        _PIPELINE = pipeline("text-classification", model=model_name)
    return _PIPELINE

def _get_newsapi():
    if newsAPI:
        return newsAPI
    global _NEWSAPI
    if _NEWSAPI is None:
        key = os.environ.get("NEWSAPI_KEY")
        if key:
            _NEWSAPI = NewsApiClient(api_key=key)
        else:
            _NEWSAPI = None
    return _NEWSAPI

def _fetch_articles(ticker: str, name: Optional[str], top_n: int = 10) -> List[Dict]:
    """
    Try NewsAPI first (if configured), otherwise fallback to yfinance.ticker.news.
    Returns list of article dicts with keys: title, description, content
    """
    newsapi = _get_newsapi()
    q = name or ticker
    articles = []
    if newsapi:
        try:
            res = newsapi.get_everything(q=q, language="en", sort_by="relevancy", page_size=top_n)
            articles = res.get("articles", [])[:top_n]
        except Exception as e:
            logger.exception("NewsAPI fetch failed, falling back to yfinance: %s", e)

    if not articles:
        try:
            t = yf.Ticker(ticker)
            yf_news = getattr(t, "news", []) or []
            # unify shape: title/description/content
            for a in yf_news[:top_n]:
                articles.append({
                    "title": a.get("title") or "",
                    "description": a.get("summary") or a.get("summary"), 
                    "content": a.get("summary") or a.get("title") or ""
                })
        except Exception:
            logger.exception("yfinance news fetch failed for %s", ticker)

    return articles

def compute_sentiment_for_ticker(ticker: str, name: Optional[str] = None, top_n: int = 10, fuzz_threshold: int = 70) -> Dict[str, float]:
    """
    Returns aggregated finBERT sentiment features for ticker:
    {
      "finbert_polarity_avg": float,  # positive score minus negative, averaged
      "finbert_count": int            # number of relevant articles scored
    }
    """
    pipe = _get_pipeline()
    articles = _fetch_articles(ticker, name, top_n=top_n)
    if not articles:
        return {"finbert_polarity_avg": 0.0, "finbert_count": 0}

    total = 0.0
    count = 0
    for a in articles:
        text = (a.get("description") or a.get("content") or a.get("title") or "").strip()
        if not text:
            continue
        # quick relevance filter using fuzzy match against company name
        if name:
            score = fuzz.partial_ratio(name.lower(), text.lower())
            if score < fuzz_threshold:
                continue
        try:
            out = pipe(text)[0]
            label = out.get("label", "").lower()
            score = float(out.get("score", 0.0))
            if "positive" in label:
                total += score
            elif "negative" in label:
                total -= score
            # neutral contributes 0
            count += 1
        except Exception:
            logger.exception("finBERT scoring failed for ticker %s", ticker)

    if count == 0:
        return {"finbert_polarity_avg": 0.0, "finbert_count": 0}
    return {"finbert_polarity_avg": round(total / count, 4), "finbert_count": count}