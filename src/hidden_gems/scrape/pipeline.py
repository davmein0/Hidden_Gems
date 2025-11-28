# -*- coding: utf-8 -*-
"""
Refactored pipeline module (moved from top-level scrape_pipeline.py).
Reads SEC_API_KEY from environment variables and exposes run_pipeline(as_of_date)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from hidden_gems.io import ensure_data_dirs, raw_path, interim_path, processed_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# =====================
# CONFIGURATION
# =====================
MIN_MARKET_CAP = float(os.environ.get("MIN_MARKET_CAP", 2e9))
MAX_MARKET_CAP = float(os.environ.get("MAX_MARKET_CAP", 1e10))
YF_SLEEP = float(os.environ.get("YF_SLEEP", 0.22))
SEC_API_KEY = os.environ.get("SEC_API_KEY")
SEC_BASE_URL = os.environ.get("SEC_BASE_URL", "https://api.sec-api.io")


def fetch_nasdaq_list() -> list[str]:
    """Retrieve NASDAQ tickers list; fallback to Wikipedia NASDAQ-100 table."""
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text
        lines = [ln for ln in text.splitlines() if "Symbol|" not in ln and "File Creation Time" not in ln]
        tickers = [ln.split("|")[0] for ln in lines if "|" in ln]
        logging.info("Retrieved %d tickers from NASDAQ HTTPS feed", len(tickers))
        return tickers
    except Exception as e:
        logging.warning("NASDAQ HTTPS fetch failed (%s). Falling back to Wikipedia list.", e)
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get("https://en.wikipedia.org/wiki/NASDAQ-100", headers=headers).text
        nasdaq = pd.read_html(html)[4]
        tickers = nasdaq["Ticker"].dropna().unique().tolist()
        logging.info("Using %d NASDAQ-100 tickers from Wikipedia fallback", len(tickers))
        return tickers


def get_historical_market_cap(ticker: str, target_date: str) -> Optional[float]:
    """Compute historical market cap as of target_date using price × sharesOutstanding."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        shares = info.get("sharesOutstanding")
        hist = tk.history(start=target_date, end=pd.to_datetime(target_date) + pd.Timedelta(days=1))
        if hist.empty or shares is None:
            return None
        price = hist["Close"].iloc[0]
        return price * shares
    except Exception as e:
        logging.debug("Historical market cap fetch failed for %s: %s", ticker, e)
        return None


def find_midcap_tickers(tickers: list[str], as_of_date: str, min_cap: float = MIN_MARKET_CAP, max_cap: float = MAX_MARKET_CAP, limit: Optional[int] = None) -> pd.DataFrame:
    results = []
    logging.info("Scanning tickers for marketCap between %s and %s as of %s", min_cap, max_cap, as_of_date)
    for t in tqdm(tickers[:limit] if limit else tickers, desc=f"Scanning tickers {as_of_date}"):
        cap = get_historical_market_cap(t, as_of_date)
        if cap and (min_cap <= cap <= max_cap):
            try:
                tk = yf.Ticker(t)
                info = tk.info
                results.append({
                    "Ticker": t,
                    "Name": info.get("shortName") or info.get("longName") or "",
                    "MarketCap": cap,
                    "Sector": info.get("sector", ""),
                    "Industry": info.get("industry", ""),
                    "AsOfDate": as_of_date,
                    "DataCollectedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                })
            except Exception as e:
                logging.debug("yfinance info error for %s: %s", t, str(e))
        time.sleep(YF_SLEEP)
    df = pd.DataFrame(results).sort_values("MarketCap", ascending=False).reset_index(drop=True)
    logging.info("Found %d midcap tickers for %s", len(df), as_of_date)
    ensure_data_dirs()
    df.to_csv(raw_path(f"midcaps_{as_of_date}.csv"), index=False)
    return df


def fetch_financials_for_ticker(ticker: str, as_of_date: str) -> Optional[dict]:
    """Retrieve basic financial ratios (current snapshot — yfinance doesn’t provide historical fundamentals)."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
    except Exception as e:
        logging.debug("yfinance.info failed for %s: %s", ticker, str(e))
        return None

    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    de = info.get("debtToEquity") or info.get("debtEquity")
    peg = info.get("pegRatio")
    fcf = info.get("freeCashflow") or info.get("freeCashFlow")

    if de is None:
        try:
            bs = tk.balance_sheet
            if not bs.empty:
                total_liab = None
                total_equity = None
                for cand in ["totalStockholderEquity", "Total Stockholder Equity"]:
                    if cand in bs.index:
                        total_equity = bs.loc[cand].iat[0]
                for cand in ["Total Liab", "TotalLiab", "totalLiab", "totalLiabilities", "Total liabilities"]:
                    if cand in bs.index:
                        total_liab = bs.loc[cand].iat[0]
                if total_equity and total_equity != 0 and total_liab:
                    de = float(total_liab) / float(total_equity)
        except Exception:
            pass

    return {
        "Ticker": ticker,
        "PE_Ratio": float(pe) if pe not in (None, "None") else None,
        "PB_Ratio": float(pb) if pb not in (None, "None") else None,
        "DE_Ratio": float(de) if de not in (None, "None") else None,
        "PEG_Ratio": float(peg) if peg not in (None, "None") else None,
        "FreeCashFlow": float(fcf) if fcf not in (None, "None") else None,
        "AsOfDate": as_of_date,
        "DataCollectedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }


def collect_financials(midcap_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    records = []
    logging.info("Collecting financial metrics for %d tickers (%s)", len(midcap_df), as_of_date)
    for t in tqdm(midcap_df["Ticker"], desc=f"Collect financials {as_of_date}"):
        entry = fetch_financials_for_ticker(t, as_of_date)
        if entry:
            records.append(entry)
        time.sleep(YF_SLEEP)
    df = pd.DataFrame(records)
    ensure_data_dirs()
    df.to_csv(raw_path(f"financials_{as_of_date}.csv"), index=False)
    logging.info("Saved financials to financials_%s.csv", as_of_date)
    return df


def fetch_10k_filings_secapi(ticker: str, as_of_date: str, limit: int = 3) -> list[dict]:
    if not SEC_API_KEY:
        return []
    query = {
        "query": {"query_string": {"query": f"ticker:{ticker} AND formType:\"10-K\""}},
        "from": "0",
        "size": limit,
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    try:
        r = requests.post(f"{SEC_BASE_URL}/query", headers={"Authorization": f"Bearer {SEC_API_KEY}"}, json=query, timeout=30)
    except Exception as e:
        logging.warning("SEC API request failed for %s: %s", ticker, e)
        return []
    if r.status_code != 200:
        logging.debug("SEC API status for %s: %s %s", ticker, r.status_code, r.text[:200])
        return []
    filings = r.json().get("filings", [])
    rows = []
    for f in filings:
        filed_at = f.get("filedAt")
        if filed_at and filed_at > f"{as_of_date}T23:59:59Z":
            continue
        rows.append({
            "Ticker": ticker,
            "FiledAt": filed_at,
            "FilingURL": f.get("linkToFiling"),
            "AsOfDate": as_of_date,
            "DataCollectedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        })
    return rows


def collect_filings(midcap_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if not SEC_API_KEY:
        logging.info("No SEC_API_KEY set. Skipping filings collection.")
        return pd.DataFrame(columns=["Ticker", "FiledAt", "FilingURL", "AsOfDate", "DataCollectedAt"])
    all_filings = []
    logging.info("Collecting 10-K filings via sec-api for %d tickers (%s)", len(midcap_df), as_of_date)
    for t in tqdm(midcap_df["Ticker"], desc=f"SEC filings {as_of_date}"):
        try:
            rows = fetch_10k_filings_secapi(t, as_of_date)
            all_filings.extend(rows)
        except Exception as e:
            logging.warning("Error fetching filings for %s: %s", t, e)
        time.sleep(0.2)
    df = pd.DataFrame(all_filings)
    ensure_data_dirs()
    df.to_csv(raw_path(f"filings_{as_of_date}.csv"), index=False)
    logging.info("Saved filings to filings_%s.csv", as_of_date)
    return df


def merge_and_save(midcap_df: pd.DataFrame, financials_df: pd.DataFrame, filings_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    merged = pd.merge(midcap_df, financials_df, on=["Ticker", "AsOfDate", "DataCollectedAt"], how="left")
    if not filings_df.empty:
        filing_links = filings_df.groupby("Ticker")[['FilingURL', 'FiledAt']].first().reset_index()
        merged = pd.merge(merged, filing_links, on="Ticker", how="left")
    out_file = interim_path(f"merged_dataset_{as_of_date}.csv")
    ensure_data_dirs()
    merged.to_csv(out_file, index=False)
    logging.info("Saved merged dataset to %s", out_file)
    return merged


def run_pipeline(as_of_date: str, limit_tickers: Optional[int] = None) -> pd.DataFrame:
    tickers = fetch_nasdaq_list()
    midcaps = find_midcap_tickers(tickers, as_of_date, limit=limit_tickers)
    financials = collect_financials(midcaps, as_of_date)
    filings = collect_filings(midcaps, as_of_date)
    merged = merge_and_save(midcaps, financials, filings, as_of_date)
    return merged


def main():
    logging.info("Starting scrape pipeline for current and last year snapshots.")
    snapshots = [
        "2024-10-17",  # last year snapshot
        datetime.today().strftime("%Y-%m-%d")  # current snapshot
    ]

    merged_files = []
    for date in snapshots:
        merged = run_pipeline(date, limit_tickers=100)  # you can adjust limit
        merged_files.append(str(interim_path(f"merged_dataset_{date}.csv")))

    # Combine both years into one dataset
    dfs = [pd.read_csv(f) for f in merged_files if os.path.exists(f)]
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        ensure_data_dirs()
        combined.to_csv(processed_path("merged_combined.csv"), index=False)
        logging.info("Saved combined dataset (both years) to %s", processed_path("merged_combined.csv"))

    print("\nDone. Files written:")
    for f in merged_files + [str(processed_path("merged_combined.csv"))]:
        if os.path.exists(f):
            print(" -", f)


if __name__ == "__main__":
    main()
import requests
import pandas as pd
import yfinance as yf
import time
import os
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# =====================
# CONFIGURATION
# =====================
MIN_MARKET_CAP = 2e9
MAX_MARKET_CAP = 1e10

SEC_API_KEY = "0311835e514a4e40f0dbcea8f2007b53e2e05ad23e34947e453e22181870f75d"
SEC_BASE_URL = "https://api.sec-api.io"

YF_SLEEP = 0.22

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# =====================
# CORE FUNCTIONS
# =====================

def fetch_nasdaq_list():
    """Retrieve NASDAQ tickers list."""
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text
        lines = [ln for ln in text.splitlines() if "Symbol|" not in ln and "File Creation Time" not in ln]
        tickers = [ln.split("|")[0] for ln in lines if "|" in ln]
        logging.info("Retrieved %d tickers from NASDAQ HTTPS feed", len(tickers))
        return tickers
    except Exception as e:
        logging.warning("NASDAQ HTTPS fetch failed (%s). Falling back to Wikipedia list.", e)
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get("https://en.wikipedia.org/wiki/NASDAQ-100", headers=headers).text
        nasdaq = pd.read_html(html)[4]
        tickers = nasdaq["Ticker"].dropna().unique().tolist()
        logging.info("Using %d NASDAQ-100 tickers from Wikipedia fallback", len(tickers))
        return tickers


def get_historical_market_cap(ticker, target_date):
    """Compute historical market cap as of target_date using price × sharesOutstanding."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        shares = info.get("sharesOutstanding")
        hist = tk.history(start=target_date, end=pd.to_datetime(target_date) + pd.Timedelta(days=1))
        if hist.empty or shares is None:
            return None
        price = hist["Close"].iloc[0]
        return price * shares
    except Exception as e:
        logging.debug("Historical market cap fetch failed for %s: %s", ticker, e)
        return None


def find_midcap_tickers(tickers, as_of_date, min_cap=MIN_MARKET_CAP, max_cap=MAX_MARKET_CAP, limit=None):
    """Find tickers whose market cap was midcap on a specific historical date."""
    results = []
    logging.info("Scanning tickers for marketCap between %s and %s as of %s", min_cap, max_cap, as_of_date)
    for t in tqdm(tickers[:limit] if limit else tickers, desc=f"Scanning tickers {as_of_date}"):
        cap = get_historical_market_cap(t, as_of_date)
        if cap and (min_cap <= cap <= max_cap):
            try:
                tk = yf.Ticker(t)
                info = tk.info
                results.append({
                    "Ticker": t,
                    "Name": info.get("shortName") or info.get("longName") or "",
                    "MarketCap": cap,
                    "Sector": info.get("sector", ""),
                    "Industry": info.get("industry", ""),
                    "AsOfDate": as_of_date,
                    "DataCollectedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                })
            except Exception as e:
                logging.debug("yfinance info error for %s: %s", t, str(e))
        time.sleep(YF_SLEEP)
    df = pd.DataFrame(results).sort_values("MarketCap", ascending=False).reset_index(drop=True)
    logging.info("Found %d midcap tickers for %s", len(df), as_of_date)
    ensure_data_dirs()
    df.to_csv(raw_path(f"midcaps_{as_of_date}.csv"), index=False)
    return df


def fetch_financials_for_ticker(ticker, as_of_date):
    """Retrieve basic financial ratios (current snapshot — yfinance doesn’t provide historical fundamentals)."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
    except Exception as e:
        logging.debug("yfinance.info failed for %s: %s", ticker, str(e))
        return None

    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    de = info.get("debtToEquity") or info.get("debtEquity")
    peg = info.get("pegRatio")
    fcf = info.get("freeCashflow") or info.get("freeCashFlow")

    if de is None:
        try:
            bs = tk.balance_sheet
            if not bs.empty:
                total_liab = None
                total_equity = None
                for cand in ["totalStockholderEquity", "Total Stockholder Equity"]:
                    if cand in bs.index:
                        total_equity = bs.loc[cand].iat[0]
                for cand in ["Total Liab", "TotalLiab", "totalLiab", "totalLiabilities", "Total liabilities"]:
                    if cand in bs.index:
                        total_liab = bs.loc[cand].iat[0]
                if total_equity and total_equity != 0 and total_liab:
                    de = float(total_liab) / float(total_equity)
        except Exception:
            pass

    return {
        "Ticker": ticker,
        "PE_Ratio": float(pe) if pe not in (None, "None") else None,
        "PB_Ratio": float(pb) if pb not in (None, "None") else None,
        "DE_Ratio": float(de) if de not in (None, "None") else None,
        "PEG_Ratio": float(peg) if peg not in (None, "None") else None,
        "FreeCashFlow": float(fcf) if fcf not in (None, "None") else None,
        "AsOfDate": as_of_date,
        "DataCollectedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }


def collect_financials(midcap_df, as_of_date):
    """Collect financial metrics for all midcaps."""
    records = []
    logging.info("Collecting financial metrics for %d tickers (%s)", len(midcap_df), as_of_date)
    for t in tqdm(midcap_df["Ticker"], desc=f"Collect financials {as_of_date}"):
        entry = fetch_financials_for_ticker(t, as_of_date)
        if entry:
            records.append(entry)
        time.sleep(YF_SLEEP)
    df = pd.DataFrame(records)
    ensure_data_dirs()
    df.to_csv(raw_path(f"financials_{as_of_date}.csv"), index=False)
    logging.info("Saved financials to financials_%s.csv", as_of_date)
    return df


def fetch_10k_filings_secapi(ticker, as_of_date, limit=3):
    """Retrieve 10-K filings before as_of_date."""
    if not SEC_API_KEY:
        return []
    query = {
        "query": {"query_string": {"query": f"ticker:{ticker} AND formType:\"10-K\""}},
        "from": "0",
        "size": limit,
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    try:
        r = requests.post(f"{SEC_BASE_URL}/query", headers={"Authorization": f"Bearer {SEC_API_KEY}"}, json=query, timeout=30)
    except Exception as e:
        logging.warning("SEC API request failed for %s: %s", ticker, e)
        return []
    if r.status_code != 200:
        logging.debug("SEC API status for %s: %s %s", ticker, r.status_code, r.text[:200])
        return []
    filings = r.json().get("filings", [])
    rows = []
    for f in filings:
        filed_at = f.get("filedAt")
        if filed_at and filed_at > f"{as_of_date}T23:59:59Z":
            continue  # skip future filings
        rows.append({
            "Ticker": ticker,
            "FiledAt": filed_at,
            "FilingURL": f.get("linkToFiling"),
            "AsOfDate": as_of_date,
            "DataCollectedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        })
    return rows


def collect_filings(midcap_df, as_of_date):
    """Collect SEC 10-K filings with timestamps."""
    if not SEC_API_KEY:
        logging.info("No SEC_API_KEY set. Skipping filings collection.")
        return pd.DataFrame(columns=["Ticker", "FiledAt", "FilingURL", "AsOfDate", "DataCollectedAt"])
    all_filings = []
    logging.info("Collecting 10-K filings via sec-api for %d tickers (%s)", len(midcap_df), as_of_date)
    for t in tqdm(midcap_df["Ticker"], desc=f"SEC filings {as_of_date}"):
        try:
            rows = fetch_10k_filings_secapi(t, as_of_date)
            all_filings.extend(rows)
        except Exception as e:
            logging.warning("Error fetching filings for %s: %s", t, e)
        time.sleep(0.2)
    df = pd.DataFrame(all_filings)
    ensure_data_dirs()
    df.to_csv(raw_path(f"filings_{as_of_date}.csv"), index=False)
    logging.info("Saved filings to filings_%s.csv", as_of_date)
    return df


def merge_and_save(midcap_df, financials_df, filings_df, as_of_date):
    """Merge datasets and save merged CSV."""
    merged = pd.merge(midcap_df, financials_df, on=["Ticker", "AsOfDate", "DataCollectedAt"], how="left")
    if not filings_df.empty:
        filing_links = filings_df.groupby("Ticker")[["FilingURL", "FiledAt"]].first().reset_index()
        merged = pd.merge(merged, filing_links, on="Ticker", how="left")
    out_file = interim_path(f"merged_dataset_{as_of_date}.csv")
    ensure_data_dirs()
    merged.to_csv(out_file, index=False)
    logging.info("Saved merged dataset to %s", out_file)
    return merged


# =====================
# MAIN PIPELINE
# =====================

def run_pipeline(as_of_date, limit_tickers=None):
    tickers = fetch_nasdaq_list()
    midcaps = find_midcap_tickers(tickers, as_of_date, limit=limit_tickers)
    financials = collect_financials(midcaps, as_of_date)
    filings = collect_filings(midcaps, as_of_date)
    merged = merge_and_save(midcaps, financials, filings, as_of_date)
    return merged


def main():
    logging.info("Starting scrape pipeline for current and last year snapshots.")
    snapshots = [
        "2024-10-17",  # last year snapshot
        datetime.today().strftime("%Y-%m-%d")  # current snapshot
    ]

    merged_files = []
    for date in snapshots:
        merged = run_pipeline(date, limit_tickers=100)  # you can adjust limit
        merged_files.append(str(interim_path(f"merged_dataset_{date}.csv")))

    # Combine both years into one dataset
    dfs = [pd.read_csv(f) for f in merged_files if os.path.exists(f)]
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv("merged_combined.csv", index=False)
        logging.info("Saved combined dataset (both years) to merged_combined.csv")

    print("\nDone. Files written:")
    for f in merged_files + ["merged_combined.csv"]:
        if os.path.exists(f):
            print(" -", f)


if __name__ == "__main__":
    main()
