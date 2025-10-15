import requests
import pandas as pd
import yfinance as yf
import time
import os
import logging
from tqdm import tqdm

MIN_MARKET_CAP = 2e9
MAX_MARKET_CAP = 1e10

MIDCAP_FILE = "midcaps.csv"
FINANCIALS_FILE = "financials.csv"
FILINGS_FILE = "10k_filings.csv"
MERGED_FILE = "merged_dataset.csv"

SEC_API_KEY = "0311835e514a4e40f0dbcea8f2007b53e2e05ad23e34947e453e22181870f75d"
SEC_BASE_URL = "https://api.sec-api.io"

YF_SLEEP = 0.22

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def fetch_nasdaq_list():
    try:
        url="https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
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
    
def find_midcap_tickers(tickers, min_cap=MIN_MARKET_CAP, max_cap=MAX_MARKET_CAP, limit=None):
    results = []
    count = 0
    logging.info("Scanning tickers for marketCap between %s and %s", min_cap, max_cap)
    for t in tqdm(tickers[:limit] if limit else tickers, desc="Scanning tickers"):
        try:
            tk = yf.Ticker(t)
            info = tk.info
            cap = info.get("marketCap")
            if cap and (min_cap <= cap <= max_cap):
                results.append({
                    "Ticker": t,
                    "Name": info.get("shortName") or info.get("longName") or "",
                    "MarketCap": cap,
                    "Sector": info.get("sector", ""),
                    "Industry": info.get("industry", "")
                })

            time.sleep(YF_SLEEP)
        except Exception as e:
            logging.debug("yfinance error for %s: %s", t, str(e))
            time.sleep(YF_SLEEP)
        count += 1

    df = pd.DataFrame(results).sort_values("MarketCap", ascending=False).reset_index(drop=True)
    logging.info("Found %d mid-cap tickers", len(df))
    df.to_csv(MIDCAP_FILE, index=False)
    return df

def fetch_financials_for_ticker(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
    except Exception as e:
        logging.debug("yfinance.info failed for %s: %s", ticker, str(e))
        return None
    
    pe = info.get("trailingPE") or info.get("forwardPE") or None
    pb = info.get("priceToBook") or None
    de = info.get("debtToEquity") or info.get("debtEquity")
    peg = info.get("pegRatio") or None
    fcf = info.get("freeCashflow") or info.get("freeCashFlow") or None

    if de is None:
        try:
            bs = tk.balance_sheet

            if not bs.empty:
                labels = bs.index.str.lower()

                total_liab = None
                total_equity = None
                for cand in ["totalStockholderEquity", "Total Stockholder Equity"]:
                    if cand in bs.index:
                        total_equity = bs.loc[cand].iat[0]
                for cand in ["Total Liab", "TotalLiab", "totalLiab", "totalLiabilities", "Total liabilities"]:
                    if cand in bs.index:
                        total_liab = bs.loc[cand].iat[0]
                if total_equity and total_equity != 0:
                    if total_liab is None:
                        try:
                            td = 0
                            for cand in ["Long Term Debt", "longTermDebt"]:
                                if cand in bs.index:
                                    td += bs.loc[cand].iat[0] or 0
                            if td and total_equity:
                                de = float(td) / float(total_equity)
                        except Exception:
                            pass
                    else:
                        de = float(total_liab) / float(total_equity)
        except Exception:
            pass

    out = {
        "Ticker": ticker,
        "PE_Ratio": float(pe) if pe not in (None, "None") else None,
        "PB_Ratio": float(pb) if pb not in (None, "None") else None,
        "DE_Ratio": float(de) if de not in (None, "None") else None,
        "PEG_Ratio": float(peg) if peg not in (None, "None") else None,
        "FreeCashFlow": float(fcf) if fcf not in (None, "None") else None
    }

    return out

def collect_financials(midcap_df):
    records = []
    logging.info("Collecting financial metrics for %d tickers", len(midcap_df))
    for t in tqdm(midcap_df["Ticker"], desc="Collect financials"):
        entry = fetch_financials_for_ticker(t)
        if entry:
            records.append(entry)
        time.sleep(YF_SLEEP)
    df = pd.DataFrame(records)
    df.to_csv(FINANCIALS_FILE, index=False)
    logging.info("Saved financials to %s", FINANCIALS_FILE)
    
    return df

def fetch_10k_filings_secapi(ticker, limit=3):
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
        rows.append({
            "Ticker": ticker,
            "FiledAt": f.get("filedAt"),
            "FilingURL": f.get("linkToFiling")
        })
    return rows

def collect_filings(midcap_df):
    if not SEC_API_KEY:
        logging.info("No SEC_API_KEY set. Skipping filings collection.")
        return pd.DataFrame(columns=["Ticker", "FiledAt", "FilingURL"])
    all_filings = []
    logging.info("Collecting 10-K filings via sec-api for %d tickers", len(midcap_df))
    for t in tqdm(midcap_df["Ticker"], desc="SEC filings"):
        try:
            rows = fetch_10k_filings_secapi(t)
            all_filings.extend(rows)
        except Exception as e:
            logging.warning("Error fetching filings for %s: %s", t, e)
        time.sleep(0.2)
    df = pd.DataFrame(all_filings)
    df.to_csv(FILINGS_FILE, index=False)
    logging.info("Saved filings to %s", FILINGS_FILE)

    return df

def merge_and_save(midcap_df, financials_df, filings_df):
    merged = pd.merge(midcap_df, financials_df, on="Ticker", how="left")
    if not filings_df.empty:
        filing_links = filings_df.groupby("Ticker")["FilingURL"].first().reset_index()
        merged = pd.merge(merged, filing_links, on="Ticker", how="left")
    merged.to_csv(MERGED_FILE, index=False)
    logging.info("Saved merged dataset to %s", MERGED_FILE)
    
    return merged

def main(limit_tickers=None):
    tickers = fetch_nasdaq_list()

    midcaps = find_midcap_tickers(tickers, limit=limit_tickers)

    financials = collect_financials(midcaps)

    filings = collect_filings(midcaps)

    merged = merge_and_save(midcaps, financials, filings)
    print("\nDone. Files written:")
    for f in [MIDCAP_FILE, FINANCIALS_FILE, FILINGS_FILE, MERGED_FILE]:
        if os.path.exists(f):
            print(" -", f)
    print("\nSample (first 10 rows):\n")
    print(merged.head(10).to_string(index=False))

if __name__ == "__main__":
    main()