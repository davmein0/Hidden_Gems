import requests
from bs4 import BeautifulSoup
import os
import re
import pandas as pd
from time import sleep
from typing import Optional

# ---- CONFIG ----
USER_AGENT = "slplayford@gmail.com"
FILING_TYPE = "10-K"
SAVE_FOLDER = "SENTIMENT"
MIDCAP_FILE = "midcaps.csv"
REQUEST_DELAY = 0.1  # SEC rate limit: 10 requests/second

# ---- SETUP ----
headers = {'User-Agent': USER_AGENT}
session = requests.Session()  # Reuse connections
session.headers.update(headers)

# Cache for CIK lookups
_cik_cache = None


def get_all_ciks():
    """Fetch and cache all CIK mappings once."""
    global _cik_cache
    if _cik_cache is None:
        _cik_cache = session.get(
            "https://www.sec.gov/files/company_tickers.json"
        ).json()
    return _cik_cache


def get_cik(ticker: str) -> Optional[str]:
    """Get 10-digit CIK for a given ticker symbol."""
    tickers = get_all_ciks()
    ticker_lower = ticker.lower()
    
    for v in tickers.values():
        if v['ticker'].lower() == ticker_lower:
            return str(v['cik_str']).zfill(10)
    return None


def get_latest_accession(cik: str, form_type: str) -> Optional[str]:
    """Get most recent accession number for specified filing type."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = session.get(url).json()
    
    filings = data['filings']['recent']
    for form, acc in zip(filings['form'], filings['accessionNumber']):
        if form == form_type:
            return acc
    return None


def get_main_html_url(cik: str, accession: str, ticker: str) -> Optional[str]:
    """Find the main HTML file URL for the filing."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    index_url = base + "index.json"
    index = session.get(index_url).json()
    
    items = index["directory"]["item"]
    ticker_lower = ticker.lower()
    
    # First pass: find HTML with ticker in name
    for item in items:
        name = item["name"].lower()
        if name.endswith(".htm") and ticker_lower in name:
            return base + item["name"]
    
    # Fallback: first .htm file
    for item in items:
        if item["name"].lower().endswith(".htm"):
            return base + item["name"]
    
    return None


def download_and_parse_html(url: str, ticker: str, form_type: str, folder: str) -> str:
    """Download HTML and parse to clean text."""
    response = session.get(url)
    response.raise_for_status()
    
    # Save raw HTML
    html_path = os.path.join(folder, f"{ticker}_{form_type}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    
    # Extract and clean text
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    
    # Save cleaned text
    txt_path = os.path.join(folder, f"{ticker}_{form_type}_clean.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"✅ {ticker}: Saved HTML and clean text")
    return text


def process_ticker(ticker: str) -> bool:
    """Process a single ticker. Returns True if successful."""
    ticker = ticker.lower()
    
    # Create folder
    company_folder = os.path.join(SAVE_FOLDER, ticker)
    os.makedirs(company_folder, exist_ok=True)
    
    # Get CIK
    cik = get_cik(ticker)
    if not cik:
        print(f"❌ {ticker}: Ticker not found")
        return False
    
    # Get latest filing
    accession = get_latest_accession(cik, FILING_TYPE)
    if not accession:
        print(f"❌ {ticker}: No {FILING_TYPE} found")
        return False
    
    # Get HTML URL
    html_url = get_main_html_url(cik, accession, ticker)
    if not html_url:
        print(f"❌ {ticker}: Main HTML file not found")
        return False
    
    # Download and parse
    download_and_parse_html(html_url, ticker, FILING_TYPE, company_folder)
    return True


def main():
    """Main execution function."""
    # Load tickers
    df_midcap = pd.read_csv(MIDCAP_FILE)
    ticker_list = df_midcap["Ticker"].dropna().unique().tolist()
    
    print(f"Processing {len(ticker_list)} tickers...\n")
    
    success_count = 0
    for i, ticker in enumerate(ticker_list, 1):
        print(f"[{i}/{len(ticker_list)}] Processing {ticker}...")
        
        try:
            if process_ticker(ticker):
                success_count += 1
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")
        
        # Rate limiting
        sleep(REQUEST_DELAY)
    
    print(f"\n{'='*50}")
    print(f"Complete: {success_count}/{len(ticker_list)} successful")


if __name__ == "__main__":
    main()