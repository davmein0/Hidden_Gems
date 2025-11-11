import requests
from bs4 import BeautifulSoup
import os
import re
import pandas as pd

# ---- CONFIG ----
USER_AGENT = "hidden@gmail.com"   
#TICKER = "aapl"                     # change as needed
FILING_TYPE = "10-K"                # change as needed
SAVE_FOLDER = "SENTIMENT"

MIDCAP_FILE = "midcaps.csv"

df_midcap = pd.read_csv(MIDCAP_FILE)

ticker_list = df_midcap["Ticker"].dropna().unique().tolist()

headers = {'User-Agent': USER_AGENT}



#Get 10-digit CIK for a given ticker symbol.
def get_cik(ticker):
    tickers = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers
    ).json()

    for v in tickers.values():
        if v['ticker'].lower() == ticker.lower():
            return str(v['cik_str']).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found.")

#Get most recent accession number - (latest filings)
def get_latest_accession(cik, form_type):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=headers).json()
    for form, acc in zip(data['filings']['recent']['form'], data['filings']['recent']['accessionNumber']):
        if form == form_type:
            return acc
    raise ValueError(f"No {form_type} found for {cik}")


#get the full company 10-k
def get_main_html_url(cik, accession, form_type, ticker):
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    index_url = base + "index.json"
    index = requests.get(index_url, headers=headers).json()

    # Find the HTML file that looks like the main report
    for item in index["directory"]["item"]:
        name = item["name"].lower()
        if name.endswith(".htm") and (ticker in name):
            return base + item["name"]

    # Fallback: return first .htm file
    for item in index["directory"]["item"]:
        if item["name"].lower().endswith(".htm"):
            return base + item["name"]

    raise ValueError("Main HTML file not found.")


#download and parse the html into .txt format
def download_and_parse_html(url, ticker, form_type, folder):
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Save raw HTML
    html_filename = os.path.join(folder, f"{ticker}_{form_type}.html")

    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    # ---- PARSE CLEAN TEXT ----
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Get text, clean extra whitespace
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()

    # Save cleaned text
    txt_filename = os.path.join(folder, f"{ticker}_{form_type}_clean.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Saved raw HTML: {html_filename}")
    print(f"✅ Saved clean text: {txt_filename}")
    return text


# ---- MAIN ----
#cik = get_cik(TICKER)
#print(f"CIK for {TICKER}: {cik}")

for t in ticker_list:

    try:
        t = t.lower()

        #MAKE SAVE FOLDER
        company_folder = os.path.join(SAVE_FOLDER, t)
        os.makedirs(company_folder, exist_ok=True)
        
        cik = get_cik(t)
        print(f"CIK FOR {t}: {cik}")

        accession = get_latest_accession(cik, FILING_TYPE)
        print(f"Latest {FILING_TYPE} accession: {accession}")

        html_url = get_main_html_url(cik, accession, FILING_TYPE, t)
        print(f"Downloading: {html_url}")

        filing_text = download_and_parse_html(html_url, t, FILING_TYPE, company_folder)

    except Exception as e:
        print("Error:", e)
#print(f"\nFirst 500 characters of cleaned text:\n")
#print(filing_text[:500])
