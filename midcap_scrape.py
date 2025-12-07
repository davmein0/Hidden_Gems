import os, time, random, pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

MIDCAP_FILE = "midcaps.csv"
OUTPUT_DIR = "midcap_financials"

df_midcap = pd.read_csv(MIDCAP_FILE)
tickers = df_midcap["Ticker"].dropna().unique().tolist()
print(f"Loaded {len(tickers)} tickers from {MIDCAP_FILE}")

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def login_once(driver):
    driver.get("https://stockanalysis.com/pro/")
    input("press enter once logged in")

def make_driver(): 
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def scrape_stockanalysis_tables(ticker, driver):
    base_url = f"https://stockanalysis.com/stocks/{ticker.lower()}/financials/"
    pages = {
        "ratios": base_url + "ratios/",
        "income": base_url + "income-statement/",
        "balance": base_url + "balance-sheet/" 
    }

    company_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(company_dir, exist_ok=True)

    for name, url in pages.items():
        out_path = os.path.join(company_dir, f"{ticker}_{name}.csv")
 
        if os.path.exists(out_path):
            print(f"[{ticker}] Skipping {name} (already exists)")
            continue

        print(f"[{ticker}] Scraping {name} page...")
        try:
            driver.get(url)
            time.sleep(4)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            table = soup.find("table")
            if not table:
                print(f"No table found on {name}")
                continue

            header_rows = table.find_all("tr")
            header_candidates = [tr.find_all("th") for tr in header_rows if tr.find_all("th")]
            headers = [th.get_text(strip=True) for th in header_candidates[-1]]

            rows = []
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if tds:
                    rows.append([td.get_text(strip=True) for td in tds])
 
            for r in rows:
                while len(r) < len(headers):
                    r.append("")

            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Saved {out_path} ({len(df)} rows, {len(headers)} cols)")
        except Exception as e:
            print(f"Error scraping {ticker} {name}: {e}")
        time.sleep(random.uniform(3, 6))

# make sure to run caffeinate before running (so that computer doesn't sleep)
driver = make_driver()
login_once(driver)
for i, t in enumerate(tickers):
    company_dir = os.path.join(OUTPUT_DIR, t)

    if os.path.exists(os.path.join(company_dir, f"{t}_ratios.csv")) and \
       os.path.exists(os.path.join(company_dir, f"{t}_income.csv")) and \
       os.path.exists(os.path.join(company_dir, f"{t}_balance.csv")):
        print(f"[{t}] All files exist, skipping")
        continue

    try:
        scrape_stockanalysis_tables(t, driver)
    except Exception as e:
        print(f"{t} failed due to {e} — restarting driver")
        driver.quit()
        time.sleep(3)
        driver = make_driver()  
    time.sleep(random.uniform(5, 9))

driver.quit()
print("\nAll midcap financials scraped successfully")
