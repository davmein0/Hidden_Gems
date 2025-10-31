from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import os, time

# ---------------- CONFIG ----------------
urls = {
    "ratios": "https://stockanalysis.com/stocks/nvda/financials/ratios/",
    "income": "https://stockanalysis.com/stocks/nvda/financials/income-statement/",
    "balance": "https://stockanalysis.com/stocks/nvda/financials/balance-sheet/",
    "cashflow": "https://stockanalysis.com/stocks/nvda/financials/cash-flow/"
}

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
os.makedirs("nvidia_financials", exist_ok=True)

# -------------- SCRAPE LOOP -------------
for name, url in urls.items():
    print(f"\nScraping {name}...")
    driver.get(url)
    time.sleep(5)  # give JS time to load fully

    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table")

    if not table:
        print(f"❌ No table found on {name}")
        continue

    # Collect only the *last* header row (closest to data)
    header_rows = table.find_all("tr")
    header_candidates = [tr.find_all("th") for tr in header_rows if tr.find_all("th")]
    headers = [th.get_text(strip=True) for th in header_candidates[-1]]  # use last header row

    # Extract data rows
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if tds:
            rows.append([td.get_text(strip=True) for td in tds])

    # Some rows may have fewer cells (missing values); pad them
    for r in rows:
        while len(r) < len(headers):
            r.append("")

    df = pd.DataFrame(rows, columns=headers)

    # Save to CSV
    path = os.path.join("nvidia_financials", f"nvidia_{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {path} ({len(df)} rows, {len(headers)} columns)")

driver.quit()
print("\nAll scraping complete! CSVs are in ./nvidia_financials/")
