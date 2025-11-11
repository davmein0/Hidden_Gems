import os
import re
import pandas as pd


#---GET TICKER LIST
MIDCAP_FILE = "midcaps.csv"
SAVE_FOLDER = "SENTIMENT"
FORM_TYPE = "10-K"
TICKER = "aapl"

df_midcap = pd.read_csv(MIDCAP_FILE)
ticker_list = df_midcap["Ticker"].dropna().unique().tolist()


#---segment the 10-K
def segment_file(filepath, ticker):

  #get save folder
  company_folder = os.path.join(filepath, ticker)
  try:
    txtFile = os.path.join(company_folder, f"{ticker}_{FORM_TYPE}_clean.txt")

    file = open(txtFile, "r")
    text = file.read()

    section_headers = [
          r"Item\s+1\.\s*Business",
          r"Item\s+1A\.\s*Risk\s+Factors",
          r"Item\s+2\.\s*Properties",
          r"Item\s+3\.\s*Legal\s+Proceedings",
          r"Item\s+7\.\s*Management's\s+Discussion\s+and\s+Analysis",
      ]
    
    # Find all section header matches
    matches = list(re.finditer("|".join(section_headers), text, flags=re.IGNORECASE))
    if not matches:
        print(f"❌No recognizable section headers found in {txtFile}")
        return
    
    for i, match in enumerate(matches):
      start = match.start()
      end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
      section_text = text[start:end].strip()

      if len(section_text) > 300:
        header = match.group().strip()
        #print(f"\n\n===== {header.upper()} =====\n")
        #print(section_text[:1000])  # print first ~1000 characters for readability
        #print("\n... [section truncated]\n")

        txt_filename = os.path.join(company_folder, f"{ticker}_{FORM_TYPE}_{header}.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(section_text)
    
    print(f"✅Saved segmented data for {ticker}.")
  except:
     print(f"❌No files for {txtFile}")


def finBert_analysis(filepath, ticker):
   pass

#--MAIN--
for t in ticker_list:
  segment_file(SAVE_FOLDER, t)
