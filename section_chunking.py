import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ---- CONFIG ----
MIDCAP_FILE = "midcaps.csv"
SAVE_FOLDER = "SENTIMENT"
FORM_TYPE = "10-K"
MIN_SECTION_LENGTH = 300

# Pre-compiled regex patterns (compiled once for performance)
SECTION_PATTERNS = {
    "Item_1_Business": re.compile(r"Item\s+1\.\s*Business", re.IGNORECASE),
    "Item_1A_Risk_Factors": re.compile(r"Item\s+1A\.\s*Risk\s+Factors", re.IGNORECASE),
    "Item_2_Properties": re.compile(r"Item\s+2\.\s*Properties", re.IGNORECASE),
    "Item_3_Legal_Proceedings": re.compile(r"Item\s+3\.\s*Legal\s+Proceedings", re.IGNORECASE),
    "Item_7_MD&A": re.compile(
        r"Item\s+7\.\s*Management'?s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition\s+and\s+Results\s+of\s+Operations",
        re.IGNORECASE
    ),
}

# Combined pattern for finding all sections at once
COMBINED_PATTERN = re.compile(
    "|".join(f"({pattern.pattern})" for pattern in SECTION_PATTERNS.values()),
    re.IGNORECASE
)


@dataclass
class Section:
    """Represents a parsed section from a filing."""
    name: str
    text: str
    start: int
    end: int


def load_ticker_list(filepath: str) -> List[str]:
    """Load and return unique ticker list from CSV."""
    df = pd.read_csv(filepath)
    return df["Ticker"].dropna().unique().tolist()


def read_filing(company_folder: Path, ticker: str) -> Optional[str]:
    """Read the clean text file for a ticker."""
    txt_file = company_folder / f"{ticker}_{FORM_TYPE}_clean.txt"
    
    if not txt_file.exists():
        return None
    
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ {ticker}: Error reading file - {e}")
        return None


def extract_sections(text: str) -> List[Section]:
    """Extract all sections from the filing text."""
    matches = list(COMBINED_PATTERN.finditer(text))
    
    if not matches:
        return []
    
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        
        # Only include sections that meet minimum length
        if len(section_text) >= MIN_SECTION_LENGTH:
            # Get clean section name
            header = match.group().strip()
            # Sanitize for filename
            clean_name = re.sub(r'[^\w\s-]', '', header).replace(' ', '_')
            
            sections.append(Section(
                name=clean_name,
                text=section_text,
                start=start,
                end=end
            ))
    
    return sections


def save_sections(sections: List[Section], company_folder: Path, ticker: str) -> int:
    """Save all sections to individual files."""
    saved_count = 0
    
    for section in sections:
        filename = company_folder / f"{ticker}_{FORM_TYPE}_{section.name}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(section.text)
            saved_count += 1
        except Exception as e:
            print(f"  ⚠️  Failed to save {section.name}: {e}")
    
    return saved_count


def segment_file(base_folder: str, ticker: str) -> bool:
    """
    Segment a filing into separate section files.
    Returns True if successful.
    """
    ticker = ticker.lower()
    company_folder = Path(base_folder) / ticker
    
    # Check if folder exists
    if not company_folder.exists():
        print(f"❌ {ticker}: Company folder not found")
        return False
    
    # Read filing
    text = read_filing(company_folder, ticker)
    if text is None:
        print(f"❌ {ticker}: Clean text file not found")
        return False
    
    # Extract sections
    sections = extract_sections(text)
    if not sections:
        print(f"❌ {ticker}: No recognizable sections found")
        return False
    
    # Save sections
    saved_count = save_sections(sections, company_folder, ticker)
    print(f"✅ {ticker}: Saved {saved_count} sections")
    
    return True


def analyze_sections(base_folder: str, ticker: str) -> dict:
    """
    Analyze segmented sections for a ticker.
    Returns dictionary with analysis results.
    """
    ticker = ticker.lower()
    company_folder = Path(base_folder) / ticker
    
    if not company_folder.exists():
        return {"error": "Folder not found"}
    
    # Find all segmented files
    section_files = list(company_folder.glob(f"{ticker}_{FORM_TYPE}_Item_*.txt"))
    
    if not section_files:
        return {"error": "No segmented files found"}
    
    results = {
        "ticker": ticker,
        "sections_found": len(section_files),
        "sections": {}
    }
    
    for section_file in section_files:
        section_name = section_file.stem.replace(f"{ticker}_{FORM_TYPE}_", "")
        
        try:
            with open(section_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            results["sections"][section_name] = {
                "length": len(text),
                "word_count": len(text.split()),
                "file": str(section_file.name)
            }
        except Exception as e:
            results["sections"][section_name] = {"error": str(e)}
    
    return results


def process_batch(ticker_list: List[str], base_folder: str) -> dict:
    """Process all tickers and return summary statistics."""
    stats = {
        "total": len(ticker_list),
        "successful": 0,
        "failed": 0,
        "sections_extracted": 0
    }
    
    for i, ticker in enumerate(ticker_list, 1):
        print(f"[{i}/{len(ticker_list)}] Processing {ticker}...")
        
        try:
            if segment_file(base_folder, ticker):
                stats["successful"] += 1
                
                # Count sections
                company_folder = Path(base_folder) / ticker.lower()
                section_count = len(list(company_folder.glob(f"{ticker.lower()}_{FORM_TYPE}_Item_*.txt")))
                stats["sections_extracted"] += section_count
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"❌ {ticker}: Unexpected error - {e}")
            stats["failed"] += 1
    
    return stats


def main():
    """Main execution function."""
    # Load tickers
    ticker_list = load_ticker_list(MIDCAP_FILE)
    print(f"Processing {len(ticker_list)} tickers...\n")
    
    # Process batch
    stats = process_batch(ticker_list, SAVE_FOLDER)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total tickers: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total sections extracted: {stats['sections_extracted']}")
    print(f"Average sections per ticker: {stats['sections_extracted'] / stats['successful']:.1f}")


if __name__ == "__main__":
    main()