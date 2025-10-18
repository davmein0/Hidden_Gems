import csv
import os

def generate_example_csv(filename="example.csv"):
    """
    Generate an example CSV with columns:
    Ticker, Name, MarketCap, Sector, Industry,
    PE_Ratio, PB_Ratio, DE_Ratio, FreeCashFlow,
    Price, FuturePrice, Label
    """
    # Example reference for how the NASDAQ did over the year, to decide Label.
    # Suppose NASDAQ went from 12,000 → 14,400 (i.e. +20%) in that year.
    nasdaq_start = 12000.0
    nasdaq_end = 12960.0
    nasdaq_return = (nasdaq_end - nasdaq_start) / nasdaq_start  # = 0.08 → +8%

    # Benchmark threshold: stock must beat NASDAQ + 10% => outperformance > 0.10 over NASDAQ
    # That means: stock_return - nasdaq_return > 0.10

    # Example rows and made up values.
    rows = [
        {
            "Ticker": "ABCD",
            "Name": "ABCD Inc.",
            "MarketCap": 3_500_000_000,  # 3.5B
            "Sector": "Technology",
            "Industry": "Software",
            "PE_Ratio": 15.2,
            "PB_Ratio": 4.5,
            "DE_Ratio": 0.8,
            "FreeCashFlow": 120_000_000,  # 120 million
            "Price": 50.0,
            "FuturePrice": 65.0,  # a +30% return
        },
        {
            "Ticker": "WXYZ",
            "Name": "WXYZ Corp.",
            "MarketCap": 6_200_000_000,  # 6.2B
            "Sector": "Industrials",
            "Industry": "Machinery",
            "PE_Ratio": 22.5,
            "PB_Ratio": 2.8,
            "DE_Ratio": 1.5,
            "FreeCashFlow": 80_000_000,  # 80 million
            "Price": 100.0,
            "FuturePrice": 115.0,  # +15% return
        },
        {
            "Ticker": "EFGH",
            "Name": "EFGH Ltd.",
            "MarketCap": 4_800_000_000,  # 4.8B
            "Sector": "Healthcare",
            "Industry": "Biotech",
            "PE_Ratio": 30.1,
            "PB_Ratio": 5.0,
            "DE_Ratio": 1.2,
            "FreeCashFlow": 50_000_000,  # 50 million
            "Price": 20.0,
            "FuturePrice": 22.0,  # +10% return
        },
        {
            "Ticker": "XYZ",
            "Name": "XyloTech Systems",
            "MarketCap": 6_800_000_000,  # 6.8B
            "Sector": "Technology",
            "Industry": "Software",
            "PE_Ratio": 18.7,
            "PB_Ratio": 2.5,
            "DE_Ratio": 0.8,
            "FreeCashFlow": 400_000_000,  # 400M
            "Price": 125.20,
            "FuturePrice": 128.00,  # +2.2%
        },
        {
            "Ticker": "GRC",
            "Name": "GreenRock Energy",
            "MarketCap": 9_200_000_000,  # 9.2B
            "Sector": "Energy",
            "Industry": "Renewable Energy",
            "PE_Ratio": 15.3,
            "PB_Ratio": 1.8,
            "DE_Ratio": 0.6,
            "FreeCashFlow": 320_000_000,  # 320M
            "Price": 62.10,
            "FuturePrice": 74.20,  # +19.5%
        },
        {
            "Ticker": "BNK",
            "Name": "Bright National Bank",
            "MarketCap": 4_100_000_000,  # 4.1B
            "Sector": "Financials",
            "Industry": "Banking",
            "PE_Ratio": 10.4,
            "PB_Ratio": 1.1,
            "DE_Ratio": 1.5,
            "FreeCashFlow": 280_000_000,  # 280M
            "Price": 45.70,
            "FuturePrice": 46.00,  # +0.7%
        },
        {
            "Ticker": "HZN",
            "Name": "Horizon Retail Group",
            "MarketCap": 2_400_000_000,  # 2.4B
            "Sector": "Consumer Discretionary",
            "Industry": "Retail",
            "PE_Ratio": 14.1,
            "PB_Ratio": 2.0,
            "DE_Ratio": 0.9,
            "FreeCashFlow": 190_000_000,  # 190M
            "Price": 32.40,
            "FuturePrice": 39.80,  # +22.8%
        },
        {
            "Ticker": "STL",
            "Name": "Stellar Manufacturing",
            "MarketCap": 7_700_000_000,  # 7.7B
            "Sector": "Industrials",
            "Industry": "Machinery",
            "PE_Ratio": 17.8,
            "PB_Ratio": 2.2,
            "DE_Ratio": 0.7,
            "FreeCashFlow": 410_000_000,  # 410M
            "Price": 85.60,
            "FuturePrice": 83.90,  # -2.0%
        },
        {
            "Ticker": "VLT",
            "Name": "Volt Communications",
            "MarketCap": 5_300_000_000,  # 5.3B
            "Sector": "Telecommunications",
            "Industry": "Wireless Services",
            "PE_Ratio": 12.9,
            "PB_Ratio": 1.9,
            "DE_Ratio": 1.1,
            "FreeCashFlow": 230_000_000,  # 230M
            "Price": 55.20,
            "FuturePrice": 60.70,  # +9.9%
        },
        {
            "Ticker": "FRM",
            "Name": "FarmFresh Foods Inc.",
            "MarketCap": 2_900_000_000,  # 2.9B
            "Sector": "Consumer Staples",
            "Industry": "Packaged Foods",
            "PE_Ratio": 11.8,
            "PB_Ratio": 1.5,
            "DE_Ratio": 0.5,
            "FreeCashFlow": 260_000_000,  # 260M
            "Price": 42.30,
            "FuturePrice": 51.40,  # +21.5%
        },

    ]

    # Compute the Label for each row
    # Label = 1 if (stock_return - nasdaq_return) > 0.10, else 0
    for r in rows:
        price = r["Price"]
        future = r["FuturePrice"]
        stock_return = (future - price) / price
        # Compute outperformance relative to NASDAQ
        outperf = stock_return - nasdaq_return
        r["Label"] = 1 if outperf > 0.10 else 0

    
    # Specify file path
    # output_dir = "data"
    # os.makedirs(output_dir, exist_ok=True)
    # file_path = os.path.join(output_dir, "example.csv")
    
    # Write CSV
    fieldnames = [
        "Ticker", "Name", "MarketCap", "Sector", "Industry",
        "PE_Ratio", "PB_Ratio", "DE_Ratio", "FreeCashFlow",
        "Price", "FuturePrice", "Label"
    ]

    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Generated example CSV: {filename}")
    print("Rows:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    generate_example_csv()
