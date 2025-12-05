import os
import re
import pandas as pd


# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")

MIDCAP_FILE = "midcaps.csv"


def sentiment_segment(
    filename, 
    max_tokens = 512, 
    overlap = 50
):
  
  text = open(filename, "r").read()

  # Split into sentences at sentence boundaries
  sentences = re.split(r'(?<=[.!?])\s+', text.strip())
  
  chunks = []
  current_chunk = []
  current_length = 0
  
  for sentence in sentences:
      words = sentence.split()
      sentence_tokens = len(words)
      
      # If adding this sentence exceeds limit, save current chunk
      if current_length + sentence_tokens > max_tokens and current_chunk:
          chunks.append(' '.join(current_chunk))
          
          # Keep last N words for overlap/context
          if overlap > 0:
              overlap_text = ' '.join(current_chunk).split()[-overlap:]
              current_chunk = [' '.join(overlap_text)]
              current_length = len(overlap_text)
          else:
              current_chunk = []
              current_length = 0
      
      current_chunk.append(sentence)
      current_length += sentence_tokens
  
  # Add final chunk
  if current_chunk:
      chunks.append(' '.join(current_chunk))
  
  return chunks


def main():

  # Load tickers
  df_midcap = pd.read_csv(MIDCAP_FILE)
  ticker_list = df_midcap["Ticker"].dropna().unique().tolist()
    
  filename = f"SENTIMENT/{ticker_list[10].lower()}/{ticker_list[10].lower()}_10-K_Item_1A_Risk_Factors.txt"

  print(sentiment_segment(filename))



if __name__ == "__main__":
    main()