import pandas as pd
import yfinance as yf
import spacy
import re
from transformers import pipeline

# --- 1. CONFIGURATION & MODELS ---
# We load these globally so they don't reload every time a function is called
print("Loading NLP models (FinBERT & spaCy)...")
try:
    nlp = spacy.load("en_core_web_trf")
    sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1) # -1 for CPU, 0 for GPU
except Exception as e:
    print(f"Error loading models: {e}. Ensure requirements are installed.")

# --- 2. CORE FUNCTIONS ---

def fetch_data(ticker):
    """Fetches news with a mock fallback."""
    try:
        news = yf.Ticker(ticker).news
        if not news: return get_mock(ticker)
        return pd.DataFrame(news)[['title', 'publisher']]
    except:
        return get_mock(ticker)

def get_mock(ticker):
    return pd.DataFrame({
        'title': [f"{ticker} surges on earnings", f"Analysts cautious on {ticker}"],
        'publisher': ['Reuters', 'CNBC']
    })

def process_nlp(df):
    """Cleans text, extracts entities, and runs sentiment."""
    # Cleaning
    df['text'] = df['title'].apply(lambda x: re.sub(r'http\S+|\$[a-zA-Z]+', '', str(x)).strip())
    
    # NER & Sentiment
    df['Company'] = df['text'].apply(lambda x: [ent.text for ent in nlp(x).ents if ent.label_ == "ORG"])
    df = df.explode('Company').dropna(subset=['Company'])
    
    results = df['text'].apply(lambda x: sentiment_pipe(x[:512])[0])
    df['sentiment'] = [r['label'] for r in results]
    df['confidence'] = [r['score'] for r in results]
    
    return df

# --- 3. EXECUTION BLOCK ---
if __name__ == "__main__":
    ticker = "NVDA"
    print(f"🚀 Starting Pipeline for {ticker}...")
    
    raw_df = fetch_data(ticker)
    final_df = process_nlp(raw_df)
    
    print("\n--- FINAL RESULTS ---")
    print(final_df[['Company', 'sentiment', 'confidence']].to_string())
