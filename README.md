# 📈 Financial Entity Sentiment Linker (NLP)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NLP](https://img.shields.io/badge/NLP-Transformers-green.svg)](https://huggingface.co/models)
[![Model](https://img.shields.io/badge/Model-FinBERT-orange.svg)](https://huggingface.co/ProsusAI/finbert)

## 🎯 Project Overview
In a fast-moving market, manual news analysis is impossible. This project implements a **state-of-the-art NLP pipeline** that automatically scrapes financial news, identifies specific organizations using **Transformer-based NER**, and calculates sentiment scores using **FinBERT**.

Unlike generic sentiment analysis, this system distinguishes between "Market Volatility" (Negative) and "Stock Volatility" (Neutral/Context-dependent) by using a model trained specifically on financial corpora.

## 🚀 Key Features
- **Real-time Scraper:** Integrated with `yfinance` to fetch live market headlines.
- **High-Accuracy NER:** Uses `spaCy` (en_core_web_trf) to isolate company names from noise.
- **Financial Sentiment:** Leverages `ProsusAI/finbert` for domain-specific classification.
- **Interactive Dashboards:** Dynamic Plotly visualizations including Heatmaps and Donut charts.

## 🏗 System Architecture


1. **Extraction:** Fetching news via API.
2. **Preprocessing:** Regex-based cleaning and noise reduction.
3. **Entity Linking:** Identifying `ORG` labels via spaCy Transformers.
4. **Sentiment Inference:** Running FinBERT for Positive/Negative/Neutral labels.
5. **Visualization:** Generating interactive insights via Plotly.


## 🛠 Setup & Installation

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/financial-sentiment-linker.git
   cd financial-sentiment-linker
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project:** Open ```notebooks/financial_sentiment_analysis.ipynb``` in Google Colab or Jupyter Notebook and run all cells.

## 💡 Future Scope
* Price Correlation: Linking sentiment shifts to 24-hour stock price movements.

* Ticker Mapping: Automatically converting "Apple" to "AAPL" using a fuzzy matching algorithm.

* Multi-lingual Support: Expanding the pipeline to analyze international news sources.

#### Author: Ruchir Agrawal | [LinkedIn](https://www.linkedin.com/in/ruchir-a-308240128/)
