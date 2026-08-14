"""
CausalFolio - News Scraper
===========================
Scrapes financial news from Economic Times and Yahoo Finance.

Usage (in Colab):
    from data.news_scraper import get_news_headlines
    
    headlines = get_news_headlines("TCS.BO", days=7)
"""

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re


# User agent to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    return text


# ============================================================
# RELIABLE NEWS SOURCES (APIs - preferred)
# ============================================================

def get_yfinance_news(ticker: str, max_headlines: int = 5) -> List[Dict]:
    """
    Get news from yfinance API (most reliable method).
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        headlines = []
        for item in news[:max_headlines]:
            # New yfinance structure: data is nested in 'content'
            content = item.get('content', item)  # Fallback to item if no 'content'
            
            title = content.get('title', '')
            provider = content.get('provider', {})
            source = provider.get('displayName', 'Yahoo Finance') if isinstance(provider, dict) else 'Yahoo Finance'
            pub_date = content.get('pubDate', '')[:10] if content.get('pubDate') else ''
            url = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
            
            if title:  # Only add if title exists
                headlines.append({
                    'headline': title,
                    'source': source,
                    'date': pub_date,
                    'url': url
                })
        
        return headlines
    except Exception as e:
        return []


def get_google_news_rss(company_name: str, max_headlines: int = 5) -> List[Dict]:
    """
    Get news from Google News RSS (free, no API key).
    """
    try:
        import urllib.parse
        
        query = company_name.replace('.BO', '').replace('.NS', '')
        query = urllib.parse.quote(f"{query} stock India")
        
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        
        headlines = []
        for item in soup.find_all('item')[:max_headlines]:
            title = item.find('title')
            source = item.find('source')
            
            if title and title.text:
                headlines.append({
                    'headline': clean_text(title.text),
                    'source': source.text if source else 'Google News',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'url': item.find('link').text if item.find('link') else ''
                })
        
        return headlines
    except Exception as e:
        return []


def get_newsapi(company_name: str, api_key: str, max_headlines: int = 5) -> List[Dict]:
    """
    Get news from NewsAPI.org (requires free API key).
    Get key at: https://newsapi.org/register
    """
    if not api_key:
        return []
    
    try:
        query = company_name.replace('.BO', '').replace('.NS', '')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{query} stock",
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': max_headlines,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') != 'ok':
            return []
        
        headlines = []
        for article in data.get('articles', []):
            headlines.append({
                'headline': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'date': article.get('publishedAt', '')[:10],
                'url': article.get('url', '')
            })
        
        return headlines
    except Exception as e:
        return []


# ============================================================
# WEB SCRAPING (fallback)
# ============================================================


def scrape_economic_times(
    company_name: str,
    max_headlines: int = 10
) -> List[Dict]:
    """
    Scrape headlines from Economic Times for a company.
    
    Args:
        company_name: Company name (e.g., 'TCS', 'Reliance')
        max_headlines: Maximum headlines to return
    
    Returns:
        List of dicts with 'headline', 'source', 'date'
    """
    # Clean company name (remove .BO suffix)
    company = company_name.replace('.BO', '').replace('.NS', '')
    
    # Economic Times search URL
    search_url = f"https://economictimes.indiatimes.com/topic/{company}"
    
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        headlines = []
        
        # Find headline elements (ET's structure may vary)
        # Try multiple selectors
        selectors = [
            'h2 a',
            '.eachStory h3 a',
            '.news_list a',
            'article h2 a',
            '.topStory h1 a'
        ]
        
        seen = set()
        for selector in selectors:
            for elem in soup.select(selector):
                text = clean_text(elem.text)
                if text and len(text) > 20 and text not in seen:
                    seen.add(text)
                    headlines.append({
                        'headline': text,
                        'source': 'Economic Times',
                        'url': elem.get('href', ''),
                        'date': datetime.now().strftime('%Y-%m-%d')
                    })
                    
                    if len(headlines) >= max_headlines:
                        break
            
            if len(headlines) >= max_headlines:
                break
        
        return headlines
        
    except Exception as e:
        print(f"  ⚠ Economic Times scrape failed: {e}")
        return []


def scrape_yahoo_finance_news(
    ticker: str,
    max_headlines: int = 10
) -> List[Dict]:
    """
    Scrape news from Yahoo Finance for a ticker.
    
    Args:
        ticker: Stock ticker (e.g., 'TCS.BO')
        max_headlines: Maximum headlines to return
    
    Returns:
        List of dicts with 'headline', 'source', 'date'
    """
    # Yahoo Finance news URL
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        headlines = []
        seen = set()
        
        # Find article headlines
        for elem in soup.select('h3 a, h2 a, li h3'):
            text = clean_text(elem.text if hasattr(elem, 'text') else elem.get_text())
            if text and len(text) > 20 and text not in seen:
                seen.add(text)
                headlines.append({
                    'headline': text,
                    'source': 'Yahoo Finance',
                    'date': datetime.now().strftime('%Y-%m-%d')
                })
                
                if len(headlines) >= max_headlines:
                    break
        
        return headlines
        
    except Exception as e:
        print(f"  ⚠ Yahoo Finance scrape failed: {e}")
        return []


def get_news_headlines(
    ticker: str,
    max_per_source: int = 3,
    newsapi_key: str = None
) -> List[Dict]:
    """
    Get news headlines from ALL sources and pool results.
    
    Queries all 3 sources in parallel and combines:
    1. yfinance API
    2. Google News RSS
    3. NewsAPI (if key provided)
    4. Economic Times (bonus fallback)
    
    Args:
        ticker: Stock ticker (e.g., 'TCS.BO')
        max_per_source: Max headlines PER source (total = 3-4x this)
        newsapi_key: Optional NewsAPI key
    
    Returns:
        List of headline dicts from ALL sources (deduplicated)
    """
    all_headlines = []
    company = ticker.replace('.BO', '').replace('.NS', '')
    
    # 1. yfinance API
    yf_headlines = get_yfinance_news(ticker, max_per_source)
    all_headlines.extend(yf_headlines)
    
    # 2. Google News RSS (always query)
    google_headlines = get_google_news_rss(company, max_per_source)
    all_headlines.extend(google_headlines)
    
    # 3. NewsAPI (always query if key provided)
    if newsapi_key:
        newsapi_headlines = get_newsapi(company, newsapi_key, max_per_source)
        all_headlines.extend(newsapi_headlines)
    
    # 4. Economic Times fallback (if we have few headlines)
    if len(all_headlines) < max_per_source:
        time.sleep(0.3)
        et_headlines = scrape_economic_times(company, max_per_source)
        all_headlines.extend(et_headlines)
    
    # Deduplicate by headline text
    seen = set()
    unique = []
    for h in all_headlines:
        if h.get('headline') and h['headline'] not in seen:
            seen.add(h['headline'])
            unique.append(h)
    
    return unique


def get_batch_news(
    tickers: List[str],
    max_per_source: int = 5,
    delay_seconds: float = 2.0
) -> Dict[str, List[Dict]]:
    """
    Get news for multiple stocks with strict rate limiting.
    """
    print(f"=" * 50)
    print(f"Fetching news for {len(tickers)} stocks...")
    print(f"=" * 50)
    
    # Optional: fetch config for default delay if not provided
    try:
        import yaml
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            delay_seconds = config.get('api_limits', {}).get('yfinance_delay_seconds', delay_seconds)
    except:
        pass
    
    all_news = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        
        # Add retry loop for resilience
        for attempt in range(3):
            try:
                headlines = get_news_headlines(ticker, max_per_source)
                all_news[ticker] = headlines
                print(f"  ✓ Total: {len(headlines)} headlines")
                break
            except Exception as e:
                print(f"  ⚠ Scrape attempt {attempt+1} failed: {e}")
                time.sleep(delay_seconds * 2) # Penalize failure with longer sleep

        if i < len(tickers):
            print(f"  [Sleep {delay_seconds}s for rate limit]")
            time.sleep(delay_seconds)
    
    print(f"\n" + "=" * 50)
    print(f"Done! Total headlines: {sum(len(h) for h in all_news.values())}")
    print(f"=" * 50)
    
    return all_news


def headlines_to_text(headlines: List[Dict]) -> List[str]:
    """
    Extract just the headline text for FinBERT input.
    """
    return [h['headline'] for h in headlines]


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing News Scraper")
    print("=" * 50)
    
    # Test single stock
    ticker = "TCS.BO"
    print(f"\nFetching news for {ticker}...")
    
    headlines = get_news_headlines(ticker, max_headlines=5)
    
    print(f"\n✓ Got {len(headlines)} headlines:")
    for i, h in enumerate(headlines, 1):
        print(f"  {i}. [{h['source']}] {h['headline'][:60]}...")
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)
