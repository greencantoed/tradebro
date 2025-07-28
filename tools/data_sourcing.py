# tools/data_sourcing.py
# FINAL VERIFIED BUILD v12.2

import yfinance as yf
import requests
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from tenacity import retry, stop_after_attempt, wait_fixed

# This global config will be set by the main orchestrator
config = {}
def set_config(app_config):
    global config
    config = app_config

class MarketDataCache:
    def __init__(self, db_path: str, staleness_hours: int):
        self.db_path = db_path
        self.staleness_limit = timedelta(hours=staleness_hours)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                Date TIMESTAMP, Symbol VARCHAR, Open DOUBLE, High DOUBLE, Low DOUBLE, Close DOUBLE, 
                "Adj Close" DOUBLE, Volume BIGINT, timestamp TIMESTAMP
            );
        """)

    def get_ohlcv(self, tickers: tuple[str], period: str):
        print(f"CACHE: Checking for {tickers} in DuckDB...")
        now = datetime.utcnow()
        symbols_str = "','".join(tickers)
        query = f"SELECT * FROM ohlcv WHERE Symbol IN ('{symbols_str}') AND timestamp > '{now - self.staleness_limit}'"
        try:
            cached_data = self.con.execute(query).fetchdf()
            if not cached_data.empty and all(t in cached_data['Symbol'].unique() for t in tickers):
                print(f"CACHE HIT: Found fresh data for {tickers}.")
                cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                return cached_data.pivot(index='Date', columns='Symbol')
        except Exception as e:
            print(f"CACHE WARN: {e}")

        print(f"CACHE MISS: Fetching fresh data for {tickers} from yfinance.")
        data = yf.download(list(tickers), period=period, progress=False)
        if data.empty: return None

        # Correctly handle single vs multi-ticker dataframes for insertion
        if len(tickers) > 1:
            df_to_insert = data.stack().reset_index()
            df_to_insert.rename(columns={'level_1': 'Symbol'}, inplace=True)
        else:
            df_to_insert = data.reset_index()
            df_to_insert.insert(1, 'Symbol', tickers[0])

        df_to_insert['timestamp'] = now
        cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'timestamp']
        df_to_insert = df_to_insert.reindex(columns=cols)

        self.con.register('df_to_insert', df_to_insert)
        self.con.execute(f"DELETE FROM ohlcv WHERE Symbol IN ('{symbols_str}');")
        self.con.execute("INSERT INTO ohlcv SELECT * FROM df_to_insert;")
        print("CACHE WRITE: Updated DuckDB.")
        return data

# --- TOOL DEFINITIONS ---

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_historical_market_data(tickers: list[str], period: str, cache):
    """Downloads historical daily stock data, utilizing a persistent disk cache."""
    try:
        max_tickers = config.get('parameters', {}).get('max_tickers_per_call', 25)
        if len(tickers) > max_tickers:
            return {"tool_error": f"Too many tickers. Max is {max_tickers}."}
        data = cache.get_ohlcv(tuple(tickers), period)
        if data is None or data.empty:
            return {"tool_error": "Failed to retrieve data from both cache and yfinance."}
        return json.loads(data.to_json(orient='split', date_format='iso'))
    except Exception as e:
        return {"tool_error": f"Data processing failed in get_historical_market_data: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_financial_statements(ticker: str, statement_type: str, period: str = 'annual', limit: int = 1):
    """Retrieves full financial statements (income_statement, balance_sheet, cash_flow) from FMP."""
    print(f"DATA TOOL: Fetching {statement_type} for {ticker}...")
    try:
        params = {'period': period, 'limit': limit, 'apikey': config['api_keys']['fmp']}
        response = requests.get(f'https://financialmodelingprep.com/api/v3/{statement_type.lower()}/{ticker}', params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP API error for {statement_type}: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_analyst_ratings(ticker: str):
    """Fetches consensus Wall Street analyst ratings for a stock from FMP."""
    print(f"DATA TOOL: Fetching analyst ratings for {ticker}...")
    try:
        params = {'apikey': config['api_keys']['fmp']}
        response = requests.get(f'https://financialmodelingprep.com/api/v3/rating/{ticker}', params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {"message": "No ratings found."}
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP API error for ratings: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_insider_transactions(ticker: str, limit: int = 10):
    """Shows if company executives are buying or selling their own stock, via FMP."""
    print(f"DATA TOOL: Fetching insider transactions for {ticker}...")
    try:
        params = {'symbol': ticker, 'limit': limit, 'apikey': config['api_keys']['fmp']}
        response = requests.get(f'https://financialmodelingprep.com/api/v4/insider-trading', params=params)
        response.raise_for_status()
        return [{"filerName": t['filerName'], "transactionType": t['transactionType'], "transactionDate": t['transactionDate'], "shares": t['shares'], "value": t['value']} for t in response.json()]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP API error for insider trading: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_financial_news_headlines(query: str, limit: int = 3):
    """Searches for recent financial news headlines for a given topic or company."""
    print(f"DATA TOOL: Searching news about '{query}'...")
    try:
        params = {'q': query, 'language': 'en', 'apiKey': config['api_keys']['news_api'], 'pageSize': limit}
        response = requests.get('https://newsapi.org/v2/everything', params=params)
        response.raise_for_status()
        return [{"title": a['title'], "source": a['source']['name']} for a in response.json().get('articles', [])]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"NewsAPI error: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def screen_for_stocks(sector: str = None, min_market_cap: int = 2000000000, min_dividend_yield: float = None, limit: int = 10):
    """Scans the market to find stocks matching specific criteria using the FMP API."""
    print(f"DATA TOOL: Screening for stocks...")
    try:
        params = {'limit': limit, 'apikey': config['api_keys']['fmp']}
        if sector: params['sector'] = sector
        if min_market_cap: params['marketCapMoreThan'] = min_market_cap
        if min_dividend_yield: params['dividendYieldMoreThan'] = min_dividend_yield
        response = requests.get(f'https://financialmodelingprep.com/api/v3/stock-screener', params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP Screener API error: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_sector_performance():
    """Provides a snapshot of U.S. stock market sector performance to understand market trends."""
    print("CONTEXT TOOL: Fetching sector performance...")
    try:
        response = requests.get(f"https://financialmodelingprep.com/api/v3/sectors-performance?apikey={config['api_keys']['fmp']}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP Sector Performance API error: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_economic_data(series_id: str):
    """Fetches a specific U.S. economic data series from FRED (e.g., 'UNRATE' for unemployment)."""
    print(f"CONTEXT TOOL: Fetching FRED data for '{series_id}'...")
    try:
        params = {'series_id': series_id, 'api_key': config['api_keys']['fred'], 'file_type': 'json', 'sort_order': 'desc', 'limit': 1}
        response = requests.get('https://api.stlouisfed.org/fred/series/observations', params=params)
        response.raise_for_status()
        return response.json()['observations'][0]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FRED API error: {e}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_market_fear_and_greed_index():
    """Retrieves the CNN Fear & Greed Index (via Alternative.me) as a proxy for market sentiment."""
    print("CONTEXT TOOL: Fetching Fear & Greed Index...")
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=1')
        response.raise_for_status()
        return response.json()['data'][0]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"Fear & Greed API error: {e}"}