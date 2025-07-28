"""
tools/data_sourcing.py
======================

This module contains all data sourcing functions used by the trading assistant.
It integrates with external APIs such as Yahoo Finance (via yfinance),
Financial Modeling Prep (FMP), the Federal Reserve Economic Data (FRED) and
alternative data providers. To minimise network usage, historical OHLCV data
is cached locally using DuckDB via the `MarketDataCache` class.

Notable improvements over previous versions:

* SQL queries in `MarketDataCache` are parameterised to prevent injection
  vulnerabilities and improve performance with prepared statements.
* All HTTP requests specify a timeout to avoid hanging the application.
* yfinance calls disable threads to reduce overhead when fetching data for
  single tickers.
* Input validation and optional arguments use explicit `is not None` checks.

Each function is decorated with `tenacity.retry` to automatically retry
failures up to a fixed number of attempts with a delay between attempts.
"""

import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any

import duckdb
import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_fixed

# This global config will be set by the main orchestrator
config: Dict[str, Any] = {}

def set_config(app_config: Dict[str, Any]) -> None:
    """Allows the main orchestrator to pass in the master config."""
    global config
    config = app_config


class MarketDataCache:
    """Persistent cache for OHLCV data using DuckDB.

    The cache stores data per symbol and updates it when stale. The
    `staleness_hours` parameter controls how long cached data is considered
    fresh. Parameterised SQL queries are used to avoid SQL injection and
    improve query planning in DuckDB.
    """

    def __init__(self, db_path: str, staleness_hours: int) -> None:
        self.db_path = db_path
        self.staleness_limit = timedelta(hours=staleness_hours)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        # Ensure the ohlcv table exists
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                Date TIMESTAMP,
                Symbol VARCHAR,
                Open DOUBLE,
                High DOUBLE,
                Low DOUBLE,
                Close DOUBLE,
                "Adj Close" DOUBLE,
                Volume BIGINT,
                timestamp TIMESTAMP
            );
            """
        )

    def get_ohlcv(self, tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
        """Retrieve OHLCV data for one or more tickers, utilising the local cache.

        If the data for all requested tickers is present in the cache and has
        been updated within the staleness window, it is returned directly.
        Otherwise, fresh data is downloaded from Yahoo Finance and the cache
        is updated. Tickers are parameterised in the SQL query to avoid
        injection and allow DuckDB to optimise the query.

        Args:
            tickers: A tuple of ticker symbols.
            period: A period string compatible with `yfinance.download`.

        Returns:
            A DataFrame containing either cached or freshly downloaded data.
        """
        print(f"CACHE: Checking for {tickers} in DuckDB...")
        now = datetime.utcnow()
        # Build a parameterised query for the list of tickers
        placeholders = ",".join(["?"] * len(tickers))
        query = f"SELECT * FROM ohlcv WHERE Symbol IN ({placeholders}) AND timestamp > ?"
        params = list(tickers) + [now - self.staleness_limit]
        try:
            cached_data = self.con.execute(query, params).fetchdf()
            # Only treat as a cache hit if all requested symbols are present
            if not cached_data.empty and all(t in cached_data['Symbol'].unique() for t in tickers):
                print(f"CACHE HIT: Found fresh data for {tickers}.")
                cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                # Return data pivoted by symbol for convenience
                return cached_data.pivot(index='Date', columns='Symbol')
        except Exception as e:
            print(f"CACHE WARN: {e}")

        print(f"CACHE MISS: Fetching fresh data for {tickers} from yfinance.")
        # Disable multithreading in yfinance to reduce overhead
        data = yf.download(list(tickers), period=period, progress=False, threads=False)
        if data.empty:
            return None
        # Flatten the data for insertion into DuckDB
        if len(tickers) > 1:
            df_to_insert = data.stack().reset_index()
            df_to_insert.rename(columns={'level_1': 'Symbol'}, inplace=True)
        else:
            df_to_insert = data.reset_index()
            df_to_insert.insert(1, 'Symbol', tickers[0])

        df_to_insert['timestamp'] = now
        cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'timestamp']
        df_to_insert = df_to_insert.reindex(columns=cols)
        # Write to the cache: replace any existing rows for these tickers
        self.con.register('df_to_insert', df_to_insert)
        placeholders_del = ",".join(["?"] * len(tickers))
        del_query = f"DELETE FROM ohlcv WHERE Symbol IN ({placeholders_del})"
        self.con.execute(del_query, list(tickers))
        self.con.execute("INSERT INTO ohlcv SELECT * FROM df_to_insert;")
        print("CACHE WRITE: Updated DuckDB.")
        return data


# -----------------------------------------------------------------------------
# Data Sourcing Tool Functions
# -----------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_historical_market_data(tickers: List[str], period: str, cache: MarketDataCache):
    """Download historical daily stock data, utilising a persistent disk cache."""
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
    """Retrieve full financial statements (income_statement, balance_sheet, cash_flow) from FMP."""
    print(f"DATA TOOL: Fetching {statement_type} for {ticker}...")
    try:
        params = {'period': period, 'limit': limit, 'apikey': config['api_keys']['fmp']}
        response = requests.get(
            f'https://financialmodelingprep.com/api/v3/{statement_type.lower()}/{ticker}',
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP API error for {statement_type}: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_analyst_ratings(ticker: str):
    """Fetch consensus Wall Street analyst ratings for a stock from FMP."""
    print(f"DATA TOOL: Fetching analyst ratings for {ticker}...")
    try:
        params = {'apikey': config['api_keys']['fmp']}
        response = requests.get(
            f'https://financialmodelingprep.com/api/v3/rating/{ticker}',
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {"message": "No ratings found."}
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP API error for ratings: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_insider_transactions(ticker: str, limit: int = 10):
    """Show if company executives are buying or selling their own stock, via FMP."""
    print(f"DATA TOOL: Fetching insider transactions for {ticker}...")
    try:
        params = {'symbol': ticker, 'limit': limit, 'apikey': config['api_keys']['fmp']}
        response = requests.get(
            'https://financialmodelingprep.com/api/v4/insider-trading',
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return [
            {
                'filerName': t['filerName'],
                'transactionType': t['transactionType'],
                'transactionDate': t['transactionDate'],
                'shares': t['shares'],
                'value': t['value'],
            }
            for t in response.json()
        ]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP API error for insider trading: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_financial_news_headlines(query: str, limit: int = 3):
    """Search for recent financial news headlines for a given topic or company."""
    print(f"DATA TOOL: Searching news about '{query}'...")
    try:
        params = {
            'q': query,
            'language': 'en',
            'apiKey': config['api_keys']['news_api'],
            'pageSize': limit,
        }
        response = requests.get('https://newsapi.org/v2/everything', params=params, timeout=10)
        response.raise_for_status()
        return [
            {'title': a['title'], 'source': a['source']['name']}
            for a in response.json().get('articles', [])
        ]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"NewsAPI error: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def screen_for_stocks(
    sector: str = None,
    min_market_cap: int = 2_000_000_000,
    min_dividend_yield: float = None,
    limit: int = 10,
) -> Any:
    """Scan the market to find stocks matching specific criteria using the FMP API."""
    print(f"DATA TOOL: Screening for stocks...")
    try:
        params: Dict[str, Any] = {'limit': limit, 'apikey': config['api_keys']['fmp']}
        if sector:
            params['sector'] = sector
        if min_market_cap is not None:
            params['marketCapMoreThan'] = min_market_cap
        if min_dividend_yield is not None:
            params['dividendYieldMoreThan'] = min_dividend_yield
        response = requests.get(
            'https://financialmodelingprep.com/api/v3/stock-screener',
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP Screener API error: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_sector_performance() -> Any:
    """Provide a snapshot of U.S. stock market sector performance to understand market trends."""
    print("CONTEXT TOOL: Fetching sector performance...")
    try:
        url = f"https://financialmodelingprep.com/api/v3/sectors-performance"
        params = {'apikey': config['api_keys']['fmp']}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FMP Sector Performance API error: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_economic_data(series_id: str) -> Any:
    """Fetch a specific U.S. economic data series from FRED (e.g., 'UNRATE')."""
    print(f"CONTEXT TOOL: Fetching FRED data for '{series_id}'...")
    try:
        params = {
            'series_id': series_id,
            'api_key': config['api_keys']['fred'],
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': 1,
        }
        response = requests.get(
            'https://api.stlouisfed.org/fred/series/observations',
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()['observations'][0]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"FRED API error: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_market_fear_and_greed_index() -> Any:
    """Retrieve the CNN Fear & Greed Index (via Alternative.me) as a proxy for market sentiment."""
    print("CONTEXT TOOL: Fetching Fear & Greed Index...")
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=10)
        response.raise_for_status()
        return response.json()['data'][0]
    except requests.exceptions.RequestException as e:
        return {"tool_error": f"Fear & Greed API error: {e}"}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_macro_signals(series_ids: List[str]) -> Any:
    """
    Fetch multiple macroeconomic indicators from FRED. Accepts a list of series
    identifiers (e.g., ['UNRATE', 'CPIAUCSL']). Returns a dictionary mapping
    each series ID to its most recent observation.
    """
    signals = {}
    try:
        for sid in series_ids:
            result = get_economic_data(sid)
            signals[sid] = result
        return signals
    except Exception as e:
        return {"tool_error": f"Macro signals retrieval failed: {e}"}
