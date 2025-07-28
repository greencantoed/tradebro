
import pandas as pd
import json
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

# UNIFIED CONFIG PROPAGATION
config = {}
def set_config(app_config):
    global config
    config = app_config

# --- TOOL DEFINITIONS ---

def perform_econometric_analysis(analysis_type: str, json_data: dict, **kwargs):
    """Performs statistical analysis like 'correlation_matrix' or 'linear_regression'."""
    print(f"ANALYSIS TOOL: Performing '{analysis_type}'...")
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        # Correctly handle single vs multi-ticker dataframes
        if isinstance(df.columns, pd.MultiIndex):
            df_close = df['Close']
        else:
            df_close = df[['Close']] # Ensure it's a DataFrame

        if analysis_type == 'correlation_matrix':
            return json.loads(df_close.corr().to_json())
        elif analysis_type == 'linear_regression':
            dependent_var = kwargs.get('dependent_var')
            independent_vars = kwargs.get('independent_vars')
            if not dependent_var or not independent_vars:
                return {"tool_error": "Dependent and independent variables are required for regression."}
            
            # Ensure we're working with a flat column structure for regression
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)

            Y = df[dependent_var]
            X = sm.add_constant(df[independent_vars])
            model = sm.OLS(Y, X).fit()
            return {"model_summary": str(model.summary())}
        else:
            return {"tool_error": "Unsupported analysis type."}
    except Exception as e:
        return {"tool_error": f"Econometric analysis failed: {e}"}

def calculate_technical_indicator(indicator_name: str, json_data: dict):
    """Calculates technical indicators like 'SMA' (Simple Moving Average) or 'RSI' (Relative Strength Index)."""
    print(f"ANALYSIS TOOL: Calculating '{indicator_name}'...")
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df['Close'].iloc[:, 0]
        else:
            close_prices = df['Close']

        indicator_name = indicator_name.upper()
        if indicator_name == 'SMA':
            return {"sma_50": round(close_prices.rolling(window=50).mean().iloc[-1], 2)}
        elif indicator_name == 'RSI':
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return {"rsi_14": round(rsi.iloc[-1], 2)}
        else:
            return {"tool_error": "Unsupported indicator."}
    except Exception as e:
        return {"tool_error": f"Indicator calculation failed: {e}"}

def analyze_portfolio_risk(portfolio_holdings: list[dict]):
    """Analyzes a user's stock portfolio to calculate its overall market risk (beta)."""
    print(f"ANALYSIS TOOL: Analyzing portfolio risk...")
    try:
        tickers = [p['ticker'] for p in portfolio_holdings]
        if not tickers: return {"tool_error": "Portfolio holdings cannot be empty."}
        
        data = yf.download(tickers + ['SPY'], period='1y', progress=False)['Adj Close'].pct_change().dropna()
        market_variance = data['SPY'].var()
        betas = {ticker: data[ticker].cov(data['SPY']) / market_variance for ticker in tickers}
        return {"individual_betas": {t: round(b, 2) for t, b in betas.items()}}
    except Exception as e:
        return {"tool_error": f"Risk analysis failed: {e}"}

def backtest_strategy(ticker: str, strategy_name: str, cache):
    """Backtests a pre-defined trading strategy ('SmaCross') on historical data."""
    print(f"ANALYSIS TOOL: Backtesting '{strategy_name}' on {ticker}...")
    try:
        # Use the cache to get data. The cache returns a standard, unpivoted dataframe.
        data_df = cache.get_ohlcv(tuple([ticker]), period='5y')
        if data_df is None or data_df.empty:
            return {"tool_error": "Could not retrieve data for backtest."}
        
        # Backtesting.py expects the index to be 'Date'
        data_df.set_index('Date', inplace=True)

        bt_config = config.get('backtesting', {})
        commission = bt_config.get('commission_bps', 5) / 10000
        slippage = bt_config.get('slippage_bps', 10) / 10000

        if strategy_name == 'SmaCross':
            class SmaCross(Strategy):
                n1, n2 = 20, 50
                def init(self):
                    self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
                    self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)
                def next(self):
                    if crossover(self.sma1, self.sma2): self.buy()
                    elif crossover(self.sma2, self.sma1): self.sell()
            
            bt = Backtest(data_df, SmaCross, cash=10000, commission=commission, slippage=slippage)
            stats = bt.run()
            
            return {
                "return_pct": f"{stats.get('Return [%]', 0):.2f}%",
                "win_rate": f"{stats.get('Win Rate [%]', 0):.2f}%",
                "sharpe_ratio": f"{stats.get('Sharpe Ratio', 0):.2f}",
                "max_drawdown_pct": f"{stats.get('Max. Drawdown [%]', 0):.2f}%"
            }
    except Exception as e:
        return {"tool_error": f"Backtesting failed for {ticker}: {e}"}

def generate_time_series_forecast(json_data: dict, forecast_periods: int = 30):
    """Generates a statistical time series forecast using the ARIMA model."""
    print(f"PREDICTION TOOL: Generating ARIMA forecast...")
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df['Close'].iloc[:, 0]
        else:
            close_prices = df['Close']
        model = ARIMA(close_prices, order=(5,1,0)).fit()
        return {"forecast": [round(val, 2) for val in model.forecast(steps=forecast_periods)]}
    except Exception as e:
        return {"tool_error": f"ARIMA forecast failed: {e}"}