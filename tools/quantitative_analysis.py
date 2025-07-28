import json
from typing import Dict, List, Any, Tuple

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import numpy as np

# UNIFIED CONFIG PROPAGATION
config: Dict[str, Any] = {}

def set_config(app_config: Dict[str, Any]) -> None:
    global config
    config = app_config


# -----------------------------------------------------------------------------
# Econometric and Technical Analysis
# -----------------------------------------------------------------------------

def perform_econometric_analysis(analysis_type: str, json_data: dict, **kwargs) -> Any:
    """Perform statistical analysis such as a correlation matrix or linear regression."""
    print(f"ANALYSIS TOOL: Performing '{analysis_type}'...")
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        # Correctly handle single vs multi-ticker dataframes
        if isinstance(df.columns, pd.MultiIndex):
            df_close = df['Close']
        else:
            df_close = df[['Close']]  # Ensure it's a DataFrame

        if analysis_type == 'correlation_matrix':
            return json.loads(df_close.corr().to_json())
        elif analysis_type == 'linear_regression':
            dependent_var = kwargs.get('dependent_var')
            independent_vars = kwargs.get('independent_vars')
            if not dependent_var or not independent_vars:
                return {"tool_error": "Dependent and independent variables are required for regression."}
            # Flatten any MultiIndex columns
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

# -----------------------------------------------------------------------------
# Performance Metrics and Portfolio Optimization
# -----------------------------------------------------------------------------

def _max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve (returns fraction, not %)."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()

def _sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Compute the Sortino ratio for a returns series."""
    downside = returns.copy()
    downside[downside > risk_free] = 0
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return float('nan')
    return (returns.mean() - risk_free) / downside_std

def _profit_factor(trades: List[float]) -> float:
    """Compute the profit factor given a list of trade returns."""
    gains = sum(x for x in trades if x > 0)
    losses = -sum(x for x in trades if x < 0)
    return gains / losses if losses != 0 else float('inf')

def calculate_performance_metrics(json_data: dict) -> Any:
    """
    Given a JSON representation of OHLCV or equity data, compute a suite of
    performance metrics including Sharpe ratio, Sortino ratio, Calmar ratio,
    maximum drawdown and annual return. Returns the metrics in a dictionary.
    """
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        if 'Close' in df:
            prices = df['Close'].copy()
        else:
            # Assume equity curve is provided directly
            prices = df.iloc[:, 0]
        returns = prices.pct_change().dropna()
        risk_free_rate = 0.0
        sharpe = (returns.mean() - risk_free_rate) / returns.std(ddof=0) * np.sqrt(252)
        sortino = _sortino_ratio(returns, risk_free=risk_free_rate) * np.sqrt(252)
        # Construct equity curve from returns
        equity = (1 + returns).cumprod()
        mdd = abs(_max_drawdown(equity))
        annual_return = equity.iloc[-1] ** (252 / len(returns)) - 1
        calmar = annual_return / mdd if mdd != 0 else float('inf')
        metrics = {
            'annual_return': round(annual_return * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'max_drawdown_pct': round(mdd * 100, 2),
            'calmar_ratio': round(calmar, 2),
        }
        return metrics
    except Exception as e:
        return {"tool_error": f"Performance metrics calculation failed: {e}"}


def optimize_portfolio(json_data: dict) -> Any:
    """
    Compute minimum-variance portfolio weights from a JSON representation of price
    data. Returns the asset weights and expected portfolio return & volatility.
    """
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        returns = df.pct_change().dropna()
        mu = returns.mean()
        cov = returns.cov()
        inv_cov = np.linalg.inv(cov.values)
        ones = np.ones(len(mu))
        weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        weights = weights / weights.sum()
        portfolio_return = float((weights @ mu.values))
        portfolio_vol = float(np.sqrt(weights @ cov.values @ weights))
        return {
            'weights': {col: round(w, 4) for col, w in zip(df.columns, weights)},
            'expected_return': round(portfolio_return * 252 * 100, 2),
            'expected_volatility': round(portfolio_vol * np.sqrt(252) * 100, 2),
        }
    except Exception as e:
        return {"tool_error": f"Portfolio optimization failed: {e}"}


def stress_test_portfolio(portfolio_holdings: List[Dict[str, Any]], shock_pct: float = -0.2) -> Any:
    """
    Simulate the impact of a market shock on a portfolio by applying a uniform
    percentage shock to all holdings' prices. Returns the portfolio loss/gain.
    """
    try:
        weights = np.array([p.get('weight', 1.0 / len(portfolio_holdings)) for p in portfolio_holdings])
        values = np.array([p.get('value', 1000.0) for p in portfolio_holdings])
        shocked_values = values * (1 + shock_pct)
        original_portfolio = np.sum(weights * values)
        shocked_portfolio = np.sum(weights * shocked_values)
        return {
            'original_value': round(original_portfolio, 2),
            'shocked_value': round(shocked_portfolio, 2),
            'portfolio_change_pct': round((shocked_portfolio / original_portfolio - 1) * 100, 2),
        }
    except Exception as e:
        return {"tool_error": f"Stress test failed: {e}"}


def calculate_technical_indicator(indicator_name: str, json_data: dict) -> Any:
    """Calculate technical indicators like Simple Moving Average (SMA) or RSI."""
    print(f"ANALYSIS TOOL: Calculating '{indicator_name}'...")
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df['Close'].iloc[:, 0]
        else:
            close_prices = df['Close']
        indicator_name = indicator_name.upper()
        if indicator_name == 'SMA':
            return {'sma_50': round(close_prices.rolling(window=50).mean().iloc[-1], 2)}
        elif indicator_name == 'RSI':
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return {'rsi_14': round(rsi.iloc[-1], 2)}
        else:
            return {"tool_error": "Unsupported indicator."}
    except Exception as e:
        return {"tool_error": f"Indicator calculation failed: {e}"}


def analyze_portfolio_risk(portfolio_holdings: List[Dict[str, Any]]) -> Any:
    """Analyze a user's stock portfolio to calculate its overall market beta."""
    print(f"ANALYSIS TOOL: Analyzing portfolio risk...")
    try:
        tickers = [p['ticker'] for p in portfolio_holdings]
        if not tickers:
            return {"tool_error": "Portfolio holdings cannot be empty."}
        data = yf.download(tickers + ['SPY'], period='1y', progress=False, threads=False)['Adj Close'].pct_change().dropna()
        market_variance = data['SPY'].var()
        betas = {ticker: data[ticker].cov(data['SPY']) / market_variance for ticker in tickers}
        return {'individual_betas': {t: round(b, 2) for t, b in betas.items()}}
    except Exception as e:
        return {"tool_error": f"Risk analysis failed: {e}"}


# -----------------------------------------------------------------------------
# Backtesting and Forecasting
# -----------------------------------------------------------------------------

def backtest_strategy(ticker: str, strategy_name: str, cache: Any) -> Any:
    """Backtest a predefined trading strategy (SmaCross, Momentum, MeanReversion) on historical data."""
    print(f"ANALYSIS TOOL: Backtesting '{strategy_name}' on {ticker}...")
    try:
        data_df = cache.get_ohlcv(tuple([ticker]), period='5y')
        if data_df is None or data_df.empty:
            return {"tool_error": "Could not retrieve data for backtest."}
        data_df.set_index('Date', inplace=True)
        bt_config = config.get('backtesting', {})
        commission = bt_config.get('commission_bps', 5) / 10_000
        slippage = bt_config.get('slippage_bps', 10) / 10_000
        strategies: Dict[str, Tuple[type, Dict[str, Any]]] = {}
        # Simple moving-average crossover (default)
        class SmaCross(Strategy):
            n1, n2 = 20, 50
            def init(self):
                self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
                self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)
            def next(self):
                if crossover(self.sma1, self.sma2):
                    self.buy()
                elif crossover(self.sma2, self.sma1):
                    self.sell()
        strategies['SmaCross'] = (SmaCross, {})
        # Momentum strategy: buy when price above 50-day MA and sell when below
        class Momentum(Strategy):
            window = 50
            def init(self):
                self.ma = self.I(lambda x: pd.Series(x).rolling(self.window).mean(), self.data.Close)
            def next(self):
                if self.data.Close[-1] > self.ma[-1]:
                    self.position.close()
                    self.buy()
                elif self.data.Close[-1] < self.ma[-1]:
                    self.position.close()
                    self.sell()
        strategies['Momentum'] = (Momentum, {})
        # Mean Reversion using Bollinger Bands
        class MeanReversion(Strategy):
            window = 20
            std_mult = 2
            def init(self):
                close = self.data.Close
                self.ma = self.I(lambda x: pd.Series(x).rolling(self.window).mean(), close)
                self.std = self.I(lambda x: pd.Series(x).rolling(self.window).std(), close)
                self.upper = self.I(lambda x: self.ma + self.std_mult * self.std, close)
                self.lower = self.I(lambda x: self.ma - self.std_mult * self.std, close)
            def next(self):
                price = self.data.Close[-1]
                if price < self.lower[-1]:
                    self.position.close()
                    self.buy()
                elif price > self.upper[-1]:
                    self.position.close()
                    self.sell()
        strategies['MeanReversion'] = (MeanReversion, {})

        if strategy_name not in strategies:
            return {"tool_error": "Unsupported strategy."}
        StratClass, strat_kwargs = strategies[strategy_name]
        bt = Backtest(data_df, StratClass, cash=10_000, commission=commission, slippage=slippage)
        stats = bt.run()
        return {
            'return_pct': f"{stats.get('Return [%]', 0):.2f}%",
            'win_rate': f"{stats.get('Win Rate [%]', 0):.2f}%",
            'sharpe_ratio': f"{stats.get('Sharpe Ratio', 0):.2f}",
            'sortino_ratio': f"{stats.get('Sortino Ratio', 0):.2f}",
            'calmar_ratio': f"{stats.get('Calmar Ratio', 0):.2f}",
            'max_drawdown_pct': f"{stats.get('Max. Drawdown [%]', 0):.2f}%",
            'profit_factor': f"{stats.get('Profit Factor', 0):.2f}",
        }
    except Exception as e:
        return {"tool_error": f"Backtesting failed for {ticker}: {e}"}


def generate_time_series_forecast(json_data: dict, forecast_periods: int = 30) -> Any:
    """Generate a statistical time series forecast using the ARIMA model."""
    print(f"PREDICTION TOOL: Generating ARIMA forecast...")
    try:
        df = pd.read_json(json.dumps(json_data), orient='split')
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df['Close'].iloc[:, 0]
        else:
            close_prices = df['Close']
        model = ARIMA(close_prices, order=(5, 1, 0)).fit()
        return {'forecast': [round(val, 2) for val in model.forecast(steps=forecast_periods)]}
    except Exception as e:
        return {"tool_error": f"ARIMA forecast failed: {e}"}