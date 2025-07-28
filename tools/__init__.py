"""Tool aggregation and configuration for the Tradebro assistant.

This module collects all available tool functions from the submodules and
exposes convenience functions to propagate the global configuration into
each tool category. Importing this module with `from tools import *` will
automatically make all tool functions available at the package level.

The `set_data_config`, `set_quant_config` and `set_cog_config` functions
are used by the main orchestrator to push the unified configuration into
the respective submodules. Without calling these functions, the tools
will not have access to API keys and other runtime settings.
"""

from .data_sourcing import (
    get_historical_market_data,
    get_financial_statements,
    get_analyst_ratings,
    get_insider_transactions,
    get_financial_news_headlines,
    screen_for_stocks,
    get_sector_performance,
    get_economic_data,
    get_market_fear_and_greed_index,
    get_macro_signals,
    MarketDataCache,
    set_config as set_data_config,
)

from .quantitative_analysis import (
    perform_econometric_analysis,
    calculate_technical_indicator,
    analyze_portfolio_risk,
    backtest_strategy,
    generate_time_series_forecast,
    calculate_performance_metrics,
    optimize_portfolio,
    stress_test_portfolio,
    optimise_moving_average_strategy,
    optimize_risk_parity_portfolio,
    set_config as set_quant_config,
)

from .cognitive_tools import (
    log_self_reflection,
    get_search_trend_analysis,
    define_user_investment_profile,
    generate_investment_thesis,
    set_config as set_cog_config,
)

from .knowledge_tools import (
    retrieve_knowledge,
    set_config as set_knowledge_config,
)

__all__ = [
    # Data sourcing tools
    'get_historical_market_data',
    'get_financial_statements',
    'get_analyst_ratings',
    'get_insider_transactions',
    'get_financial_news_headlines',
    'screen_for_stocks',
    'get_sector_performance',
    'get_economic_data',
    'get_market_fear_and_greed_index',
    'get_macro_signals',
    'MarketDataCache',
    # Quantitative analysis tools
    'perform_econometric_analysis',
    'calculate_technical_indicator',
    'analyze_portfolio_risk',
    'backtest_strategy',
    'generate_time_series_forecast',
    'calculate_performance_metrics',
    'optimize_portfolio',
    'stress_test_portfolio',
    'optimise_moving_average_strategy',
    'optimize_risk_parity_portfolio',
    # Cognitive tools
    'log_self_reflection',
    'get_search_trend_analysis',
    'define_user_investment_profile',
    'generate_investment_thesis',
    # Knowledge tools
    'retrieve_knowledge',
    # Config setters
    'set_data_config',
    'set_quant_config',
    'set_cog_config',
    'set_knowledge_config',
]
