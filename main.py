# ================================================================= #
# == Legion Framework v12.5 - Main Orchestrator (Pro Build)      == #
# == This version directly upgrades v12.2 with the secure        == #
# == two-file config system and Gemini 2.5 Pro specifications.   == #
# ================================================================= #

import toml
import os
from dotenv import load_dotenv
import google.generativeai as genai
import functools
import re
import sys

# --- 1. CONFIGURATION & SETUP (UPGRADED FOR SECURITY) ---
try:
    # Step 1: Load non-secret settings from config.toml
    config = toml.load("config.toml")
    
    # Step 2: Load secret API keys from .env file
    load_dotenv()
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "fmp": os.getenv("FMP_API_KEY"),
        "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "fred": os.getenv("FRED_API_KEY"),
        "news_api": os.getenv("NEWS_API_KEY")
    }
    
    # Step 3: Check if all keys were loaded successfully
    if not all(api_keys.values()):
        raise KeyError("One or more API keys are missing from your .env file.")

    # Step 4: Combine settings and secrets into one config object
    config['api_keys'] = api_keys

    # Configure the primary Google AI client
    genai.configure(api_key=config['api_keys']['gemini'])

except (FileNotFoundError, KeyError) as e:
    raise SystemExit(f"CRITICAL: Configuration error. Ensure `config.toml` and `.env` are present and correct. Error: {e}")

# --- 2. BOOTSTRAP & IMPORTS ---
try:
    from tools import *
    from tools.data_sourcing import MarketDataCache
    
    # Propagate the complete config object to all tool modules
    set_data_config(config)
    set_quant_config(config)
    set_cog_config(config)
except ImportError as e:
    raise SystemExit(f"CRITICAL: Failed to import tool modules. Ensure `tools/__init__.py` is correct. Error: {e}")


# --- 3. MODEL INITIALIZATION FACTORY (UPGRADED FOR GEMINI 2.5 PRO) ---
@functools.lru_cache(maxsize=2)
def get_model(operating_mode: str):
    """
    Factory function to create, configure, and return the GenerativeModel
    with Pro settings from the config file.
    """
    prompt_file = f"system_prompts/{operating_mode}_mode.md"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        raise SystemExit(f"CRITICAL: System prompt file not found at '{prompt_file}'.")

    # UPGRADE: Create the GenerationConfig object from the .toml file
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=config['generation_config']['max_output_tokens']
    )

    cache = MarketDataCache(
        db_path=config['cache']['db_path'],
        staleness_hours=config['cache']['staleness_hours']
    )

    get_historical_market_data_tool = functools.partial(get_historical_market_data, cache=cache)
    get_historical_market_data_tool.__doc__ = get_historical_market_data.__doc__

    backtest_strategy_tool = functools.partial(backtest_strategy, cache=cache)
    backtest_strategy_tool.__doc__ = backtest_strategy.__doc__

    all_tools = [
        get_historical_market_data_tool, get_financial_statements, get_analyst_ratings, 
        get_insider_transactions, get_financial_news_headlines, screen_for_stocks, 
        get_sector_performance, get_economic_data, get_market_fear_and_greed_index,
        perform_econometric_analysis, calculate_technical_indicator, analyze_portfolio_risk, 
        backtest_strategy_tool, generate_time_series_forecast, log_self_reflection,
        get_search_trend_analysis, define_user_investment_profile, generate_investment_thesis, 
        retrieve_knowledge
    ]
    
    return genai.GenerativeModel(
        model_name=config['model']['name'],
        system_instruction=system_prompt,
        tools=all_tools,
        generation_config=generation_config # Pass the new Pro configuration here
    )

# --- 4. MAIN APPLICATION ENGINE (UNCHANGED CORE LOGIC) ---
def run_engine():
    """Initializes and runs the main conversation loop for the operator."""
    mode_override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--mode' else None
    mode = mode_override or config['system']['operating_mode']
    
    model = get_model(mode)
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    total_tokens = 0
    turn_count = 0
    print("---" * 20)
    print(f"Legion v12.5 (Pro Engine) online. Model: {config['model']['name']}.")
    print(f"OPERATING MODE: {mode.upper()}. Max Output Tokens: {config['generation_config']['max_output_tokens']}")
    print("---" * 20)

    while True:
        try:
            token_budget = config['features'].get('token_budget', 1e6)
            if total_tokens > token_budget:
                print("SYSTEM HALT: Session token budget has been exceeded.")
                break
            
            user_input = input("Operator: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            turn_count += 1
            token_mode = config['features'].get('token_counter_mode', 'off')
            
            if token_mode != 'off':
                prompt_tokens = model.count_tokens(chat.history + [{'role':'user', 'parts': [user_input]}]).total_tokens
                if token_mode == 'turn':
                    print(f"// Sending {prompt_tokens} tokens... //")

            response = chat.send_message(user_input)
            
            validated_text = response.text
            if mode == 'rigorous' and not re.match(r'^\[LEVEL [1-3]\]', response.text):
                print("// PROTOCOL VIOLATION: Dispatch missing Level header. //")

            print(f"\nDispatch:\n{validated_text}\n")

            if token_mode != 'off':
                response_tokens = model.count_tokens(response.parts).total_tokens
                total_tokens += prompt_tokens + response_tokens
                if token_mode == 'turn':
                    print(f"// Turn tokens: {prompt_tokens+response_tokens}. Session total: {total_tokens} //")
                elif token_mode == 'summary' and turn_count % 5 == 0:
                    print(f"// Session total after {turn_count} turns: {total_tokens} //")

        except KeyboardInterrupt:
            print("\nOperator abort. Shutting down engine.")
            break
        except Exception as e:
            print(f"\nSYSTEM ALERT: A critical error occurred in the main loop: {e}\n")
            
    print("Legion Engine offline.")

if __name__ == "__main__":
    run_engine()