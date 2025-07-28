# tools/cognitive_tools.py
# FINAL VERIFIED BUILD for AEGIS v11

import json
import os
from datetime import datetime
import random

# This global config will be set by the main orchestrator
config = {}
def set_config(app_config):
    """Allows the main orchestrator to pass in the master config."""
    global config
    config = app_config

REFLECTION_LOG_PATH = "logs/reflection_log.json"

def log_self_reflection(reflection_text: str):
    """
    A pure side-effect tool to append the model's self-critique to a disk log.
    Used for the Automated Reflection Protocol (ARP).
    """
    print("COGNITIVE TOOL: Logging self-reflection...")
    try:
        max_len = config.get('parameters', {}).get('max_reflection_tokens', 200)
        # Simple word count as a proxy for token count to enforce the limit
        if len(reflection_text.split()) > max_len * 0.75:
             reflection_text = " ".join(reflection_text.split()[:int(max_len * 0.75)]) + "... (truncated)"
        
        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(REFLECTION_LOG_PATH), exist_ok=True)
        
        entry = {"timestamp_utc": datetime.utcnow().isoformat(), "reflection": reflection_text}
        data = []
        if os.path.exists(REFLECTION_LOG_PATH):
            with open(REFLECTION_LOG_PATH, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError: # Handle empty or corrupted file
                    pass
        data.append(entry)
        with open(REFLECTION_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        return {"status": "Reflection logged successfully."}
    except Exception as e:
        return {"tool_error": f"Failed to log reflection: {e}"}

def get_search_trend_analysis(keywords: list[str]):
    """
    SIMULATED: Analyzes Google search trend volume for keywords to gauge public interest or panic.
    A full implementation would require a library like 'pytrends'.
    """
    print(f"COGNITIVE TOOL (SIMULATED): Analyzing search trends for {keywords}...")
    # This simulates a successful response for demonstration purposes.
    return {
        "note": "This is a simulated tool. It returns random data.",
        "trends": {kw: random.randint(30, 100) for kw in keywords}
    }

def define_user_investment_profile(risk_tolerance: str, investment_horizon_years: int, primary_goal: str):
    """
    Stores the user's investment profile to personalize future advice.
    This is a simple state-management tool.
    """
    print("COGNITIVE TOOL: Defining user investment profile...")
    profile = {
        "risk_tolerance": risk_tolerance,
        "investment_horizon_years": investment_horizon_years,
        "primary_goal": primary_goal
    }
    # In a more advanced system, this would be saved to a persistent user profile file.
    # For a solo operator, returning it is sufficient for the session.
    return {
        "status": "success",
        "profile_summary": f"Profile set: {risk_tolerance} risk, {investment_horizon_years}-year horizon for {primary_goal}."
    }

def generate_investment_thesis(ticker: str, bull_case_points: list[str], bear_case_points: list[str], final_recommendation: str):
    """
    A meta-tool that structures provided points into a formal investment thesis.
    The LLM calls this to format its own final output.
    """
    print("COGNITIVE TOOL: Structuring investment thesis...")
    return {
        "ticker": ticker.upper(),
        "bull_case": bull_case_points,
        "bear_case": bear_case_points,
        "recommendation": final_recommendation,
        "disclaimer": "This is a synthesized argument based on provided data and does not constitute financial advice."
    }