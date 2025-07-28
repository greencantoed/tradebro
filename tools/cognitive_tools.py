"""
tools/cognitive_tools.py
=======================

This module provides a collection of cognitive helper utilities for the
trading assistant. These tools deal with logging of self-reflection,
simulation of search trend analysis, capturing user investment profiles,
and formatting investment theses. While not computationally intensive,
these functions help structure and record the assistant's reasoning.

The `log_self_reflection` function writes reflections to a JSON log file,
truncating entries that exceed a configurable token limit. The simulation
functions demonstrate how one might integrate additional data sources such
as Google Trends in a future version.
"""

import json
import os
from datetime import datetime
import random
from typing import Dict, List, Any

# This global config will be set by the main orchestrator
config: Dict[str, Any] = {}

def set_config(app_config: Dict[str, Any]) -> None:
    """Allows the main orchestrator to pass in the master config."""
    global config
    config = app_config

REFLECTION_LOG_PATH = "logs/reflection_log.json"

def log_self_reflection(reflection_text: str) -> Dict[str, Any]:
    """
    Append the model's self-critique to a disk log. Used for the Automated
    Reflection Protocol (ARP). Entries longer than 75% of the configured
    maximum reflection tokens are truncated.
    """
    print("COGNITIVE TOOL: Logging self-reflection...")
    try:
        max_len = config.get('parameters', {}).get('max_reflection_tokens', 200)
        if len(reflection_text.split()) > max_len * 0.75:
            reflection_text = " ".join(reflection_text.split()[: int(max_len * 0.75)]) + "... (truncated)"
        os.makedirs(os.path.dirname(REFLECTION_LOG_PATH), exist_ok=True)
        entry = {"timestamp_utc": datetime.utcnow().isoformat(), "reflection": reflection_text}
        data: List[Any] = []
        if os.path.exists(REFLECTION_LOG_PATH):
            with open(REFLECTION_LOG_PATH, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass
        data.append(entry)
        with open(REFLECTION_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return {"status": "Reflection logged successfully."}
    except Exception as e:
        return {"tool_error": f"Failed to log reflection: {e}"}


def get_search_trend_analysis(keywords: List[str]) -> Dict[str, Any]:
    """
    SIMULATED: Analyse Google search trend volume for keywords to gauge public interest.
    A full implementation would require a library like `pytrends`. This function
    returns random values for demonstration purposes.
    """
    print(f"COGNITIVE TOOL (SIMULATED): Analyzing search trends for {keywords}...")
    return {
        'note': 'This is a simulated tool. It returns random data.',
        'trends': {kw: random.randint(30, 100) for kw in keywords},
    }


def define_user_investment_profile(risk_tolerance: str, investment_horizon_years: int, primary_goal: str) -> Dict[str, Any]:
    """
    Store the user's investment profile to personalise future advice. This is a
    simple state-management tool; in a more advanced system, the profile would
    be persisted between sessions.
    """
    print("COGNITIVE TOOL: Defining user investment profile...")
    profile = {
        'risk_tolerance': risk_tolerance,
        'investment_horizon_years': investment_horizon_years,
        'primary_goal': primary_goal,
    }
    return {
        'status': 'success',
        'profile_summary': f"Profile set: {risk_tolerance} risk, {investment_horizon_years}-year horizon for {primary_goal}.",
    }


def generate_investment_thesis(
    ticker: str,
    bull_case_points: List[str],
    bear_case_points: List[str],
    final_recommendation: str,
) -> Dict[str, Any]:
    """
    Structure provided points into a formal investment thesis. The LLM calls
    this tool to format its own final output. A disclaimer is always
    appended to clarify that the output does not constitute financial advice.
    """
    print("COGNITIVE TOOL: Structuring investment thesis...")
    return {
        'ticker': ticker.upper(),
        'bull_case': bull_case_points,
        'bear_case': bear_case_points,
        'recommendation': final_recommendation,
        'disclaimer': 'This is a synthesized argument based on provided data and does not constitute financial advice.',
    }
