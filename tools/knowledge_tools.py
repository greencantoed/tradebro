# tools/knowledge_tools.py
# FINAL VERIFIED BUILD for AEGIS v11 - With Functional Knowledge Base

import json
import os
import re

# This global config will be set by the main orchestrator
config = {}
def set_config(app_config):
    """Allows the main orchestrator to pass in the master config."""
    global config
    config = app_config

KNOWLEDGE_BASE_FILE = "knowledge_base/financial_concepts.json"

def retrieve_knowledge(query: str):
    """
    Searches a local knowledge base for conceptual definitions and principles.
    This is the primary tool for 'what is' or 'explain' questions.
    """
    print(f"KNOWLEDGE TOOL: Searching for '{query}' in local knowledge base...")
    
    try:
        if not os.path.exists(KNOWLEDGE_BASE_FILE):
             return {"note": f"Knowledge base file not found at '{KNOWLEDGE_BASE_FILE}'. RAG is disabled."}

        with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        # --- Upgraded Search Logic ---
        # Instead of a simple keyword match, we'll score each entry based on relevance.
        query_words = set(re.findall(r'\w+', query.lower()))
        scored_results = []

        for entry in kb:
            score = 0
            # Higher score for matching the main concept title
            if any(word in entry.get("concept", "").lower() for word in query_words):
                score += 10
            # Standard score for matching keywords
            keywords = set(entry.get("keywords", []))
            score += len(query_words.intersection(keywords)) * 2
            
            if score > 0:
                scored_results.append({"score": score, "entry": entry})
        
        # Sort by the highest score to get the most relevant results first
        if scored_results:
            sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)
            # Return the full entry for the top results (up to 3)
            top_entries = [res['entry'] for res in sorted_results[:3]]
            return {"status": "success", "retrieved_snippets": top_entries}
        else:
            return {"status": "no_results", "message": "No relevant snippets found in the local knowledge base."}
            
    except Exception as e:
        return {"tool_error": f"Failed to access knowledge base: {e}"}<