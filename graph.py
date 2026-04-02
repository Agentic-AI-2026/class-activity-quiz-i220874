### Graph
import json
import re
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# TODO: LLM INITIALIZATION
# We will define the actual LLM here later. 
# For now, we are just setting up the structure.
# Example: llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
# ==========================================

# ─── 1. STATE DEFINITION ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    goal: str
    plan: List[Dict[str, Any]]
    current_step: int
    results: List[Dict[str, Any]]

# ─── 2. PLANNER PROMPT ──────────────────────────────────────────────────────────
PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available MCP tools and their EXACT argument names:
  - fetch_wikipedia(topic: str)       → look up a topic on Wikipedia
  - fetch_data_source(source: str)    → source must be one of: sales, customers, expenses
  - get_weather(city: str)            → get real weather for a city

Use null for tool/args on synthesis or writing steps.
Return ONLY a valid JSON array. No markdown, no explanation."""

# ─── 3. PLANNER NODE ────────────────────────────────────────────────────────────
def planner_node(state: AgentState):
    print("\n--- PLANNER NODE ---")
    goal = state["goal"]
    
    # ⚠️ Note: This will use the 'llm' variable we define at the top later
    plan_resp = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=goal)
    ])
    
    # Extract text safely
    raw_text = plan_resp.content if isinstance(plan_resp.content, str) else plan_resp.content[0].get("text", "")
    
    # Clean and Parse JSON
    clean_json = re.sub(r"```json|```", "", raw_text).strip()
    plan = json.loads(clean_json)
    
    print(f"Generated Plan with {len(plan)} steps.")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
        
    # Update the LangGraph State
    return {
        "plan": plan, 
        "current_step": 0, # Starts the executor at step 0
        "results": []      # Starts with an empty list of results
    }