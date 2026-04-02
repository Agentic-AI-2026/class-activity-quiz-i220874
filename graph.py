### Graph
import json
import re
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

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
  - calculator(expression: str)       → use this to calculate numbers
  - search_web(query: str)            → use this to search the internet for real-time data
  - get_current_weather(city: str)→ get real weather for a city

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

# ─── 4. TOOL HELPERS & EXECUTOR NODE ────────────────────────────────────────────

TOOL_ARG_MAP = {
    "fetch_wikipedia":  "topic",
    "fetch_data_source": "source",
    "get_weather":      "city",
}

def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Remap hallucinated arg names to the correct parameter."""
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}

# We make this async because MCP tools (like your weather API) are called asynchronously
async def executor_node(state: AgentState):
    print("\n--- EXECUTOR NODE ---")
    plan = state["plan"]
    current_step_idx = state["current_step"]
    results = state["results"]
    
    # Identify which step we are currently on
    step = plan[current_step_idx]
    print(f"Executing Step {step['step']}: {step['description']}")
    
    tool_name = step.get("tool")
    
    # ⚠️ Note: 'tools_map' and 'llm' will be injected/imported when we run the main script
    if tool_name and tool_name in tools_map:
        corrected = safe_args(tool_name, step.get("args") or {})
        # Execute the specific MCP tool
        result = await tools_map[tool_name].ainvoke(corrected)
    else:
        # Synthesis step — use the LLM to write a summary based on prior results
        context  = "\n".join([f"Step {r['step']}: {r['result']}" for r in results])
        response = llm.invoke([
            HumanMessage(content=f"{step['description']}\n\nContext:\n{context}")
        ])
        result = response.content

    print(f"Result: {str(result)[:100]}...\n")
    
    # Store the result
    results.append({"step": step["step"], "description": step["description"], "result": str(result)})
    
    # Increment the step tracker and update the state
    return {
        "results": results,
        "current_step": current_step_idx + 1
    }

# ─── 5. GRAPH WIRING & ROUTING ──────────────────────────────────────────────────

def should_continue(state: AgentState):
    """Routing logic to determine if we are done or need to execute the next step."""
    plan = state["plan"]
    current_step = state["current_step"]
    
    # If we haven't reached the end of the plan, keep looping
    if current_step < len(plan):
        return "continue"
    else:
        return "end"

# Initialize the graph with our state schema
workflow = StateGraph(AgentState)

# Add our two nodes
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

# Define the standard edges (how it starts)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")

# Define the conditional edge (the loop)
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue": "executor", # Loop back to execute the next step
        "end": END              # Finish the graph
    }
)

# Compile the graph into a runnable application
app = workflow.compile()
print("\n✅ LangGraph compiled successfully!")