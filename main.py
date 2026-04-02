import asyncio
import os
from dotenv import load_dotenv
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Import your local modules
from MCP_code import get_mcp_tools
import graph

# Load environment variables (pulls your GOOGLE_API_KEY from .env)
load_dotenv()

async def main():
    print("Initializing LLM and Tools...")
    
    # 1. Initialize the LLM (Using the free GROQ tier)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # 2. Load the MCP Tools (using your existing MCP setup)
    # We are loading math, search, and weather as requested by the activity
    tools, tools_map = await get_mcp_tools(["math", "search", "weather"])
    # 3. Inject the LLM and Tools into our graph module's namespace
    graph.llm = llm
    graph.tools_map = tools_map
    
    # 4. Define the exact test case goal from the PDF [cite: 36, 41, 42]
    #goal = "Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize."
    goal = "Plan an outdoor event for 150 people in Lahore: calculate tables/chairs (assume 10 people per table), use the web to find the average outdoor event ticket price, check the current weather, and summarize."
    print("\n=======================================================")
    print(f" GOAL: {goal}")
    print("=======================================================\n")
    
    # 5. Run the LangGraph Agent!
    # ainvoke() is the asynchronous way to trigger our compiled graph
    final_state = await graph.app.ainvoke({"goal": goal})
    
    print("\n=======================================================")
    print(" FINAL AGENT EXECUTION RESULTS")
    print("=======================================================\n")
    
    # Loop through the final results array and print them nicely
    for step in final_state["results"]:
        print(f"✅ Step {step['step']}: {step['description']}")
        print(f"   Result: {step['result']}\n")

if __name__ == "__main__":
    # Ensure nested async loops (like your MCP code) work correctly
    import nest_asyncio
    nest_asyncio.apply()
    
    asyncio.run(main())