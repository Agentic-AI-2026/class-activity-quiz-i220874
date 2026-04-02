# 🧠 LangGraph Planner-Executor Agent with MCP Tools

This repository contains a state-driven AI agent workflow built using **LangGraph**. It upgrades a legacy LangChain Planner-Executor architecture into a robust, cyclic graph capable of generating multi-step plans and executing them sequentially using local and remote **Model Context Protocol (MCP)** tools.

## ✨ Features
* **State-Driven Workflow:** Utilizes LangGraph to manage agent state (`goal`, `plan`, `current_step`, `results`) across distinct Planner and Executor nodes.
* **Intelligent Planning:** Uses Groq (Llama 3.3) to break down complex user prompts into structured, sequential JSON steps.
* **MCP Tool Integration:** Seamlessly interacts with background server tools using `langchain-mcp-adapters` via `stdio` and HTTP transports.
* **Available Tools:**
  * 🧮 `calculator` (Local Math Server)
  * 🌐 `search_web` (Tavily Search API)
  * 🌤️ `get_current_weather` (Local Uvicorn/FastAPI Weather Server)

---

## 🛠️ Prerequisites
* Python 3.9+
* [Groq API Key](https://console.groq.com/) (Free)
* [Tavily API Key](https://app.tavily.com/) (Free)

---

## 🚀 Installation & Setup

**1. Clone the repository and navigate to the project directory:**
```bash
git clone <your-repo-url>
cd class-activity-quiz-i220874

