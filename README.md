# Multi-Agent Research System

A production-ready multi-agent research system with clean architecture, flexible provider support, and an intuitive web interface. Built with LangGraph, LangChain, and Gradio.

## 🎯 Features

- **Multi-Agent Orchestration**: Specialized agents (Planner, Researcher, Writer, Reviewer) work together
- **Provider Flexibility**: Support for multiple LLM and search providers
- **Clean Architecture**: Proper separation of concerns with modular design
- **Web Interface**: Intuitive Gradio UI with real-time results
- **Error Resilience**: Graceful degradation and helpful error messages
- **Type Safety**: Full type hints throughout codebase

## 🏗️ Architecture

```
multi-agent-research/
├── agents/                 # Agent implementations
│   ├── planner.py         # Query decomposition
│   ├── researcher.py      # Search execution
│   ├── writer.py          # Brief synthesis
│   └── reviewer.py        # Citation validation
├── providers/             # Provider abstractions
│   ├── llm.py            # LLM provider factory
│   └── search.py         # Search provider factory
├── workflow.py           # LangGraph orchestration
├── app.py               # Gradio web interface
└── multi_agent_research.py  # Legacy CLI (deprecated)
```

### Workflow

```
Entry → Planner → Researcher → Writer → Reviewer → END
                               ↑          |
                               └──────────┘
                           (max 1 revision)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/kaiser-data/multi-agent-research.git
cd multi-agent-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**Minimum configuration (free options):**

```env
LLM_PROVIDER=ollama
MODEL_NAME=llama3.1:70b
OLLAMA_BASE_URL=http://localhost:11434/v1
SEARCH_PROVIDER=duckduckgo
```

### 3. Run

**Web Interface (Recommended):**
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

**Python API:**
```python
from workflow import run_research

result = run_research(
    query="What are the latest developments in AI safety?",
    llm_provider="ollama",
    model_name="llama3.1:70b",
    search_provider="duckduckgo",
    num_results=5
)

print(result["final"])
```

## 🔧 Provider Configuration

### LLM Providers

| Provider | Cost | Setup | Configuration |
|----------|------|-------|---------------|
| **Anthropic Claude** | $0.50-2.00/query | Get API key | `LLM_PROVIDER=anthropic`<br>`ANTHROPIC_API_KEY=sk-ant-...` |
| **OpenAI GPT** | $0.10-1.00/query | Get API key | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=sk-...` |
| **Ollama** | FREE | Install locally | `LLM_PROVIDER=ollama`<br>`OLLAMA_BASE_URL=http://localhost:11434/v1` |
| **Self-Hosted** | Server costs | Deploy server | `LLM_PROVIDER=custom`<br>`OPENAI_BASE_URL=http://your-server:8000/v1` |

### Search Providers

| Provider | Cost | Setup | Configuration |
|----------|------|-------|---------------|
| **DuckDuckGo** | FREE | No setup needed | `SEARCH_PROVIDER=duckduckgo` |
| **Brave Search** | 2K free/month | Get API key | `SEARCH_PROVIDER=brave`<br>`BRAVE_API_KEY=...` |
| **Serper.dev** | 2.5K free/month | Get API key | `SEARCH_PROVIDER=serper`<br>`SERPER_API_KEY=...` |

## 📖 Detailed Setup Guides

### Ollama Setup (Recommended for Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended model
ollama pull llama3.1:70b

# Configure .env
cat > .env << EOF
LLM_PROVIDER=ollama
MODEL_NAME=llama3.1:70b
OLLAMA_BASE_URL=http://localhost:11434/v1
SEARCH_PROVIDER=duckduckgo
EOF

# Run
python app.py
```

### Anthropic Claude Setup

```bash
# 1. Sign up at https://console.anthropic.com
# 2. Add credits ($5-20 to start)
# 3. Create API key
# 4. Configure .env

cat > .env << EOF
LLM_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-4-5-20250929
ANTHROPIC_API_KEY=sk-ant-your-key-here
SEARCH_PROVIDER=duckduckgo
EOF

# Run
python app.py
```

### Self-Hosted LLM Setup

For vLLM, Text Generation WebUI, or other OpenAI-compatible servers:

```env
LLM_PROVIDER=custom
MODEL_NAME=your-model-name
OPENAI_BASE_URL=http://your-server:8000/v1
OPENAI_API_KEY=your-key-if-needed
SEARCH_PROVIDER=duckduckgo
```

## 💡 Usage Examples

### Web Interface

1. Start the server: `python app.py`
2. Open http://localhost:7860
3. Select providers from dropdowns
4. Enter research query
5. Click "Start Research"
6. View formatted results

### Python API

```python
from workflow import run_research

# Basic usage
result = run_research(
    query="Impact of quantum computing on cryptography"
)

# Custom providers
result = run_research(
    query="Latest AI safety research",
    llm_provider="ollama",
    model_name="llama3.1:70b",
    search_provider="brave",
    num_results=8
)

# Access results
print("Plan:", result["plan"])
print("Brief:", result["final"])
print("References:", result["references"])
```

### Extending the System

**Add a new agent:**

```python
# agents/custom_agent.py
def custom_agent_node(state):
    # Your logic here
    return {"custom_field": value, "messages": [...]}

# workflow.py
from agents.custom_agent import custom_agent_node
workflow.add_node("custom", custom_agent_node)
workflow.add_edge("reviewer", "custom")
```

**Add a new LLM provider:**

```python
# providers/llm.py
elif provider == "new_provider":
    return YourLLMClass(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("YOUR_API_KEY"),
    )
```

## 🛡️ Error Handling

The system handles errors gracefully:

- **Missing API keys**: Clear error messages with setup instructions
- **Provider failures**: Automatic retry with exponential backoff
- **Empty results**: Continues with partial data when possible
- **LLM errors**: Falls back to simpler operations

## 📊 Cost Comparison

**Free Options (Recommended for Testing):**
- Ollama (local) + DuckDuckGo: $0/query
- Total setup time: ~10 minutes

**Low-Cost Cloud:**
- OpenAI GPT-4o-mini + DuckDuckGo: ~$0.10/query
- Anthropic Claude Haiku + DuckDuckGo: ~$0.15/query

**High-Quality Cloud:**
- Anthropic Claude Sonnet 4 + Brave: ~$1.50/query
- OpenAI GPT-4o + Serper: ~$0.75/query

## 🧪 Testing

```bash
# Test provider connections
python -c "from providers.llm import get_llm; llm = get_llm('ollama', 'llama3.1:70b'); print('LLM OK')"
python -c "from providers.search import search; results = search('test', 'duckduckgo', 3); print(f'Search OK: {len(results)} results')"

# Test workflow
python -c "from workflow import run_research; result = run_research('AI test query'); print('Workflow OK')"
```

## 🐛 Troubleshooting

### "ANTHROPIC_API_KEY required"
- Add API key to `.env` file
- Get key at https://console.anthropic.com

### "duckduckgo-search required"
- Install: `pip install duckduckgo-search`

### "Ollama connection failed"
- Ensure Ollama is running: `ollama serve`
- Check URL in `.env` matches Ollama endpoint

### "No search results found"
- Try different search provider
- Check internet connection
- Verify API keys if using Brave/Serper

### Gradio not loading
- Check port 7860 is available
- Try: `python app.py` with different port
- Firewall may be blocking connection

## 📝 Development

**Code Style:**
- PEP 8 formatting
- Type hints on all functions
- Docstrings with Args/Returns
- Max file length: 250 lines

**Architecture Principles:**
- Single Responsibility: Each module has one job
- Dependency Inversion: Depend on abstractions (providers)
- Error Resilience: Never crash, always degrade gracefully
- Testability: Pure functions, clear interfaces

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style guidelines
4. Add tests for new functionality
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/claude), [OpenAI](https://openai.com), and [Ollama](https://ollama.com)
- Search by [DuckDuckGo](https://duckduckgo.com), [Brave](https://brave.com/search/api/), and [Serper](https://serper.dev)
- UI by [Gradio](https://gradio.app)

## 📞 Support

- Issues: https://github.com/kaiser-data/multi-agent-research/issues
- Discussions: https://github.com/kaiser-data/multi-agent-research/discussions

---

**Ready to start researching?** Run `python app.py` and open http://localhost:7860 🚀
