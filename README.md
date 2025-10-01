# Multi-Agent Research System

A sophisticated research automation tool using LangGraph + LangChain with Claude AI and Google Search to orchestrate multi-agent workflows for comprehensive query analysis.

## 🎯 Features

- **Multi-Agent Orchestration**: Coordinates specialized agents (Planner, Researcher, Writer, Reviewer)
- **Intelligent Research Planning**: Decomposes queries into focused research steps
- **Automated Search**: Executes Google searches via SerpAPI with retry logic
- **Citation-Based Synthesis**: Generates concise briefs with inline citations
- **Quality Review**: Validates factual claims and triggers revisions when needed
- **State Persistence**: Uses LangGraph checkpointing for workflow reliability

## 🏗️ Architecture

```
┌─────────┐    ┌────────────┐    ┌────────┐    ┌──────────┐
│ Planner │ -> │ Researcher │ -> │ Writer │ -> │ Reviewer │ -> END
└─────────┘    └────────────┘    └────────┘    └──────────┘
                                       ↑            │
                                       └────────────┘
                                    (revision loop)
```

### Agents

1. **Planner**: Decomposes research queries into 2-5 focused steps using Claude's extended thinking
2. **Researcher**: Executes Google searches for each step, normalizes results with error handling
3. **Writer**: Synthesizes findings into ≤200-word briefs with inline citations [1], [2], etc.
4. **Reviewer**: Validates citations and triggers one revision if needed

## 🚀 Installation

### Prerequisites

- Python 3.8+
- **One of:** Anthropic API key, Ollama, LM Studio, or self-hosted LLM server
- SerpAPI key (Google Search access)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd multi-agent-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install 'langchain>=0.3.27,<0.4' \
            'langgraph>=0.6.8,<0.7' \
            'langchain-anthropic>=0.3.21,<0.4' \
            'langchain-openai>=0.3.0' \
            python-dotenv \
            google-search-results \
            requests
```

### Configuration

Create a `.env` file in the project root. Choose one option:

#### Option 1: Anthropic Claude (Cloud API)
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
MODEL_NAME=claude-sonnet-4-5-20250929
SERPAPI_API_KEY=your-serpapi-key-here
```

#### Option 2: Ollama (Local - FREE)
```env
LLM_PROVIDER=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3.1:70b
SERPAPI_API_KEY=your-serpapi-key-here
```

#### Option 3: LM Studio (Local - FREE)
```env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:1234/v1
MODEL_NAME=local-model
SERPAPI_API_KEY=your-serpapi-key-here
```

#### Option 4: Self-Hosted Server (Remote)
```env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://your-server:8000/v1
OPENAI_API_KEY=your-api-key  # if required
MODEL_NAME=mistral-large
SERPAPI_API_KEY=your-serpapi-key-here
```

## 📖 Usage

### Basic Usage

```bash
python multi_agent_research.py --query "impact of quantum computing on cryptography"
```

### Advanced Options

```bash
# Specify number of results per search step (1-8)
python multi_agent_research.py --query "latest AI safety research" --results 8

# Override Claude model
python multi_agent_research.py --query "climate change solutions" --model claude-3-7-sonnet-20250219
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--query` | Yes | - | Research query to investigate |
| `--results` | No | 5 | Number of search results per step (1-8) |
| `--model` | No | `claude-sonnet-4-5-20250929` | Claude model override |

## 📊 Output Format

The script produces a structured output:

1. **Research Plan**: Numbered list of research steps
2. **Top Results**: Title and URL for each search result
3. **Final Brief**: Synthesized findings with inline citations (≤200 words)
4. **References**: Numbered list of source URLs

### Example Output

```
======================================================================
🔍 RESEARCH QUERY: impact of quantum computing on cryptography
======================================================================

📋 RESEARCH PLAN:
  1. Current state of quantum computing capabilities
  2. Vulnerable cryptographic algorithms
  3. Post-quantum cryptography solutions

🔗 TOP RESULTS:
  [1] Quantum Computing and Cryptography
      https://example.com/quantum-crypto
  ...

📝 FINAL BRIEF:
Quantum computing poses significant threats to current cryptographic
systems [1][2]. RSA and ECC algorithms are particularly vulnerable to
Shor's algorithm [3]...

📚 REFERENCES:
  [1] https://example.com/quantum-crypto
  [2] https://example.com/post-quantum
  ...
```

## 🔧 Model Selection & Self-Hosted Setup

### Supported Providers

| Provider | Cost | Setup Difficulty | Performance |
|----------|------|------------------|-------------|
| **Anthropic Claude** | $0.50-2.00/query | Easy | Excellent |
| **Ollama** | FREE | Easy | Good |
| **LM Studio** | FREE | Easy | Good |
| **Self-Hosted vLLM** | Server costs | Medium | Excellent |
| **Self-Hosted Text Generation WebUI** | Server costs | Medium | Good |

### Quick Setup Guides

#### Ollama (Recommended for Local)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:70b  # or mistral, phi3, etc.

# Run Ollama server (runs automatically on install)
ollama serve

# Configure .env
echo "LLM_PROVIDER=ollama" > .env
echo "OPENAI_BASE_URL=http://localhost:11434/v1" >> .env
echo "MODEL_NAME=llama3.1:70b" >> .env
echo "SERPAPI_API_KEY=your-key" >> .env
```

#### LM Studio
```bash
# 1. Download from https://lmstudio.ai
# 2. Load a model (e.g., Mistral, Llama)
# 3. Start local server (default: localhost:1234)
# 4. Configure .env
echo "LLM_PROVIDER=custom" > .env
echo "OPENAI_BASE_URL=http://localhost:1234/v1" >> .env
echo "MODEL_NAME=local-model" >> .env
echo "SERPAPI_API_KEY=your-key" >> .env
```

#### Self-Hosted Server (vLLM, TGI, etc.)
```env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://your-server:8000/v1
OPENAI_API_KEY=your-optional-key
MODEL_NAME=your-model-name
SERPAPI_API_KEY=your-serpapi-key
```

### Model Recommendations

**For Best Results:**
- Claude Sonnet 4 (paid, excellent)
- Llama 3.1 70B (free via Ollama, very good)
- Mistral Large (free via Ollama, good)

**For Faster/Lower Resource:**
- Llama 3.1 8B (free, decent)
- Phi-3 Medium (free, decent)
- Mistral 7B (free, decent)

## 🛡️ Error Handling

- **API Failures**: Automatic retry with exponential backoff (max 3 attempts)
- **Missing Keys**: Clear error messages with exit codes
- **Empty Results**: Graceful handling with warnings
- **Rate Limits**: Automatic delay between search requests

## 🧪 Development

### Project Structure

```
multi-agent-research/
├── multi_agent_research.py  # Main script
├── .env                      # Environment variables (not in git)
├── .gitignore
├── README.md
└── requirements.txt          # Coming soon
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/claude)
- Search provided by [SerpAPI](https://serpapi.com/)

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This tool is for research purposes. Ensure compliance with API terms of service and rate limits.
