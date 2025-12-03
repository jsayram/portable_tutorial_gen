# Portable Tutorial Generator

A self-contained, cross-platform tool to generate beginner-friendly tutorials from local codebases.

## Supported Platforms
- **Windows** (10/11)
- **macOS** (Intel & Apple Silicon)
- **Linux** (Ubuntu, Debian, Fedora, etc.)

## Quick Start

### 1. Install Python 3.8+
Make sure Python 3.8 or higher is installed on your system.

### 2. Install Dependencies

**Windows (PowerShell):**
```powershell
cd portable_tutorial_gen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
cd portable_tutorial_gen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in this directory:

```bash
# Choose ONE provider (OpenAI is recommended and checked first)
OPENAI_API_KEY=sk-your-key-here

# Optional: Other providers (checked in this order if OpenAI not set)
# GEMINI_API_KEY=your-gemini-key
# OPENROUTER_API_KEY=your-openrouter-key
# LLM_API_BASE_URL=http://localhost:11434  # For local models like Ollama

# Optional: Model configuration
# OPENAI_MODEL=gpt-4o  # Default: gpt-3.5-turbo
```

### 4. Run the Generator

```bash
# Generate tutorial from a local directory
python run.py --dir /path/to/your/code

# With custom output directory
python run.py --dir /path/to/your/code --output ./my_tutorials

# Specify file patterns
python run.py --dir /path/to/your/code --include "*.py" "*.js" --exclude "*test*"

# Generate in different language
python run.py --dir /path/to/your/code --language spanish
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dir` | Path to local directory (required) | - |
| `--name` | Project name | Derived from directory |
| `--output` | Output directory | `./output` |
| `--include` | File patterns to include | Common code files |
| `--exclude` | File patterns to exclude | Tests, builds, etc. |
| `--max-size` | Max file size in bytes | 100000 |
| `--language` | Tutorial language | english |
| `--no-cache` | Disable LLM caching | Caching enabled |
| `--max-abstractions` | Max concepts to identify | 10 |

## File Structure

```
portable_tutorial_gen/
├── run.py              # Main entry point
├── flow.py             # PocketFlow workflow
├── nodes.py            # Processing nodes
├── utils/
│   ├── __init__.py
│   ├── call_llm.py     # LLM wrapper (OpenAI, Gemini, etc.)
│   ├── crawl_github_files.py  # GitHub repository crawler
│   └── crawl_local_files.py   # Local directory crawler
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment config
├── logs/               # LLM call logs (auto-created)
├── llm_cache.json      # Response cache (auto-created)
└── README.md           # This file
```

## Caching System

The tool includes an intelligent caching system to:
- **Save money**: Avoid redundant API calls for the same prompts
- **Speed up development**: Re-run on same codebase without waiting
- **Enable debugging**: Review previous LLM responses

### Cache Behavior
- **Location**: `portable_tutorial_gen/llm_cache.json`
- **Format**: JSON mapping prompts to responses
- **Automatic**: Cache is checked before every LLM call
- **Disable**: Use `--no-cache` flag to bypass

### Logs
- **Location**: `portable_tutorial_gen/logs/llm_calls_YYYYMMDD.log`
- **Content**: All LLM prompts and responses with timestamps
- **Useful for**: Debugging, auditing, prompt optimization

## LLM Provider Priority

The tool checks for API keys in this order:
1. `OPENAI_API_KEY` → OpenAI
2. `GEMINI_API_KEY` → Google Gemini
3. `OPENROUTER_API_KEY` → OpenRouter
4. `LLM_API_BASE_URL` → Generic (Ollama, local models)

## Local LLM Support (Ollama)

This tool supports **offline operation** using [Ollama](https://ollama.com). When you run the generator, it automatically detects if Ollama (or other local LLM servers) is running and prompts you to use it.

### Install Ollama

**macOS:**
```bash
brew install ollama
```

**Windows/Linux:**
Download from [ollama.com/download](https://ollama.com/download)

### Recommended Models by RAM

| RAM | Best Model | Size | Context | Command |
|-----|------------|------|---------|---------|
| **64GB+** | `qwen2.5-coder:32b` | 18GB | 128K | `ollama pull qwen2.5-coder:32b` |
| **32GB** | `qwen2.5-coder:7b` | 4.7GB | 128K | `ollama pull qwen2.5-coder:7b` |
| **16GB** | `qwen2.5-coder:3b` | 1.9GB | 32K | `ollama pull qwen2.5-coder:3b` |
| **8GB** | `qwen2.5-coder:1.5b` | 1.0GB | 32K | `ollama pull qwen2.5-coder:1.5b` |
| **4GB** | `tinyllama:latest` | 0.6GB | 2K | `ollama pull tinyllama:latest` |

### Model Recommendations by Use Case

| Use Case | Recommended Model | Notes |
|----------|-------------------|-------|
| **Large codebases (500+ files)** | `qwen2.5-coder:32b` or `qwen2.5-coder:14b` | Needs 64GB+ RAM |
| **Medium projects** | `qwen2.5-coder:7b` | Best balance of quality/speed |
| **Quick testing** | `llama3.2:latest` | General purpose, fast |
| **Code-focused analysis** | `qwen2.5-coder:*` | Trained on 5.5T code tokens |
| **Budget hardware** | `qwen2.5-coder:1.5b` | Works on 8GB RAM |

### Running with Ollama

1. Start Ollama (runs in background):
   ```bash
   ollama serve
   ```

2. Pull a model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

3. Run the generator - it will auto-detect Ollama:
   ```bash
   python run.py --dir /path/to/code
   ```

4. When prompted, select the local LLM option.

### Skip Local LLM Detection

To skip the local LLM prompt and use cloud providers directly:
```bash
python run.py --dir /path/to/code --skip-local-llm
```

Or set the environment variable:
```bash
export SKIP_LOCAL_LLM_DETECTION=1
```

## Troubleshooting

### "No LLM provider configured"
Make sure your `.env` file exists and contains a valid API key.

### Rate limits
Use `--no-cache` sparingly. The caching helps avoid repeated API calls.

### Large codebases
Use `--exclude` to skip unnecessary files and `--max-size` to limit file sizes.
