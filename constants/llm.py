"""
================================================================================
LLM PROVIDER CONSTANTS
================================================================================
This file contains all constants related to LLM providers, API configuration,
and model defaults. This is the single source of truth for LLM configuration.

PROVIDER PRIORITY (checked in this order):
==========================================
1. OPENAI_API_KEY     → Uses OpenAI API
2. GEMINI_API_KEY     → Uses Google Gemini API
3. GEMINI_PROJECT_ID  → Uses Vertex AI (requires ADC setup)
4. OPENROUTER_API_KEY → Uses OpenRouter (access to many models)
5. LLM_API_BASE_URL   → Uses any OpenAI-compatible API (Ollama, etc.)
================================================================================
"""

# =============================================================================
# LLM PROVIDER NAMES
# =============================================================================
LLM_PROVIDER_OPENAI = "OPENAI"
LLM_PROVIDER_GEMINI = "GEMINI"
LLM_PROVIDER_OPENROUTER = "OPENROUTER"
LLM_PROVIDER_GENERIC = "GENERIC"
LLM_PROVIDER_OLLAMA = "OLLAMA"

# =============================================================================
# ENVIRONMENT VARIABLE NAMES
# =============================================================================
# OpenAI
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"

# Gemini / Vertex AI
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GEMINI_PROJECT_ID = "GEMINI_PROJECT_ID"
ENV_GEMINI_LOCATION = "GEMINI_LOCATION"
ENV_GEMINI_MODEL = "GEMINI_MODEL"

# OpenRouter
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
ENV_OPENROUTER_MODEL = "OPENROUTER_MODEL"
ENV_OPENROUTER_REFERER = "OPENROUTER_REFERER"
ENV_OPENROUTER_TITLE = "OPENROUTER_TITLE"

# Generic OpenAI-compatible API
ENV_LLM_API_BASE_URL = "LLM_API_BASE_URL"
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"

# Logging
ENV_LOG_DIR = "LOG_DIR"

# =============================================================================
# DEFAULT MODEL VALUES
# =============================================================================
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GEMINI_LOCATION = "us-central1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"
DEFAULT_GENERIC_MODEL = "llama3.2"
DEFAULT_GENERIC_BASE_URL = "http://localhost:11434"

# =============================================================================
# API URLs
# =============================================================================
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# =============================================================================
# LOCAL LLM DETECTION
# =============================================================================
# Common local LLM servers and their default ports
LOCAL_LLM_ENDPOINTS = [
    {"name": "Ollama", "url": "http://localhost:11434", "health_path": "/api/tags"},
    {"name": "LM Studio", "url": "http://localhost:1234", "health_path": "/v1/models"},
    {"name": "LocalAI", "url": "http://localhost:8080", "health_path": "/v1/models"},
    {"name": "Text Generation WebUI", "url": "http://localhost:5000", "health_path": "/v1/models"},
    {"name": "vLLM", "url": "http://localhost:8000", "health_path": "/v1/models"},
]

# Timeout for local LLM detection (in seconds)
LOCAL_LLM_DETECTION_TIMEOUT = 1.0

# Environment variable to skip local LLM detection
ENV_SKIP_LOCAL_LLM_DETECTION = "SKIP_LOCAL_LLM_DETECTION"

# =============================================================================
# OLLAMA RECOMMENDED MODELS
# =============================================================================
# Models optimized for code analysis with large context windows
# Ordered by capability (best first)

OLLAMA_MODELS_HIGH_RAM = [
    # For 64GB+ RAM (Mac M1 Max, high-end workstations)
    {
        "name": "qwen2.5-coder:32b",
        "size_gb": 18,
        "context": 131072,
        "ram_required": 30,
        "description": "Best code model - 32B params, 128K context",
    },
    {
        "name": "qwen2.5-coder:14b",
        "size_gb": 9,
        "context": 131072,
        "ram_required": 20,
        "description": "Excellent code model - 14B params, 128K context",
    },
    {
        "name": "deepseek-coder-v2:16b",
        "size_gb": 9,
        "context": 131072,
        "ram_required": 20,
        "description": "Great for code analysis - 16B params",
    },
]

OLLAMA_MODELS_MEDIUM_RAM = [
    # For 32GB RAM (Windows PCs, mid-range Macs)
    {
        "name": "qwen2.5-coder:7b",
        "size_gb": 4.7,
        "context": 131072,
        "ram_required": 12,
        "description": "Best for 32GB RAM - 7B params, 128K context",
    },
    {
        "name": "deepseek-coder:6.7b",
        "size_gb": 3.8,
        "context": 65536,
        "ram_required": 10,
        "description": "Lightweight code model - 6.7B params",
    },
    {
        "name": "codellama:7b",
        "size_gb": 3.8,
        "context": 100000,
        "ram_required": 10,
        "description": "Meta's code model - 7B params",
    },
]

OLLAMA_MODELS_LOW_RAM = [
    # For 16GB RAM
    {
        "name": "qwen2.5-coder:3b",
        "size_gb": 1.9,
        "context": 32768,
        "ram_required": 8,
        "description": "Small code model - 3B params, 32K context",
    },
    {
        "name": "llama3.2:latest",
        "size_gb": 2.0,
        "context": 131072,
        "ram_required": 6,
        "description": "General purpose - 3B params",
    },
    {
        "name": "phi3:mini",
        "size_gb": 2.2,
        "context": 131072,
        "ram_required": 6,
        "description": "Microsoft mini model - 3.8B params",
    },
]

OLLAMA_MODELS_SMALL_RAM = [
    # For 8GB RAM (budget laptops, older machines)
    {
        "name": "qwen2.5-coder:1.5b",
        "size_gb": 1.0,
        "context": 32768,
        "ram_required": 4,
        "description": "Tiny code model - 1.5B params",
    },
    {
        "name": "qwen2.5-coder:0.5b",
        "size_gb": 0.4,
        "context": 32768,
        "ram_required": 2,
        "description": "Ultra-tiny code model - 0.5B params",
    },
    {
        "name": "codegemma:2b",
        "size_gb": 1.4,
        "context": 8192,
        "ram_required": 4,
        "description": "Google's tiny code model - 2B params",
    },
]

OLLAMA_MODELS_XSMALL_RAM = [
    # For 4GB RAM or less (Raspberry Pi, very old machines)
    {
        "name": "tinyllama:latest",
        "size_gb": 0.6,
        "context": 2048,
        "ram_required": 2,
        "description": "Tiniest model - 1.1B params, limited context",
    },
    {
        "name": "phi:latest",
        "size_gb": 1.6,
        "context": 2048,
        "ram_required": 3,
        "description": "Microsoft Phi-1 - 1.3B params",
    },
    {
        "name": "stablelm2:1.6b",
        "size_gb": 1.0,
        "context": 4096,
        "ram_required": 3,
        "description": "Stability AI - 1.6B params",
    },
]

# All recommended models combined
OLLAMA_RECOMMENDED_MODELS = (
    OLLAMA_MODELS_HIGH_RAM + 
    OLLAMA_MODELS_MEDIUM_RAM + 
    OLLAMA_MODELS_LOW_RAM +
    OLLAMA_MODELS_SMALL_RAM +
    OLLAMA_MODELS_XSMALL_RAM
)

# =============================================================================
# OLLAMA CONTEXT CONFIGURATION
# =============================================================================
# Context length based on available system RAM
# Higher context = more RAM for KV cache (~1GB per 8K tokens)

OLLAMA_CONTEXT_BY_RAM = {
    2: 2048,       # 2GB RAM   → 2K context (Raspberry Pi, minimal)
    4: 4096,       # 4GB RAM   → 4K context (very old machines)
    8: 8192,       # 8GB RAM   → 8K context (budget laptops)
    16: 16384,     # 16GB RAM  → 16K context
    32: 32768,     # 32GB RAM  → 32K context (Windows PC)
    64: 65536,     # 64GB RAM  → 64K context (your Mac M1 Max)
    128: 131072,   # 128GB RAM → 128K context (full)
}

# Default context length for unknown RAM configurations
# 64K is optimal for 64GB RAM machines
OLLAMA_CONTEXT_LENGTH = 65536  # 64K tokens

# =============================================================================
# SYSTEM REQUIREMENTS
# =============================================================================
# Minimum requirements for processing large codebases (500+ files)

REQUIREMENTS_LARGE_CODEBASE = {
    "min_ram_gb": 32,
    "recommended_ram_gb": 64,
    "min_context_tokens": 32768,
    "recommended_context_tokens": 65536,
    "recommended_model_high_ram": "qwen2.5-coder:32b",  # For 64GB+
    "recommended_model_medium_ram": "qwen2.5-coder:7b",  # For 32GB
}

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
DEFAULT_TEMPERATURE = 0.7  # Balanced creativity vs consistency

# =============================================================================
# APP METADATA (for OpenRouter tracking)
# =============================================================================
DEFAULT_OPENROUTER_REFERER = "https://github.com"
DEFAULT_OPENROUTER_TITLE = "PocketFlow Tutorial Generator"
