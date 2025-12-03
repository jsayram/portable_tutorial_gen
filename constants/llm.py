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
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro-exp-03-25"
DEFAULT_GEMINI_LOCATION = "us-central1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-3.5-turbo"
DEFAULT_GENERIC_MODEL = "llama2"
DEFAULT_GENERIC_BASE_URL = "http://localhost:11434"

# =============================================================================
# API URLs
# =============================================================================
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
DEFAULT_TEMPERATURE = 0.7  # Balanced creativity vs consistency

# =============================================================================
# APP METADATA (for OpenRouter tracking)
# =============================================================================
DEFAULT_OPENROUTER_REFERER = "https://github.com"
DEFAULT_OPENROUTER_TITLE = "PocketFlow Tutorial Generator"
