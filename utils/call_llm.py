"""
================================================================================
PORTABLE TUTORIAL GENERATOR - LLM WRAPPER
================================================================================
This is the BRAIN of the tutorial generator - the LLM calling interface.
It's an EXACT COPY of the main utils/call_llm.py with added comments.

PROVIDER PRIORITY (checked in this order):
==========================================
1. OPENAI_API_KEY     → Uses OpenAI API (gpt-3.5-turbo by default)
2. GEMINI_API_KEY     → Uses Google Gemini API
3. GEMINI_PROJECT_ID  → Uses Vertex AI (requires ADC setup)
4. OPENROUTER_API_KEY → Uses OpenRouter (access to many models)
5. LLM_API_BASE_URL   → Uses any OpenAI-compatible API (Ollama, etc.)

CACHING:
========
Responses are cached to llm_cache.json to avoid redundant API calls.
This saves money and time when re-running on the same codebase.
Use --no-cache flag to disable caching.

LOGGING:
========
All prompts and responses are logged to logs/llm_calls_YYYYMMDD.log
This is essential for debugging and understanding LLM behavior.

IMPORTANT: This file must stay in sync with the main project's call_llm.py!
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import logging
import json
import requests
import sys
from datetime import datetime

# Try to load dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv is optional - env vars can be set manually

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Get the portable package root directory (parent of utils/)
# This ensures logs and cache stay in the portable directory
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory relative to package root
log_directory = os.getenv("LOG_DIR", os.path.join(_PACKAGE_DIR, "logs"))
os.makedirs(log_directory, exist_ok=True)

# Create log file with today's date
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger - we use a named logger to avoid conflicts
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger (avoids duplicate logs)

# Only add handler if not already present (prevents duplicate handlers on reimport)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
# Cache file stores prompt -> response mappings as JSON
# Located in the portable package directory (not current working directory)
cache_file = os.path.join(_PACKAGE_DIR, "llm_cache.json")


def load_cache() -> dict:
    """
    Load the LLM response cache from disk.
    
    The cache is a simple JSON file mapping prompts to their responses.
    This avoids redundant API calls for the same prompts.
    
    Returns:
        dict: The cache dictionary, or empty dict if cache doesn't exist
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return {}


def save_cache(cache: dict) -> None:
    """
    Save the LLM response cache to disk.
    
    Args:
        cache: The cache dictionary to save
    """
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


# =============================================================================
# PROVIDER DETECTION
# =============================================================================
def get_llm_provider() -> str:
    """
    Determine which LLM provider to use based on environment variables.
    
    This function checks for API keys in a specific priority order.
    The FIRST provider with a valid key will be used.
    
    PRIORITY ORDER:
    1. OPENAI_API_KEY     → "OPENAI"  (Most common, recommended)
    2. GEMINI_API_KEY     → "GEMINI"  (Google's Gemini)
    3. GEMINI_PROJECT_ID  → "GEMINI"  (Vertex AI - requires ADC)
    4. OPENROUTER_API_KEY → "OPENROUTER" (Access to many models)
    5. LLM_API_BASE_URL   → "GENERIC" (Ollama, local models, etc.)
    
    Returns:
        str: The provider name ("OPENAI", "GEMINI", "OPENROUTER", or "GENERIC")
        
    Raises:
        ValueError: If no provider is configured
    """
    # Check OpenAI first (most common choice)
    if os.getenv("OPENAI_API_KEY"):
        return "OPENAI"
    # Check Gemini next (API key or Vertex AI project)
    elif os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_PROJECT_ID"):
        return "GEMINI"
    # Check OpenRouter (multi-model gateway)
    elif os.getenv("OPENROUTER_API_KEY"):
        return "OPENROUTER"
    # Check for generic OpenAI-compatible API (Ollama, etc.)
    elif os.getenv("LLM_API_BASE_URL"):
        return "GENERIC"
    else:
        raise ValueError(
            "No LLM provider configured. Set one of: "
            "OPENAI_API_KEY, GEMINI_API_KEY, GEMINI_PROJECT_ID, "
            "OPENROUTER_API_KEY, or LLM_API_BASE_URL"
        )


# =============================================================================
# MAIN LLM CALLING FUNCTION
# =============================================================================
def call_llm(prompt: str, use_cache: bool = True) -> str:
    """
    Main LLM calling function that routes to the appropriate provider.
    
    This is the primary interface for all LLM calls in the application.
    It handles:
    1. Logging the prompt
    2. Checking the cache (if enabled)
    3. Routing to the correct provider
    4. Logging the response
    5. Updating the cache (if enabled)
    
    Args:
        prompt: The prompt to send to the LLM
        use_cache: Whether to use caching (default: True)
                   Set to False when retrying to get fresh response
        
    Returns:
        str: The LLM response text
        
    Raises:
        Various exceptions depending on the provider
    """
    # Log the prompt for debugging
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled - avoids redundant API calls
    if use_cache:
        cache = load_cache()
        if prompt in cache:
            logger.info("CACHE HIT: Using cached response")
            return cache[prompt]

    # Get the configured provider and route to appropriate function
    provider = get_llm_provider()
    
    # Route to the correct provider-specific function
    # IMPORTANT: This order must match the priority in get_llm_provider()!
    if provider == "OPENAI":
        response_text = _call_llm_openai(prompt)
    elif provider == "GEMINI":
        response_text = _call_llm_gemini(prompt)
    elif provider == "OPENROUTER":
        response_text = _call_llm_openrouter(prompt)
    else:  # GENERIC - OpenAI-compatible API
        response_text = _call_llm_generic(prompt)

    # Log the response for debugging
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        cache = load_cache()
        cache[prompt] = response_text
        save_cache(cache)

    return response_text


# =============================================================================
# PROVIDER-SPECIFIC IMPLEMENTATIONS
# =============================================================================

def _call_llm_openai(prompt: str) -> str:
    """
    Call OpenAI API directly using the official SDK.
    
    Environment variables:
    - OPENAI_API_KEY: Required - your OpenAI API key
    - OPENAI_MODEL: Optional - model to use (default: gpt-3.5-turbo)
    
    Args:
        prompt: The prompt to send
        
    Returns:
        str: The response text
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Use environment variable for model, with sensible default
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Create client and make the API call
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # Balanced creativity vs consistency
    )
    
    return response.choices[0].message.content


def _call_llm_gemini(prompt: str) -> str:
    """
    Call Google Gemini API.
    
    Supports two modes:
    1. API Key mode (GEMINI_API_KEY) - Simpler, recommended
    2. Vertex AI mode (GEMINI_PROJECT_ID) - Requires ADC setup
    
    Environment variables:
    - GEMINI_API_KEY: API key for Gemini (preferred)
    - GEMINI_PROJECT_ID: For Vertex AI (requires gcloud auth)
    - GEMINI_LOCATION: Vertex AI location (default: us-central1)
    - GEMINI_MODEL: Model to use (default: gemini-2.5-pro-exp-03-25)
    
    IMPORTANT: API key is checked FIRST to avoid Vertex AI ADC issues!
    
    Args:
        prompt: The prompt to send
        
    Returns:
        str: The response text
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError("Google GenAI package not installed. Run: pip install google-genai")
    
    # IMPORTANT: Check API key FIRST (simpler, no ADC required)
    # This avoids the "DefaultCredentialsError" when using API key
    if os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    elif os.getenv("GEMINI_PROJECT_ID"):
        # Vertex AI mode - requires Application Default Credentials
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID"),
            location=os.getenv("GEMINI_LOCATION", "us-central1")
        )
    else:
        raise ValueError("Either GEMINI_API_KEY or GEMINI_PROJECT_ID must be set")
    
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    response = client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    return response.text


def _call_llm_openrouter(prompt: str) -> str:
    """
    Call OpenRouter API - a gateway to many LLM providers.
    
    OpenRouter provides access to models from OpenAI, Anthropic, Google,
    Meta, and many others through a single API.
    
    Environment variables:
    - OPENROUTER_API_KEY: Required - your OpenRouter API key
    - OPENROUTER_MODEL: Model to use (default: openai/gpt-3.5-turbo)
    - OPENROUTER_REFERER: HTTP referer for tracking (default: https://github.com)
    - OPENROUTER_TITLE: App title for tracking
    
    Args:
        prompt: The prompt to send
        
    Returns:
        str: The response text
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    # OpenRouter requires specific headers for tracking
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "PocketFlow Tutorial Generator")
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    
    response = requests.post(base_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_llm_generic(prompt: str) -> str:
    """
    Call a generic OpenAI-compatible API.
    
    This works with:
    - Ollama (local LLMs)
    - LM Studio
    - vLLM
    - Any other OpenAI-compatible server
    
    Environment variables:
    - LLM_API_BASE_URL: The base URL (default: http://localhost:11434)
    - LLM_API_KEY: Optional API key (not needed for local models)
    - LLM_MODEL: Model to use (default: llama2)
    
    Args:
        prompt: The prompt to send
        
    Returns:
        str: The response text
    """
    base_url = os.getenv("LLM_API_BASE_URL", "http://localhost:11434")
    api_key = os.getenv("LLM_API_KEY", "")  # Optional for local models
    model = os.getenv("LLM_MODEL", "llama2")
    
    # OpenAI-compatible endpoint
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling LLM API at {url}: {e}")


# =============================================================================
# TEST SCRIPT
# =============================================================================
if __name__ == "__main__":
    """
    Test the LLM configuration.
    
    Run this file directly to verify your API key is working:
        python utils/call_llm.py
    """
    try:
        provider = get_llm_provider()
        print(f"Using LLM provider: {provider}")
        
        test_prompt = "Say hello in one sentence."
        print(f"Testing with prompt: {test_prompt}")
        
        response = call_llm(test_prompt, use_cache=False)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
