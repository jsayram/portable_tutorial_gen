"""
Portable Tutorial Generator - Constants Package

This package contains all configuration constants, magic numbers,
and default values used throughout the application.
"""

from .llm import (
    # LLM Provider Names
    LLM_PROVIDER_OPENAI,
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_OPENROUTER,
    LLM_PROVIDER_GENERIC,
    
    # Environment Variable Names
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_MODEL,
    ENV_GEMINI_API_KEY,
    ENV_GEMINI_PROJECT_ID,
    ENV_GEMINI_LOCATION,
    ENV_GEMINI_MODEL,
    ENV_OPENROUTER_API_KEY,
    ENV_OPENROUTER_MODEL,
    ENV_OPENROUTER_REFERER,
    ENV_OPENROUTER_TITLE,
    ENV_LLM_API_BASE_URL,
    ENV_LLM_API_KEY,
    ENV_LLM_MODEL,
    ENV_LOG_DIR,
    
    # Default Model Values
    DEFAULT_OPENAI_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_LOCATION,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_GENERIC_MODEL,
    DEFAULT_GENERIC_BASE_URL,
    
    # API URLs
    OPENROUTER_API_URL,
    
    # LLM Configuration
    DEFAULT_TEMPERATURE,
    
    # App Metadata
    DEFAULT_OPENROUTER_REFERER,
    DEFAULT_OPENROUTER_TITLE,
)

from .paths import (
    # Directory Names
    LOGS_DIR_NAME,
    CACHE_FILE_NAME,
    DEFAULT_OUTPUT_DIR,
    
    # Log File Format
    LOG_FILE_PREFIX,
    LOG_DATE_FORMAT,
)

from .defaults import (
    # Default Argument Values
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_ABSTRACTIONS,
    
    # File Patterns
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
)

__all__ = [
    # LLM Provider Names
    'LLM_PROVIDER_OPENAI',
    'LLM_PROVIDER_GEMINI',
    'LLM_PROVIDER_OPENROUTER',
    'LLM_PROVIDER_GENERIC',
    
    # Environment Variable Names
    'ENV_OPENAI_API_KEY',
    'ENV_OPENAI_MODEL',
    'ENV_GEMINI_API_KEY',
    'ENV_GEMINI_PROJECT_ID',
    'ENV_GEMINI_LOCATION',
    'ENV_GEMINI_MODEL',
    'ENV_OPENROUTER_API_KEY',
    'ENV_OPENROUTER_MODEL',
    'ENV_OPENROUTER_REFERER',
    'ENV_OPENROUTER_TITLE',
    'ENV_LLM_API_BASE_URL',
    'ENV_LLM_API_KEY',
    'ENV_LLM_MODEL',
    'ENV_LOG_DIR',
    
    # Default Model Values
    'DEFAULT_OPENAI_MODEL',
    'DEFAULT_GEMINI_MODEL',
    'DEFAULT_GEMINI_LOCATION',
    'DEFAULT_OPENROUTER_MODEL',
    'DEFAULT_GENERIC_MODEL',
    'DEFAULT_GENERIC_BASE_URL',
    
    # API URLs
    'OPENROUTER_API_URL',
    
    # LLM Configuration
    'DEFAULT_TEMPERATURE',
    
    # App Metadata
    'DEFAULT_OPENROUTER_REFERER',
    'DEFAULT_OPENROUTER_TITLE',
    
    # Directory Names
    'LOGS_DIR_NAME',
    'CACHE_FILE_NAME',
    'DEFAULT_OUTPUT_DIR',
    
    # Log File Format
    'LOG_FILE_PREFIX',
    'LOG_DATE_FORMAT',
    
    # Default Argument Values
    'DEFAULT_MAX_FILE_SIZE',
    'DEFAULT_LANGUAGE',
    'DEFAULT_MAX_ABSTRACTIONS',
    
    # File Patterns
    'DEFAULT_INCLUDE_PATTERNS',
    'DEFAULT_EXCLUDE_PATTERNS',
]
