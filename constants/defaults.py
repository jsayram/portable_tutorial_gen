"""
================================================================================
DEFAULT VALUES CONSTANTS
================================================================================
This file contains all default values for command-line arguments and file patterns.
This is the single source of truth for application defaults.
================================================================================
"""

# =============================================================================
# DEFAULT ARGUMENT VALUES
# =============================================================================
DEFAULT_MAX_FILE_SIZE = 100000  # Maximum file size in bytes (about 100KB)
DEFAULT_LANGUAGE = "english"    # Default tutorial language
DEFAULT_MAX_ABSTRACTIONS = 10   # Maximum number of abstractions to identify

# =============================================================================
# FILE PATTERNS
# =============================================================================
DEFAULT_INCLUDE_PATTERNS = {
    "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java", "*.pyi", "*.pyx",
    "*.c", "*.cs", "*.cc", "*.cpp", "*.h", "*.md", "*.rst", "Dockerfile",
    "Makefile", "*.yaml", "*.yml",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "assets/*", "data/*", "images/*", "public/*", "static/*", "temp/*",
    "*docs/*",
    "*venv/*",
    "*.venv/*",
    "*test*",
    "*tests/*",
    "*examples/*",
    "v1/*",
    "*dist/*",
    "*build/*",
    "*experimental/*",
    "*deprecated/*",
    "*misc/*",
    "*legacy/*",
    ".git/*", ".github/*", ".next/*", ".vscode/*",
    "*obj/*",
    "*bin/*",
    "*node_modules/*",
    "*.log"
}
