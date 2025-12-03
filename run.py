#!/usr/bin/env python3
"""
Portable Tutorial Generator - Main Entry Point
Cross-platform compatible (Windows, macOS, Linux)

Usage:
    python run.py --dir /path/to/code
    python run.py --dir . --output ./tutorials
"""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")

from flow import create_tutorial_flow

# Default file patterns
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a beginner-friendly tutorial from a local codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --dir /path/to/project
  python run.py --dir . --output ./tutorials
  python run.py --dir ./src --include "*.py" --exclude "*test*"
  python run.py --dir . --language spanish
        """
    )

    parser.add_argument(
        "--dir", 
        required=True, 
        help="Path to local directory to analyze."
    )
    parser.add_argument(
        "-n", "--name", 
        help="Project name (optional, derived from directory if omitted)."
    )
    parser.add_argument(
        "-o", "--output", 
        default="output", 
        help="Output directory for the tutorial (default: ./output)."
    )
    parser.add_argument(
        "-i", "--include", 
        nargs="+", 
        help="Include file patterns (e.g., '*.py' '*.js'). Defaults to common code files."
    )
    parser.add_argument(
        "-e", "--exclude", 
        nargs="+", 
        help="Exclude file patterns (e.g., 'tests/*' 'docs/*'). Defaults to test/build directories."
    )
    parser.add_argument(
        "-s", "--max-size", 
        type=int, 
        default=100000, 
        help="Maximum file size in bytes (default: 100000, about 100KB)."
    )
    parser.add_argument(
        "--language", 
        default="english", 
        help="Language for the generated tutorial (default: english)."
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true", 
        help="Disable LLM response caching (default: caching enabled)."
    )
    parser.add_argument(
        "--max-abstractions", 
        type=int, 
        default=10, 
        help="Maximum number of abstractions to identify (default: 10)."
    )

    args = parser.parse_args()

    # Validate directory exists
    dir_path = Path(args.dir).resolve()
    if not dir_path.is_dir():
        print(f"Error: Directory does not exist: {args.dir}")
        sys.exit(1)

    # Initialize the shared dictionary
    shared = {
        "local_dir": str(dir_path),
        "project_name": args.name,
        "output_dir": args.output,
        "include_patterns": set(args.include) if args.include else DEFAULT_INCLUDE_PATTERNS,
        "exclude_patterns": set(args.exclude) if args.exclude else DEFAULT_EXCLUDE_PATTERNS,
        "max_file_size": args.max_size,
        "language": args.language,
        "use_cache": not args.no_cache,
        "max_abstraction_num": args.max_abstractions,
        # Outputs will be populated by the nodes
        "files": [],
        "abstractions": [],
        "relationships": {},
        "chapter_order": [],
        "chapters": [],
        "final_output_dir": None
    }

    # Display starting message
    print(f"=" * 60)
    print(f"Portable Tutorial Generator")
    print(f"=" * 60)
    print(f"Directory: {dir_path}")
    print(f"Language: {args.language.capitalize()}")
    print(f"LLM Caching: {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"Output: {args.output}")
    print(f"=" * 60)

    # Create and run the flow
    tutorial_flow = create_tutorial_flow()
    tutorial_flow.run(shared)

    print(f"\n{'=' * 60}")
    print(f"Tutorial generated successfully!")
    print(f"Output directory: {shared['final_output_dir']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
