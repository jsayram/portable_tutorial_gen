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
from constants.defaults import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_ABSTRACTIONS,
)
from constants.paths import DEFAULT_OUTPUT_DIR
from constants.llm import ENV_SKIP_LOCAL_LLM_DETECTION


def check_for_local_llm() -> dict | None:
    """
    Check for running local LLM servers and prompt user if they want to use one.
    
    If Ollama is detected but not running, offers to start it.
    
    Returns:
        dict with 'url' and 'name' if user chooses a local LLM, None otherwise
    """
    # Skip if environment variable is set
    if os.getenv(ENV_SKIP_LOCAL_LLM_DETECTION, "").lower() in ("1", "true", "yes"):
        return None
    
    # Import detection and health check functions
    from utils.call_llm import detect_local_llms, is_ollama_running, ensure_ollama_running
    
    print("Checking for local LLM servers...")
    detected = detect_local_llms()
    
    if not detected:
        # Check if Ollama is installed but not running
        import shutil
        if shutil.which("ollama"):
            print("\n⚠️  Ollama is installed but not running.")
            try:
                start_choice = input("Would you like to start Ollama? [Y/n]: ").strip().lower()
                if start_choice in ("", "y", "yes"):
                    if ensure_ollama_running():
                        # Re-detect after starting
                        detected = detect_local_llms()
                    else:
                        print("Failed to start Ollama. Using cloud provider.")
                        return None
                else:
                    print("Using cloud provider.")
                    return None
            except KeyboardInterrupt:
                print("\nUsing cloud provider.")
                return None
        else:
            print("No local LLM servers detected.")
            return None
    
    if not detected:
        print("No local LLM servers detected.")
        return None
    
    # Display detected LLMs
    print(f"\n{'=' * 60}")
    print("🖥️  Local LLM Server(s) Detected!")
    print(f"{'=' * 60}")
    
    for i, llm in enumerate(detected, 1):
        models_str = ", ".join(llm["models"][:3]) if llm["models"] else "models available"
        if len(llm["models"]) > 3:
            models_str += f" (+{len(llm['models']) - 3} more)"
        print(f"  [{i}] {llm['name']} at {llm['url']}")
        print(f"      Models: {models_str}")
    
    print(f"  [0] Skip - Use configured cloud provider")
    print(f"{'=' * 60}")
    
    # Prompt user
    while True:
        try:
            choice = input("\nUse local LLM? Enter number (0 to skip): ").strip()
            if choice == "" or choice == "0":
                print("Using configured cloud provider.")
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(detected):
                selected = detected[idx]
                
                # Verify the selected LLM is still running
                if "ollama" in selected["name"].lower():
                    if not is_ollama_running(selected["url"]):
                        print(f"\n⚠️  {selected['name']} stopped responding.")
                        if not ensure_ollama_running(selected["url"]):
                            print("Failed to restart. Using cloud provider.")
                            return None
                
                # Use first available model if any
                model = selected["models"][0] if selected["models"] else None
                print(f"✓ Using {selected['name']} at {selected['url']}")
                if model:
                    print(f"  Model: {model}")
                return {"url": selected["url"], "name": selected["name"], "model": model}
            else:
                print(f"Invalid choice. Enter 0-{len(detected)}")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nUsing configured cloud provider.")
            return None


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
        default=DEFAULT_OUTPUT_DIR, 
        help=f"Output directory for the tutorial (default: ./{DEFAULT_OUTPUT_DIR})."
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
        default=DEFAULT_MAX_FILE_SIZE, 
        help=f"Maximum file size in bytes (default: {DEFAULT_MAX_FILE_SIZE}, about 100KB)."
    )
    parser.add_argument(
        "--language", 
        default=DEFAULT_LANGUAGE, 
        help=f"Language for the generated tutorial (default: {DEFAULT_LANGUAGE})."
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true", 
        help="Disable LLM response caching (default: caching enabled)."
    )
    parser.add_argument(
        "--max-abstractions", 
        type=int, 
        default=DEFAULT_MAX_ABSTRACTIONS, 
        help=f"Maximum number of abstractions to identify (default: {DEFAULT_MAX_ABSTRACTIONS})."
    )
    parser.add_argument(
        "--skip-local-llm", 
        action="store_true", 
        help="Skip local LLM detection and use configured cloud provider."
    )

    args = parser.parse_args()

    # Check for local LLM servers (unless skipped)
    local_llm = None
    if not args.skip_local_llm:
        local_llm = check_for_local_llm()
        if local_llm:
            from utils.call_llm import set_local_llm_override
            set_local_llm_override(local_llm["url"], local_llm["name"], local_llm.get("model"))

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
    if local_llm:
        print(f"LLM Provider: {local_llm['name']} (local) at {local_llm['url']}")
    else:
        from utils.call_llm import get_llm_provider
        try:
            provider = get_llm_provider()
            print(f"LLM Provider: {provider}")
        except ValueError as e:
            print(f"LLM Provider: Not configured - {e}")
            sys.exit(1)
    print(f"Output: {args.output}")
    print(f"=" * 60)

    # Create and run the flow with timing
    import time
    start_time = time.time()
    
    tutorial_flow = create_tutorial_flow()
    tutorial_flow.run(shared)

    # Calculate elapsed time
    elapsed = time.time() - start_time
    if elapsed >= 60:
        time_str = f"{elapsed/60:.1f} minutes"
    else:
        time_str = f"{elapsed:.1f} seconds"

    print(f"\n{'=' * 60}")
    print(f"✅ Tutorial generated successfully!")
    print(f"   Output: {shared['final_output_dir']}")
    print(f"   Time: {time_str}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
