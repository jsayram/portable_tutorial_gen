"""
Local Directory Crawler - Cross-platform compatible (Windows, macOS, Linux)
"""

import os
import fnmatch
from pathlib import Path

# Try to import pathspec for .gitignore support
try:
    import pathspec
    HAS_PATHSPEC = True
except ImportError:
    HAS_PATHSPEC = False


def crawl_local_files(
    directory,
    include_patterns=None,
    exclude_patterns=None,
    max_file_size=None,
    use_relative_paths=True,
):
    """
    Crawl files in a local directory with cross-platform support.
    
    Args:
        directory (str): Path to local directory
        include_patterns (set): File patterns to include (e.g. {"*.py", "*.js"})
        exclude_patterns (set): File patterns to exclude (e.g. {"tests/*"})
        max_file_size (int): Maximum file size in bytes
        use_relative_paths (bool): Whether to use paths relative to directory

    Returns:
        dict: {"files": {filepath: content}}
    """
    # Use Path for cross-platform compatibility
    directory = Path(directory).resolve()
    
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    files_dict = {}

    # --- Load .gitignore ---
    gitignore_path = directory / ".gitignore"
    gitignore_spec = None
    if HAS_PATHSPEC and gitignore_path.exists():
        try:
            with open(gitignore_path, "r", encoding="utf-8-sig") as f:
                gitignore_patterns = f.readlines()
            gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_patterns)
            print(f"Loaded .gitignore patterns from {gitignore_path}")
        except Exception as e:
            print(f"Warning: Could not read or parse .gitignore file {gitignore_path}: {e}")

    all_files = []
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        
        # Filter directories using .gitignore and exclude_patterns early
        excluded_dirs = set()
        for d in dirs:
            dirpath = root_path / d
            dirpath_rel = str(dirpath.relative_to(directory))
            # Normalize path separators for cross-platform
            dirpath_rel = dirpath_rel.replace(os.sep, '/')

            if gitignore_spec and gitignore_spec.match_file(dirpath_rel + '/'):
                excluded_dirs.add(d)
                continue

            if exclude_patterns:
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(dirpath_rel, pattern) or fnmatch.fnmatch(d, pattern):
                        excluded_dirs.add(d)
                        break

        for d in list(dirs):
            if d in excluded_dirs:
                dirs.remove(d)

        for filename in files:
            filepath = root_path / filename
            all_files.append(filepath)

    total_files = len(all_files)
    processed_files = 0

    for filepath in all_files:
        if use_relative_paths:
            relpath = str(filepath.relative_to(directory))
        else:
            relpath = str(filepath)
        
        # Normalize path separators for consistent output
        relpath_normalized = relpath.replace(os.sep, '/')

        # --- Exclusion check ---
        excluded = False
        if gitignore_spec and gitignore_spec.match_file(relpath_normalized):
            excluded = True

        if not excluded and exclude_patterns:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(relpath_normalized, pattern) or fnmatch.fnmatch(relpath, pattern):
                    excluded = True
                    break

        included = False
        if include_patterns:
            for pattern in include_patterns:
                if fnmatch.fnmatch(relpath_normalized, pattern) or fnmatch.fnmatch(filepath.name, pattern):
                    included = True
                    break
        else:
            included = True

        processed_files += 1

        status = "processed"
        if not included or excluded:
            status = "skipped (excluded)"
            if total_files > 0:
                percentage = int((processed_files / total_files) * 100)
                print(f"\033[92mProgress: {processed_files}/{total_files} ({percentage}%) {relpath} [{status}]\033[0m")
            continue

        if max_file_size and filepath.stat().st_size > max_file_size:
            status = "skipped (size limit)"
            if total_files > 0:
                percentage = int((processed_files / total_files) * 100)
                print(f"\033[92mProgress: {processed_files}/{total_files} ({percentage}%) {relpath} [{status}]\033[0m")
            continue

        # --- File is being processed ---        
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                content = f.read()
            # Use normalized path as key for consistency
            files_dict[relpath_normalized] = content
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}")
            status = "skipped (read error)"

        if total_files > 0:
            percentage = int((processed_files / total_files) * 100)
            print(f"\033[92mProgress: {processed_files}/{total_files} ({percentage}%) {relpath} [{status}]\033[0m")

    return {"files": files_dict}


if __name__ == "__main__":
    import sys
    
    # Test with current directory or provided path
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"--- Crawling directory: {test_dir} ---")
    files_data = crawl_local_files(
        test_dir,
        include_patterns={"*.py", "*.md"},
        exclude_patterns={
            "*.pyc",
            "__pycache__/*",
            ".venv/*",
            ".git/*",
        },
    )
    print(f"\nFound {len(files_data['files'])} files:")
    for path in files_data["files"]:
        print(f"  {path}")
