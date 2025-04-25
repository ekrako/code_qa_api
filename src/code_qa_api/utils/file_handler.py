import fnmatch
from pathlib import Path
from typing import Generator

# Default patterns to ignore (can be expanded)
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
]


def find_python_files(start_path: Path, ignore_patterns: list[str] = DEFAULT_IGNORE_PATTERNS) -> Generator[Path, None, None]:
    for item in start_path.rglob("*"):
        if any(fnmatch.fnmatch(str(item), pattern) or pattern in item.parts for pattern in ignore_patterns):
            continue
        if item.is_file() and item.suffix == ".py":
            yield item


def find_markdown_files(start_path: Path, ignore_patterns: list[str] = DEFAULT_IGNORE_PATTERNS) -> Generator[Path, None, None]:
    for item in start_path.rglob("*"):
        if any(fnmatch.fnmatch(str(item), pattern) or pattern in item.parts for pattern in ignore_patterns):
            continue
        if item.is_file() and item.suffix.lower() == ".md":
            yield item


def read_file_content(file_path: Path) -> str | None:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Warning: Could not decode {file_path} as UTF-8. Trying with errors ignored.")
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e_inner:
            print(f"Error reading {file_path} even with errors ignored: {e_inner}")
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
