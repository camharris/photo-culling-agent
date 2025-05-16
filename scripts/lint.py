#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linting script for the photo-culling-agent project.
This script runs various linting and code formatting tools.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Define the project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return whether it succeeded."""
    print(f"\n\033[1;34m>>> {description}...\033[0m")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main() -> int:
    """Run linting tools and return exit code."""
    parser = argparse.ArgumentParser(description="Run code linting tools")
    parser.add_argument("--check", action="store_true", help="Check only, don't modify files")
    parser.add_argument("--fix", action="store_true", help="Fix issues automatically")
    parser.add_argument("--path", default="src", help="Path to check (default: src)")

    args = parser.parse_args()
    path = args.path
    check_only = args.check
    fix = args.fix

    # Default to fix mode if neither --check nor --fix is specified
    if not check_only and not fix:
        fix = True

    success = True

    # Remove unused imports with autoflake
    if fix:
        autoflake_cmd = [
            "autoflake",
            "--recursive",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--in-place",
            path,
        ]
        success = run_command(autoflake_cmd, "Removing unused imports and variables") and success

    # Sort imports with isort
    isort_cmd = ["isort", path]
    if check_only:
        isort_cmd.append("--check-only")
    success = run_command(isort_cmd, "Sorting imports") and success

    # Format with black
    black_cmd = ["black", path]
    if check_only:
        black_cmd.append("--check")
    success = run_command(black_cmd, "Formatting code with Black") and success

    # Check with flake8
    flake8_cmd = ["flake8", path]
    success = run_command(flake8_cmd, "Checking code with flake8") and success

    if success:
        print("\n\033[1;32m>>> All linting checks passed!\033[0m")
        return 0
    else:
        print("\n\033[1;31m>>> Some linting checks failed.\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
