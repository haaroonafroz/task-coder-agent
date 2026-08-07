<!-- SKILL_START: read_file -->
## Skill Name: read_file
- **Description:** Reads the raw string content of an individual file in the workspace. Always call this before writing or patching a file to understand its current state.
- **Keywords:** read, file, content, inspect
- **Parameters:**
  - `file_path` (string): Workspace-relative path of the file to read (e.g. `src/utils.py`).
- **Returns:** String content of the file or an error message if the file does not exist.
- **When to use:** Before any write or patch operation; when validating that a file was correctly written; when debugging unexpected behaviour.
- **Example:**
  ```json
  {"tool": "read_file", "args": {"file_path": "src/utils.py"}, "reasoning": "Need to check existing imports before adding one."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: write_file -->
## Skill Name: write_file
- **Description:** Creates a completely new file or replaces the entire content of an existing file. Use only for new files or complete rewrites — prefer `patch_file` for targeted edits. Full rewrites of existing files larger than ~60 lines are rejected unless the file was read first this milestone or `rewrite` is true.
- **Keywords:** write, create, file, new
- **Parameters:**
  - `file_path` (string): Workspace-relative destination path. Must be one of the milestone's target files — other paths are rejected.
  - `content` (string): Full file content to write.
  - `rewrite` (boolean, optional): Set true to force a full rewrite of a large existing file without a prior read. Escape hatch — prefer read_file + patch_file.
- **Returns:** Confirmation message with the written byte count.
- **When to use:** Creating new Python modules, config files, or test files from scratch.
- **Example:**
  ```json
  {"tool": "write_file", "args": {"file_path": "src/math_ops.py", "content": "def add(a, b):\n    return a + b\n"}, "reasoning": "Creating the new math_ops module."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: patch_file -->
## Skill Name: patch_file
- **Description:** Performs a precise search-and-replace inside an existing file without touching surrounding code. Safer than write_file for incremental changes.
- **Keywords:** patch, edit, replace, modify
- **Parameters:**
  - `file_path` (string): Workspace-relative path to the file. Must be one of the milestone's target files — other paths are rejected.
  - `search_string` (string): Exact text to find (must be unique in the file).
  - `replace_string` (string): Text to substitute in place of `search_string`.
- **Returns:** Success confirmation or error if `search_string` was not found.
- **When to use:** Adding an import, fixing a single function, correcting a syntax error without rewriting the whole file.
- **Example:**
  ```json
  {"tool": "patch_file", "args": {"file_path": "src/app.py", "search_string": "import os", "replace_string": "import os\nimport sys"}, "reasoning": "Adding missing sys import."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: list_directory -->
## Skill Name: list_directory
- **Description:** Recursively maps out and lists all files and subdirectories inside a given project path. Returns an indented tree representation.
- **Keywords:** list, directory, tree, structure
- **Parameters:**
  - `target_dir` (string): Directory path to inspect (e.g. `.` or `src/`).
- **Returns:** Indented directory tree as a string.
- **When to use:** Orienting yourself at the start of a milestone; verifying that a newly created file appears in the correct location.
- **Example:**
  ```json
  {"tool": "list_directory", "args": {"target_dir": "."}, "reasoning": "Understanding current project layout before writing new files."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: search_grep -->
## Skill Name: search_grep
- **Description:** Performs a regex-based keyword search across all files in the workspace and returns matching lines with their file paths and line numbers.
- **Keywords:** search, grep, regex, find
- **Parameters:**
  - `query` (string): Python regex pattern to search for.
  - `target_dir` (string, optional): Restrict search to this directory (default: `.`).
- **Returns:** Matching lines grouped by file path and line number.
- **When to use:** Finding where a function or class is defined; locating all usages of a variable; checking if an import already exists.
- **Example:**
  ```json
  {"tool": "search_grep", "args": {"query": "def validate_input", "target_dir": "/src"}, "reasoning": "Checking if validate_input already exists before creating it."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: run_pytest -->
## Skill Name: run_pytest
- **Description:** Runs the Python test suite (pytest) on a targeted test file or directory. Captures stdout, stderr, and exit code.
- **Keywords:** test, pytest, assert, validation
- **Parameters:**
  - `test_path` (string): Path to a test file or directory (e.g. `/tests/test_math.py`).
  - `extra_args` (string, optional): Additional pytest flags (e.g. `-v --tb=short`).
- **Returns:** Dict with `stdout`, `stderr`, `returncode`, and `passed` (bool).
- **When to use:** Verifying that an implementation passes its test contract; debugging failing tests after a code change.
- **Example:**
  ```json
  {"tool": "run_pytest", "args": {"test_path": "tests/test_math.py", "extra_args": "-v"}, "reasoning": "Running milestone validation tests."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: run_linter -->
## Skill Name: run_linter
- **Description:** Executes flake8 or black --check on a target file or directory to detect syntax and style issues.
- **Keywords:** lint, flake8, style, quality
- **Parameters:**
  - `target_path` (string): File or directory to lint.
  - `tool` (string, optional): `"flake8"` (default) or `"black"`.
  - `extra_args` (string, optional): Additional linter flags (e.g. `--max-line-length=120`).
- **Returns:** Dict with `stdout`, `stderr`, `returncode`, and `clean` (bool — True if no issues).
- **When to use:** Before finalising any Python file; as part of the validation contract execution.
- **Example:**
  ```json
  {"tool": "run_linter", "args": {"target_path": "/src/utils.py", "tool": "flake8", "extra_args": "--max-line-length=120"}, "reasoning": "Checking PEP8 compliance before signalling completion."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: git_commit -->
## Skill Name: git_commit
- **Description:** Stages all modified workspace files and creates a git commit with the supplied message. Called automatically by the runtime upon a PASS verdict — workers should not call this directly.
- **Keywords:** commit, save, checkpoint, git
- **Parameters:**
  - `message` (string): Conventional-commit-style message (e.g. `feat(M2): implement input validation`).
- **Returns:** Confirmation with the new commit hash.
- **When to use:** Only called by the runtime pipeline after a validator PASS; never call manually during implementation.
- **Example:**
  ```json
  {"tool": "git_commit", "args": {"message": "feat(M1): scaffold project structure"}, "reasoning": "Saving validated milestone as a permanent checkpoint."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: git_diff -->
## Skill Name: git_diff
- **Description:** Shows the pending diff between the workspace and the last committed HEAD, helping both the worker and validator understand what changed.
- **Keywords:** diff, changes, git, review
- **Parameters:** None
- **Returns:** Unified diff string of all uncommitted changes.
- **When to use:** Before signalling completion to verify that only intended changes were made; during adversarial validation to inspect exact edits.
- **Example:**
  ```json
  {"tool": "git_diff", "args": {}, "reasoning": "Reviewing all changes before completion signal."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: install_dependency -->
## Skill Name: install_dependency
- **Description:** Installs a Python package into the session venv via pip. Also appends the package to `requirements.txt` if it exists. **Required before complete** when target files import third-party libraries.
- **Keywords:** install, pip, package, dependency, pygame, flask, httpx, requirements, third-party, runtime
- **Parameters:**
  - `package_name` (string): Package identifier, optionally with version pin (e.g. `httpx>=0.27.0`, `pygame`).
- **Returns:** pip install stdout/stderr and success flag.
- **When to use:** Before signalling `complete` whenever you add a non-stdlib third-party import (e.g. `import pygame`) in a target file. The session venv only includes pytest, flake8, and black by default.
- **Example:**
  ```json
  {"tool": "install_dependency", "args": {"package_name": "pygame"}, "reasoning": "main.py imports pygame; install before complete."}
  ```
  ```json
  {"tool": "install_dependency", "args": {"package_name": "httpx>=0.27.0"}, "reasoning": "Need httpx for async HTTP client in the new module."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: uninstall_dependency -->
## Skill Name: uninstall_dependency
- **Description:** Uninstalls a Python package via pip and removes it from `requirements.txt` if present. Cleans both the Python environment and the dependency list, fully removing all matching lines by base name (e.g., "httpx" will remove "httpx==...", "httpx>=...", etc.).
- **Keywords:** uninstall, pip, package, dependency, remove, requirements.txt
- **Parameters:**
  - `package_name` (string): Package name with optional version specifier (e.g. `"httpx>=0.27.0"`).
- **Returns:** Dict containing `success` (bool), `stdout` (string), and `stderr` (string), indicating removal status and output.
- **When to use:** When a package should be removed from both the runtime and the workspace's requirements manifest.
- **Example:**
  ```json
  {"tool": "uninstall_dependency", "args": {"package_name": "httpx"}, "reasoning": "The httpx package is now unused and should be fully removed from the environment and requirements.txt."}
  ```
<!-- SKILL_END -->

<!-- SKILL_START: run_shellscript -->
## Skill Name: run_shellscript
- **Description:** Executes an arbitrary shell script or command inside the workspace directory with a configurable timeout. Do not run `cd workspace`; tool calls already start at the workspace root. Useful for build steps, data generation, or environment checks that don't map to a dedicated tool.
- **Keywords:** shell, bash, script, execute
- **Parameters:**
  - `script` (string): The shell command or multi-line bash script to run.
  - `timeout` (integer, optional): Maximum execution time in seconds (default: 30).
- **Returns:** Dict with `stdout`, `stderr`, `returncode`, and `timed_out` (bool).
- **When to use:** Running build scripts, initialising a database schema, generating fixtures, or any bespoke validation step not covered by run_pytest or run_linter.
- **Example:**
  ```json
  {
  "tool": "run_shellscript", "args": {"script": "python -c 'import app; print(app.VERSION)'", "timeout": 10},"reasoning": "Check that the local app module imports successfully."}
  ```
<!-- SKILL_END -->
