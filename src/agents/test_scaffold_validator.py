"""
Deterministic validation for test-scaffolding milestones.

The validator checks that spec/test milestones describe an external contract
for future implementation code instead of embedding production logic inside
the tests. For Python scaffolds it can also create a temporary API stub overlay
so tests can import future modules during collect/red-phase validation.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScaffoldCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_replan: bool = False


def validate_test_scaffold_structure(
    milestone: dict[str, Any],
    workspace_root: Path,
) -> ScaffoldCheck:
    """Validate test-scaffold files without executing the implementation."""
    contract = milestone.get("validation_contract", {}) or {}
    language = _contract_language(contract, milestone)
    if language != "python":
        return ScaffoldCheck(
            ok=False,
            needs_replan=True,
            errors=[
                (
                    f"test_scaffold validation currently supports Python, got {language!r}. "
                    "Add a language-specific scaffold validator or use an integration contract."
                )
            ],
        )

    public_api = normalize_public_api(contract, milestone)
    if not public_api:
        return ScaffoldCheck(
            ok=False,
            needs_replan=True,
            errors=[
                (
                    "test_scaffold contract is missing public_api. The orchestrator must "
                    "declare the future module symbols that tests should import."
                )
            ],
        )

    target_files = _test_target_files(milestone)
    if not target_files:
        return ScaffoldCheck(
            ok=False,
            needs_replan=True,
            errors=["test_scaffold milestone must target at least one test file."],
        )

    errors: list[str] = []
    warnings: list[str] = []
    required_imports = set(_required_imports(contract, public_api))
    forbidden_defs = set(_forbidden_definitions(contract, public_api))
    min_assertions = int(contract.get("min_assertions", 1))
    min_tests = int(contract.get("min_tests", max(1, len(contract.get("required_behaviors", [])))))

    for rel_path in target_files:
        path = (workspace_root / rel_path).resolve()
        try:
            path.relative_to(workspace_root.resolve())
        except ValueError:
            errors.append(f"Test path escapes workspace: {rel_path}")
            continue
        if not path.exists():
            errors.append(f"Test file does not exist: {rel_path}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            errors.append(f"{rel_path} has syntax error: {exc}")
            continue
        except OSError as exc:
            errors.append(f"Could not read {rel_path}: {exc}")
            continue

        imports = _collect_imports(tree)
        missing_imports = [
            required for required in required_imports
            if required not in imports and not _module_import_satisfies(required, imports)
        ]
        errors.extend(f"{rel_path} missing required import: {item}" for item in missing_imports)

        defs = _collect_definitions(tree)
        embedded = sorted(forbidden_defs.intersection(defs))
        errors.extend(
            f"{rel_path} defines production object inside tests: {name}"
            for name in embedded
        )

        test_count = _count_tests(tree)
        if test_count < min_tests:
            errors.append(
                f"{rel_path} has {test_count} test function(s), expected at least {min_tests}."
            )

        assertion_count = _count_assertions(tree)
        if assertion_count < min_assertions:
            errors.append(
                f"{rel_path} has {assertion_count} assertion(s), expected at least {min_assertions}."
            )

        skip_markers = _collect_skip_markers(tree)
        if skip_markers:
            errors.append(f"{rel_path} contains skip/xfail markers: {', '.join(skip_markers)}")

        helper_errors = _check_large_helpers(tree, rel_path, int(contract.get("max_helper_statements", 8)))
        errors.extend(helper_errors)

        missing_keywords = _missing_behavior_keywords(source, contract)
        warnings.extend(
            f"{rel_path} may not cover behavior keyword: {keyword}"
            for keyword in missing_keywords
        )

    return ScaffoldCheck(ok=not errors, errors=errors, warnings=warnings)


def build_python_stub_overlay(
    milestone: dict[str, Any],
    workspace_root: Path,
    stub_root: Path,
) -> Path:
    """Create temporary Python modules matching contract public_api."""
    contract = milestone.get("validation_contract", {}) or {}
    public_api = normalize_public_api(contract, milestone)
    if not public_api:
        raise ValueError("Cannot build stub overlay without public_api")

    if stub_root.exists():
        shutil.rmtree(stub_root)
    stub_root.mkdir(parents=True, exist_ok=True)

    by_module: dict[str, list[dict[str, Any]]] = {}
    for item in public_api:
        module = str(item.get("module", "")).strip()
        if not module:
            continue
        by_module.setdefault(module, []).append(item)

    for module, symbols in by_module.items():
        module_path = _module_to_path(stub_root, module)
        module_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_package_inits(stub_root, module_path.parent)
        module_path.write_text(_render_python_stub_module(symbols), encoding="utf-8")

    return stub_root


def python_stub_env_overlay(stub_root: Path, workspace_root: Path) -> dict[str, str]:
    """Prepend generated stubs ahead of the real workspace on PYTHONPATH."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(stub_root), str(workspace_root)]
    if existing:
        parts.append(existing)
    return {"PYTHONPATH": os.pathsep.join(parts)}


def collect_contract(milestone: dict[str, Any]) -> dict[str, Any]:
    """Return a collect-only pytest contract for the scaffold target files."""
    contract = dict(milestone.get("validation_contract", {}) or {})
    targets = _test_target_files(milestone)
    target = targets[0] if targets else "tests"

    if contract.get("target"):
        args = str(contract.get("args", ""))
        if "--collect-only" not in args:
            args = f"{args} --collect-only".strip()
        if "-q" not in args.split():
            args = f"{args} -q".strip()
        contract.update({"type": "pytest", "args": args})
        return contract

    command = str(contract.get("command", "")).strip()
    if command:
        if "--collect-only" not in command:
            command = f"{command} --collect-only"
        if " -q" not in f" {command} ":
            command = f"{command} -q"
    else:
        command = f"python -m pytest {target} --collect-only -q"
    contract.update({"type": "pytest", "command": command})
    return contract


def red_phase_contract(milestone: dict[str, Any]) -> dict[str, Any]:
    """Return a pytest contract that executes tests against stubs."""
    contract = dict(milestone.get("validation_contract", {}) or {})
    targets = _test_target_files(milestone)
    target = targets[0] if targets else "tests"

    if contract.get("target"):
        args = _strip_collect_args(str(contract.get("args", "")))
        if not args:
            args = "-q"
        contract.update({"type": "pytest", "args": args})
        return contract

    command = str(contract.get("command", "")).strip()
    if command:
        command = _strip_collect_args(command)
    else:
        command = f"python -m pytest {target} -q"
    contract.update({"type": "pytest", "command": command})
    return contract


def normalize_public_api(
    contract: dict[str, Any],
    milestone: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize public_api from either contract or milestone."""
    raw = contract.get("public_api") or milestone.get("public_api") or []
    if isinstance(raw, dict):
        module = raw.get("module")
        symbols = raw.get("symbols", [])
        normalized = []
        for symbol in symbols:
            if isinstance(symbol, str):
                normalized.append({"module": module, "name": symbol, "kind": "function"})
            elif isinstance(symbol, dict):
                normalized.append({"module": symbol.get("module", module), **symbol})
        return [item for item in normalized if item.get("module") and item.get("name")]
    if isinstance(raw, list):
        return [
            item for item in raw
            if isinstance(item, dict) and item.get("module") and item.get("name")
        ]
    return []


def _contract_language(contract: dict[str, Any], milestone: dict[str, Any]) -> str:
    raw = contract.get("language") or milestone.get("language")
    if raw:
        return str(raw).strip().lower()
    targets = _test_target_files(milestone)
    if targets and all(path.endswith(".py") for path in targets):
        return "python"
    return "unknown"


def _test_target_files(milestone: dict[str, Any]) -> list[str]:
    return [
        str(path)
        for path in milestone.get("target_files", [])
        if str(path).endswith(".py") and ("test_" in str(path) or str(path).startswith("tests/"))
    ]


def _required_imports(contract: dict[str, Any], public_api: list[dict[str, Any]]) -> list[str]:
    raw = contract.get("required_imports")
    if isinstance(raw, list) and raw:
        normalized: list[str] = []
        for item in raw:
            normalized.extend(_normalize_required_import(str(item)))
        return normalized
    return [f"{item['module']}.{item['name']}" for item in public_api]


def _forbidden_definitions(contract: dict[str, Any], public_api: list[dict[str, Any]]) -> list[str]:
    names = {str(item["name"]) for item in public_api}
    for item in public_api:
        methods = item.get("methods", [])
        if isinstance(methods, list):
            names.update(str(method) for method in methods)
    raw = contract.get("forbidden_definitions", [])
    if isinstance(raw, list):
        names.update(_normalize_forbidden_definition(str(item)) for item in raw)
    return sorted(names)


def _normalize_required_import(value: str) -> list[str]:
    """Accept canonical dotted imports and common Python import statements."""
    value = value.strip()
    from_match = re.fullmatch(r"from\s+([A-Za-z_][\w.]*)\s+import\s+(.+)", value)
    if from_match:
        module = from_match.group(1)
        names = []
        for raw_name in from_match.group(2).split(","):
            name = raw_name.strip().split(" as ", 1)[0].strip()
            if name and name != "*":
                names.append(f"{module}.{name}")
        return names

    import_match = re.fullmatch(r"import\s+([A-Za-z_][\w.]*)(?:\s+as\s+\w+)?", value)
    if import_match:
        return [import_match.group(1)]

    return [value]


def _normalize_forbidden_definition(value: str) -> str:
    """Convert prompt-style entries like 'class Foo' or 'def bar' to bare names."""
    value = value.strip()
    match = re.fullmatch(r"(?:class|def|enum|function|method)\s+([A-Za-z_][A-Za-z0-9_]*)", value)
    if match:
        return match.group(1)
    return value


def _collect_imports(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
            for alias in node.names:
                imports.add(f"{node.module}.{alias.name}")
    return imports


def _module_import_satisfies(required: str, imports: set[str]) -> bool:
    module = required.rsplit(".", 1)[0]
    return module in imports or f"{module}.*" in imports


def _collect_definitions(tree: ast.AST) -> set[str]:
    defs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.add(node.name)
    return defs


def _count_tests(tree: ast.AST) -> int:
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )


def _count_assertions(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            count += 1
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name.startswith("assert") or name in {"pytest.raises", "raises"}:
                count += 1
    return count


def _collect_skip_markers(tree: ast.AST) -> list[str]:
    markers: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"pytest.skip", "skip", "pytest.xfail", "xfail"}:
                markers.append(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                name = _call_name(decorator)
                if "skip" in name or "xfail" in name:
                    markers.append(name)
    return sorted(set(markers))


def _check_large_helpers(tree: ast.AST, rel_path: str, max_statements: int) -> list[str]:
    errors: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") or _has_fixture_decorator(node):
                continue
            if len(node.body) > max_statements:
                errors.append(
                    f"{rel_path} helper {node.name!r} is too large for a test scaffold "
                    f"({len(node.body)} statements > {max_statements})."
                )
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                continue
            if len(node.body) > max_statements:
                errors.append(
                    f"{rel_path} helper class {node.name!r} is too large for a test scaffold "
                    f"({len(node.body)} statements > {max_statements})."
                )
    return errors


def _has_fixture_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any("fixture" in _call_name(decorator) for decorator in node.decorator_list)


def _missing_behavior_keywords(source: str, contract: dict[str, Any]) -> list[str]:
    keywords = contract.get("required_behavior_keywords") or contract.get("required_behaviors") or []
    if not isinstance(keywords, list):
        return []
    lowered = source.lower()
    missing = []
    for keyword in keywords:
        token = str(keyword).strip().lower()
        if token and token not in lowered:
            missing.append(token)
    return missing


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _module_to_path(stub_root: Path, module: str) -> Path:
    safe_parts = [
        part for part in module.split(".")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part)
    ]
    if not safe_parts or len(safe_parts) != len(module.split(".")):
        raise ValueError(f"Invalid Python module path in public_api: {module!r}")
    return stub_root.joinpath(*safe_parts).with_suffix(".py")


def _ensure_package_inits(stub_root: Path, module_dir: Path) -> None:
    current = module_dir
    while current != stub_root and stub_root in current.parents:
        init = current / "__init__.py"
        init.touch(exist_ok=True)
        current = current.parent


def _render_python_stub_module(symbols: list[dict[str, Any]]) -> str:
    lines = [
        '"""Generated validation stubs for test-scaffold collection."""',
        "from __future__ import annotations",
        "from enum import Enum",
        "",
    ]
    for item in symbols:
        name = str(item["name"])
        kind = str(item.get("kind", "function")).lower()
        if kind == "enum":
            members = item.get("members") or item.get("values") or ["VALUE"]
            lines.append(f"class {name}(Enum):")
            for member in members:
                member_name = re.sub(r"\W+", "_", str(member)).upper().strip("_") or "VALUE"
                lines.append(f"    {member_name} = {member_name!r}")
            lines.append("")
        elif kind == "class":
            lines.append(f"class {name}:")
            lines.append("    def __init__(self, *args, **kwargs):")
            lines.append("        raise NotImplementedError")
            methods = item.get("methods", [])
            for method in (methods if isinstance(methods, list) else []):
                method_name = str(method)
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", method_name):
                    lines.append("")
                    lines.append(f"    def {method_name}(self, *args, **kwargs):")
                    lines.append("        raise NotImplementedError")
            lines.append("")
        elif kind in {"constant", "value"}:
            lines.append(f"{name} = None")
            lines.append("")
        else:
            lines.append(f"def {name}(*args, **kwargs):")
            lines.append("    raise NotImplementedError")
            lines.append("")
    return "\n".join(lines)


def _strip_collect_args(command: str) -> str:
    parts = command.split()
    stripped = [part for part in parts if part != "--collect-only"]
    if "-q" not in stripped:
        stripped.append("-q")
    return " ".join(stripped)
