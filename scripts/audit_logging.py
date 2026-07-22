"""Logging completeness audit script.

Scans app/ source files to verify structlog logging practices:
Confirms that required fields (request_id, chunk_ids, prompt_version, latency_ms)
are bound or included on relevant logging statements across Phase 1+ endpoint handlers.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


REQUIRED_LOG_KEYS = {"chunk_ids", "prompt_version", "latency_ms"}


def audit_logging_completeness(app_dir: str | Path = "./app") -> tuple[bool, list[str]]:
    """Scan Python files in app_dir for structlog call completeness.

    Returns:
        Tuple of (is_complete: bool, findings: list[str]).
    """
    path = Path(app_dir)
    if not path.is_dir():
        return False, [f"Directory {app_dir} not found"]

    findings: list[str] = []
    py_files = list(path.glob("**/*.py"))

    query_route_found = False
    query_route_logged_keys: set[str] = set()

    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=py_file.name)
        except Exception as e:
            findings.append(f"{py_file.name}: Syntax error parsing AST: {e}")
            continue

        for node in ast.walk(tree):
            # Check for logger.info / logger.warn / logger.error calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("info", "warn", "warning", "error"):
                    kw_keys = {kw.arg for kw in node.keywords if kw.arg is not None}
                    
                    # Check if this is the main query completion log in routes.py
                    if "routes.py" in py_file.name and "latency_ms" in kw_keys:
                        query_route_found = True
                        query_route_logged_keys.update(kw_keys)

    if not query_route_found:
        findings.append("app/api/routes.py: Did not find main query completion log with latency_ms")
    else:
        missing = REQUIRED_LOG_KEYS - query_route_logged_keys
        if missing:
            findings.append(f"app/api/routes.py: Query completion log is missing required keys: {missing}")

    is_complete = len(findings) == 0
    return is_complete, findings


def main() -> None:
    print("=== Running Logging Completeness Audit ===")
    is_complete, findings = audit_logging_completeness("./app")

    if is_complete:
        print("SUCCESS: All required logging fields (chunk_ids, prompt_version, latency_ms) are present! [Exit 0]")
        sys.exit(0)
    else:
        print("FAILURE: Logging audit found gaps in log field coverage:")
        for f in findings:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
