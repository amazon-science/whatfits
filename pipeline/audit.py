# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Auditor for reproducer train.py — checks for validation data leakage.

Static analysis: scans the reproducer's code for patterns that indicate
direct access to validation data, bypassing the prepare_reproduce.py lockdown.

Usage:
    python audit.py <reproduce_dir>
    python audit.py --all <experiment_dir>    # audit all reproduce-iter* dirs

Exit codes:
    0 = CLEAN (no leakage detected)
    1 = TAINTED (leakage detected)
    2 = ERROR (could not audit)
"""

import ast
import sys
import os
import re
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Banned patterns
# ---------------------------------------------------------------------------

# Default functions that expose val data directly.
# Can be extended per-experiment via experiment.conf BANNED_VAL_FUNCTIONS.
BANNED_IMPORTS = {
    "load_val", "_load_val_private", "_load_val_internal",
    "make_val_loader", "_make_val_loader_private", "_make_val_loader_internal",
    "evaluate_val",  # reproducer should use save_model(), not evaluate_val()
    "_load_all",
}


def _load_experiment_conf(reproduce_dir):
    """Try to find experiment.conf by walking up from reproduce_dir."""
    d = Path(reproduce_dir).resolve()
    # reproduce-iterN is inside the experiment dir
    parent = d.parent
    conf = parent / "experiment.conf"
    if conf.exists():
        extra_banned = set()
        for line in conf.read_text().splitlines():
            line = line.strip()
            if line.startswith("BANNED_VAL_FUNCTIONS="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                for fn in val.split(","):
                    fn = fn.strip()
                    if fn:
                        extra_banned.add(fn)
        return extra_banned
    return set()

# Modules that should not be imported from (only prepare_reproduce is allowed)
BANNED_IMPORT_SOURCES = {
    "prepare",  # should use prepare_reproduce, not prepare directly
}

# File patterns that could contain val/holdout data
SUSPICIOUS_FILE_PATTERNS = [
    r"embeddings\.npz",
    r"train_embeddings\.npz",
    r"test_embeddings\.npz",
    r"holdout",
    r"val.*\.npy",
    r"val.*\.npz",
    r"\.cache.*autoresearch",
    r"autoresearch-cifar-embed",
    r"autoresearch-cifar-cert",
]

# Suspicious array sizes (val=200, holdout=10000)
SUSPICIOUS_ARRAY_SIZES = {200, 10000}


class AuditFinding:
    def __init__(self, severity, message, file, line=None):
        self.severity = severity  # "CRITICAL", "WARNING", "INFO"
        self.message = message
        self.file = file
        self.line = line

    def __str__(self):
        loc = f"{self.file}"
        if self.line:
            loc += f":{self.line}"
        return f"[{self.severity}] {loc} — {self.message}"


class Auditor:
    def __init__(self, reproduce_dir):
        self.dir = Path(reproduce_dir)
        self.findings = []
        # Merge experiment-specific banned functions
        self.banned_imports = BANNED_IMPORTS | _load_experiment_conf(reproduce_dir)

    def audit(self):
        """Run all checks. Returns (verdict, findings)."""
        # Check that prepare.py does NOT exist (isolation violation)
        if (self.dir / "prepare.py").exists():
            self.findings.append(AuditFinding(
                "CRITICAL",
                "prepare.py exists in reproduce directory — isolation violation",
                "prepare.py"))

        py_files = list(self.dir.glob("*.py"))
        # Exclude prepare.py and prepare_reproduce.py (infrastructure, not agent code)
        agent_files = [f for f in py_files
                       if f.name not in ("prepare.py", "prepare_reproduce.py")]

        if not agent_files:
            self.findings.append(AuditFinding(
                "WARNING", "No agent Python files found", str(self.dir)))
            return "ERROR", self.findings

        for f in agent_files:
            source = f.read_text()
            self._check_imports_regex(f.name, source)
            self._check_file_io(f.name, source)
            self._check_hardcoded_arrays(f.name, source)
            try:
                tree = ast.parse(source)
                self._check_imports_ast(f.name, tree)
                self._check_open_calls(f.name, tree)
                self._check_np_load(f.name, tree)
            except SyntaxError:
                self.findings.append(AuditFinding(
                    "WARNING", "Could not parse AST (syntax error)", f.name))

        critical = [f for f in self.findings if f.severity == "CRITICAL"]
        verdict = "TAINTED" if critical else "CLEAN"
        return verdict, self.findings

    def _check_imports_regex(self, filename, source):
        """Quick regex scan for banned function names."""
        for banned in self.banned_imports:
            pattern = rf'\b{re.escape(banned)}\b'
            for i, line in enumerate(source.splitlines(), 1):
                if re.search(pattern, line) and not line.strip().startswith('#'):
                    self.findings.append(AuditFinding(
                        "CRITICAL",
                        f"References banned function '{banned}'",
                        filename, i))

    def _check_imports_ast(self, filename, tree):
        """AST-level check for imports from banned modules."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in BANNED_IMPORT_SOURCES:
                    names = [a.name for a in node.names]
                    # Allow if only importing non-val functions via prepare
                    # (but flag it as warning regardless — should use prepare_reproduce)
                    val_names = [n for n in names if n in self.banned_imports]
                    if val_names:
                        self.findings.append(AuditFinding(
                            "CRITICAL",
                            f"Imports {val_names} from '{node.module}' (val data access)",
                            filename, node.lineno))
                    else:
                        self.findings.append(AuditFinding(
                            "WARNING",
                            f"Imports from '{node.module}' instead of 'prepare_reproduce': {names}",
                            filename, node.lineno))

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in BANNED_IMPORT_SOURCES:
                        self.findings.append(AuditFinding(
                            "WARNING",
                            f"Imports module '{alias.name}' directly (should use prepare_reproduce)",
                            filename, node.lineno))

    def _check_file_io(self, filename, source):
        """Check for file paths that could point to val/holdout data."""
        for pattern in SUSPICIOUS_FILE_PATTERNS:
            for i, line in enumerate(source.splitlines(), 1):
                if re.search(pattern, line) and not line.strip().startswith('#'):
                    self.findings.append(AuditFinding(
                        "CRITICAL",
                        f"References suspicious file pattern: {pattern}",
                        filename, i))

    def _check_open_calls(self, filename, tree):
        """Check for open() or np.load() with suspicious paths."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name == "open" and node.args:
                    self.findings.append(AuditFinding(
                        "WARNING",
                        "Uses open() — verify not reading val/holdout data",
                        filename, node.lineno))

    def _check_np_load(self, filename, tree):
        """Check for np.load() calls that could load cached embeddings."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and
                    node.func.attr == "load" and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "np"):
                    self.findings.append(AuditFinding(
                        "CRITICAL",
                        "Uses np.load() — could load cached val/holdout embeddings",
                        filename, node.lineno))

    def _check_hardcoded_arrays(self, filename, source):
        """Check for hardcoded arrays that could encode val labels."""
        # Look for list literals with ~200 or ~10000 elements
        # This is a heuristic — catches [0,1,3,2,5,...] style encoding
        bracket_pattern = re.compile(r'\[[\d,\s]{400,}\]')
        for i, line in enumerate(source.splitlines(), 1):
            if bracket_pattern.search(line):
                self.findings.append(AuditFinding(
                    "CRITICAL",
                    "Large hardcoded array detected — may encode val/holdout labels",
                    filename, i))

        # Also check for long string constants that could be encoded labels
        for i, line in enumerate(source.splitlines(), 1):
            strings = re.findall(r'["\']([^"\']{200,})["\']', line)
            for s in strings:
                if re.match(r'^[\d,\s]+$', s):
                    self.findings.append(AuditFinding(
                        "CRITICAL",
                        "Long numeric string constant — may encode val/holdout labels",
                        filename, i))


def audit_directory(reproduce_dir):
    """Audit a single reproduce directory. Returns (verdict, findings)."""
    auditor = Auditor(reproduce_dir)
    return auditor.audit()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <reproduce_dir>")
        print(f"       {sys.argv[0]} --all <experiment_dir>")
        sys.exit(2)

    if sys.argv[1] == "--all":
        if len(sys.argv) < 3:
            print("Usage: --all <experiment_dir>")
            sys.exit(2)
        exp_dir = Path(sys.argv[2])
        dirs = sorted(exp_dir.glob("reproduce-iter*"))
        if not dirs:
            print(f"No reproduce-iter* directories found in {exp_dir}")
            sys.exit(2)

        all_clean = True
        print(f"{'Directory':<30s}  {'Verdict':<10s}  {'Findings'}")
        print("-" * 80)
        for d in dirs:
            verdict, findings = audit_directory(d)
            n_crit = sum(1 for f in findings if f.severity == "CRITICAL")
            n_warn = sum(1 for f in findings if f.severity == "WARNING")
            summary = f"{n_crit} critical, {n_warn} warnings"
            print(f"{d.name:<30s}  {verdict:<10s}  {summary}")
            if verdict == "TAINTED":
                all_clean = False
                for f in findings:
                    if f.severity == "CRITICAL":
                        print(f"  {f}")

        sys.exit(0 if all_clean else 1)

    else:
        reproduce_dir = sys.argv[1]
        if not os.path.isdir(reproduce_dir):
            print(f"ERROR: {reproduce_dir} is not a directory")
            sys.exit(2)

        verdict, findings = audit_directory(reproduce_dir)
        print(f"Verdict: {verdict}")
        print()
        for f in findings:
            print(f"  {f}")

        sys.exit(0 if verdict == "CLEAN" else 1)


if __name__ == "__main__":
    main()
