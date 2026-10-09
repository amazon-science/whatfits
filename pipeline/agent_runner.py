#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run an agent prompt through the configured CLI provider.

The pipeline defaults to Claude Code, matching the paper runs:

    AGENT_PROVIDER=claude

Set AGENT_PROVIDER=codex to run the same rendered prompts through Codex directly,
without putting a fake `claude` executable on PATH.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def get_agent_provider() -> str:
    provider = os.environ.get("AGENT_PROVIDER", "claude").strip().lower()
    if provider in {"claude", "anthropic"}:
        return "claude"
    if provider in {"codex", "gpt", "openai"}:
        return "codex"
    raise ValueError(f"Unsupported AGENT_PROVIDER={provider!r}; expected claude or codex")


def get_agent_model(provider: str | None = None) -> str:
    provider = provider or get_agent_provider()
    if provider == "claude":
        return os.environ.get("AGENT_MODEL") or os.environ.get("CLAUDE_MODEL", "opus")
    if provider == "codex":
        return os.environ.get("AGENT_MODEL") or os.environ.get("CODEX_MODEL", "openai.gpt-5.5")
    raise ValueError(f"Unsupported provider: {provider}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_agent_command(prompt: str, cwd: str, max_turns: int) -> list[str]:
    provider = get_agent_provider()
    model = get_agent_model(provider)

    if provider == "claude":
        return [
            os.environ.get("CLAUDE_BIN", "claude"),
            "--dangerously-skip-permissions",
            "--max-turns",
            str(max_turns),
            "--model",
            model,
            "-p",
            prompt,
        ]

    codex_bin = os.environ.get("CODEX_BIN", "codex")
    model_provider = os.environ.get("CODEX_MODEL_PROVIDER", "amazon-bedrock")
    sandbox = os.environ.get("CODEX_SANDBOX", "workspace-write")

    cmd = [codex_bin, "-a", os.environ.get("CODEX_APPROVAL", "never")]
    if model_provider:
        cmd.extend(["-c", f'model_provider="{model_provider}"'])
    if sandbox == "workspace-write" and _env_bool("CODEX_NETWORK_ACCESS", True):
        cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])
    extra_global = os.environ.get("CODEX_GLOBAL_ARGS", "")
    if extra_global:
        cmd.extend(shlex.split(extra_global))

    cmd.append("exec")
    if _env_bool("CODEX_SKIP_GIT_REPO_CHECK", True):
        cmd.append("--skip-git-repo-check")
    extra_exec = os.environ.get("CODEX_EXEC_ARGS", "")
    if extra_exec:
        cmd.extend(shlex.split(extra_exec))

    cmd.extend(["-C", str(Path(cwd).resolve()), "-m", model, "-s", sandbox, prompt])
    return cmd


def run_agent(prompt: str, cwd: str, max_turns: int = 30,
              timeout: int | None = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    cmd = build_agent_command(prompt, cwd, max_turns)
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
    )


def run_agent_text(prompt: str, cwd: str, max_turns: int = 30,
                   timeout: int | None = None) -> str:
    result = run_agent(prompt, cwd, max_turns=max_turns, timeout=timeout, capture_output=True)
    return (result.stdout or "") + (result.stderr or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for the agent")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="Claude max-turns value; Codex ignores this and uses its own loop")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Optional wall-clock timeout in seconds")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Read prompt text from this file instead of stdin")
    parser.add_argument("--print-command", action="store_true",
                        help="Print the provider command and exit without running it")
    args = parser.parse_args()

    prompt = args.prompt_file.read_text() if args.prompt_file else sys.stdin.read()
    cmd = build_agent_command(prompt, args.cwd, args.max_turns)
    if args.print_command:
        print(shlex.join(cmd))
        return 0

    result = subprocess.run(cmd, cwd=args.cwd, timeout=args.timeout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
