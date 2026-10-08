#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Iterative compression with token budget enforcement, fidelity audit, and dry-run check.

Multi-turn process:
1. Agent reads research transcript and writes first compression attempt
2. We count tokens. If over budget, show the truncated version and ask for rewrite.
3. Repeat up to MAX_COMPRESS_TURNS times. Hard truncate on final turn if still over.

With --audit:
4. An auditor agent reads compressed_prompt.md + explorer's train.py and checks for
   missing, ambiguous, or incorrect details. It also reports token usage.
5. If issues found, the compressor revises with targeted guidance.
6. Repeat until audit passes or MAX_AUDIT_TURNS exhausted.
7. [Dry-run] A fresh agent reads ONLY the compressed prompt and describes the code it
   would write. A comparator diffs those decisions against the actual train.py. If
   mismatches are found, one final revision round fixes them. This catches interpretation
   ambiguities that text-level auditing misses (e.g. "cosine schedule" → wrong total_steps).

Usage:
    python compress_iterative.py <compress_dir> <max_tokens> [--audit]

The compress_dir should already contain:
    - compress_prompt.md (the full compression prompt with transcript)
    - best_train.py (explorer's code at the best checkpoint)
    - results.tsv, iterations/, etc.

Output:
    - compressed_prompt.md (the final compressed prompt, guaranteed ≤ max_tokens)
    - compress_log.json (metadata about each turn)
"""

import json
import os
import re
import subprocess
import sys
import time

MAX_COMPRESS_TURNS = 5
MAX_AUDIT_TURNS = 5
MAX_DRYRUN_REVISIONS = 2
HISTORY_FILE = "attempt_history.jsonl"


def load_history(compress_dir: str) -> list:
    """Load previous attempt results from history file."""
    path = os.path.join(compress_dir, HISTORY_FILE)
    history = []
    if not os.path.exists(path):
        return history
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def record_result(compress_dir: str, val_loss: float):
    """Append current prompt + result to history file."""
    prompt_path = os.path.join(compress_dir, "compressed_prompt.md")
    if not os.path.exists(prompt_path):
        print(f"ERROR: {prompt_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(prompt_path) as f:
        prompt_text = strip_comments(f.read())
    n_tokens = count_tokens(prompt_text)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tokens": n_tokens,
        "val_loss": val_loss,
        "prompt": prompt_text,
    }
    path = os.path.join(compress_dir, HISTORY_FILE)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Recorded: val_loss={val_loss}, tokens={n_tokens}, history={path}")


def format_history(history: list) -> str:
    """Format attempt history for inclusion in prompts."""
    if not history:
        return ""
    # Sort by val_loss (best first for lower-is-better; caller can adjust)
    lines = []
    for i, h in enumerate(history, 1):
        prompt_preview = h.get("prompt", "")[:120]
        val = h.get("val_loss", "?")
        tok = h.get("tokens", "?")
        lines.append(f"  Attempt {i}: [{tok}tok] \"{prompt_preview}...\" → val_loss={val}")
    best = min(history, key=lambda h: h.get("val_loss", float("inf")))
    text = f"""
PREVIOUS ATTEMPTS on this checkpoint (use this to judge what matters):
{chr(10).join(lines)}

Best result so far: val_loss={best.get('val_loss', '?')} (attempt with: "{best.get('prompt', '')[:80]}...")
Keep details that correlated with better results. Avoid changes that made things worse.
"""
    return text


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def truncate_to_budget(text: str, budget: int) -> str:
    """Truncate text to budget tokens, snapping to last sentence/line boundary."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= budget:
        return text
    raw = enc.decode(tokens[:budget])
    # Snap back to last sentence or line boundary to avoid mid-word cuts
    for sep in ['. ', '.\n', '\n']:
        idx = raw.rfind(sep)
        if idx > len(raw) * 0.5:  # don't lose more than half
            return raw[:idx + len(sep)].rstrip()
    return raw.rstrip()


def strip_comments(text: str) -> str:
    """Remove HTML comments and collapse blank lines."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_claude(prompt: str, cwd: str, max_turns: int = 30, timeout: int = 600) -> str:
    """Run claude CLI with a prompt and return stdout."""
    result = subprocess.run(
        [
            "claude", "--dangerously-skip-permissions",
            "--max-turns", str(max_turns),
            "--model", os.environ.get("CLAUDE_MODEL", "opus"),
            "-p", prompt,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def run_audit(compressed_text: str, train_py: str, budget: int, n_tokens: int,
              cwd: str) -> dict:
    """Run the auditor: check compressed prompt against explorer's code.

    Returns a dict with:
        pass: bool — True if no critical issues and within budget
        issues: list of {severity, detail, suggestion}
        revision_guidance: str — what to fix in the next revision
    """
    remaining = budget - n_tokens

    audit_prompt = f"""You are auditing a compressed training strategy prompt for information fidelity.
A compressor agent read the full research transcript and distilled it into a short prompt.
A reproducer agent will ONLY see this compressed prompt and must recreate the training strategy.

COMPRESSED PROMPT ({n_tokens}/{budget} tokens, {remaining} tokens remaining):
---
{compressed_text}
---

EXPLORER'S ACTUAL train.py (ground truth):
---
{train_py}
---

Your job: check whether the compressed prompt contains enough information for a fresh agent
to faithfully reproduce the explorer's training strategy. Focus on details that would cause
a significant performance difference if wrong.

Check for:
1. **Missing critical details** — architecture choices, hyperparameters, training procedures,
   or implementation details present in train.py but absent from the compressed prompt.
   Focus on things that would change the result by >1%, not cosmetic differences.
2. **Ambiguous descriptions** — phrasing that could reasonably be interpreted two different ways
   (e.g., "GN+SiLU" could mean pre-norm or post-norm; "SiLU-MLP" could mean gated or ungated).
3. **Incorrect information** — details in the compressed prompt that contradict the code.
4. **Token budget** — if under budget, suggest what to add with remaining tokens.
   If over budget, suggest what to cut (lowest-priority details first).

Output ONLY valid JSON (no markdown fencing, no commentary):
{{
    "pass": true or false,
    "token_count": {n_tokens},
    "budget": {budget},
    "tokens_remaining": {remaining},
    "issues": [
        {{"severity": "critical" or "warning", "detail": "what's wrong", "suggestion": "how to fix in ≤N tokens"}}
    ],
    "revision_guidance": "Concise instructions for the compressor to fix all issues while staying within budget. If under budget, suggest additions. If over budget, suggest cuts."
}}

Set "pass" to true ONLY if there are no critical issues AND token count is within budget.
Warnings alone should not cause a fail UNLESS there are 3+ warnings.

Severity guidelines — err on the side of "critical":
- Any ambiguity where two common interpretations produce structurally different code
  (e.g., pre-norm vs post-norm, gated vs ungated MLP, operation ordering) → "critical"
- Any missing detail that changes architecture shape, parameter count, or optimization
  dynamics (e.g., missing augmentation, missing weight init scheme, wrong batch size) → "critical"
- Only truly cosmetic differences (variable naming, code style, comments) → "warning"

The reproducer agent has NO context beyond this prompt. When in doubt, mark it critical."""

    output = run_claude(audit_prompt, cwd, max_turns=1, timeout=300)

    # Parse JSON from output — try all candidate JSON blocks, prefer the one with "issues"
    parsed = None
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output):
        try:
            candidate = json.loads(match.group())
            if "issues" in candidate or "pass" in candidate:
                parsed = candidate
                break  # first valid match with expected keys
        except (json.JSONDecodeError, ValueError):
            continue

    if parsed is None:
        # Last resort: greedy match (old behavior)
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                parsed = json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    if parsed:
        # Guard: pass=false with 0 issues is not actionable — treat as pass
        if not parsed.get("pass", True) and len(parsed.get("issues", [])) == 0:
            parsed["pass"] = True
            parsed.setdefault("revision_guidance", "No issues found.")
        return parsed

    # Fallback: treat as failed audit with raw output
    return {
        "pass": False,
        "issues": [{"severity": "critical", "detail": "Audit parse error", "suggestion": output[:500]}],
        "revision_guidance": output[:500],
    }


def run_dryrun_interpret(compressed_text: str, cwd: str) -> str:
    """Ask a fresh Claude to write the train.py it would implement from ONLY the compressed prompt.

    Returns the generated code as a string for structural comparison against the actual train.py.
    """
    interpret_prompt = f"""You are a reproducer agent. You will be given a compressed strategy prompt
and must write the complete train.py you would implement. You have never seen the original
training code — you only have this prompt.

COMPRESSED PROMPT:
---
{compressed_text}
---

Write the complete train.py that implements this prompt to a file called `dryrun_train.py`.
Include all imports, model definition, optimizer setup, LR scheduler, training loop, and model saving.

Where the prompt is ambiguous, use your best judgment and add a comment like:
# ASSUMPTION: <what you assumed and why>

Write the code to `dryrun_train.py`. Do NOT write to `train.py`."""

    output = run_claude(interpret_prompt, cwd, max_turns=5, timeout=600)
    # Extract code from output — look for python code block or raw code
    # Try to find a ```python ... ``` block first
    code_match = re.search(r'```python\s*\n(.*?)```', output, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # Try bare ``` block
    code_match = re.search(r'```\s*\n(.*?)```', output, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # Check if agent wrote a file — read it
    dryrun_path = os.path.join(cwd, "dryrun_train.py")
    if os.path.exists(dryrun_path):
        with open(dryrun_path) as f:
            code = f.read()
        os.remove(dryrun_path)  # clean up
        return code.strip()
    # Fallback: return raw output
    return output.strip()


def run_dryrun_compare(interpretation: str, train_py: str, compressed_text: str,
                       budget: int, n_tokens: int, cwd: str) -> dict:
    """Compare the reproducer's generated code against the actual train.py.

    Returns dict with:
        pass: bool
        mismatches: list of {decision, interpreted_as, actual, impact, suggestion}
        revision_guidance: str
    """
    remaining = budget - n_tokens

    compare_prompt = f"""You are comparing two train.py implementations. The REPRODUCER wrote its code
based ONLY on a compressed prompt. The EXPLORER's code is the ground truth. Your job: find
behavioral differences that would change the training metric by >1%.

COMPRESSED PROMPT ({n_tokens}/{budget} tokens):
---
{compressed_text}
---

REPRODUCER'S train.py (written from compressed prompt only):
---
{interpretation}
---

EXPLORER'S ACTUAL train.py (ground truth):
---
{train_py}
---

Compare the two implementations structurally. For each behavioral difference, determine:
- Is it because the prompt is MISSING information (reproducer guessed a default)?
- Is it because the prompt is AMBIGUOUS (reproducer chose a valid but wrong interpretation)?
- Is it because the prompt is INCORRECT (states something different from the code)?

Focus on differences that affect the trained model:
- Optimizer class, LR, weight decay, betas, amsgrad
- LR scheduler: type, total_steps scope (per-epoch vs all training), warmup, decay target
- Model architecture: layers, dimensions, activation, normalization, init
- Training loop: epochs vs time-budget, batch size, gradient clipping, mixed precision
- Data: sequence length, tokenizer, preprocessing

Ignore: variable names, code style, import order, comments, logging, error handling.

Output ONLY valid JSON (no markdown fencing):
{{
    "pass": true or false,
    "mismatches": [
        {{
            "decision": "what aspect (e.g., 'LR schedule total_steps')",
            "interpreted_as": "what the reproducer's code does (cite the line)",
            "actual": "what the explorer's code does (cite the line)",
            "cause": "missing" or "ambiguous" or "incorrect",
            "impact": "high" or "medium" or "low",
            "suggestion": "how to fix the compressed prompt in ≤N tokens"
        }}
    ],
    "revision_guidance": "Concise instructions to fix the prompt. Budget: {remaining} tokens remaining."
}}

Set "pass" to true ONLY if there are no high-impact mismatches.
Mark impact as "high" if the mismatch would likely change the metric by >2%.
Mark as "medium" for 1-2% impact, "low" for <1%."""

    output = run_claude(compare_prompt, cwd, max_turns=1, timeout=300)

    # Parse JSON — same robust parsing as run_audit
    parsed = None
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output):
        try:
            candidate = json.loads(match.group())
            if "mismatches" in candidate or "pass" in candidate:
                parsed = candidate
                break
        except (json.JSONDecodeError, ValueError):
            continue

    if parsed is None:
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                parsed = json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    if parsed:
        if not parsed.get("pass", True) and len(parsed.get("mismatches", [])) == 0:
            parsed["pass"] = True
        return parsed

    return {
        "pass": False,
        "mismatches": [{"decision": "parse_error", "interpreted_as": "N/A",
                        "actual": "N/A", "cause": "N/A", "impact": "high",
                        "suggestion": output[:500]}],
        "revision_guidance": output[:500],
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: compress_iterative.py <compress_dir> <max_tokens> [--audit] [--record-result <val_loss>]",
              file=sys.stderr)
        sys.exit(1)

    compress_dir = sys.argv[1]
    budget = int(sys.argv[2])
    do_audit = "--audit" in sys.argv

    # Handle --record-result mode
    if "--record-result" in sys.argv:
        idx = sys.argv.index("--record-result")
        if idx + 1 >= len(sys.argv):
            print("ERROR: --record-result requires a val_loss value", file=sys.stderr)
            sys.exit(1)
        val_loss = float(sys.argv[idx + 1])
        record_result(compress_dir, val_loss)
        return

    if not os.path.exists(os.path.join(compress_dir, "compress_prompt.md")):
        print(f"ERROR: {compress_dir}/compress_prompt.md not found", file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(compress_dir, "compress_prompt.md")) as f:
        compress_prompt = f.read()

    # Load attempt history for this checkpoint
    history = load_history(compress_dir)
    history_text = format_history(history)
    if history:
        print(f"Loaded {len(history)} previous attempts (best val_loss="
              f"{min(h.get('val_loss', float('inf')) for h in history):.4f})")

    # Load explorer's train.py for audit (best_train.py is the best checkpoint's code)
    train_py = None
    if do_audit:
        train_py_path = os.path.join(compress_dir, "best_train.py")
        if not os.path.exists(train_py_path):
            train_py_path = os.path.join(compress_dir, "train.py")
        if os.path.exists(train_py_path):
            with open(train_py_path) as f:
                train_py = f.read()
        else:
            print("WARNING: No train.py found for audit. Disabling audit.", file=sys.stderr)
            do_audit = False

    log = {"budget": budget, "audit": do_audit, "turns": []}

    mode_str = "budget + audit" if do_audit else "budget only"
    print(f"Iterative compression: budget={budget} tokens, mode={mode_str}")
    print(f"Working dir: {compress_dir}")
    print()

    # ── Phase 1: Initial compression with budget enforcement ──────────────
    clean_text = None
    n_tokens = 0

    for turn in range(1, MAX_COMPRESS_TURNS + 1):
        print(f"=== Compress Turn {turn}/{MAX_COMPRESS_TURNS} ===")

        if turn == 1 or clean_text is None:
            # clean_text is None when a prior turn's agent failed to write the
            # file at all; fall back to the original compress prompt rather than
            # crashing in truncate_to_budget(None). (GPT/codex is less reliable
            # than Claude about always emitting the file on turn 1.)
            prompt = compress_prompt
        else:
            truncated = truncate_to_budget(clean_text, budget)
            prompt = f"""Your previous compressed prompt was {n_tokens} tokens, but the budget is {budget} tokens.

After hard truncation to {budget} tokens, the reproducer would see ONLY this:

---
{truncated}
---

Everything after that is LOST. The reproducer never sees it.

Rewrite the compressed prompt to fit within {budget} tokens. Key principles:
- **Front-load by importance**: Write the most critical fact first (model class), then next most critical (key hyperparameter), etc. If truncation happens at ANY point, the surviving prefix should be the best possible partial description.
- **Drop rather than abbreviate**: It's better to say "RandomForest depth=12" clearly than to cram "RFC(500,d12,leaf5,rs7,tgtenc,poly)" which might confuse the reproducer.
- **Omit rather than truncate**: If a detail doesn't fit, leave it out entirely. Don't let it get cut mid-word.

Write ONLY the compressed prompt to `compressed_prompt.md`. Nothing else."""

        t0 = time.time()
        output = run_claude(prompt, compress_dir, max_turns=30)
        elapsed = time.time() - t0

        prompt_path = os.path.join(compress_dir, "compressed_prompt.md")
        if not os.path.exists(prompt_path):
            print(f"  Agent didn't write compressed_prompt.md. Retrying...")
            log["turns"].append({
                "turn": turn, "phase": "compress", "status": "no_output",
                "elapsed": round(elapsed, 1),
            })
            continue

        with open(prompt_path) as f:
            raw_text = f.read()

        clean_text = strip_comments(raw_text)
        n_tokens = count_tokens(clean_text)

        print(f"  Tokens: {n_tokens}/{budget}", end="")
        if n_tokens <= budget:
            print(" [within budget]")
            with open(prompt_path, "w") as f:
                f.write(clean_text + "\n")
            log["turns"].append({
                "turn": turn, "phase": "compress", "tokens": n_tokens, "status": "pass",
                "elapsed": round(elapsed, 1),
                "prompt_preview": clean_text[:200],
            })
            break
        else:
            print(f" [over by {n_tokens - budget}]")
            log["turns"].append({
                "turn": turn, "phase": "compress", "tokens": n_tokens, "status": "over",
                "elapsed": round(elapsed, 1),
                "prompt_preview": clean_text[:200],
            })

            if turn == MAX_COMPRESS_TURNS:
                print(f"  Max compress turns reached. Hard truncating to {budget} tokens.")
                truncated = truncate_to_budget(clean_text, budget)
                with open(prompt_path, "w") as f:
                    f.write(truncated + "\n")
                clean_text = truncated
                n_tokens = count_tokens(truncated)
                log["turns"][-1]["status"] = "hard_truncated"
                log["turns"][-1]["final_tokens"] = n_tokens

    # ── Phase 2: Audit loop (budget + fidelity) ──────────────────────────
    if do_audit and clean_text:
        print(f"\n{'='*60}")
        print(f"AUDIT PHASE: checking fidelity against explorer's train.py")
        print(f"{'='*60}\n")

        audit_history = []  # Track previous issues to prevent oscillation

        for audit_turn in range(1, MAX_AUDIT_TURNS + 1):
            print(f"=== Audit Turn {audit_turn}/{MAX_AUDIT_TURNS} ===")

            # Re-read in case the compressor revised it
            prompt_path = os.path.join(compress_dir, "compressed_prompt.md")
            with open(prompt_path) as f:
                clean_text = strip_comments(f.read())
            n_tokens = count_tokens(clean_text)

            t0 = time.time()
            audit_result = run_audit(clean_text, train_py, budget, n_tokens, compress_dir)
            audit_elapsed = time.time() - t0

            n_issues = len(audit_result.get("issues", []))
            n_critical = sum(1 for i in audit_result.get("issues", []) if i.get("severity") == "critical")
            passed = audit_result.get("pass", False)

            print(f"  Audit: {'PASS' if passed else 'FAIL'} "
                  f"({n_issues} issues, {n_critical} critical)")
            for issue in audit_result.get("issues", []):
                sev = issue.get("severity", "?")
                det = issue.get("detail", "")
                print(f"    [{sev}] {det}")

            log["turns"].append({
                "turn": audit_turn, "phase": "audit",
                "tokens": n_tokens,
                "status": "pass" if passed else "fail",
                "n_issues": n_issues,
                "n_critical": n_critical,
                "issues": audit_result.get("issues", []),
                "elapsed": round(audit_elapsed, 1),
            })

            if passed:
                print("  Audit passed. Compression complete.")
                break

            if audit_turn == MAX_AUDIT_TURNS:
                print("  Max audit turns reached. Keeping current version.")
                break

            # Revise based on audit feedback
            print(f"\n  Revising based on audit feedback...")
            guidance = audit_result.get("revision_guidance", "Fix the issues listed above.")
            issues_text = "\n".join(
                f"  - [{i.get('severity', '?')}] {i.get('detail', '')} → {i.get('suggestion', '')}"
                for i in audit_result.get("issues", [])
            )
            audit_history.append({
                "turn": audit_turn,
                "issues": [i.get("detail", "") for i in audit_result.get("issues", [])],
            })

            # Build history context to prevent oscillation
            oscillation_text = ""
            if len(audit_history) > 1:
                prev_issues = []
                for h in audit_history[:-1]:
                    prev_issues.extend(h["issues"])
                if prev_issues:
                    oscillation_text = f"""
PREVIOUS AUDIT ISSUES (already flagged in earlier rounds — do NOT regress these):
{chr(10).join(f'  - {i}' for i in prev_issues)}

Fix the NEW issues below WITHOUT reintroducing the old ones.
"""

            revise_prompt = f"""Your compressed prompt was audited against the explorer's actual train.py.
The audit found these issues:

{issues_text}
{oscillation_text}
{history_text}
Token budget: {n_tokens}/{budget} tokens ({budget - n_tokens} remaining).

Auditor's guidance: {guidance}

Rewrite `compressed_prompt.md` to fix the critical issues while staying within {budget} tokens.

CRITICAL CONSTRAINT: Your output MUST be ≤{budget} tokens. If you go over, the end will be
hard-truncated and information will be lost mid-word. To make room for fixes:
- Drop the LEAST important details (seeds, implied conventions, discoverable API signatures)
- Use terse notation (e.g., "lr.06" not "learning_rate=0.06")
- Front-load the most critical information so truncation damages least

Write ONLY the revised compressed prompt to `compressed_prompt.md`. Nothing else."""

            t0 = time.time()
            output = run_claude(revise_prompt, compress_dir, max_turns=30)
            revise_elapsed = time.time() - t0

            # Re-read and verify budget
            with open(prompt_path) as f:
                raw_text = f.read()
            clean_text = strip_comments(raw_text)
            n_tokens = count_tokens(clean_text)

            print(f"  Revised: {n_tokens}/{budget} tokens")

            # Hard truncate if revision went over budget
            if n_tokens > budget:
                print(f"  Revision over budget. Hard truncating.")
                clean_text = truncate_to_budget(clean_text, budget)
                n_tokens = count_tokens(clean_text)
                with open(prompt_path, "w") as f:
                    f.write(clean_text + "\n")
            else:
                with open(prompt_path, "w") as f:
                    f.write(clean_text + "\n")

            log["turns"].append({
                "turn": audit_turn, "phase": "revise",
                "tokens": n_tokens,
                "status": "revised",
                "elapsed": round(revise_elapsed, 1),
                "prompt_preview": clean_text[:200],
            })

    # ── Phase 3: Dry-run interpretation check ─────────────────────────────
    if do_audit and clean_text and train_py:
        print(f"\n{'='*60}")
        print(f"DRY-RUN PHASE: fresh agent writes code from prompt, diff against actual")
        print(f"{'='*60}\n")

        prompt_path = os.path.join(compress_dir, "compressed_prompt.md")

        for dryrun_round in range(1, MAX_DRYRUN_REVISIONS + 1):
            # Step 1: Get interpretation from a fresh agent
            print(f"=== Dry-run Round {dryrun_round}/{MAX_DRYRUN_REVISIONS} ===")
            print("  Fresh agent writing train.py from prompt...")

            with open(prompt_path) as f:
                clean_text = strip_comments(f.read())
            n_tokens = count_tokens(clean_text)

            t0 = time.time()
            interpretation = run_dryrun_interpret(clean_text, compress_dir)
            interp_elapsed = time.time() - t0

            print(f"  Code generated ({len(interpretation)} chars, {interp_elapsed:.0f}s)")

            log["turns"].append({
                "turn": dryrun_round, "phase": "dryrun_interpret",
                "tokens": n_tokens,
                "elapsed": round(interp_elapsed, 1),
                "interpretation_preview": interpretation[:300],
            })

            # Step 2: Compare interpretation against actual code
            print("  Diffing generated code against explorer's train.py...")
            t0 = time.time()
            compare_result = run_dryrun_compare(
                interpretation, train_py, clean_text, budget, n_tokens, compress_dir
            )
            compare_elapsed = time.time() - t0

            mismatches = compare_result.get("mismatches", [])
            high_impact = [m for m in mismatches if m.get("impact") == "high"]
            passed = compare_result.get("pass", False)

            print(f"  Dry-run: {'PASS' if passed else 'FAIL'} "
                  f"({len(mismatches)} mismatches, {len(high_impact)} high-impact)")
            for m in mismatches:
                imp = m.get("impact", "?")
                dec = m.get("decision", "?")
                interp = m.get("interpreted_as", "?")[:80]
                actual = m.get("actual", "?")[:80]
                cause = m.get("cause", "?")
                print(f"    [{imp}] {dec}: ({cause})")
                print(f"      reproducer: {interp}")
                print(f"      actual:     {actual}")

            log["turns"].append({
                "turn": dryrun_round, "phase": "dryrun_compare",
                "tokens": n_tokens,
                "status": "pass" if passed else "fail",
                "n_mismatches": len(mismatches),
                "n_high_impact": len(high_impact),
                "mismatches": mismatches,
                "elapsed": round(compare_elapsed, 1),
            })

            if passed:
                print("  Dry-run passed. No high-impact interpretation mismatches.")
                break

            if dryrun_round == MAX_DRYRUN_REVISIONS:
                print("  Max dry-run rounds reached. Keeping current version.")
                break

            # Step 3: Revise prompt to fix interpretation mismatches
            print(f"\n  Revising to fix code mismatches...")
            mismatch_text = "\n".join(
                f"  - [{m.get('impact', '?')}] {m.get('decision', '?')}: "
                f"reproducer code does '{m.get('interpreted_as', '?')[:100]}' "
                f"but explorer code does '{m.get('actual', '?')[:100]}' "
                f"(cause: {m.get('cause', '?')}) → {m.get('suggestion', '')}"
                for m in mismatches if m.get("impact") in ("high", "medium")
            )
            guidance = compare_result.get("revision_guidance", "Fix the mismatches listed above.")

            revise_prompt = f"""A fresh reproducer agent read your compressed prompt and wrote a complete
train.py. We diffed it against the explorer's actual train.py and found these MISMATCHES —
places where the reproducer wrote DIFFERENT code:

{mismatch_text}

Comparator's guidance: {guidance}

EXPLORER'S ACTUAL train.py (use this to judge what's critical vs expendable):
---
{train_py}
---
{history_text}
Token budget: {n_tokens}/{budget} tokens ({budget - n_tokens} remaining).

Rewrite `compressed_prompt.md` to eliminate the high-impact mismatches. The goal: a fresh agent
reading ONLY this prompt should make the SAME key decisions as the explorer's code.

CRITICAL CONSTRAINT: Stay within {budget} tokens. When you need to make room:
- Consult the train.py to judge which existing details matter most for performance
- If previous attempts exist, learn from what helped and what hurt
- Drop details the reproducer would likely get right by default (common conventions)
- Keep details where the explorer's choice differs from the obvious default
- Be explicit about defaults that differ from common assumptions
- Use precise language: "cosine decay over all epochs" not just "cosine schedule"

Write ONLY the revised compressed prompt to `compressed_prompt.md`. Nothing else."""

            t0 = time.time()
            output = run_claude(revise_prompt, compress_dir, max_turns=30)
            revise_elapsed = time.time() - t0

            with open(prompt_path) as f:
                raw_text = f.read()
            clean_text = strip_comments(raw_text)
            n_tokens = count_tokens(clean_text)

            print(f"  Revised: {n_tokens}/{budget} tokens")

            if n_tokens > budget:
                print(f"  Over budget. Hard truncating.")
                clean_text = truncate_to_budget(clean_text, budget)
                n_tokens = count_tokens(clean_text)
                with open(prompt_path, "w") as f:
                    f.write(clean_text + "\n")
            else:
                with open(prompt_path, "w") as f:
                    f.write(clean_text + "\n")

            log["turns"].append({
                "turn": dryrun_round, "phase": "dryrun_revise",
                "tokens": n_tokens,
                "status": "revised",
                "elapsed": round(revise_elapsed, 1),
                "prompt_preview": clean_text[:200],
            })

    # ── Write log and verify ─────────────────────────────────────────────
    log_path = os.path.join(compress_dir, "compress_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    with open(os.path.join(compress_dir, "compressed_prompt.md")) as f:
        final_text = strip_comments(f.read())
    final_tokens = count_tokens(final_text)
    print(f"\nFinal: {final_tokens}/{budget} tokens")
    print(f"Preview: {final_text[:300]}")
    print(f"Log: {log_path}")

    if final_tokens > budget:
        print(f"WARNING: Still over budget! This shouldn't happen.")
        sys.exit(1)


if __name__ == "__main__":
    main()
