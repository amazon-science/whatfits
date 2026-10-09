#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: run_reproduce.sh <experiment_dir> [CHECKPOINT]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"
LABEL="${2:-latest}"

CONDITION="$(get_condition_suffix)"
COMPRESS_DIR="${EXPERIMENT_DIR}/compress-${CONDITION}-${LABEL}"
REPRODUCE_DIR="${EXPERIMENT_DIR}/reproduce-${CONDITION}-${LABEL}"

if [ ! -f "$COMPRESS_DIR/compressed_prompt.md" ]; then
    echo "ERROR: Compressed prompt not found: $COMPRESS_DIR/compressed_prompt.md"
    echo "Run run_compress.sh first."
    exit 1
fi

# --- Verify token budget ---
TOKEN_BUDGET="${COMPRESS_TOKENS:-128}"
python3 "$PIPELINE_DIR/enforce_token_budget.py" "$COMPRESS_DIR/compressed_prompt.md" "$TOKEN_BUDGET" || true
echo ""

# Build the reproduce prompt from template + compressed prompt
COMPRESSED=$(cat "$COMPRESS_DIR/compressed_prompt.md")
export COMPRESSED_PROMPT="$COMPRESSED"
"$PIPELINE_DIR/render_prompt.sh" "$EXPERIMENT_DIR" reproduce > /tmp/reproduce_prompt_rendered.md

if [ ! -d "$REPRODUCE_DIR" ]; then
    echo "Creating reproducer agent directory: $REPRODUCE_DIR"
    mkdir -p "$REPRODUCE_DIR"

    # Copy reproduce files — NOTE: prepare.py is NOT copied (isolation)
    for f in $(get_reproduce_files); do
        if [ -f "$EXPERIMENT_DIR/$f" ]; then
            cp "$EXPERIMENT_DIR/$f" "$REPRODUCE_DIR/"
        fi
    done

    cp /tmp/reproduce_prompt_rendered.md "$REPRODUCE_DIR/reproduce_prompt.md"

    cd "$REPRODUCE_DIR"
    git init
    git add -A
    git commit -m "baseline"
else
    echo "Reproducer agent directory already exists: $REPRODUCE_DIR"
    cd "$REPRODUCE_DIR"
fi

echo "Starting reproducer agent in $REPRODUCE_DIR"
echo "Checkpoint: $LABEL"
echo "Agent provider: $(get_agent_provider)"
echo "Agent model: $(get_agent_model)"
echo ""
echo "Compressed prompt being used:"
cat "$COMPRESS_DIR/compressed_prompt.md"
echo ""

python3 "$PIPELINE_DIR/agent_runner.py" \
    --cwd "$PWD" \
    --max-turns 100 \
    --prompt-file reproduce_prompt.md \
    2>&1 | stdbuf -oL tee agent.log

# ── Post-agent evaluation ────────────────────────
# The reproducer saved a model; now evaluate it externally on val + holdout.
# The agent never sees these results.
echo ""
echo "=== POST-AGENT EVALUATION ==="
cd "$REPRODUCE_DIR"
if [ -f "model.pkl" ] || [ -f "model.pt" ]; then
    # Run prepare_reproduce.py as a script to trigger _evaluate_saved_model()
    if echo "$RUN_CMD" | grep -q "sbatch"; then
        # GPU experiment — need to run eval on GPU via sbatch
        cat > _eval.py << 'EVAL_EOF'
import sys, os, torch, ast, types
sys.path.insert(0, ".")
# Disable torch.compile so eval doesn't spend 30+ min compiling
_orig_compile = torch.compile
torch.compile = lambda model=None, *a, **kw: model if model is not None else (lambda fn: fn)
# Import train.py but prevent top-level training from running
if os.path.exists("train.py"):
    with open("train.py") as f:
        src = f.read()
    if '__name__' in src and '__main__' in src:
        import train
    else:
        # No __main__ guard — extract only definitions via AST
        tree = ast.parse(src)
        filtered = [n for n in tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Import, ast.ImportFrom))]
        new_tree = ast.Module(body=filtered, type_ignores=[])
        ast.fix_missing_locations(new_tree)
        train_mod = types.ModuleType("train")
        train_mod.__file__ = os.path.abspath("train.py")
        code = compile(new_tree, "train.py", "exec")
        exec(code, train_mod.__dict__)
        sys.modules['train'] = train_mod
    # Inject train module classes into __main__ so torch.load can find them
    # (models saved from train.py running as __main__ pickle classes as __main__.ClassName)
    train_mod = sys.modules.get('train')
    if train_mod:
        import __main__
        for attr in dir(train_mod):
            obj = getattr(train_mod, attr)
            if isinstance(obj, type) and not attr.startswith('_'):
                setattr(__main__, attr, obj)
from prepare_reproduce import _evaluate_saved_model
_evaluate_saved_model()
EVAL_EOF
        # Build eval submit script from experiment's submit.sh (copy SBATCH headers)
        {
            echo '#!/bin/bash'
            echo '#SBATCH --job-name=eval'
            echo '#SBATCH --output=eval.log'
            # Copy partition, nodes, GPU-related SBATCH lines from experiment submit.sh
            grep '#SBATCH.*--partition\|#SBATCH.*--gres\|#SBATCH.*--gpus\|#SBATCH.*--exclusive' "$EXPERIMENT_DIR/submit.sh" 2>/dev/null
            echo '#SBATCH --ntasks=1'
            echo '#SBATCH --time=01:00:00'
            echo ''
            echo 'export PATH="$HOME/.local/bin:$PATH"'
            echo 'uv run python _eval.py'
        } > _eval_submit.sh
        chmod +x _eval_submit.sh
        sbatch --wait _eval_submit.sh 2>&1 || echo "WARNING: eval sbatch failed (exit $?)"
        cat eval.log 2>/dev/null
        # Keep eval artifacts on failure for debugging
        if [ -f "holdout_results.tsv" ]; then
            rm -f _eval.py _eval_submit.sh eval.log
        else
            echo "WARNING: holdout_results.tsv not written. Keeping eval.log for debugging."
        fi
    else
        uv run python prepare_reproduce.py || echo "WARNING: eval failed (exit $?)"
    fi
    echo ""
    echo "Results:"
    cat holdout_results.tsv 2>/dev/null || echo "(no holdout_results.tsv)"
else
    echo "WARNING: No saved model found (model.pkl or model.pt)."
    echo "The reproducer may have called evaluate_val() directly (old interface)."
    echo "Check holdout_results.tsv for results."
    cat holdout_results.tsv 2>/dev/null
fi
