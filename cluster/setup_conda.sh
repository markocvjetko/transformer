#!/bin/bash
# Register env-var overrides for data/experiment paths on the cluster.
# These activate/deactivate automatically with `conda activate transformer`.
#
# Usage:
#   1. Update CONDA_ENV, DATA, and EXPERIMENTS below to match your setup.
#   2. Run this script once after creating the conda env.
#
# See src/utils/paths.py for how these variables are consumed.

CONDA_ENV="$CONDA_PREFIX"  # run with the target env active, or set manually
DATA="/path/to/your/data"
EXPERIMENTS="/path/to/your/experiments"

if [ -z "$CONDA_ENV" ]; then
    echo "Error: No conda env active. Activate one first or set CONDA_ENV manually."
    exit 1
fi

mkdir -p "$CONDA_ENV/etc/conda/activate.d"
mkdir -p "$CONDA_ENV/etc/conda/deactivate.d"

cat > "$CONDA_ENV/etc/conda/activate.d/paths.sh" << EOF
export TRANSFORMER_DATA_DIR="$DATA"
export TRANSFORMER_EXPERIMENTS_DIR="$EXPERIMENTS"
EOF

cat > "$CONDA_ENV/etc/conda/deactivate.d/paths.sh" << 'EOF'
unset TRANSFORMER_DATA_DIR
unset TRANSFORMER_EXPERIMENTS_DIR
EOF