#!/bin/bash




mkdir -p /linkhome/rech/genluq01/uix29qp/.conda/envs/transformer/etc/conda/activate.d
mkdir -p /linkhome/rech/genluq01/uix29qp/.conda/envs/transformer/etc/conda/deactivate.d

cat > /linkhome/rech/genluq01/uix29qp/.conda/envs/transformer/etc/conda/activate.d/paths.sh << 'EOF'
export TRANSFORMER_DATA_DIR=/lustre/fsn1/projects/rech/imi/uix29qp/proj/transformer/data
export TRANSFORMER_EXPERIMENTS_DIR=/lustre/fsn1/projects/rech/imi/uix29qp/proj/transformer/experiments
EOF

cat > /linkhome/rech/genluq01/uix29qp/.conda/envs/transformer/etc/conda/deactivate.d/paths.sh << 'EOF'
unset TRANSFORMER_DATA_DIR
unset TRANSFORMER_EXPERIMENTS_DIR
EOF