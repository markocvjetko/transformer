module purge
module use /opt/software/gaia-external/IA/venv_AI/modulefiles/Core/
module load AI-6.0.0
export AI_VENV_PATH=$WORK/venv/transformer
module load venv/0.8.0+torch2.11.0+rocm7.2.0     # creates the venv on first load
source $AI_VENV_PATH/bin/activate
python -c "import torch; print(f'torch=={torch.__version__}')" > $AI_VENV_PATH/constraints.txt
pip install -c $AI_VENV_PATH/constraints.txt lightning sacrebleu tensorboard wandb submitit \
    lxml langid omegaconf randomname translate-toolkit optuna-integration
