module purge
module use /opt/software/gaia-external/IA/venv_AI/modulefiles/Core/
module load AI-6.0.0
export AI_VENV_PATH=$WORK/venv/transformer
module load venv/0.8.0+torch2.11.0+rocm7.2.0
source $AI_VENV_PATH/bin/activate
cd $WORK/proj/transformer
export PYTHONPATH=$PWD:$PYTHONPATH
python slurm/new_submit.py src.scripts.gpt2-test-large.train_lm --batch ... --variant ... -- <config> ...
