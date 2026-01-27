#!/bin/bash -l
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# TODO: adjust paths according to your setup
cd $HOME/reading-between-tokens

module purge
module load cuda/12.6.2
module load openmpi/4.1.7-gcc11.4.1-cuda
module load python/3.12-conda

export STORAGE_DIR="./"
export TMPDIR=${SLURM_TMPDIR:-/tmp}

export HF_HOME="$TMPDIR/huggingface"
export TORCH_HOME="$TMPDIR/torch"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p $HF_HOME $TORCH_HOME $TRITON_CACHE_DIR

if ! conda env list | awk '{print $1}' | grep -qx "reading-between-tokens"; then
  conda env create -f conda_env.yml
fi
conda activate reading-between-tokens

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# sanity checks
python - <<'EOF'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
EOF

deepspeed ./src/run.py --path_dir $STORAGE_DIR
