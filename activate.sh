#!/bin/bash
export REPO_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# curl --max-time 1 http://1.1.1.1

# # If not internet proxy it
# if [ $? -ne 0 ]
# then
# echo 'Setting up proxy...'
# source $REPO_DIR/proxy_activate.sh
# # else
# # <other commands>
# fi

module load rust/1.53.0 python/3.8 StdEnv/2020 gcc/9.3.0 arrow/8.0.0 cuda/11.4

if [ -n "$SLURM_TMPDIR" ]; then
  MEM_DIR="$SLURM_TMPDIR"
else
  MEM_DIR=/dev/shm/vonderoh
fi

# MEM_DIR=/dev/shm/vonderoh

mkdir -p $MEM_DIR

if [ ! -d "$MEM_DIR/EG" ]; then
  time lz4 -d /scratch/vonderoh/env/EG.tar.lz4 -c | tar xf - -C $MEM_DIR/
fi

alias python3="$MEM_DIR/EG/bin/python3.8"
alias pip="$MEM_DIR/EG/bin/python3.8 -m pip"
alias pip3="$MEM_DIR/EG/bin/python3.8 -m pip"

# ! Stop module from messing with deps
export PYTHONPATH=""

# pip install -r $REPO_DIR/requirements.txt

mkdir -p /scratch/vonderoh/datasets/
mkdir -p /scratch/vonderoh/transformers-cache/

export HF_DATASETS_CACHE=/scratch/vonderoh/datasets/
export TRANSFORMERS_CACHE=/scratch/vonderoh/transformers-cache/
export XDG_CACHE_HOME=/scratch/vonderoh/.cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline