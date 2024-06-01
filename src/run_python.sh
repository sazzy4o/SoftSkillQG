#!/bin/bash
shopt -s expand_aliases
export REPO_DIR=/scratch/vonderoh/EG-uni-cedar
source $REPO_DIR/activate.sh
time python3 $@