#!/bin/bash

cd $HOME

source ~/conda.sh

conda activate jupyter

which jupyter

jupyter lab \
  --ip=0.0.0.0 \
  --collaborative \
