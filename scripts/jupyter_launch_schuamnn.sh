#!/bin/bash

cd $HOME

source ~/conda.sh

conda activate jupyter

which jupyter

jupyter lab \
  --ip=0.0.0.0 \
  --collaborative \
  --SQLiteYStore.db_path="$HOME/data1/.cache/jupyter/.jupyter_ystore.db"
