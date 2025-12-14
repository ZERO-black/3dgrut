#!/bin/bash

set -e

VER="v1.2"
RESULT_DIR="./results/$VER"
if [[ -z $RESULT_DIR ]]; then
    echo "Error: Result directory is not provided. Aborting execution."
    echo "Usage: $0 <result-directory>"
    exit 1
fi

SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
# SCENE_LIST="bicycle"

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    python report_wandb.py --checkpoint $(find $RESULT_DIR/$SCENE -name ckpt_last.pt) --out-dir $RESULT_DIR/$SCENE/eval --scene $SCENE --method "ours_$VER"

done
