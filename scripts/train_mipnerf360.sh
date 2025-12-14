#!/bin/bash

set -e


CONFIG="cvm/erank-3dgrt.yaml"
if [[ -z $CONFIG ]]; then
    echo "Error: Configuration is not provided. Aborting execution."
    echo "Usage: $0 <config-yaml>"
    exit 1
fi

EXP_NAME="3dgrt_erank"
VERSION="v1.2"

RESULT_DIR=${RESULT_DIR:-"results/$VERSION"}
EXTRA_ARGS=${@:2} # any extra arguments to pass to the script

# if the result directory already exists, warn user and aport execution
# if [ -d "$RESULT_DIR" ]; then
#     echo "Result directory $RESULT_DIR already exists. Aborting execution."
#     exit 1
# fi

mkdir -p $RESULT_DIR
export TORCH_EXTENSIONS_DIR=$RESULT_DIR/.cache

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="counter flowers garden kitchen room stump treehill bonsai"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running: $SCENE, Configuration: $CONFIG"

    # train without eval
    nvidia-smi > $RESULT_DIR/train_$SCENE.log
    CUDA_VISIBLE_DEVICES=0 python train.py --config-name $CONFIG \
        use_wandb=True with_gui=False out_dir=$RESULT_DIR \
        path=data/mipnerf360/$SCENE experiment_name=$SCENE \
        dataset.downsample_factor=$DATA_FACTOR \
        wandb_project="$EXP_NAME_$VERSION" \
        $EXTRA_ARGS >> $RESULT_DIR/train_$SCENE.log

done
