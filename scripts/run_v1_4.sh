#!/bin/bash

set -e

VERSION="v1.4"
# VERSION="debug"

EXP_NAME="ours_$VERSION"
RESULT_DIR="./results/$VERSION"
CONFIG="./cvm/erank-3dgrt-full.yaml"

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="stump treehill"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    nvidia-smi > $RESULT_DIR/train_$SCENE.log
    CUDA_VISIBLE_DEVICES=0 python train.py --config-name $CONFIG \
        use_wandb=True with_gui=False out_dir=$RESULT_DIR \
        path=data/mipnerf360/$SCENE experiment_name=$SCENE \
        dataset.downsample_factor=$DATA_FACTOR \
        render.primitive_type=instances \
        wandb_project=$EXP_NAME\
        $EXTRA_ARGS >> $RESULT_DIR/train_$SCENE.log
    # CUDA_VISIBLE_DEVICES=0 python train.py --config-name $CONFIG \
    # use_wandb=False with_gui=False out_dir=$RESULT_DIR \
    # path=data/mipnerf360/$SCENE experiment_name=$SCENE \
    # dataset.downsample_factor=$DATA_FACTOR \
    # render.primitive_type=instances \
    $EXTRA_ARGS >> $RESULT_DIR/train_$SCENE.log
    python report_wandb.py --checkpoint $(find $RESULT_DIR/$SCENE -name ckpt_last.pt) --out-dir $RESULT_DIR/$SCENE/eval --scene $SCENE --method $EXP_NAME
    python report_erank.py --checkpoint $(find $RESULT_DIR/$SCENE -name ckpt_last.pt) --out-dir $RESULT_DIR/$SCENE/eval --scene $SCENE --method $EXP_NAME

done
