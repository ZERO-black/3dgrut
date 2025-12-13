#!/bin/bash
EXTRA_ARGS=${@:2} # any extra arguments to pass to the script

RESULT_DIR="temp"
CONFIG="cvm/erank-3dgrt-full.yaml"
SCENE="garden"


if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
    DATA_FACTOR=2
else
    DATA_FACTOR=4
fi

nvidia-smi > $RESULT_DIR/train_$SCENE.log
CUDA_VISIBLE_DEVICES=0 python train.py --config-name $CONFIG \
    use_wandb=True with_gui=False out_dir=$RESULT_DIR \
    path=data/mipnerf360/$SCENE experiment_name=$SCENE \
    dataset.downsample_factor=$DATA_FACTOR \
    $EXTRA_ARGS >> $RESULT_DIR/train_$SCENE.log