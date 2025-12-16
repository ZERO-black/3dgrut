# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -e


RESULT_DIR="results/mipnerf360-instances"
if [[ -z $RESULT_DIR ]]; then
    echo "Error: Result directory is not provided. Aborting execution."
    echo "Usage: $0 <result-directory>"
    exit 1
fi

CONFIG="paper/3dgrt/colmap_ours.yaml"

SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
# SCENE_LIST="garden kitchen room treehill flowers"

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
        $EXTRA_ARGS >> $RESULT_DIR/train_$SCENE.log
    python report_wandb.py --checkpoint $(find $RESULT_DIR/$SCENE -name ckpt_last.pt) --out-dir $RESULT_DIR/$SCENE/eval --scene $SCENE --method "3dgrt"
    python report_erank.py --checkpoint $(find $RESULT_DIR/$SCENE -name ckpt_last.pt) --out-dir $RESULT_DIR/$SCENE/eval --scene $SCENE --method "3dgrt"

done

# To grep results from log files, run the following command:
# grep "Test Metrics"        -A 5 train_*.log | awk 'NR % 7 == 5'
