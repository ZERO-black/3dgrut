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


CONFIG=$1
if [[ -z $CONFIG ]]; then
    echo "Error: Configuration is not provided. Aborting execution."
    echo "Usage: $0 <config-yaml>"
    exit 1
fi

RESULT_DIR=${RESULT_DIR:-"results/mipnerf360"}
EXTRA_ARGS=${@:2} # any extra arguments to pass to the script

# if the result directory already exists, warn user and aport execution
# if [ -d "$RESULT_DIR" ]; then
#     echo "Result directory $RESULT_DIR already exists. Aborting execution."
#     exit 1
# fi

mkdir -p $RESULT_DIR
export TORCH_EXTENSIONS_DIR=$RESULT_DIR/.cache

# SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENE_LIST="kitchen room stump"


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
        use_wandb=False with_gui=False out_dir=$RESULT_DIR \
        path=data/mipnerf360/$SCENE experiment_name=$SCENE \
        dataset.downsample_factor=$DATA_FACTOR \
        $EXTRA_ARGS >> $RESULT_DIR/train_$SCENE.log

done
