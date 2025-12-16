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

import argparse

from threedgrut.render import Renderer
import wandb
from torch.utils.tensorboard.writer import SummaryWriter
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.erank import get_effective_rank
from datetime import datetime
import numpy as np
import torch

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument(
        "--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt"
    )
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--iteration", type=int, default=30000)

    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    run_name = f"{args.scene}-hist-" + timestamp

    wandb.login()
    wandb.init(
        project="mipnerf360",
        tags=[args.scene],
        group=args.method,
        name=run_name,
        config={
            "scene": args.scene,
            "method": args.method,
        }
    )
    wandb.tensorboard.patch(root_logdir=args.out_dir, save=False)

    writer = SummaryWriter(log_dir=args.out_dir)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    # overrides
    if conf["render"]["method"] == "3dgrt":
        conf["render"]["particle_kernel_density_clamping"] = True
        conf["render"]["min_transmittance"] = 0.03
    conf["render"]["enable_kernel_timings"] = True
    # Initialize the model and the optix context
    model = MixtureOfGaussians(conf)
    # Initialize the parameters from checkpoint
    model.init_from_checkpoint(checkpoint)

    name = f"erank_{args.iteration}" 
    values = get_effective_rank(model.get_scale()).detach().cpu().numpy()
    hist, _ = np.histogram(values, bins=50, range=(1,3))
    for idx, item in enumerate(hist.tolist()):
        wandb.log({name: item})
    wandb.run.summary["erank_mean"] = float(values.mean())
    wandb.run.summary["erank_std"] = float(values.std())
    wandb.run.summary["num"] = int(model.num_gaussians)
    wandb.finish()