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

import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.lod_model import MixtureOfGaussiansWithAnchor
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer


class Renderer:
    def __init__(
        self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None, compute_extra_metrics=True
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False, collate_fn=None)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            if conf.get("lod", False):
                ModelClass = MixtureOfGaussiansWithAnchor
            else:
                ModelClass = MixtureOfGaussians
            model = ModelClass(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint, False)
        model.build_acc()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
        )

    @classmethod
    def from_preloaded_model(
        cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None, compute_extra_metrics=False
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
        )

    @classmethod
    def from_ply(self, config_path, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True):
        def load_default_config():
            from hydra.compose import compose
            from hydra.initialize import initialize
            with initialize(version_base=None, config_path='../configs'):
                conf = compose(config_name=config_path)
            return conf

        global_step = 0

        conf = load_default_config()

        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, conf['out_dir'], experiment_name, use_wandb=False)

        if model is None:
            if (conf.get('lod', False)):
                ModelClass = MixtureOfGaussiansWithAnchor
            else:
                ModelClass = MixtureOfGaussians
            model = ModelClass(conf)
            model.init_from_ply(conf['initial_ply'], init_model=False)
        model.build_acc()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
        )
    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]} inference time: {outputs['frame_time_ms']}ms")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time

    @torch.no_grad()
    def render_from_saved_poses(self, save_dir="./camera_poses"):
        import glob, json
        import polyscope as ps
        from threedgrut_playground.utils.kaolin_future.conversions import (
            polyscope_to_kaolin_camera,
        )

        output_dir = os.path.join(
            self.out_dir, f"ours_{int(self.global_step)}", "saved_renders"
        )
        os.makedirs(output_dir, exist_ok=True)

        # 저장된 pose 파일 리스트
        pose_files = sorted(glob.glob(os.path.join(save_dir, "pose_*.pt")))
        inference_times = []

        # dataloader 길이와 pose 파일 개수가 같다고 가정
        for idx, (batch, pose_file) in enumerate(zip(self.dataloader, pose_files)):
            # 1) 원래 뱃치를 가져옴
            batch_dict = batch

            # 2) 저장된 pose 로드
            saved = torch.load(pose_file)
            view_json = saved["view_json"]
            w, h = saved["width"], saved["height"]

            # 3) Polyscope에 뷰 복원
            ps.set_view_from_json(view_json)
            ps.set_window_size(w, h)

            camera = polyscope_to_kaolin_camera(
                ps.get_view_camera_parameters(), w, h, device="cuda"
            )

            # 5) batch_dict["pose"]만 덮어쓰기 (shape: [1,4,4], float32)
            #    intr, data, mask 등 나머지는 그대로
            pose_mat = camera.extrinsics.inv_view_matrix()
            flip = torch.diag(
                torch.tensor(
                    [1.0, -1.0, -1.0, 1.0],
                    device=pose_mat.device,
                    dtype=pose_mat.dtype,
                )
            )
            pose_mat = pose_mat @ flip

            batch_dict["pose"] = pose_mat.unsqueeze(0).to(torch.float32).to("cuda")

            # 6) GPU로 옮기고 rays/origins/directions 등 세팅
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch_dict)

            # 7) 모델 추론
            outputs = self.model(gpu_batch)
            pred_rgb = outputs["pred_rgb"].squeeze(0)
            inference_times.append(outputs["frame_time_ms"])
            logger.info(f"min inference time: {inference_times[-1]}")

            # 8) 결과 이미지 저장
            torchvision.utils.save_image(
                pred_rgb.permute(2, 0, 1).clip(0, 1),
                os.path.join(output_dir, f"{idx:05d}.png"),
            )

        # 평균 추론 시간 로깅
        mean_time = sum(inference_times) / len(inference_times)
        logger.info(
            f"Rendered {len(pose_files)} frames, mean inference time: {mean_time:.2f} ms"
        )
