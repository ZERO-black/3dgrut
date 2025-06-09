// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <3dgrt/pipelineDefinitions.h>
#include <3dgrt/optixTracer.h>

class OptixLoDTracer : public OptixTracer {

public:
    OptixLoDTracer(
        const std::string& path,
        const std::string& cuda_path,
        const std::string& pipeline,
        const std::string& backwardPipeline,
        const std::string& primitive,
        float particleKernelDegree,
        float particleKernelMinResponse,
        bool particleKernelDensityClamping,
        int particleRadianceSphDegree,
        bool enableNormals,
        bool enableHitCounts)
        : OptixTracer(path, cuda_path, pipeline, backwardPipeline, primitive, particleKernelDegree, particleKernelMinResponse, particleKernelDensityClamping, particleRadianceSphDegree, enableNormals, enableHitCounts, true) {
    }
    ~OptixLoDTracer() {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> trace(uint32_t frameNumber,
                  torch::Tensor rayToWorld,
                  torch::Tensor rayOri,
                  torch::Tensor rayDir,
                  torch::Tensor particleDensity,
                  torch::Tensor particleRadiance,
                  torch::Tensor particleLevels,
                  torch::Tensor particleExtraLevels,
                  uint32_t renderOpts,
                  int sphDegree,
                  float minTransmittance, float stdDist);


    std::tuple<torch::Tensor, torch::Tensor> traceBwd(uint32_t frameNumber,
                        torch::Tensor rayToWorld,
                        torch::Tensor rayOri,
                        torch::Tensor rayDir,
                        torch::Tensor rayRad,
                        torch::Tensor rayDns,
                        torch::Tensor rayHit,
                        torch::Tensor rayNrm,
                        torch::Tensor particleDensity,
                        torch::Tensor particleRadiance,
                        torch::Tensor lodMask,
                        torch::Tensor rayRadGrd,
                        torch::Tensor rayDnsGrd,
                        torch::Tensor rayHitGrd,
                        torch::Tensor rayNrmGrd,
                        uint32_t renderOpts,
                        int sphDegree,
                        float minTransmittance);
};
