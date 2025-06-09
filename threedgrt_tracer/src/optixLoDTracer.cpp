

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgrt/cuoptixMacros.h>
#include <3dgrt/optixLoDTracer.h>
#include <3dgrt/particlePrimitives.h>
#include <3dgrt/visibilityKernel.h>
#include <3dgrt/pipelineParameters.h>
#include <3dgrt/tensorBuffering.h>
#include <3dgrt/particleDensity.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <nvrtc.h>
#include <optix.h>

#include <chrono>
#include <iostream>

//------------------------------------------------------------------------------
// OptixTracer
//------------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OptixLoDTracer::trace(uint32_t frameNumber,
                                                                                                                                          torch::Tensor rayToWorld,
                                                                                                                                          torch::Tensor rayOri,
                                                                                                                                          torch::Tensor rayDir,
                                                                                                                                          torch::Tensor particleDensity,
                                                                                                                                          torch::Tensor particleRadiance,
                                                                                                                                          torch::Tensor particleLevels,
                                                                                                                                          torch::Tensor particleExtraLevels,
                                                                                                                                          uint32_t renderOpts,
                                                                                                                                          int sphDegree,
                                                                                                                                          float minTransmittance, float std_dist) {

                                                                                                                                            const torch::TensorOptions opts  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rayRad             = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayDns             = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayHit             = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 2}, opts);
    torch::Tensor rayNrm             = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayHitsCount       = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor particleVisibility = torch::zeros({particleDensity.size(0), 1}, opts);

    PipelineParameters paramsHost;

    std::chrono::high_resolution_clock::time_point frame_start = std::chrono::high_resolution_clock::now();
    torch::Tensor lodMask                                      = torch::empty({particleDensity.size(0)}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    static double total_time                                   = 0.0;
    static int frame_count                                     = 0;
    float host_eye[3]                                          = {0, 0, 0};
    CUDA_CHECK(cudaMemcpy(
        host_eye,
        rayOri.data_ptr<float>(), // origin x,y,z
        3 * sizeof(float),
        cudaMemcpyDeviceToHost));

    float3 cam_center = make_float3(
        host_eye[0], host_eye[1], host_eye[2]);

    // 4. execute Visibility kernel
    launchVisibilityKernel(
        /* lods          */ particleLevels.data_ptr<float>(),
        /* extra_levels  */ particleExtraLevels.data_ptr<float>(),
        /* gPos          */ getPtr<const ParticleDensity>(particleDensity),
        /* mask          */ reinterpret_cast<unsigned char*>(lodMask.data_ptr<uint8_t>()),
        /* count         */ static_cast<int>(particleDensity.size(0)),
        /* eye           */ cam_center,
        /* standard_dist */ std_dist);

    paramsHost.lodMask = reinterpret_cast<const unsigned char*>(lodMask.data_ptr<uint8_t>());

    auto frame_end     = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    total_time += duration_ms;
    frame_count++;

    double avg_time = total_time / frame_count;
    // std::cout << "[Frame " << frame_count << "] LOD mask generation time: "
    //           << duration_ms << " ms, average: " << avg_time << " ms\n";

    paramsHost.handle = _state->gasHandle;
    paramsHost.aabb   = _state->gasAABB;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber   = frameNumber;
    paramsHost.gPrimNumTri   = _state->gPrimNumTri;

    paramsHost.minTransmittance       = minTransmittance;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold      = 1.0f / 255.0f;
    paramsHost.sphDegree              = sphDegree;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));
    paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    paramsHost.particleDensity      = getPtr<const ParticleDensity>(particleDensity);
    paramsHost.particleRadiance     = getPtr<const float>(particleRadiance);
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);
    paramsHost.particleVisibility   = getPtr<int32_t>(particleVisibility);

    paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);
    paramsHost.rayHitsCount   = packed_accessor32<float, 4>(rayHitsCount);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();
    reallocateParamsDevice(sizeof(paramsHost), cudaStream);

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    OPTIX_CHECK(optixLaunch(_state->pipelineTracingFwd, cudaStream, _state->paramsDevice,
                            sizeof(PipelineParameters), &_state->sbtTracingFwd, rayRad.size(2),
                            rayRad.size(1), rayRad.size(0)));

    CUDA_CHECK_LAST();

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(rayRad, rayDns, rayHit, rayNrm, rayHitsCount, particleVisibility, lodMask);
}

std::tuple<torch::Tensor, torch::Tensor>
OptixLoDTracer::traceBwd(uint32_t frameNumber,
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
                         float minTransmittance) {

    const torch::TensorOptions opts    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor particleDensityGrad  = torch::zeros({particleDensity.size(0), particleDensity.size(1)}, opts);
    torch::Tensor particleRadianceGrad = torch::zeros({particleRadiance.size(0), particleRadiance.size(1)}, opts);

    PipelineBackwardParameters paramsHost;
    paramsHost.handle = _state->gasHandle;
    paramsHost.aabb   = _state->gasAABB;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber   = frameNumber;
    paramsHost.gPrimNumTri   = _state->gPrimNumTri;

    paramsHost.minTransmittance       = minTransmittance;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold      = 1.0f / 255.0f;
    paramsHost.sphDegree              = sphDegree;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));
    paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    paramsHost.particleDensity      = getPtr<const ParticleDensity>(particleDensity);
    paramsHost.particleRadiance     = getPtr<const float>(particleRadiance);
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);

    paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);

    paramsHost.particleDensityGrad  = getPtr<ParticleDensity>(particleDensityGrad);
    paramsHost.particleRadianceGrad = getPtr<float>(particleRadianceGrad);

    paramsHost.rayRadianceGrad    = packed_accessor32<float, 4>(rayRadGrd);
    paramsHost.rayDensityGrad     = packed_accessor32<float, 4>(rayDnsGrd);
    paramsHost.rayHitDistanceGrad = packed_accessor32<float, 4>(rayHitGrd);
    paramsHost.rayNormalGrad      = packed_accessor32<float, 4>(rayNrmGrd);

    paramsHost.lodMask = reinterpret_cast<const unsigned char*>(lodMask.data_ptr<uint8_t>());

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    OPTIX_CHECK(optixLaunch(_state->pipelineTracingBwd, cudaStream, _state->paramsDevice,
                            sizeof(PipelineBackwardParameters), &_state->sbtTracingBwd,
                            rayRad.size(2), rayRad.size(1), rayRad.size(0)));

    return std::tuple<torch::Tensor, torch::Tensor>(particleDensityGrad, particleRadianceGrad);
}