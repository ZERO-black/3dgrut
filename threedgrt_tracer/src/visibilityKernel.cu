#include <cuda_runtime.h>
#include <3dgrt/mathUtils.h>
#include <ATen/cuda/CUDAContext.h>  // for at::cuda::getCurrentCUDAStream()
#include <3dgrt/visibilityKernel.h> // declare computeVisibilityKernel()
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include <3dgrt/particleDensity.h>

constexpr size_t stride = sizeof(ParticleDensity) / sizeof(float);
__global__ void computeVisibilityKernel(
    const float* levels,
    const float* extraLevels,
    const ParticleDensity* particles,
    unsigned char* mask,
    int count,
    float3 eye,
    float std_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    float3 anchor   = particles[idx].position;
    float dist      = length(anchor - eye);
    float predLevel = log2f(std_dist / dist) + extraLevels[idx];
    mask[idx]       = (levels[idx] <= predLevel) ? 1 : 0;
}

inline uint32_t div_round_up(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

void launchVisibilityKernel(
    const float* levels,
    const float* extraLevels,
    const ParticleDensity* particles,
    unsigned char* mask,
    int count,
    float3 eye,
    float std_dist) {
    // 1. CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 2. Grid/block dimensions
    constexpr uint32_t threads = 1024;
    uint32_t blocks            = div_round_up(static_cast<uint32_t>(count), threads);

    // 3. Launch the visibility kernel
    computeVisibilityKernel<<<blocks, threads, 0, stream>>>(
        levels,
        extraLevels,
        particles,
        mask,
        count,
        eye,
        std_dist);

    // 4. Error check and synchronize
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);
}
