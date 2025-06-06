#pragma once

#include <optix.h>
#include <3dgrt/particleDensity.h>

void launchVisibilityKernel(
    const float* levels,
    const float* extraLevels,
    const ParticleDensity* particles,
    unsigned char* mask,
    int count,
    float3 eye,
    float std_dist);