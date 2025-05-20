#pragma once

#include <optix.h>

void launchVisibilityKernel(
    const float* lods,
    const float* extra_levels,
    const float3* gPos,
    unsigned char* mask,
    int count,
    float3 eye,
    float std_dist);