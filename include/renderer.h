#pragma once

#include "frustum_culling.h"
#include "octree.h"

#include <cstdint>
#include <vector>

struct RenderTimings {
    double soa_ms      = 0.0;
    double upload_ms   = 0.0;
    double forward_ms  = 0.0;
    double download_ms = 0.0;
    double pack_ms     = 0.0;
    double cleanup_ms  = 0.0;
    int    P           = 0;
};

struct RenderedImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb; // pixel-interleaved RGB, size = 3 * width * height
    RenderTimings timings;
};

RenderedImage render_frame(
    const GaussianLOD& lod,
    const GatherResult& gather,
    const Camera& cam
);
