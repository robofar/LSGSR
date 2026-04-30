#pragma once

#include "gaussian.h"
#include "octree.h"

#include <array>
#include <cstdint>
#include <vector>

struct Camera {
    float R[3][3]; // w2c
    float t[3]; // w2c
    float position[3]; // c2w

    float fx, fy;
    float cx, cy;
    int image_width;
    int image_height;

    float near_plane;
    float far_plane;
};

struct Plane {
    float a, b, c, d;
};

struct Frustum {
    std::array<Plane, 6> planes; // left, right, bottom, top, near, far
};

struct GatherRange {
    uint8_t level;     // which level in GaussianLOD
    uint32_t offset;   // start index
    uint32_t count;    // number of gaussians
};

struct GatherResult {
    std::vector<GatherRange> ranges;
    size_t total_gaussians = 0;
    size_t leaf_gaussians = 0;
    size_t summary_gaussians = 0;
    size_t nodes_frustum_culled = 0;
    size_t nodes_lod_selected = 0;
};

Camera make_camera(
    const float R[3][3],
    const float t[3],
    float fx, float fy,
    float cx, float cy,
    int image_width,
    int image_height,
    float near_plane,
    float far_plane
);

Camera make_camera_lookat(
    const float position[3],
    const float target[3],
    const float world_up[3],
    float fx, float fy,
    float cx, float cy,
    int image_width,
    int image_height,
    float near_plane,
    float far_plane
);

void compute_view_matrix(const Camera& cam, float out[4][4]);
void compute_projection_matrix(const Camera& cam, float out[4][4]);
void multiply_4x4(const float A[4][4], const float B[4][4], float out[4][4]);

Frustum extract_frustum(const float VP[4][4]);

bool aabb_in_frustum(const Frustum& frustum,
                     float cx, float cy, float cz,
                     float half_size);

bool should_use_summary(const OctreeNode* node,
                       const Camera& cam,
                       float lod_threshold);

GatherResult gather_visible_gaussians(
    const OctreeNode* root,
    const GaussianLOD& lod,
    const Frustum& frustum,
    const Camera& cam,
    float lod_threshold
);

// Debug-only: walk the octree once and bucket every node by fate.
// `all` = every internal+leaf node in the tree.
// `frustum_passed` = nodes whose AABB survived the frustum cull (regardless of whether
//                    they were further recursed into or terminally selected).
// `selected` = the terminal nodes the gather ended up choosing (summary or leaf).
struct OctreeDebugTrace {
    std::vector<const OctreeNode*> all;
    std::vector<const OctreeNode*> frustum_passed;
    std::vector<const OctreeNode*> selected;
};

OctreeDebugTrace debug_gather_trace(
    const OctreeNode* root,
    const Frustum& frustum,
    const Camera& cam,
    float lod_threshold
);
