#pragma once

#include "gaussian.h"
#include "morton.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

inline constexpr uint32_t MAX_LEAF_SIZE = 256;

struct OctreeNode {
    float center_x, center_y, center_z;
    float half_size;

    uint8_t depth;
    bool is_leaf;
    
    uint32_t gauss_offset;
    uint32_t gauss_count;

    std::array<OctreeNode*, 8> children;

    OctreeNode();
};

struct GaussianLOD {
    std::vector<std::vector<Gaussian>> levels; // levels[max_depth] = original, levels[0..max_depth-1] = summaries
    std::array<std::mutex, 22> level_mutexes;   // one per possible depth level
    int max_depth;
    int sh_degree;
    int sh_rest_count;
};

AABB child_aabb(const AABB& parent, int child_index);

OctreeNode* build_octree_parallel(
    const std::vector<uint64_t>& morton_codes,
    const AABB& scene_bbox,
    uint8_t max_depth,
    GaussianLOD& lod
);

void print_stats(const OctreeNode* root, const GaussianLOD& lod);
void delete_octree(OctreeNode* root);
