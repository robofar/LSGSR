#pragma once

#include "gaussian.h"

#include <cstdint>
#include <vector>

struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

AABB compute_bounding_box(const std::vector<Gaussian>& gaussians);
AABB make_bbox_cubic(const AABB& bbox);

std::vector<uint64_t> compute_morton_codes(const std::vector<Gaussian>& gaussians, const AABB& bbox);
void sort_gaussians_spatially(std::vector<Gaussian>& gaussians, std::vector<uint64_t>& morton_codes);