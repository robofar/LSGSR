#include "morton.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

AABB compute_bounding_box(const std::vector<Gaussian>& gaussians) {
    const size_t n = gaussians.size();
    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t chunk = (n + num_threads - 1) / num_threads;

    std::vector<AABB> local_boxes(num_threads);
    std::vector<std::thread> threads;

    for (unsigned t = 0; t < num_threads; t++) {
        const size_t start = t * chunk;
        const size_t end = std::min(start + chunk, n);

        threads.emplace_back([&gaussians, &local_boxes, t, start, end]() {
            float lmin_x = std::numeric_limits<float>::max();
            float lmin_y = std::numeric_limits<float>::max();
            float lmin_z = std::numeric_limits<float>::max();
            float lmax_x = std::numeric_limits<float>::lowest();
            float lmax_y = std::numeric_limits<float>::lowest();
            float lmax_z = std::numeric_limits<float>::lowest();

            for (size_t i = start; i < end; i++) {
                const Gaussian& g = gaussians[i];
                lmin_x = std::min(lmin_x, g.x);
                lmin_y = std::min(lmin_y, g.y);
                lmin_z = std::min(lmin_z, g.z);
                lmax_x = std::max(lmax_x, g.x);
                lmax_y = std::max(lmax_y, g.y);
                lmax_z = std::max(lmax_z, g.z);
            }

            local_boxes[t] = {lmin_x, lmin_y, lmin_z, lmax_x, lmax_y, lmax_z};
        });
    }

    for (auto& th : threads) th.join();

    // Reduce
    AABB bbox = local_boxes[0];
    for (size_t t = 1; t < threads.size(); t++) {
        bbox.min_x = std::min(bbox.min_x, local_boxes[t].min_x);
        bbox.min_y = std::min(bbox.min_y, local_boxes[t].min_y);
        bbox.min_z = std::min(bbox.min_z, local_boxes[t].min_z);
        bbox.max_x = std::max(bbox.max_x, local_boxes[t].max_x);
        bbox.max_y = std::max(bbox.max_y, local_boxes[t].max_y);
        bbox.max_z = std::max(bbox.max_z, local_boxes[t].max_z);
    }

    // Epsilon padding so no point sits exactly on max boundary
    constexpr float eps = 1e-5f;
    bbox.max_x += eps;
    bbox.max_y += eps;
    bbox.max_z += eps;

    return bbox;
}

AABB make_bbox_cubic(const AABB& bbox) {
    float cx = (bbox.min_x + bbox.max_x) * 0.5f;
    float cy = (bbox.min_y + bbox.max_y) * 0.5f;
    float cz = (bbox.min_z + bbox.max_z) * 0.5f;

    float half = std::max({bbox.max_x - bbox.min_x,
                           bbox.max_y - bbox.min_y,
                           bbox.max_z - bbox.min_z}) * 0.5f;

    return {cx - half, cy - half, cz - half,
            cx + half, cy + half, cz + half};
}

// --- Morton code helpers ---

// Spread 21 bits across 63 bit positions (2 zero-gaps between each bit)
static uint64_t expand_bits(uint64_t v) {
    v &= 0x1FFFFF; // keep only 21 bits
    v = (v | (v << 32)) & 0x1F00000000FFFF;
    v = (v | (v << 16)) & 0x1F0000FF0000FF;
    v = (v | (v <<  8)) & 0x100F00F00F00F00F;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3;
    v = (v | (v <<  2)) & 0x1249249249249249;
    return v;
}

static uint64_t morton_code(float x, float y, float z, const AABB& bbox) {
    // Normalize to [0, 1]
    float nx = (x - bbox.min_x) / (bbox.max_x - bbox.min_x);
    float ny = (y - bbox.min_y) / (bbox.max_y - bbox.min_y);
    float nz = (z - bbox.min_z) / (bbox.max_z - bbox.min_z);

    // Quantize to 21-bit integer
    constexpr uint64_t max_val = (1 << 21) - 1; // 2097152 - 1 = 2097151
    uint64_t ix = static_cast<uint64_t>(nx * max_val); // [0, 2097151]
    uint64_t iy = static_cast<uint64_t>(ny * max_val); // [0, 2097151]
    uint64_t iz = static_cast<uint64_t>(nz * max_val); // [0, 2097151]

    // Interleave bits. Result: z20y20x20...z1y1x1z0y0x0 (63 bits)
    return expand_bits(ix) | (expand_bits(iy) << 1) | (expand_bits(iz) << 2);
}

std::vector<uint64_t> compute_morton_codes(const std::vector<Gaussian>& gaussians, const AABB& bbox) {
    const size_t n = gaussians.size();
    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t chunk = n / num_threads;

    std::cout << "Computing Morton codes..." << std::endl;
    std::vector<uint64_t> codes(n);
    {
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < num_threads; t++) {
            const size_t start = t * chunk;
            const size_t end = std::min(start + chunk, n);

            threads.emplace_back([&gaussians, &codes, &bbox, start, end]() {
                for (size_t i = start; i < end; i++) {
                    const Gaussian& g = gaussians[i];
                    codes[i] = morton_code(g.x, g.y, g.z, bbox);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    return codes;
}


void sort_gaussians_spatially(std::vector<Gaussian>& gaussians, std::vector<uint64_t>& morton_codes) {
    const size_t n = gaussians.size();

    std::cout << "Sorting Gaussians spatially, based on their Morton codes ..." << std::endl;

    // Build (morton, index) pairs
    std::vector<std::pair<uint64_t, uint32_t>> morton_pairs(n);
    for (size_t i = 0; i < n; i++) {
        morton_pairs[i] = {morton_codes[i], static_cast<uint32_t>(i)};
    }

    // Sort by morton code
    std::sort(morton_pairs.begin(), morton_pairs.end());

    // Reorder both gaussians and morton codes
    std::vector<Gaussian> sorted_g(n);
    std::vector<uint64_t> sorted_m(n);
    for (size_t i = 0; i < n; i++) {
        sorted_g[i] = gaussians[morton_pairs[i].second];
        sorted_m[i] = morton_pairs[i].first;
    }

    gaussians = std::move(sorted_g);
    morton_codes = std::move(sorted_m);
}
