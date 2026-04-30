#include "gaussian.h"

#include <algorithm>
#include <cmath>
#include <iostream>

void GaussianCloud::filter_gaussians(float min_opacity, float sigma_threshold) {
    std::cout << "Filtering Gaussians based on opacity and outlier removal..." << std::endl;
    const size_t original_size = this->gaussians.size();

    // Step 1: Filter by opacity
    auto opacity_end = std::remove_if(this->gaussians.begin(), this->gaussians.end(),
        [min_opacity](const Gaussian& g) {
            return g.opacity < min_opacity;
        }
    );
    size_t removed_opacity = std::distance(opacity_end, this->gaussians.end());
    this->gaussians.erase(opacity_end, this->gaussians.end());
    std::cout << "Opacity filter: removed " << removed_opacity << " Gaussians (opacity < " << min_opacity << ")" << std::endl;

    // Step 2: Compute center of mass
    const size_t n = this->gaussians.size();
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (const auto& g : this->gaussians) {
        cx += g.x;
        cy += g.y;
        cz += g.z;
    }
    cx /= n;
    cy /= n;
    cz /= n;

    // Step 3: Compute standard deviation of distances from center
    double dist_sum_sq = 0.0;
    for (const auto& g : gaussians) {
        double dx = g.x - cx;
        double dy = g.y - cy;
        double dz = g.z - cz;
        dist_sum_sq += dx * dx + dy * dy + dz * dz;
    }
    double sigma = std::sqrt(dist_sum_sq / n);
    double max_dist = sigma_threshold * sigma;

    // Step 4: Filter by distance
    auto dist_end = std::remove_if(gaussians.begin(), gaussians.end(),
        [cx, cy, cz, max_dist](const Gaussian& g) {
            double dx = g.x - cx;
            double dy = g.y - cy;
            double dz = g.z - cz;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            return dist > max_dist;
        }
    );
    size_t removed_distance = std::distance(dist_end, gaussians.end());
    gaussians.erase(dist_end, gaussians.end());

    std::cout << "Distance filter: removed " << removed_distance << " Gaussians (> " << sigma_threshold << " sigma)" << std::endl;
    std::cout << "Remaining Gaussians: " << gaussians.size() << " / " << original_size << std::endl;
}
