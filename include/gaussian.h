#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Maximum SH rest coefficients (SH degree 3 = 45)
inline constexpr int MAX_SH_REST = 45;

struct Gaussian {
    float x, y, z;                        // position (3)
    float opacity;                        // activated opacity in (0,1)
    std::array<float, 3> scale;           // activated scales, positive
    std::array<float, 4> rotation;        // normalized quaternion wxyz
    std::array<float, 3> sh_dc;           // SH degree 0, RGB
    std::array<float, MAX_SH_REST> sh_rest; // SH higher degrees (up to degree 3)
};

class GaussianCloud {
public:
    std::vector<Gaussian> gaussians;
    int sh_degree;    // detected SH degree (0, 1, 2, or 3)
    int sh_rest_count; // number of valid sh_rest coefficients per Gaussian

    void filter_gaussians(float min_opacity = 0.01f, float sigma_threshold = 3.0f);
};
