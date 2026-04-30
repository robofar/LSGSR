#include "ply_loader.h"
#include "happly.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// --- Activation functions ---

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static std::array<float, 3> exp_activate(const std::array<float, 3>& raw) {
    return {std::exp(raw[0]), std::exp(raw[1]), std::exp(raw[2])};
}

static std::array<float, 4> normalize_quaternion(const std::array<float, 4>& raw) {
    float norm = std::sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]);
    float inv_norm = 1.0f / norm;
    return {raw[0] * inv_norm, raw[1] * inv_norm, raw[2] * inv_norm, raw[3] * inv_norm};
}

// --- Helpers ---

static int detect_sh_rest_count(happly::Element& vertex) {
    int count = 0;
    for (const auto& name : vertex.getPropertyNames()) {
        if (name.rfind("f_rest_", 0) == 0) count++;
    }
    return count;
}

static int sh_rest_count_to_degree(int rest_count) {
    // rest = 3 * ((deg+1)^2 - 1)
    if (rest_count == 0)  return 0;
    if (rest_count == 9)  return 1;
    if (rest_count == 24) return 2;
    if (rest_count == 45) return 3;
    throw std::runtime_error("Unexpected f_rest count: " + std::to_string(rest_count) + ". Expected 0, 9, 24, or 45.");
}

// --- Public API ---

GaussianCloud load_ply(const std::string& filepath) {
    std::cout << "Loading PLY: " << filepath << std::endl;

    happly::PLYData ply(filepath);
    auto& vertex = ply.getElement("vertex");
    const size_t n = vertex.count;

    // Detect SH degree
    int rest_count = detect_sh_rest_count(vertex);
    int sh_degree = sh_rest_count_to_degree(rest_count);

    auto px = vertex.getProperty<float>("x");
    auto py = vertex.getProperty<float>("y");
    auto pz = vertex.getProperty<float>("z");

    auto opacity_raw = vertex.getProperty<float>("opacity");

    auto s0 = vertex.getProperty<float>("scale_0");
    auto s1 = vertex.getProperty<float>("scale_1");
    auto s2 = vertex.getProperty<float>("scale_2");

    auto r0 = vertex.getProperty<float>("rot_0");
    auto r1 = vertex.getProperty<float>("rot_1");
    auto r2 = vertex.getProperty<float>("rot_2");
    auto r3 = vertex.getProperty<float>("rot_3");

    auto dc0 = vertex.getProperty<float>("f_dc_0");
    auto dc1 = vertex.getProperty<float>("f_dc_1");
    auto dc2 = vertex.getProperty<float>("f_dc_2");

    std::vector<std::vector<float>> f_rest(rest_count);
    for (int i = 0; i < rest_count; i++) {
        f_rest[i] = vertex.getProperty<float>("f_rest_" + std::to_string(i));
    }

    // Build Gaussian vector
    GaussianCloud cloud;
    cloud.sh_degree = sh_degree;
    cloud.sh_rest_count = rest_count;
    cloud.gaussians.resize(n);
    std::cout << "Loading Gaussians..." << std::endl;

    const size_t progress_step = n / 10;

    for (size_t i = 0; i < n; i++) {
        if (progress_step > 0 && i % progress_step == 0) {
            std::cout << "  " << (i * 100) / n << "%\r" << std::flush;
        }

        Gaussian& g = cloud.gaussians[i];

        // Position — direct
        g.x = px[i];
        g.y = py[i];
        g.z = pz[i];

        // Opacity — sigmoid activation
        g.opacity = sigmoid(opacity_raw[i]);

        // Scale — exp activation
        g.scale = exp_activate({s0[i], s1[i], s2[i]});

        // Rotation — normalize quaternion (w, x, y, z)
        g.rotation = normalize_quaternion({r0[i], r1[i], r2[i], r3[i]});

        // SH DC — direct
        g.sh_dc = {dc0[i], dc1[i], dc2[i]};

        // SH rest — direct copy, zero-fill remainder
        g.sh_rest.fill(0.0f);
        for (int j = 0; j < rest_count; j++) {
            g.sh_rest[j] = f_rest[j][i];
        }
    }

    std::cout << "  100%" << std::endl;
    std::cout << "Total Gaussians: " << n << std::endl;
    std::cout << "Detected SH degree: " << cloud.sh_degree << " (" << cloud.sh_rest_count << " rest coefficients)" << std::endl;

    return cloud;
}

