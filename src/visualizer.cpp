#include "visualizer.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kSH_C0 = 0.28209479177387814f;

uint8_t to_u8(float v) {
    v = std::min(std::max(v, 0.0f), 1.0f);
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

rerun::Color depth_color(int depth, int max_depth) {
    const float t = (max_depth > 0) ? static_cast<float>(depth) / static_cast<float>(max_depth) : 0.0f;
    return rerun::Color(to_u8(1.0f - t), 50, to_u8(t));
}

}  // namespace

Visualizer::Visualizer(const std::string& app_name) : rec(app_name) {
    rec.spawn().exit_on_failure();
}

void Visualizer::log_point_cloud(const std::string& path,
                                 const std::vector<Gaussian>& gaussians,
                                 int stride) {
    if (gaussians.empty()) return;
    stride = std::max(1, stride);
    const size_t n_out = (gaussians.size() + stride - 1) / stride;

    std::vector<rerun::Position3D> positions;
    std::vector<rerun::Color> colors;
    positions.reserve(n_out);
    colors.reserve(n_out);
    for (size_t i = 0; i < gaussians.size(); i += stride) {
        const Gaussian& g = gaussians[i];
        positions.push_back(std::array<float, 3>{g.x, g.y, g.z});
        colors.push_back(rerun::Color(
            to_u8(0.5f + kSH_C0 * g.sh_dc[0]),
            to_u8(0.5f + kSH_C0 * g.sh_dc[1]),
            to_u8(0.5f + kSH_C0 * g.sh_dc[2])
        ));
    }
    rec.log(path, rerun::Points3D(positions).with_colors(colors).with_radii(0.02f));
}

void Visualizer::log_camera(const std::string& path, const Camera& cam,
                            float image_plane_distance) {
    // Pinhole intrinsics — RUB matches our OpenGL convention (-Z forward, +Y up).
    rec.log(path,
            rerun::Pinhole::from_focal_length_and_resolution(
                {cam.fx, cam.fy},
                {static_cast<float>(cam.image_width), static_cast<float>(cam.image_height)})
                .with_camera_xyz(rerun::components::ViewCoordinates::RUB)
                .with_image_plane_distance(image_plane_distance));

    // c2w transform: rotation columns are the rows of cam.R (w2c).
    const rerun::datatypes::Vec3D columns[3] = {
        {cam.R[0][0], cam.R[0][1], cam.R[0][2]},
        {cam.R[1][0], cam.R[1][1], cam.R[1][2]},
        {cam.R[2][0], cam.R[2][1], cam.R[2][2]},
    };
    const rerun::components::Translation3D translation(cam.position[0], cam.position[1], cam.position[2]);
    rec.log(path, rerun::Transform3D(translation, columns));
}

void Visualizer::log_octree_boxes(const std::string& path,
                                  const std::vector<const OctreeNode*>& nodes,
                                  int max_depth) {
    if (nodes.empty()) return;

    std::vector<rerun::components::Translation3D> centers;
    std::vector<rerun::components::HalfSize3D> half_sizes;
    std::vector<rerun::Color> colors;
    centers.reserve(nodes.size());
    half_sizes.reserve(nodes.size());
    colors.reserve(nodes.size());

    for (const OctreeNode* n : nodes) {
        centers.emplace_back(n->center_x, n->center_y, n->center_z);
        half_sizes.emplace_back(n->half_size, n->half_size, n->half_size);
        colors.push_back(depth_color(n->depth, max_depth));
    }

    rec.log(path,
            rerun::Boxes3D::from_centers_and_half_sizes(centers, half_sizes)
                .with_colors(colors));
}

void Visualizer::set_step(int64_t step) {
    rec.set_time_sequence("step", step);
}
