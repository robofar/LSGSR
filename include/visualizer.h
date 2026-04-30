#pragma once

#include "frustum_culling.h"
#include "gaussian.h"
#include "octree.h"

#include <rerun.hpp>
#include <string>
#include <vector>

class Visualizer {
public:
    Visualizer(const std::string& app_name);

    void log_point_cloud(const std::string& path,
                         const std::vector<Gaussian>& gaussians,
                         int stride = 1);

    void log_camera(const std::string& path, const Camera& cam,
                    float image_plane_distance = 100.0f);

    // Logs an entire collection of octree nodes as Boxes3D, depth-graded color
    // (shallow=red -> deep=blue). `max_depth` is used to normalize the gradient.
    void log_octree_boxes(const std::string& path,
                          const std::vector<const OctreeNode*>& nodes,
                          int max_depth);

    void set_step(int64_t step);

private:
    rerun::RecordingStream rec;
};
