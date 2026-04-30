#include "ply_loader.h"
#include "interactive_viewer.h"
#include "morton.h"
#include "octree.h"
#include "visualizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ply> [--up X Y Z] [--vis] [--vis_stride N]"
                  << std::endl;
        return 1;
    }

    // ------------- Loading PLY file -------------
    std::string filepath = argv[1];

    float user_up[3] = {0.0f, 0.0f, 1.0f};
    bool enable_vis = false;
    int vis_stride = 50;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--up" && i + 3 < argc) {
            user_up[0] = std::atof(argv[++i]);
            user_up[1] = std::atof(argv[++i]);
            user_up[2] = std::atof(argv[++i]);
        } else if (a == "--vis") {
            enable_vis = true;
        } else if (a == "--vis_stride" && i + 1 < argc) {
            vis_stride = std::max(1, std::atoi(argv[++i]));
        } else if (a[0] == '-') {
            std::cerr << "Unknown flag: " << a << std::endl;
            return 1;
        }
    }
    GaussianCloud cloud = load_ply(filepath);

    // ------------- Filter outliers -------------
    cloud.filter_gaussians();


    // ------------- Spatial sort -------------

    AABB cubic_bbox = make_bbox_cubic(compute_bounding_box(cloud.gaussians));
    auto morton_codes = compute_morton_codes(cloud.gaussians, cubic_bbox);
    sort_gaussians_spatially(cloud.gaussians, morton_codes);

    // ------------- Octree construction + LoD summaries -------------
    uint8_t max_depth = 21;

    GaussianLOD lod;
    lod.max_depth = max_depth;
    lod.sh_degree = cloud.sh_degree;
    lod.sh_rest_count = cloud.sh_rest_count;
    lod.levels.resize(max_depth + 1);
    lod.levels[max_depth] = std::move(cloud.gaussians);

    std::cout << "\nBuilding octree with max_depth=" << static_cast<int>(max_depth) << std::endl;
    OctreeNode* root = build_octree_parallel(morton_codes, cubic_bbox, max_depth, lod);

    // ------------- Scene framing + interactive viewer -------------
    {
        float scene_center[3] = {
            (cubic_bbox.min_x + cubic_bbox.max_x) * 0.5f,
            (cubic_bbox.min_y + cubic_bbox.max_y) * 0.5f,
            (cubic_bbox.min_z + cubic_bbox.max_z) * 0.5f
        };

        // 95th-percentile radius from scene center — bbox diagonal would frame for
        // outliers that survived the 3σ filter.
        const auto& all_g = lod.levels[lod.max_depth];
        std::vector<float> dists;
        dists.reserve(all_g.size());
        for (const auto& g : all_g) {
            float dx = g.x - scene_center[0];
            float dy = g.y - scene_center[1];
            float dz = g.z - scene_center[2];
            dists.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        size_t k = static_cast<size_t>(dists.size() * 0.95);
        std::nth_element(dists.begin(), dists.begin() + k, dists.end());
        const float scene_radius = dists[k];
        printf("Scene: center=(%.2f, %.2f, %.2f) radius(p95)=%.2f, up=(%.2f, %.2f, %.2f)\n",
               scene_center[0], scene_center[1], scene_center[2], scene_radius,
               user_up[0], user_up[1], user_up[2]);

        std::unique_ptr<Visualizer> vis;
        if (enable_vis) {
            vis = std::make_unique<Visualizer>("LSGS");
            vis->log_point_cloud("world/scene", lod.levels[lod.max_depth], vis_stride);
            printf("Visualizer enabled (point cloud stride=%d)\n", vis_stride);
        }

        InteractiveViewerConfig cfg{};
        std::copy(user_up,      user_up      + 3, cfg.world_up);
        std::copy(scene_center, scene_center + 3, cfg.scene_center);
        cfg.scene_radius  = scene_radius;
        cfg.lod_threshold = 16.0f;
        cfg.image_width   = 1920;
        cfg.image_height  = 1080;
        cfg.fx = 1000.0f; cfg.fy = 1000.0f;
        cfg.cx = 960.0f;  cfg.cy = 540.0f;
        cfg.near_plane = 1.0f;
        cfg.far_plane  = 10000.0f;
        run_interactive_viewer(lod, root, cfg, vis.get());
    }

    print_stats(root, lod);
    delete_octree(root);

    return 0;
}
