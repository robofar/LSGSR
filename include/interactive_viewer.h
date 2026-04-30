#pragma once

#include "frustum_culling.h"
#include "octree.h"
#include "visualizer.h"

struct InteractiveViewerConfig {
    float world_up[3];
    float scene_center[3];
    float scene_radius;
    float lod_threshold;
    int   image_width;
    int   image_height;
    float fx, fy, cx, cy;
    float near_plane, far_plane;
};

// Blocks until the window is closed. `vis` may be null.
void run_interactive_viewer(const GaussianLOD& lod,
                            const OctreeNode* root,
                            const InteractiveViewerConfig& cfg,
                            Visualizer* vis);
