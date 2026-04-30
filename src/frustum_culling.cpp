#include "frustum_culling.h"

#include <cmath>
#include <cstring>

// --- Local math helpers ---

static float dot3(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void normalize3(float v[3]) {
    float n = std::sqrt(dot3(v, v));
    if (n > 0.0f) {
        v[0] /= n; v[1] /= n; v[2] /= n;
    }
}

// --- Camera construction ---

Camera make_camera(
    const float R[3][3],
    const float t[3],
    float fx, float fy,
    float cx, float cy,
    int image_width,
    int image_height,
    float near_plane,
    float far_plane
) {
    Camera cam;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) cam.R[i][j] = R[i][j];
        cam.t[i] = t[i];
    }
    cam.fx = fx; cam.fy = fy;
    cam.cx = cx; cam.cy = cy;
    cam.image_width = image_width;
    cam.image_height = image_height;
    cam.near_plane = near_plane;
    cam.far_plane = far_plane;

    // position = -R^T * t
    for (int i = 0; i < 3; i++) {
        cam.position[i] = -(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]);
    }
    return cam;
}

Camera make_camera_lookat(
    const float position[3],
    const float target[3],
    const float world_up[3],
    float fx, float fy,
    float cx, float cy,
    int image_width,
    int image_height,
    float near_plane,
    float far_plane
) {
    float forward[3] = {
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2]
    };
    normalize3(forward);

    float right[3];
    cross3(forward, world_up, right);
    normalize3(right);

    float up[3];
    cross3(right, forward, up);

    float R[3][3];
    R[0][0] = right[0];    R[0][1] = right[1];    R[0][2] = right[2];
    R[1][0] = up[0];       R[1][1] = up[1];       R[1][2] = up[2];
    R[2][0] = -forward[0]; R[2][1] = -forward[1]; R[2][2] = -forward[2];

    float t[3];
    for (int i = 0; i < 3; i++) {
        t[i] = -(R[i][0] * position[0] + R[i][1] * position[1] + R[i][2] * position[2]);
    }

    Camera cam = make_camera(R, t, fx, fy, cx, cy,
                             image_width, image_height,
                             near_plane, far_plane);

    return cam;
}

// --- View / Projection / Multiply ---

void compute_view_matrix(const Camera& cam, float out[4][4]) {
    out[0][0] = cam.R[0][0]; out[0][1] = cam.R[0][1]; out[0][2] = cam.R[0][2]; out[0][3] = cam.t[0];
    out[1][0] = cam.R[1][0]; out[1][1] = cam.R[1][1]; out[1][2] = cam.R[1][2]; out[1][3] = cam.t[1];
    out[2][0] = cam.R[2][0]; out[2][1] = cam.R[2][1]; out[2][2] = cam.R[2][2]; out[2][3] = cam.t[2];
    out[3][0] = 0;           out[3][1] = 0;           out[3][2] = 0;           out[3][3] = 1;
}

void compute_projection_matrix(const Camera& cam, float out[4][4]) {
    float n = cam.near_plane;
    float f = cam.far_plane;
    float w = static_cast<float>(cam.image_width);
    float h = static_cast<float>(cam.image_height);

    std::memset(out, 0, sizeof(float) * 16);
    out[0][0] = 2.0f * cam.fx / w;
    out[1][1] = 2.0f * cam.fy / h;
    out[0][2] = 1.0f - 2.0f * cam.cx / w;
    out[1][2] = 1.0f - 2.0f * cam.cy / h;
    out[2][2] = -(f + n) / (f - n);
    out[2][3] = -2.0f * f * n / (f - n);
    out[3][2] = -1.0f;
}

void multiply_4x4(const float A[4][4], const float B[4][4], float out[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                out[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// --- Frustum extraction ---

Frustum extract_frustum(const float VP[4][4]) {
    Frustum f;
    // Left
    f.planes[0] = { VP[3][0]+VP[0][0], VP[3][1]+VP[0][1], VP[3][2]+VP[0][2], VP[3][3]+VP[0][3] };
    // Right
    f.planes[1] = { VP[3][0]-VP[0][0], VP[3][1]-VP[0][1], VP[3][2]-VP[0][2], VP[3][3]-VP[0][3] };
    // Bottom
    f.planes[2] = { VP[3][0]+VP[1][0], VP[3][1]+VP[1][1], VP[3][2]+VP[1][2], VP[3][3]+VP[1][3] };
    // Top
    f.planes[3] = { VP[3][0]-VP[1][0], VP[3][1]-VP[1][1], VP[3][2]-VP[1][2], VP[3][3]-VP[1][3] };
    // Near
    f.planes[4] = { VP[3][0]+VP[2][0], VP[3][1]+VP[2][1], VP[3][2]+VP[2][2], VP[3][3]+VP[2][3] };
    // Far
    f.planes[5] = { VP[3][0]-VP[2][0], VP[3][1]-VP[2][1], VP[3][2]-VP[2][2], VP[3][3]-VP[2][3] };

    for (auto& p : f.planes) {
        float n = std::sqrt(p.a*p.a + p.b*p.b + p.c*p.c);
        if (n > 0.0f) {
            p.a /= n; p.b /= n; p.c /= n; p.d /= n;
        }
    }
    return f;
}

// --- AABB / frustum test ---

bool aabb_in_frustum(const Frustum& frustum,
                     float cx, float cy, float cz,
                     float half) {
    for (const auto& plane : frustum.planes) {
        float px = (plane.a > 0) ? cx + half : cx - half;
        float py = (plane.b > 0) ? cy + half : cy - half;
        float pz = (plane.c > 0) ? cz + half : cz - half;

        if (plane.a * px + plane.b * py + plane.c * pz + plane.d < 0) {
            return false;
        }
    }
    return true;
}

// --- LoD selection ---

bool should_use_summary(const OctreeNode* node,
                       const Camera& cam,
                       float lod_threshold) {
    float dx = node->center_x - cam.position[0];
    float dy = node->center_y - cam.position[1];
    float dz = node->center_z - cam.position[2];
    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (distance < node->half_size) return false;

    float projected_size = node->half_size * cam.fy / distance;
    return projected_size < lod_threshold;
}

// --- Traversal ---

static void traverse(
    const OctreeNode* node,
    const GaussianLOD& lod,
    const Frustum& frustum,
    const Camera& cam,
    float lod_threshold,
    GatherResult& result
) {
    if (!node) return;

    if (!aabb_in_frustum(frustum, node->center_x, node->center_y, node->center_z, node->half_size)) {
        result.nodes_frustum_culled++;
        return;
    }

    if (node->is_leaf) {
        result.ranges.push_back({
            static_cast<uint8_t>(lod.max_depth),
            node->gauss_offset,
            node->gauss_count
        });
        result.leaf_gaussians += node->gauss_count;
        result.total_gaussians += node->gauss_count;
        return;
    }

    if (should_use_summary(node, cam, lod_threshold)) {
        result.ranges.push_back({
            node->depth,
            node->gauss_offset,
            node->gauss_count
        });
        result.summary_gaussians += node->gauss_count;
        result.total_gaussians += node->gauss_count;
        result.nodes_lod_selected++;
        return;
    }

    for (int i = 0; i < 8; i++) {
        traverse(node->children[i], lod, frustum, cam, lod_threshold, result);
    }
}

GatherResult gather_visible_gaussians(
    const OctreeNode* root,
    const GaussianLOD& lod,
    const Frustum& frustum,
    const Camera& cam,
    float lod_threshold
) {
    GatherResult result;
    result.ranges.reserve(8192);
    traverse(root, lod, frustum, cam, lod_threshold, result);
    return result;
}

// --- Debug traversal (visualization only) ---

static void debug_walk_all(const OctreeNode* node, std::vector<const OctreeNode*>& out) {
    if (!node) return;
    out.push_back(node);
    if (!node->is_leaf) {
        for (int i = 0; i < 8; i++) debug_walk_all(node->children[i], out);
    }
}

static void debug_walk_gather(
    const OctreeNode* node,
    const Frustum& frustum,
    const Camera& cam,
    float lod_threshold,
    std::vector<const OctreeNode*>& frustum_passed,
    std::vector<const OctreeNode*>& selected
) {
    if (!node) return;
    if (!aabb_in_frustum(frustum, node->center_x, node->center_y, node->center_z, node->half_size)) {
        return;
    }
    frustum_passed.push_back(node);
    if (node->is_leaf) {
        selected.push_back(node);
        return;
    }
    if (should_use_summary(node, cam, lod_threshold)) {
        selected.push_back(node);
        return;
    }
    for (int i = 0; i < 8; i++) {
        debug_walk_gather(node->children[i], frustum, cam, lod_threshold, frustum_passed, selected);
    }
}

OctreeDebugTrace debug_gather_trace(
    const OctreeNode* root,
    const Frustum& frustum,
    const Camera& cam,
    float lod_threshold
) {
    OctreeDebugTrace t;
    debug_walk_all(root, t.all);
    debug_walk_gather(root, frustum, cam, lod_threshold, t.frustum_passed, t.selected);
    return t;
}
