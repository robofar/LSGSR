#include "octree.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

// --- OctreeNode constructor ---

OctreeNode::OctreeNode()
    : center_x(0), center_y(0), center_z(0), half_size(0),
      children{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
      gauss_offset(0), gauss_count(0),
      depth(0), is_leaf(false) {}

// --- Helpers ---

AABB child_aabb(const AABB& parent, int child_index) {
    float mid_x = (parent.min_x + parent.max_x) * 0.5f;
    float mid_y = (parent.min_y + parent.max_y) * 0.5f;
    float mid_z = (parent.min_z + parent.max_z) * 0.5f;

    // bit 0 = x, bit 1 = y, bit 2 = z
    AABB child;
    child.min_x = (child_index & 1) ? mid_x : parent.min_x;
    child.max_x = (child_index & 1) ? parent.max_x : mid_x;
    child.min_y = (child_index & 2) ? mid_y : parent.min_y;
    child.max_y = (child_index & 2) ? parent.max_y : mid_y;
    child.min_z = (child_index & 4) ? mid_z : parent.min_z;
    child.max_z = (child_index & 4) ? parent.max_z : mid_z;
    return child;
}

// Find where each of the 8 octants starts within [offset, offset+count).
// Returns 9 boundaries: 
// child_ranges[i] is the start of octant i (i = {0,1,2,3,4,5,6,7}),
// child_ranges[8] = offset + count.
static std::array<uint32_t, 9> find_child_ranges(
    const std::vector<uint64_t>& morton_codes,
    uint32_t offset,
    uint32_t count,
    uint8_t depth
) {
    std::array<uint32_t, 9> ranges; // 0-1 ; 1-2 ; 2-3 ; 3-4 ; 4-5 ; 5-6 ; 6-7 ; 7-8 (that's why 9 - ranges)
    ranges[0] = offset;
    ranges[8] = offset + count;

    int bit_pos = (20 - depth) * 3; // shift

    // Binary search for where each octant starts.
    // Since the array is sorted, octant values (0-7) appear in order.
    // For octant boundary `oct`, find first code where the 3 bits at this
    // depth are >= oct.
    for (int oct = 1; oct <= 7; oct++) {
        // Search within remaining range
        auto it = std::lower_bound(
            morton_codes.begin() + ranges[oct - 1],
            morton_codes.begin() + ranges[8],
            static_cast<uint8_t>(oct),
            [bit_pos](uint64_t code, uint8_t octant) {
                return ((code >> bit_pos) & 0x7) < octant; // extracts the 3 octant bits from a Morton code (right shift + zeroing rest bits)
            }
        );
        ranges[oct] = static_cast<uint32_t>(it - morton_codes.begin());
    }

    return ranges;
}

// Compute max scale value across all 3 scale components for gaussians in [offset, offset+count)
static float compute_max_scale(const std::vector<Gaussian>& gaussians, uint32_t offset, uint32_t count) {
    float max_s = 0.0f;
    for (uint32_t i = offset; i < offset + count; i++) {
        for (int j = 0; j < 3; j++) {
            max_s = std::max(max_s, gaussians[i].scale[j]);
        }
    }
    return max_s;
}

// --- Moment-matched gaussian merge ---

// Per-axis cell count when binning a node's source gaussians for summarization.
// Up to kSummaryAxisDiv^3 representatives per node.
static constexpr int kSummaryAxisDiv = 6;

// Symmetric 3x3 packed as [xx, yy, zz, xy, xz, yz].
// Jacobi rotation method. Eigenvectors are columns of `eigvecs`.
static void eig3_symmetric(const float Sigma[6], float eigvecs[3][3], float eigvals[3]) {
    float A[3][3] = {
        { Sigma[0], Sigma[3], Sigma[4] },
        { Sigma[3], Sigma[1], Sigma[5] },
        { Sigma[4], Sigma[5], Sigma[2] }
    };
    float V[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};

    constexpr int max_iter = 32;
    constexpr float eps = 1e-12f;
    for (int iter = 0; iter < max_iter; iter++) {
        int p = 0, q = 1;
        float max_off = std::fabs(A[0][1]);
        if (std::fabs(A[0][2]) > max_off) { p = 0; q = 2; max_off = std::fabs(A[0][2]); }
        if (std::fabs(A[1][2]) > max_off) { p = 1; q = 2; max_off = std::fabs(A[1][2]); }
        if (max_off < eps) break;

        float app = A[p][p], aqq = A[q][q], apq = A[p][q];
        float theta = (aqq - app) / (2.0f * apq);
        float t = (theta >= 0.0f)
            ? 1.0f / (theta + std::sqrt(1.0f + theta * theta))
            : 1.0f / (theta - std::sqrt(1.0f + theta * theta));
        float c = 1.0f / std::sqrt(1.0f + t * t);
        float s = t * c;

        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = A[q][p] = 0.0f;
        for (int i = 0; i < 3; i++) {
            if (i != p && i != q) {
                float aip = A[i][p], aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip - s * aiq;
                A[i][q] = A[q][i] = s * aip + c * aiq;
            }
        }
        for (int i = 0; i < 3; i++) {
            float vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }
    eigvals[0] = A[0][0];
    eigvals[1] = A[1][1];
    eigvals[2] = A[2][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            eigvecs[i][j] = V[i][j];
}

// Quaternion (w, x, y, z) -> 3x3 rotation. Assumes q is unit-length.
static void quat_to_mat3(const float q[4], float R[3][3]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1.0f - 2.0f * (y * y + z * z);
    R[0][1] = 2.0f * (x * y - w * z);
    R[0][2] = 2.0f * (x * z + w * y);
    R[1][0] = 2.0f * (x * y + w * z);
    R[1][1] = 1.0f - 2.0f * (x * x + z * z);
    R[1][2] = 2.0f * (y * z - w * x);
    R[2][0] = 2.0f * (x * z - w * y);
    R[2][1] = 2.0f * (y * z + w * x);
    R[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

// 3x3 rotation -> quaternion (w, x, y, z). Shoemake.
static void mat3_to_quat(const float R[3][3], float q[4]) {
    float trace = R[0][0] + R[1][1] + R[2][2];
    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        q[0] = 0.25f * s;
        q[1] = (R[2][1] - R[1][2]) / s;
        q[2] = (R[0][2] - R[2][0]) / s;
        q[3] = (R[1][0] - R[0][1]) / s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        float s = std::sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]) * 2.0f;
        q[0] = (R[2][1] - R[1][2]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[0][1] + R[1][0]) / s;
        q[3] = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        float s = std::sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]) * 2.0f;
        q[0] = (R[0][2] - R[2][0]) / s;
        q[1] = (R[0][1] + R[1][0]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[1][2] + R[2][1]) / s;
    } else {
        float s = std::sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]) * 2.0f;
        q[0] = (R[1][0] - R[0][1]) / s;
        q[1] = (R[0][2] + R[2][0]) / s;
        q[2] = (R[1][2] + R[2][1]) / s;
        q[3] = 0.25f * s;
    }
}

// Moment-match a set of source gaussians: produce one representative whose
// 1st moment (mean) and 2nd moment (covariance) match the source mixture.
// SH coefs are weighted-averaged in the world frame (3DGS convention -> no
// rotation correction needed). Opacity uses the existing transmittance combine.
// `min_scale` floors each per-axis scale so the rep covers at least its sub-cell
// (avoids tiny PCA-derived reps from blowing out as sub-pixel splats at distance).
static Gaussian moment_match(const Gaussian* src, size_t n, float min_scale) {
    Gaussian result{};
    result.rotation = {1.0f, 0.0f, 0.0f, 0.0f};

    if (n == 0) return result;
    if (n == 1) {
        result = src[0];
        for (int j = 0; j < 3; j++) result.scale[j] = std::max(result.scale[j], min_scale);
        return result;
    }

    double total_weight = 0.0;
    for (size_t i = 0; i < n; i++) {
        const Gaussian& g = src[i];
        total_weight += static_cast<double>(g.opacity) * g.scale[0] * g.scale[1] * g.scale[2];
    }
    const bool use_weights = total_weight > 1e-10;
    auto weight_of = [&](const Gaussian& g) -> double {
        return use_weights
            ? (static_cast<double>(g.opacity) * g.scale[0] * g.scale[1] * g.scale[2]) / total_weight
            : 1.0 / static_cast<double>(n);
    };

    // 1st moment: mean of position, SH DC, SH rest.
    double px = 0, py = 0, pz = 0;
    double dc[3] = {0, 0, 0};
    double rest[MAX_SH_REST] = {0};
    for (size_t i = 0; i < n; i++) {
        const Gaussian& g = src[i];
        double w = weight_of(g);
        px += w * g.x;
        py += w * g.y;
        pz += w * g.z;
        for (int c = 0; c < 3; c++) dc[c] += w * g.sh_dc[c];
        for (int k = 0; k < MAX_SH_REST; k++) rest[k] += w * g.sh_rest[k];
    }
    result.x = static_cast<float>(px);
    result.y = static_cast<float>(py);
    result.z = static_cast<float>(pz);
    for (int c = 0; c < 3; c++) result.sh_dc[c] = static_cast<float>(dc[c]);
    for (int k = 0; k < MAX_SH_REST; k++) result.sh_rest[k] = static_cast<float>(rest[k]);

    // 2nd moment: Σ = Σ wᵢ · (Σᵢ + (μᵢ − μ)(μᵢ − μ)ᵀ),  Σᵢ = Rᵢ · diag(sᵢ²) · Rᵢᵀ.
    double C[6] = {0};  // xx, yy, zz, xy, xz, yz
    for (size_t i = 0; i < n; i++) {
        const Gaussian& g = src[i];
        double w = weight_of(g);

        float qn[4] = { g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3] };
        float qmag = std::sqrt(qn[0]*qn[0] + qn[1]*qn[1] + qn[2]*qn[2] + qn[3]*qn[3]);
        if (qmag > 0.0f) {
            for (int j = 0; j < 4; j++) qn[j] /= qmag;
        } else {
            qn[0] = 1.0f; qn[1] = qn[2] = qn[3] = 0.0f;
        }
        float Ri[3][3];
        quat_to_mat3(qn, Ri);

        double s2[3] = {
            static_cast<double>(g.scale[0]) * g.scale[0],
            static_cast<double>(g.scale[1]) * g.scale[1],
            static_cast<double>(g.scale[2]) * g.scale[2]
        };
        double Si[3][3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                double sum = 0;
                for (int k = 0; k < 3; k++) sum += static_cast<double>(Ri[a][k]) * s2[k] * Ri[b][k];
                Si[a][b] = sum;
            }
        }

        double dx = static_cast<double>(g.x) - px;
        double dy = static_cast<double>(g.y) - py;
        double dz = static_cast<double>(g.z) - pz;

        C[0] += w * (Si[0][0] + dx * dx);
        C[1] += w * (Si[1][1] + dy * dy);
        C[2] += w * (Si[2][2] + dz * dz);
        C[3] += w * (Si[0][1] + dx * dy);
        C[4] += w * (Si[0][2] + dx * dz);
        C[5] += w * (Si[1][2] + dy * dz);
    }

    float Sigma[6] = {
        static_cast<float>(C[0]), static_cast<float>(C[1]), static_cast<float>(C[2]),
        static_cast<float>(C[3]), static_cast<float>(C[4]), static_cast<float>(C[5])
    };
    float eigvecs[3][3], eigvals[3];
    eig3_symmetric(Sigma, eigvecs, eigvals);

    // Sort eigenvalues descending so axis 0 = largest.
    int order[3] = {0, 1, 2};
    if (eigvals[order[0]] < eigvals[order[1]]) std::swap(order[0], order[1]);
    if (eigvals[order[1]] < eigvals[order[2]]) std::swap(order[1], order[2]);
    if (eigvals[order[0]] < eigvals[order[1]]) std::swap(order[0], order[1]);

    // Build R with columns = sorted eigenvectors.
    float R[3][3];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            R[i][j] = eigvecs[i][order[j]];

    // Force right-handed (det = +1); flip last column otherwise.
    float det =
        R[0][0]*(R[1][1]*R[2][2] - R[1][2]*R[2][1])
      - R[0][1]*(R[1][0]*R[2][2] - R[1][2]*R[2][0])
      + R[0][2]*(R[1][0]*R[2][1] - R[1][1]*R[2][0]);
    if (det < 0.0f) {
        for (int i = 0; i < 3; i++) R[i][2] = -R[i][2];
    }

    float q[4];
    mat3_to_quat(R, q);
    float qmag = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (qmag > 0.0f) for (int j = 0; j < 4; j++) q[j] /= qmag;
    result.rotation = { q[0], q[1], q[2], q[3] };

    for (int j = 0; j < 3; j++) {
        float lam = eigvals[order[j]];
        if (lam < 1e-12f) lam = 1e-12f;
        result.scale[j] = std::max(std::sqrt(lam), min_scale);
    }

    double transmittance = 1.0;
    for (size_t i = 0; i < n; i++) {
        transmittance *= (1.0 - static_cast<double>(src[i].opacity));
    }
    float combined = static_cast<float>(1.0 - transmittance);
    result.opacity = std::max(0.01f, std::min(0.99f, combined));

    return result;
}

// Bin sources into kSummaryAxisDiv^3 cells over `node_bbox`. For each non-empty
// cell, append one moment-matched representative to `out`.
static void compute_node_summaries(
    const AABB& node_bbox,
    const Gaussian* src,
    size_t n_src,
    std::vector<Gaussian>& out
) {
    if (n_src == 0) return;

    constexpr int M = kSummaryAxisDiv;
    constexpr int M3 = M * M * M;

    const float min_x = node_bbox.min_x;
    const float min_y = node_bbox.min_y;
    const float min_z = node_bbox.min_z;
    const float ex = node_bbox.max_x - min_x;
    const float ey = node_bbox.max_y - min_y;
    const float ez = node_bbox.max_z - min_z;
    const float inv_ex = ex > 0.0f ? static_cast<float>(M) / ex : 0.0f;
    const float inv_ey = ey > 0.0f ? static_cast<float>(M) / ey : 0.0f;
    const float inv_ez = ez > 0.0f ? static_cast<float>(M) / ez : 0.0f;

    // Bbox is cubic per build_octree (make_bbox_cubic), so a single per-axis
    // cell size suffices. Floor each rep's per-axis scale to cell_size / 2.
    const float cell_size = std::max({ex, ey, ez}) / static_cast<float>(M);
    const float min_scale = 0.5f * cell_size;

    // Counting sort: bucketize sources by cell index.
    std::vector<uint16_t> cell_of(n_src);
    std::array<uint32_t, M3 + 1> bucket = {};
    for (size_t i = 0; i < n_src; i++) {
        const Gaussian& g = src[i];
        int ix = static_cast<int>((g.x - min_x) * inv_ex);
        int iy = static_cast<int>((g.y - min_y) * inv_ey);
        int iz = static_cast<int>((g.z - min_z) * inv_ez);
        if (ix < 0) ix = 0; else if (ix >= M) ix = M - 1;
        if (iy < 0) iy = 0; else if (iy >= M) iy = M - 1;
        if (iz < 0) iz = 0; else if (iz >= M) iz = M - 1;
        uint16_t c = static_cast<uint16_t>(ix + M * (iy + M * iz));
        cell_of[i] = c;
        bucket[c + 1]++;
    }
    for (int c = 1; c <= M3; c++) bucket[c] += bucket[c - 1];

    std::vector<uint32_t> sorted_idx(n_src);
    std::array<uint32_t, M3> write_ptr;
    for (int c = 0; c < M3; c++) write_ptr[c] = bucket[c];
    for (size_t i = 0; i < n_src; i++) {
        sorted_idx[write_ptr[cell_of[i]]++] = static_cast<uint32_t>(i);
    }

    std::vector<Gaussian> tmp;
    tmp.reserve(64);
    for (int c = 0; c < M3; c++) {
        uint32_t start = bucket[c];
        uint32_t end = bucket[c + 1];
        if (start == end) continue;
        tmp.clear();
        for (uint32_t k = start; k < end; k++) tmp.push_back(src[sorted_idx[k]]);
        out.push_back(moment_match(tmp.data(), tmp.size(), min_scale));
    }
}

// Copy a child's source gaussians into `dst`, holding the source level's mutex
// only for the read window. Leaves draw from the immutable original cloud.
static void append_child_sources(
    const OctreeNode* child,
    GaussianLOD& lod,
    std::vector<Gaussian>& dst
) {
    const int src_depth = child->is_leaf ? lod.max_depth : child->depth;
    const auto& level = lod.levels[src_depth];
    const uint32_t off = child->gauss_offset;
    const uint32_t cnt = child->gauss_count;
    if (child->is_leaf) {
        // levels[max_depth] is set once at startup and never written again.
        dst.insert(dst.end(), level.begin() + off, level.begin() + off + cnt);
    } else {
        std::lock_guard<std::mutex> lock(lod.level_mutexes[src_depth]);
        dst.insert(dst.end(), level.begin() + off, level.begin() + off + cnt);
    }
}

static OctreeNode* build_octree_recursive(
    const std::vector<uint64_t>& morton_codes,
    const AABB& node_bbox,
    uint32_t offset,
    uint32_t count,
    uint8_t depth,
    uint8_t max_depth,
    GaussianLOD& lod
) {

    OctreeNode* node = new OctreeNode();
    node->center_x = (node_bbox.min_x + node_bbox.max_x) * 0.5f;
    node->center_y = (node_bbox.min_y + node_bbox.max_y) * 0.5f;
    node->center_z = (node_bbox.min_z + node_bbox.max_z) * 0.5f;
    node->half_size = (node_bbox.max_x - node_bbox.min_x) * 0.5f;
    node->depth = depth;
    node->gauss_offset = offset;
    node->gauss_count = count;

    // Leaf condition
    if (count <= MAX_LEAF_SIZE || depth >= max_depth) {
        node->is_leaf = true;
        const auto& leaf_gaussians = lod.levels[lod.max_depth];
        return node;
    }

    // Internal node: subdivide
    node->is_leaf = false;

    auto child_ranges = find_child_ranges(morton_codes, offset, count, depth);

    for (int i = 0; i < 8; i++) {
        uint32_t child_offset = child_ranges[i];
        uint32_t child_count = child_ranges[i + 1] - child_ranges[i];

        if (child_count == 0) continue;

        AABB cbox = child_aabb(node_bbox, i);

        node->children[i] = build_octree_recursive(
            morton_codes, cbox,
            child_offset, child_count,
            depth + 1, max_depth, lod
        );

 
    }

    // Bottom-up: gather child sources, bin, moment-match, append to this depth.
    {
        std::vector<Gaussian> sources;
        sources.reserve(8 * MAX_LEAF_SIZE);
        for (int i = 0; i < 8; i++) {
            if (!node->children[i]) continue;
            append_child_sources(node->children[i], lod, sources);
        }

        std::vector<Gaussian> reps;
        compute_node_summaries(node_bbox, sources.data(), sources.size(), reps);

        std::lock_guard<std::mutex> lock(lod.level_mutexes[depth]);
        node->gauss_offset = static_cast<uint32_t>(lod.levels[depth].size());
        node->gauss_count = static_cast<uint32_t>(reps.size());
        lod.levels[depth].insert(lod.levels[depth].end(), reps.begin(), reps.end());
    }

    return node;
}


OctreeNode* build_octree_parallel(
    const std::vector<uint64_t>& morton_codes,
    const AABB& scene_bbox,
    uint8_t max_depth,
    GaussianLOD& lod
) {

    OctreeNode* root = new OctreeNode();
    root->center_x = (scene_bbox.min_x + scene_bbox.max_x) * 0.5f;
    root->center_y = (scene_bbox.min_y + scene_bbox.max_y) * 0.5f;
    root->center_z = (scene_bbox.min_z + scene_bbox.max_z) * 0.5f;
    root->half_size = (scene_bbox.max_x - scene_bbox.min_x) * 0.5f;
    root->depth = 0;
    root->is_leaf = false;

    const size_t n = lod.levels[lod.max_depth].size(); // original sorted gaussians are in max_depth lod vector
    std::array<uint32_t, 9> child_ranges = find_child_ranges(morton_codes, 0, n, 0);

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++) {
        uint32_t child_offset = child_ranges[i];
        uint32_t child_count = child_ranges[i + 1] - child_ranges[i];
        if (child_count == 0) continue;

        AABB cbox = child_aabb(scene_bbox, i);

        threads.emplace_back([&, i, child_offset, child_count, cbox]() {
            root->children[i] = build_octree_recursive(
                morton_codes, cbox,
                child_offset, child_count,
                1, max_depth, lod
            );
        });
    }
    for (auto& t : threads) t.join();


    // Root's own summary representatives.
    {
        std::vector<Gaussian> sources;
        sources.reserve(8 * MAX_LEAF_SIZE);
        for (int i = 0; i < 8; i++) {
            if (!root->children[i]) continue;
            append_child_sources(root->children[i], lod, sources);
        }

        std::vector<Gaussian> reps;
        compute_node_summaries(scene_bbox, sources.data(), sources.size(), reps);

        std::lock_guard<std::mutex> lock(lod.level_mutexes[0]);
        root->gauss_offset = static_cast<uint32_t>(lod.levels[0].size());
        root->gauss_count = static_cast<uint32_t>(reps.size());
        lod.levels[0].insert(lod.levels[0].end(), reps.begin(), reps.end());
    }

    // Compact: move leaf gaussians from levels[max_depth] down to the actual deepest leaf depth, then shrink the levels vector.
    int reached = 0;
    for (int i = lod.max_depth - 1; i >= 0; i--) { // -1 cuz max_depth contains original set of gaussians
        if (!lod.levels[i].empty()) { reached = i + 1; break; }
    }
    if (reached > 0 && reached < lod.max_depth) {
        lod.levels[reached] = std::move(lod.levels[lod.max_depth]);
        lod.max_depth = reached;
        lod.levels.resize(reached + 1);
    }

    return root;
}

// --- Octree + LoD stats and verification ---

struct TreeStats {
    size_t total_nodes = 0;
    size_t leaf_nodes = 0;
    size_t internal_nodes = 0;
    uint32_t min_leaf_size = UINT32_MAX;
    uint32_t max_leaf_size = 0;
    uint64_t total_leaf_gaussians = 0;
    uint8_t max_depth_reached = 0;
};

static void walk_tree(const OctreeNode* node, const GaussianLOD& lod, TreeStats& stats, bool& ok) {
    if (!node) return;
    stats.total_nodes++;
    stats.max_depth_reached = std::max(stats.max_depth_reached, node->depth);

    if (node->is_leaf) {
        stats.leaf_nodes++;
        stats.min_leaf_size = std::min(stats.min_leaf_size, node->gauss_count);
        stats.max_leaf_size = std::max(stats.max_leaf_size, node->gauss_count);
        stats.total_leaf_gaussians += node->gauss_count;

        // Leaf bounds check against original gaussian array
        if (node->gauss_offset + node->gauss_count > lod.levels[lod.max_depth].size()) {
            std::cerr << "ERROR: leaf at depth " << static_cast<int>(node->depth)
                      << " has out-of-bounds range [" << node->gauss_offset
                      << ", " << node->gauss_offset + node->gauss_count << ")" << std::endl;
            ok = false;
        }
        return;
    }

    stats.internal_nodes++;

    // Non-leaf: summary count is bounded by the per-node sub-cell grid (M^3).
    constexpr uint32_t kMaxSummariesPerNode = kSummaryAxisDiv * kSummaryAxisDiv * kSummaryAxisDiv;
    if (node->gauss_count < 1 || node->gauss_count > kMaxSummariesPerNode) {
        std::cerr << "ERROR: non-leaf at depth " << static_cast<int>(node->depth)
                  << " has gauss_count=" << node->gauss_count
                  << " (expected 1-" << kMaxSummariesPerNode << ")" << std::endl;
        ok = false;
    }

    // Non-leaf: check summary bounds and values
    const auto& level = lod.levels[node->depth];
    if (node->gauss_offset + node->gauss_count > level.size()) {
        std::cerr << "ERROR: non-leaf at depth " << static_cast<int>(node->depth)
                  << " has out-of-bounds range [" << node->gauss_offset
                  << ", " << node->gauss_offset + node->gauss_count
                  << ") in level of size " << level.size() << std::endl;
        ok = false;
    } else {
        for (uint32_t i = node->gauss_offset; i < node->gauss_offset + node->gauss_count; i++) {
            const Gaussian& g = level[i];
            if (g.opacity <= 0.0f || g.opacity >= 1.0f) {
                std::cerr << "ERROR: summary at depth " << static_cast<int>(node->depth)
                          << " index " << i << " has invalid opacity=" << g.opacity << std::endl;
                ok = false;
            }
            for (int j = 0; j < 3; j++) {
                if (g.scale[j] <= 0.0f) {
                    std::cerr << "ERROR: summary at depth " << static_cast<int>(node->depth)
                              << " index " << i << " has invalid scale[" << j << "]=" << g.scale[j] << std::endl;
                    ok = false;
                }
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        walk_tree(node->children[i], lod, stats, ok);
    }
}

void print_stats(const OctreeNode* root, const GaussianLOD& lod) {
    TreeStats stats;
    bool ok = true;
    walk_tree(root, lod, stats, ok);

    double avg_leaf = stats.leaf_nodes > 0
        ? static_cast<double>(stats.total_leaf_gaussians) / stats.leaf_nodes
        : 0.0;
    double memory_mb = stats.total_nodes * sizeof(OctreeNode) / (1024.0 * 1024.0);

    std::cout << "\n--- Octree Statistics ---" << std::endl;
    std::cout << "Total nodes: " << stats.total_nodes << std::endl;
    std::cout << "  Internal: " << stats.internal_nodes << std::endl;
    std::cout << "  Leaves: " << stats.leaf_nodes << std::endl;
    std::cout << "Leaf size: min=" << stats.min_leaf_size
              << ", max=" << stats.max_leaf_size
              << ", avg=" << avg_leaf << std::endl;
    std::cout << "Total leaf Gaussians: " << stats.total_leaf_gaussians << std::endl;
    std::cout << "Max depth reached: " << static_cast<int>(stats.max_depth_reached) << std::endl;
    std::cout << "Tree memory: " << memory_mb << " MB" << std::endl;

    // Leaf count check
    const size_t expected = lod.levels[lod.max_depth].size();
    if (stats.total_leaf_gaussians != expected) {
        std::cerr << "Count check FAILED: leaf sum = " << stats.total_leaf_gaussians
                  << ", expected " << expected << std::endl;
        ok = false;
    }

    std::cout << "\n--- LoD Level Stats ---" << std::endl;
    for (int i = 0; i <= lod.max_depth; i++) {
        size_t count = lod.levels[i].size();
        if (count == 0) continue;
        double mb = count * sizeof(Gaussian) / (1024.0 * 1024.0);
        std::cout << "  Level " << i << ": " << count << " Gaussians (" << mb << " MB)" << std::endl;
    }

    std::cout << "\n" << (ok ? "All checks PASSED." : "Some checks FAILED.") << std::endl;
}

// --- Cleanup ---

void delete_octree(OctreeNode* root) {
    if (!root) return;
    for (int i = 0; i < 8; i++) {
        delete_octree(root->children[i]);
    }
    delete root;
}
