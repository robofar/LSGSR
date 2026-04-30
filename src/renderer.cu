#include "renderer.h"

#include "rasterizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <thread>

namespace {

struct DeviceBuffer {
    char* data = nullptr;
    size_t capacity = 0;
    char* ensure(size_t bytes) {
        if (bytes > capacity) {
            if (data) cudaFree(data);
            cudaMalloc(&data, bytes);
            capacity = bytes;
        }
        return data;
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

// Pinned host buffer. Backs the SoA staging arrays with cudaMallocHost-allocated
// memory so cudaMemcpy can DMA directly (no internal pageable->pinned staging).
// API mirrors the std::vector<T> calls used in the staging path: resize/data/
// size/operator[]. Unlike std::vector, resize discards prior contents on grow
// (the caller refills every frame, so we don't pay to copy old data).
template <typename T>
struct PinnedBuffer {
    T* ptr = nullptr;
    size_t capacity_ = 0;
    size_t size_ = 0;

    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }

    void resize(size_t n) {
        if (n > capacity_) {
            if (ptr) cudaFreeHost(ptr);
            size_t new_cap = std::max(n, capacity_ + capacity_ / 2);
            cudaMallocHost(&ptr, new_cap * sizeof(T));
            capacity_ = new_cap;
        }
        size_ = n;
    }

    T*       data()       { return ptr; }
    const T* data() const { return ptr; }
    size_t   size() const { return size_; }
    T&       operator[](size_t i)       { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
};

// Upload host data into a persistent device buffer, growing it if needed.
// Returns a typed pointer to the device buffer.
template <typename T>
T* upload_to(DeviceBuffer& b, const T* host, size_t n) {
    T* d_ptr = reinterpret_cast<T*>(b.ensure(n * sizeof(T)));
    cudaMemcpy(d_ptr, host, n * sizeof(T), cudaMemcpyHostToDevice);
    return d_ptr;
}

// Re-pack a row-major 4x4 (M_ij at offset i*4+j) as column-major (M_ij at offset j*4+i).
void to_column_major(const float in[4][4], float out[16]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            out[j * 4 + i] = in[i][j];
}

}  // namespace

// Plane-major float [R|G|B] (each W*H) -> pixel-interleaved uint8 [RGB RGB ...].
__global__ void plane_to_interleaved_uint8_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    int total_pixels
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total_pixels) return;

    float r = in[0 * total_pixels + p];
    float g = in[1 * total_pixels + p];
    float b = in[2 * total_pixels + p];
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    out[p * 3 + 0] = static_cast<uint8_t>(r * 255.0f + 0.5f);
    out[p * 3 + 1] = static_cast<uint8_t>(g * 255.0f + 0.5f);
    out[p * 3 + 2] = static_cast<uint8_t>(b * 255.0f + 0.5f);
}

RenderedImage render_frame(
    const GaussianLOD& lod,
    const GatherResult& gather,
    const Camera& cam
) {
    const int D = lod.sh_degree;
    const int M = (D + 1) * (D + 1);
    const int P = static_cast<int>(gather.total_gaussians);
    const int W = cam.image_width;
    const int H = cam.image_height;

    RenderedImage img;
    img.width = W;
    img.height = H;
    img.rgb.assign(static_cast<size_t>(3) * W * H, 0);
    img.timings.P = P;
    if (P == 0) return img;

    using clk = std::chrono::high_resolution_clock;
    auto ms_between = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Persistent buffers — allocated once, grown across calls. Avoids per-frame
    // cudaMalloc/Free (~5 ms) and host buffer reallocation. Host SoA arrays are
    // pinned so cudaMemcpyAsync DMAs straight to device.
    static PinnedBuffer<float> h_means3D, h_shs, h_opacities, h_scales, h_rotations;
    static DeviceBuffer means_buf, shs_buf, opac_buf, scales_buf, rot_buf;
    static DeviceBuffer view_buf, proj_buf, campos_buf, bg_buf;
    static DeviceBuffer out_color_buf, out_color_u8_buf;
    static DeviceBuffer geom_buf, bin_buf, img_buf;

    // Dedicated upload stream so async H2D copies overlap with subsequent
    // CPU staging chunks within the same frame.
    static cudaStream_t upload_stream = nullptr;
    if (!upload_stream) cudaStreamCreate(&upload_stream);

    auto t0 = clk::now();

    // ----- 1. SoA staging + chunked async upload (CPU/GPU overlap) -----
    h_means3D  .resize(static_cast<size_t>(P) * 3);
    h_shs      .resize(static_cast<size_t>(P) * M * 3);
    h_opacities.resize(static_cast<size_t>(P));
    h_scales   .resize(static_cast<size_t>(P) * 3);
    h_rotations.resize(static_cast<size_t>(P) * 4);

    // Prefix sum: range_starts[i] = sum of counts of ranges [0, i). Gives each range
    // its destination index in the output arrays without any cross-thread coordination.
    std::vector<uint32_t> range_starts(gather.ranges.size() + 1);
    range_starts[0] = 0;
    for (size_t i = 0; i < gather.ranges.size(); i++) {
        range_starts[i + 1] = range_starts[i] + gather.ranges[i].count;
    }

    // Pre-size device buffers; chunked async copies write into slices of these.
    float* d_means3D   = reinterpret_cast<float*>(means_buf .ensure(static_cast<size_t>(P) * 3 * sizeof(float)));
    float* d_shs       = reinterpret_cast<float*>(shs_buf   .ensure(static_cast<size_t>(P) * M * 3 * sizeof(float)));
    float* d_opacities = reinterpret_cast<float*>(opac_buf  .ensure(static_cast<size_t>(P) * sizeof(float)));
    float* d_scales    = reinterpret_cast<float*>(scales_buf.ensure(static_cast<size_t>(P) * 3 * sizeof(float)));
    float* d_rotations = reinterpret_cast<float*>(rot_buf   .ensure(static_cast<size_t>(P) * 4 * sizeof(float)));

    const int rest_per_channel = M - 1;
    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t n_ranges = gather.ranges.size();

    // Chunk gather.ranges into K slices of approximately equal P. Within the loop:
    // chunk c's CPU staging runs while chunk c-1's H2D copy is in flight on the
    // upload stream → max(stage_total, upload_total) instead of sum.
    constexpr int kPipelineChunks = 4;
    std::array<size_t, kPipelineChunks + 1> chunk_starts{};
    chunk_starts[0] = 0;
    chunk_starts[kPipelineChunks] = n_ranges;
    for (int c = 1; c < kPipelineChunks; c++) {
        const uint32_t target_p = static_cast<uint32_t>((static_cast<size_t>(P) * c) / kPipelineChunks);
        auto it = std::lower_bound(range_starts.begin(), range_starts.end(), target_p);
        size_t idx = static_cast<size_t>(it - range_starts.begin());
        if (idx > n_ranges) idx = n_ranges;
        chunk_starts[c] = idx;
    }

    auto stage_subset = [&](size_t r_start, size_t r_end) {
        const size_t n_local = r_end - r_start;
        const size_t per_thread = (n_local + num_threads - 1) / num_threads;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (unsigned t = 0; t < num_threads; t++) {
            const size_t l_start = r_start + t * per_thread;
            const size_t l_end   = std::min(l_start + per_thread, r_end);
            if (l_start >= l_end) continue;
            threads.emplace_back([&, l_start, l_end]() {
                for (size_t ri = l_start; ri < l_end; ri++) {
                    const GatherRange& r = gather.ranges[ri];
                    const auto& level = lod.levels[r.level];
                    uint32_t out_idx = range_starts[ri];
                    for (uint32_t i = r.offset; i < r.offset + r.count; i++, out_idx++) {
                        const Gaussian& g = level[i];

                        h_means3D[out_idx * 3 + 0] = g.x;
                        h_means3D[out_idx * 3 + 1] = g.y;
                        h_means3D[out_idx * 3 + 2] = g.z;

                        // SH: dc first, then transpose channel-major (3, M-1) -> coefficient-major (M-1, 3)
                        const size_t sh_base = static_cast<size_t>(out_idx) * M * 3;
                        h_shs[sh_base + 0] = g.sh_dc[0];
                        h_shs[sh_base + 1] = g.sh_dc[1];
                        h_shs[sh_base + 2] = g.sh_dc[2];
                        for (int k = 0; k < rest_per_channel; k++) {
                            const size_t off = sh_base + 3 + k * 3;
                            h_shs[off + 0] = g.sh_rest[k];
                            h_shs[off + 1] = g.sh_rest[k + rest_per_channel];
                            h_shs[off + 2] = g.sh_rest[k + 2 * rest_per_channel];
                        }

                        h_opacities[out_idx] = g.opacity;

                        h_scales[out_idx * 3 + 0] = g.scale[0];
                        h_scales[out_idx * 3 + 1] = g.scale[1];
                        h_scales[out_idx * 3 + 2] = g.scale[2];

                        h_rotations[out_idx * 4 + 0] = g.rotation[0];
                        h_rotations[out_idx * 4 + 1] = g.rotation[1];
                        h_rotations[out_idx * 4 + 2] = g.rotation[2];
                        h_rotations[out_idx * 4 + 3] = g.rotation[3];
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
    };

    for (int c = 0; c < kPipelineChunks; c++) {
        const size_t r_start = chunk_starts[c];
        const size_t r_end   = chunk_starts[c + 1];
        if (r_start >= r_end) continue;

        stage_subset(r_start, r_end);

        const uint32_t p_start = range_starts[r_start];
        const uint32_t p_end   = range_starts[r_end];
        const size_t n = p_end - p_start;
        if (n == 0) continue;

        cudaMemcpyAsync(d_means3D   + p_start * 3,    h_means3D  .data() + p_start * 3,
                        n * 3 * sizeof(float),     cudaMemcpyHostToDevice, upload_stream);
        cudaMemcpyAsync(d_shs       + p_start * M*3, h_shs      .data() + p_start * M*3,
                        n * M*3 * sizeof(float),   cudaMemcpyHostToDevice, upload_stream);
        cudaMemcpyAsync(d_opacities + p_start,        h_opacities.data() + p_start,
                        n * sizeof(float),         cudaMemcpyHostToDevice, upload_stream);
        cudaMemcpyAsync(d_scales    + p_start * 3,    h_scales   .data() + p_start * 3,
                        n * 3 * sizeof(float),     cudaMemcpyHostToDevice, upload_stream);
        cudaMemcpyAsync(d_rotations + p_start * 4,    h_rotations.data() + p_start * 4,
                        n * 4 * sizeof(float),     cudaMemcpyHostToDevice, upload_stream);
    }

    auto t1 = clk::now();  // CPU loop done; some uploads may still be in flight on the GPU.

    // ----- 2. Camera marshaling: OpenGL -> COLMAP -----
    // Rasterizer expects +Z forward, +Y down (COLMAP). Our cam helpers build OpenGL
    // (-Z forward, +Y up). Flip Y and Z rows of view, and rebuild projection with
    // COLMAP signs.
    float view_gl[4][4], view_c[4][4];
    compute_view_matrix(cam, view_gl);
    for (int j = 0; j < 4; j++) {
        view_c[0][j] =  view_gl[0][j];
        view_c[1][j] = -view_gl[1][j];
        view_c[2][j] = -view_gl[2][j];
        view_c[3][j] =  view_gl[3][j];
    }

    const float n = cam.near_plane;
    const float f = cam.far_plane;
    const float tan_fovx = static_cast<float>(W) / (2.0f * cam.fx);
    const float tan_fovy = static_cast<float>(H) / (2.0f * cam.fy);
    float proj_c[4][4] = {};
    proj_c[0][0] = 1.0f / tan_fovx;
    proj_c[1][1] = 1.0f / tan_fovy;
    proj_c[2][2] = f / (f - n);
    proj_c[2][3] = -(f * n) / (f - n);
    proj_c[3][2] = 1.0f;

    float vp_c[4][4];
    multiply_4x4(proj_c, view_c, vp_c);

    float h_view[16], h_projfull[16];
    to_column_major(view_c, h_view);
    to_column_major(vp_c,   h_projfull);

    float h_campos[3]     = { cam.position[0], cam.position[1], cam.position[2] };
    float h_background[3] = { 0.0f, 0.0f, 0.0f };

    // Tiny matrix uploads (sync, default stream — kilobytes total, sub-ms).
    float* d_viewmatrix = upload_to(view_buf,   h_view,       16);
    float* d_projmatrix = upload_to(proj_buf,   h_projfull,   16);
    float* d_campos     = upload_to(campos_buf, h_campos,      3);
    float* d_background = upload_to(bg_buf,     h_background,  3);

    // Wait for all chunked SoA uploads to land before forward reads them.
    cudaStreamSynchronize(upload_stream);

    float*   d_out_color    = reinterpret_cast<float*>  (out_color_buf   .ensure(static_cast<size_t>(3) * W * H * sizeof(float)));
    uint8_t* d_out_color_u8 = reinterpret_cast<uint8_t*>(out_color_u8_buf.ensure(static_cast<size_t>(3) * W * H * sizeof(uint8_t)));

    auto t2 = clk::now();

    // ----- 4. Forward call (persistent geom/bin/img buffers from above) -----
    auto make_alloc = [](DeviceBuffer& b) {
        return [&b](size_t n) -> char* { return b.ensure(n); };
    };

    CudaRasterizer::Rasterizer::forward(
        make_alloc(geom_buf), make_alloc(bin_buf), make_alloc(img_buf),
        P, D, M,
        d_background,
        W, H,
        d_means3D, d_shs, /*colors_precomp=*/nullptr,
        d_opacities, d_scales, /*scale_modifier=*/1.0f,
        d_rotations, /*cov3D_precomp=*/nullptr,
        d_viewmatrix, d_projmatrix, d_campos,
        tan_fovx, tan_fovy,
        /*prefiltered=*/false,
        d_out_color, /*radii=*/nullptr, /*debug=*/false);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("[renderer] CUDA error after forward: %s\n", cudaGetErrorString(err));
    }

    auto t3 = clk::now();

    // ----- 5. GPU pack: float plane-major -> uint8 interleaved -----
    {
        const int total_pixels = W * H;
        const int block = 256;
        const int grid  = (total_pixels + block - 1) / block;
        plane_to_interleaved_uint8_kernel<<<grid, block>>>(d_out_color, d_out_color_u8, total_pixels);
        cudaDeviceSynchronize();
    }

    auto t4 = clk::now();

    // ----- 6. Download uint8 framebuffer (4x smaller than the float version) -----
    cudaMemcpy(img.rgb.data(), d_out_color_u8,
               static_cast<size_t>(3) * W * H * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    auto t5 = clk::now();

    // ----- 6. Cleanup (no-op: persistent buffers stay allocated across frames) -----
    auto t6 = clk::now();

    img.timings.soa_ms      = ms_between(t0, t1);
    img.timings.upload_ms   = ms_between(t1, t2);
    img.timings.forward_ms  = ms_between(t2, t3);
    img.timings.pack_ms     = ms_between(t3, t4);  // GPU convert kernel + sync
    img.timings.download_ms = ms_between(t4, t5);  // uint8 DtoH
    img.timings.cleanup_ms  = ms_between(t5, t6);

    return img;
}
