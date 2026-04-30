#include "interactive_viewer.h"
#include "renderer.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kSensitivity = 0.002f;
constexpr float kFastMul = 5.0f;

struct ViewerState {
    float e1[3], e2[3], up_n[3];
    float pos[3];
    float yaw = 0.0f;
    float pitch = 0.0f;

    double mouse_dx = 0.0;
    double mouse_dy = 0.0;
    bool first_mouse = true;
    double last_x = 0.0;
    double last_y = 0.0;

    float move_speed = 1.0f;
    bool cursor_captured = true;

    bool request_T = false;
    bool request_V = false;
    bool request_P = false;
    bool request_esc = false;
    int64_t vis_step = 0;
};

void normalize3(float v[3]) {
    float n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n > 0.0f) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

void build_basis(const float world_up[3], float e1[3], float e2[3], float up_n[3]) {
    up_n[0] = world_up[0]; up_n[1] = world_up[1]; up_n[2] = world_up[2];
    normalize3(up_n);
    float ref[3] = {1.0f, 0.0f, 0.0f};
    if (std::abs(up_n[0]) > 0.9f) { ref[0] = 0.0f; ref[1] = 1.0f; ref[2] = 0.0f; }
    cross3(ref, up_n, e1);
    normalize3(e1);
    cross3(up_n, e1, e2);
}

void compute_forward(float yaw, float pitch,
                     const float e1[3], const float e2[3], const float up_n[3],
                     float out[3]) {
    // yaw=0 -> -e1 (looking toward scene center from +e1 side)
    // yaw=π/2 -> +e2 (look right)
    // pitch tilts toward up_n.
    float c = std::cos(yaw), s = std::sin(yaw);
    float horiz[3];
    for (int i = 0; i < 3; i++) horiz[i] = -c * e1[i] + s * e2[i];
    float cp = std::cos(pitch), sp = std::sin(pitch);
    for (int i = 0; i < 3; i++) out[i] = cp * horiz[i] + sp * up_n[i];
}

void cursor_pos_callback(GLFWwindow* win, double x, double y) {
    auto* st = static_cast<ViewerState*>(glfwGetWindowUserPointer(win));
    if (!st->cursor_captured) {
        st->last_x = x;
        st->last_y = y;
        return;
    }
    if (st->first_mouse) {
        st->last_x = x;
        st->last_y = y;
        st->first_mouse = false;
        return;
    }
    st->mouse_dx += (x - st->last_x);
    st->mouse_dy += (y - st->last_y);
    st->last_x = x;
    st->last_y = y;
}

void scroll_callback(GLFWwindow* win, double /*xoff*/, double yoff) {
    auto* st = static_cast<ViewerState*>(glfwGetWindowUserPointer(win));
    st->move_speed *= std::pow(1.1f, static_cast<float>(yoff));
    if (st->move_speed < 1e-6f) st->move_speed = 1e-6f;
    printf("[scroll] move_speed=%.4f units/s\n", st->move_speed);
}

void key_callback(GLFWwindow* win, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) return;
    auto* st = static_cast<ViewerState*>(glfwGetWindowUserPointer(win));
    if      (key == GLFW_KEY_T)      st->request_T = true;
    else if (key == GLFW_KEY_V)      st->request_V = true;
    else if (key == GLFW_KEY_P)      st->request_P = true;
    else if (key == GLFW_KEY_ESCAPE) st->request_esc = true;
}

void mouse_button_callback(GLFWwindow* win, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
    auto* st = static_cast<ViewerState*>(glfwGetWindowUserPointer(win));
    if (!st->cursor_captured) {
        glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        st->cursor_captured = true;
        st->first_mouse = true;
    }
}

}  // namespace

void run_interactive_viewer(const GaussianLOD& lod,
                            const OctreeNode* root,
                            const InteractiveViewerConfig& cfg,
                            Visualizer* vis) {
    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return;
    }

    int win_w = cfg.image_width;
    int win_h = cfg.image_height;
    if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
        if (const GLFWvidmode* vm = glfwGetVideoMode(mon)) {
            win_w = std::min(win_w, vm->width  - 100);
            win_h = std::min(win_h, vm->height - 100);
        }
    }

    GLFWwindow* win = glfwCreateWindow(win_w, win_h, "LSGS", nullptr, nullptr);
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    ViewerState st;
    build_basis(cfg.world_up, st.e1, st.e2, st.up_n);
    for (int i = 0; i < 3; i++) {
        st.pos[i] = cfg.scene_center[i] + 1.5f * cfg.scene_radius * st.e1[i];
    }
    st.move_speed = std::max(cfg.scene_radius / 10.0f, 1e-3f);

    glfwSetWindowUserPointer(win, &st);
    glfwSetCursorPosCallback(win, cursor_pos_callback);
    glfwSetScrollCallback(win, scroll_callback);
    glfwSetKeyCallback(win, key_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(win, &fb_w, &fb_h);
    glViewport(0, 0, fb_w, fb_h);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 cfg.image_width, cfg.image_height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    printf("Interactive viewer: WASD move, Space/LShift up/down, mouse look, scroll speed,\n"
           "                    LCtrl fast, T timing, V dump-to-rerun, P print-pose,\n"
           "                    Esc release-cursor / quit, click to recapture cursor\n");
    printf("Initial pos=(%.2f, %.2f, %.2f) move_speed=%.3f units/s\n",
           st.pos[0], st.pos[1], st.pos[2], st.move_speed);

    double prev_time = glfwGetTime();
    double last_cam_ms = 0.0, last_gather_ms = 0.0;
    RenderTimings last_timings{};

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - prev_time);
        prev_time = now;

        st.yaw   += static_cast<float>(st.mouse_dx) * kSensitivity;
        st.pitch -= static_cast<float>(st.mouse_dy) * kSensitivity;
        st.pitch = std::max(-kPi * 0.5f + 1e-3f, std::min(kPi * 0.5f - 1e-3f, st.pitch));
        st.mouse_dx = 0.0;
        st.mouse_dy = 0.0;

        float forward[3], right[3];
        compute_forward(st.yaw, st.pitch, st.e1, st.e2, st.up_n, forward);
        cross3(forward, st.up_n, right);
        normalize3(right);

        const bool fast = glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
        const float v = st.move_speed * dt * (fast ? kFastMul : 1.0f);
        if (glfwGetKey(win, GLFW_KEY_W)          == GLFW_PRESS) for (int i=0;i<3;i++) st.pos[i] += forward[i] * v;
        if (glfwGetKey(win, GLFW_KEY_S)          == GLFW_PRESS) for (int i=0;i<3;i++) st.pos[i] -= forward[i] * v;
        if (glfwGetKey(win, GLFW_KEY_D)          == GLFW_PRESS) for (int i=0;i<3;i++) st.pos[i] += right[i]   * v;
        if (glfwGetKey(win, GLFW_KEY_A)          == GLFW_PRESS) for (int i=0;i<3;i++) st.pos[i] -= right[i]   * v;
        if (glfwGetKey(win, GLFW_KEY_SPACE)      == GLFW_PRESS) for (int i=0;i<3;i++) st.pos[i] += st.up_n[i] * v;
        if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) for (int i=0;i<3;i++) st.pos[i] -= st.up_n[i] * v;

        const float target[3] = {
            st.pos[0] + forward[0],
            st.pos[1] + forward[1],
            st.pos[2] + forward[2]
        };

        auto t_cam_a = std::chrono::high_resolution_clock::now();
        Camera cam = make_camera_lookat(st.pos, target, cfg.world_up,
                                        cfg.fx, cfg.fy, cfg.cx, cfg.cy,
                                        cfg.image_width, cfg.image_height,
                                        cfg.near_plane, cfg.far_plane);
        float view_m[4][4], proj_m[4][4], vp_m[4][4];
        compute_view_matrix(cam, view_m);
        compute_projection_matrix(cam, proj_m);
        multiply_4x4(proj_m, view_m, vp_m);
        Frustum frustum = extract_frustum(vp_m);
        auto t_cam_b = std::chrono::high_resolution_clock::now();
        last_cam_ms = std::chrono::duration<double, std::milli>(t_cam_b - t_cam_a).count();

        auto t_g_a = std::chrono::high_resolution_clock::now();
        GatherResult result = gather_visible_gaussians(root, lod, frustum, cam, cfg.lod_threshold);
        auto t_g_b = std::chrono::high_resolution_clock::now();
        last_gather_ms = std::chrono::duration<double, std::milli>(t_g_b - t_g_a).count();

        RenderedImage img = render_frame(lod, result, cam);
        last_timings = img.timings;

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        cfg.image_width, cfg.image_height,
                        GL_RGB, GL_UNSIGNED_BYTE, img.rgb.data());

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
            // Flip V: image row 0 is the top of the framebuffer, GL origin is bottom-left.
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(win);
        glfwPollEvents();

        if (st.request_T) {
            const RenderTimings& tt = last_timings;
            const double render_total = tt.soa_ms + tt.upload_ms + tt.forward_ms +
                                        tt.download_ms + tt.pack_ms + tt.cleanup_ms;
            const double total = last_cam_ms + last_gather_ms + render_total;
            printf("[T] cam=%.2f gather=%.2f | soa=%.2f upload=%.2f forward=%.2f"
                   " dl=%.2f pack=%.2f cleanup=%.2f | render=%.2f TOTAL=%.2f ms (%.1f FPS)"
                   " [P=%d, ranges=%zu, lod-used=%zu]\n",
                   last_cam_ms, last_gather_ms,
                   tt.soa_ms, tt.upload_ms, tt.forward_ms,
                   tt.download_ms, tt.pack_ms, tt.cleanup_ms,
                   render_total, total, total > 0 ? 1000.0 / total : 0.0,
                   tt.P, result.ranges.size(), result.nodes_lod_selected);
            printf("  culled: frustum=%zu\n", result.nodes_frustum_culled);

            // Per-level breakdown. Tag: L = leaf range (references original gaussians,
            // always tagged at lod.max_depth). S = summary range at the node's depth.
            std::vector<uint32_t> nodes_per_level(lod.max_depth + 1, 0);
            std::vector<uint64_t> g_per_level(lod.max_depth + 1, 0);
            for (const auto& r : result.ranges) {
                nodes_per_level[r.level]++;
                g_per_level[r.level] += r.count;
            }
            printf("  levels:");
            for (int d = 0; d <= lod.max_depth; d++) {
                if (nodes_per_level[d] == 0) continue;
                const char tag = (d == lod.max_depth) ? 'L' : 'S';
                printf(" d%d%c(n=%u,g=%llu)", d, tag,
                       nodes_per_level[d],
                       static_cast<unsigned long long>(g_per_level[d]));
            }
            printf("\n");

            st.request_T = false;
        }
        if (st.request_P) {
            printf("[P] pos=(%.4f, %.4f, %.4f) yaw=%.4f pitch=%.4f fwd=(%.4f, %.4f, %.4f)\n",
                   st.pos[0], st.pos[1], st.pos[2], st.yaw, st.pitch,
                   forward[0], forward[1], forward[2]);
            st.request_P = false;
        }
        if (st.request_V) {
            if (vis) {
                OctreeDebugTrace trace = debug_gather_trace(root, frustum, cam, cfg.lod_threshold);
                vis->set_step(st.vis_step++);
                vis->log_octree_boxes("world/octree", trace.selected, lod.max_depth);
                vis->log_camera("world/camera", cam, 100.0f);
                printf("[V] step=%lld trace: all=%zu frustum_passed=%zu selected=%zu\n",
                       static_cast<long long>(st.vis_step - 1),
                       trace.all.size(), trace.frustum_passed.size(), trace.selected.size());
            } else {
                printf("[V] no visualizer (--vis not set)\n");
            }
            st.request_V = false;
        }
        if (st.request_esc) {
            if (st.cursor_captured) {
                glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                st.cursor_captured = false;
                st.first_mouse = true;
                printf("[Esc] cursor released (Esc again to quit, click to recapture)\n");
            } else {
                glfwSetWindowShouldClose(win, GLFW_TRUE);
            }
            st.request_esc = false;
        }
    }

    glDeleteTextures(1, &tex);
    glfwDestroyWindow(win);
    glfwTerminate();
}
