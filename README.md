# LSGS — Large-Scale Gaussian Splatting

Renderer for 3D Gaussian Splatting scenes that are too large to fit fully in GPU memory. Each frame, only the **LOD-appropriate, frustum-visible subset** of gaussians is selected on the CPU, streamed to GPU, and rasterized with the Inria `diff-gaussian-rasterization` kernel.

## Demo

<!-- Replace with your own recording -->
![LSGS interactive viewer](docs/demo.gif)

<!-- Or for a video:
<video src="docs/demo.mp4" controls></video>
-->

## How LOD works

The pipeline uses a classic **octree + distance-based voxel LOD selection + Gaussian summaries**:

1. **Octree partitioning.** The scene is wrapped in a cubic bounding box and recursively split into 8 children. Cells stop splitting once they contain few enough gaussians (or hit a max depth). The original gaussians live only at the bottom of the tree.

2. **Per-node Gaussian summaries.** Every internal node stores a small set of "summary gaussians" that approximate all the gaussians inside it. They are computed by fitting Gaussians to the statistical moments (mean, covariance, color) of the children, so each summary captures the rough position, orientation, size, and color of a chunk of the scene without having to draw thousands of individualP gaussians.

3. **Per-frame selection (the LOD decision).** For each frame, the renderer walks the octree and asks two questions per node:
   - **Is the node visible?** Cull it against the camera frustum if not.
   - **Is the node small on screen?** If the node's projected size on screen is below a threshold, draw its summary instead of recursing further. If it's still large, recurse into the children — and eventually fall through to the original gaussians for nearby cells.

   This means **far / coarse parts of the scene get summaries** (cheap), and **only what's close to the camera gets the full-detail gaussians** (expensive but worth it).

4. **Streaming to the GPU.** The selected subset is uploaded to the GPU each frame using overlapped chunked transfers, so the upload is happening at the same time as the next chunk is being prepared on the CPU. Nothing is uploaded once-and-for-all — large scenes that wouldn't fit in VRAM are still rendered because only a fraction is resident at any moment.

5. **Rasterization.** The Inria `diff-gaussian-rasterization` forward kernel rasterizes the selected gaussians, and the resulting image is shown via OpenGL.

## Requirements

- **CUDA Toolkit** 12.x or 13.x and a compatible NVIDIA driver
- **NVIDIA GPU** (Compute Capability detected automatically via `CUDA_ARCHITECTURES native`)
- **OpenGL** (system package)
- **CMake** ≥ 3.24
- **C++17** compiler (GCC 11+ tested)
- Linux (developed on Ubuntu)

GLFW and the Rerun C++ SDK are pulled in automatically by CMake `FetchContent` — no system install needed.

## Clone and build

The project uses git submodules for `happly` (PLY parser) and Inria's `diff-gaussian-rasterization`, so clone with `--recursive`:

```bash
git clone --recursive https://github.com/<your-user>/LSGS.git
cd LSGS
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Then build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The executable is at `build/lsgs`.

## Usage

```bash
./build/lsgs <path/to/scene.ply> [--up X Y Z]
```

| Flag | Default | Description |
|---|---|---|
| `<ply>` | required | Path to a 3DGS `.ply` (positions, opacity, scale, rotation, SH DC + rest). |
| `--up X Y Z` | `0 0 1` | World up vector. Use `--up 0 1 0` for Y-up datasets. |

Example:

```bash
./build/lsgs data/point_cloud_truck.ply --up 0 1 0
```

## Interactive viewer controls

The viewer captures the cursor on launch (FPS-style mouse look). Press **Esc** once to release the cursor; click in the window to recapture.

| Input | Action |
|---|---|
| `W` / `A` / `S` / `D` | Move forward / left / back / right |
| `Space` | Move up (along world up) |
| `Left Shift` | Move down (along world up) |
| Mouse | Look around (yaw / pitch) |
| Scroll wheel | Increase / decrease move speed (×1.1 per tick) |
| `Left Ctrl` (held) | Fast move (5× current speed) |
| `T` | Print per-frame timing breakdown + per-LOD-level stats |
| `P` | Print current camera pose (position, yaw, pitch, forward) |
| `Esc` | First press: release cursor. Second press: quit. |
| Left click | Recapture cursor after release |