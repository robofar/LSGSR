# LSGS — Large-Scale Gaussian Splatting

Renderer for 3D Gaussian Splatting scenes that are too large to fit fully in GPU memory. Each frame, only the **LOD-appropriate, frustum-visible subset** of gaussians is selected on the CPU, streamed to GPU, and rasterized with the Inria `diff-gaussian-rasterization` kernel.

## Demo

<video src="https://github.com/robofar/LSGSR/releases/download/demo/demo.mp4" controls width="720"></video>

## How LOD works

The pipeline uses a classic **octree + distance-based voxel LOD selection + Gaussian summaries**:

1. **Octree build.** The scene is partitioned into an octree where leaf nodes contain no more than 256 Gaussians or are created upon reaching a maximum depth. The original Gaussians are stored in the leaf nodes.

2. **Per-node Gaussian summaries.** Each internal node approximates the Gaussians in its eight children using a small set of *summary Gaussians*. The node’s volume is split into a `6×6×6` grid, and each cell fits a representative Gaussian from the Gaussians it contains (via mean, covariance, and color). These summaries capture the coarse structure and appearance of the region.

3. **Per-frame Gaussians selection (the LOD decision).** For each frame, the renderer traverses the octree and checks:
   - **Is the node inside camera frustum?** If not, skip the node and its subtree.
   - **Is the node small enough on screen?** If yes, use that node’s Gaussians. If not, recurse into its children.

   Result: distant areas use coarse summaries, while nearby regions use fine Gaussians.

4. **Streaming to the GPU.** Each frame, only the selected Gaussians are uploaded to the GPU. Nothing is kept permanently in VRAM—only a small, temporary working set is resident at any time.

5. **Rasterization.** The selected Gaussians are rasterized using Inria’s `diff-gaussian-rasterization` forward kernel, and the final image is displayed via OpenGL. GLFW is used to handle the window and input, allowing real-time navigation through the scene.

## Requirements

- Linux (developed on Ubuntu)
- **NVIDIA GPU**
- **CUDA Toolkit** 12.x or 13.x and a compatible NVIDIA driver
- **OpenGL** (system package)
- **CMake** ≥ 3.24
- **C++17** compiler (GCC 11+ tested)

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