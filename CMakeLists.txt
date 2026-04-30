cmake_minimum_required(VERSION 3.24)
project(LSGS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Rerun SDK
include(FetchContent)
FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.30.2/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

# GLFW (window + input for interactive viewer)
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

find_package(OpenGL REQUIRED)

# Inria diff-gaussian-rasterization (provides CudaRasterizer static lib)
add_subdirectory(third_party/diff-gaussian-rasterization)
# Override the upstream's hardcoded "70;75;86" — CUDA 13 dropped sm_70.
# `native` targets the host GPU (resolved at configure time).
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES native)
# rasterizer_impl.h uses uint32_t / std::uintptr_t without including <cstdint>.
# Older toolchains pulled it in transitively; force-include it for CUDA 13 + GCC 13.
target_compile_options(CudaRasterizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:--pre-include cstdint>
)

# PLY loader library (pure C++, no CUDA)
add_library(ply_loader
    src/ply_loader.cpp
    src/morton.cpp
    src/visualizer.cpp
    src/octree.cpp
    src/frustum_culling.cpp
    src/interactive_viewer.cpp
)
target_include_directories(ply_loader PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/happly
    ${CMAKE_SOURCE_DIR}/third_party/stb
)
target_link_libraries(ply_loader PUBLIC Threads::Threads rerun_sdk glfw OpenGL::GL)

add_library(gaussian src/gaussian.cpp)
target_include_directories(gaussian PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# CUDA-side renderer
add_library(gpu_renderer src/renderer.cu)
target_include_directories(gpu_renderer PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_renderer PUBLIC ply_loader CudaRasterizer)

# Main executable
add_executable(lsgs src/main.cpp)
target_link_libraries(lsgs PRIVATE ply_loader gaussian gpu_renderer glfw OpenGL::GL)
