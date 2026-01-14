#include "ChCudaBroadphase.h"
#include "ChCudaCollisionTypes.h"
#include "ChCudaUtils.h"
#include <cooperative_groups.h>
#include <algorithm>

namespace chrono {
namespace collision {
namespace cuda {

namespace cg = cooperative_groups;

// Kernel to count objects per grid cell
__global__ void count_objects_per_cell_kernel(
    const GPU_AABB* __restrict__ aabbs,
    int numObjects,
    int* __restrict__ cellCounts,
    float3 gridMin,
    float3 gridMax,
    float cellSize,
    int gridResolution) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;

    const GPU_AABB& aabb = aabbs[idx];

    // Calculate grid cell bounds for this AABB
    int3 cellMin = make_int3(
        max(0, min(gridResolution - 1, (int)floorf((aabb.min.x - gridMin.x) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.min.y - gridMin.y) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.min.z - gridMin.z) / cellSize))));

    int3 cellMax = make_int3(
        max(0, min(gridResolution - 1, (int)floorf((aabb.max.x - gridMin.x) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.max.y - gridMin.y) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.max.z - gridMin.z) / cellSize))));

    // Count objects in each overlapping cell
    for (int x = cellMin.x; x <= cellMax.x; ++x) {
        for (int y = cellMin.y; y <= cellMax.y; ++y) {
            for (int z = cellMin.z; z <= cellMax.z; ++z) {
                int cellIndex = x + y * gridResolution + z * gridResolution * gridResolution;
                atomicAdd(&cellCounts[cellIndex], 1);
            }
        }
    }
}

// Kernel to assign objects to grid cells
__global__ void assign_objects_to_cells_kernel(
    const GPU_AABB* __restrict__ aabbs,
    int numObjects,
    int* __restrict__ cellCounters,
    int* __restrict__ objectIndices,
    float3 gridMin,
    float3 gridMax,
    float cellSize,
    int gridResolution) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;

    const GPU_AABB& aabb = aabbs[idx];

    // Calculate grid cell bounds for this AABB
    int3 cellMin = make_int3(
        max(0, min(gridResolution - 1, (int)floorf((aabb.min.x - gridMin.x) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.min.y - gridMin.y) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.min.z - gridMin.z) / cellSize))));

    int3 cellMax = make_int3(
        max(0, min(gridResolution - 1, (int)floorf((aabb.max.x - gridMin.x) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.max.y - gridMin.y) / cellSize))),
        max(0, min(gridResolution - 1, (int)floorf((aabb.max.z - gridMin.z) / cellSize))));

    // Assign object to each overlapping cell
    for (int x = cellMin.x; x <= cellMax.x; ++x) {
        for (int y = cellMin.y; y <= cellMax.y; ++y) {
            for (int z = cellMin.z; z <= cellMax.z; ++z) {
                int cellIndex = x + y * gridResolution + z * gridResolution * gridResolution;
                // Get position within this cell's range and store object index
                int position = atomicAdd(&cellCounters[cellIndex], 1);
                objectIndices[position] = idx;
            }
        }
    }
}

// Simple prefix sum kernel (works for small arrays like grid cells)
__global__ void prefix_sum_kernel(const int* __restrict__ input, int* __restrict__ output, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    // Simple sequential prefix sum per thread (inefficient but correct for small grids)
    int sum = 0;
    for (int i = 0; i <= idx; ++i) {
        if (i < idx) sum += input[i];
    }
    output[idx] = sum;
}

// Broadphase collision detection kernel
__global__ void broadphase_collision_kernel(
    const GPU_AABB* __restrict__ aabbs,
    const int* __restrict__ cellStarts,
    const int* __restrict__ cellCounts,
    const int* __restrict__ objectIndices,
    GPU_ContactManifold* __restrict__ manifolds,
    int* __restrict__ manifoldCount,
    int gridResolution) {

    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numCells = gridResolution * gridResolution * gridResolution;
    if (cellIdx >= numCells) return;

    int start = cellStarts[cellIdx];
    int count = cellCounts[cellIdx];

    // Process all pairs in this cell
    for (int i = 0; i < count; ++i) {
        int objA = objectIndices[start + i];
        const GPU_AABB& aabbA = aabbs[objA];

        for (int j = i + 1; j < count; ++j) {
            int objB = objectIndices[start + j];
            const GPU_AABB& aabbB = aabbs[objB];

            // Check collision groups/masks
            if ((aabbA.collisionGroup & aabbB.collisionMask) &&
                (aabbB.collisionGroup & aabbA.collisionMask)) {

                // AABB overlap test
                if (aabbA.max.x >= aabbB.min.x && aabbA.min.x <= aabbB.max.x &&
                    aabbA.max.y >= aabbB.min.y && aabbA.min.y <= aabbB.max.y &&
                    aabbA.max.z >= aabbB.min.z && aabbA.min.z <= aabbB.max.z) {

                    // Add to potential contact manifold
                    int idx = atomicAdd(manifoldCount, 1);
                    if (idx < MAX_MANIFOLDS) {
                        manifolds[idx].objectIdA = min(objA, objB);
                        manifolds[idx].objectIdB = max(objA, objB);
                        manifolds[idx].contactCount = 0;
                        manifolds[idx].contactOffset = 0;
                    }
                }
            }
        }
    }
}

void ChCudaBroadphase::AllocateDeviceMemory() {
    // Allocate cell data using RAII buffers
    d_cellCounts.Resize(m_numCells);
    d_cellStarts.Resize(m_numCells);
    d_cellCounters.Resize(m_numCells);

    // Estimate max object indices (each object can be in multiple cells)
    m_maxObjectIndices = MAX_OBJECTS * 8;  // Assume max 8 cells per object
    d_objectIndices.Resize(m_maxObjectIndices);
}

void ChCudaBroadphase::FreeDeviceMemory() {
    // RAII buffers automatically free memory
    d_cellCounts = ChCudaDeviceBuffer<int>();
    d_cellStarts = ChCudaDeviceBuffer<int>();
    d_cellCounters = ChCudaDeviceBuffer<int>();
    d_objectIndices = ChCudaDeviceBuffer<int>();
}

void ChCudaBroadphase::BuildGrid(const GPU_AABB* d_aabbs, int numObjects) {
    // Reset cell counts
    CUDA_CHECK(cudaMemset(d_cellCounts.Get(), 0, m_numCells * sizeof(int)));

    // Launch count kernel
    int blocks = (numObjects + 255) / 256;
    count_objects_per_cell_kernel<<<blocks, 256>>>(
        d_aabbs, numObjects, d_cellCounts.Get(),
        m_gridMin, m_gridMax, m_cellSize, m_gridResolution);
    CUDA_CHECK(cudaGetLastError());

    // Compute cell start indices (prefix sum)
    prefix_sum_kernel<<<(m_numCells + 255) / 256, 256>>>(
        d_cellCounts.Get(), d_cellStarts.Get(), m_numCells);
    CUDA_CHECK(cudaGetLastError());

    // Initialize cell counters to starting positions
    CUDA_CHECK(cudaMemcpy(d_cellCounters.Get(), d_cellStarts.Get(), m_numCells * sizeof(int), cudaMemcpyDeviceToDevice));

    // Launch assignment kernel
    assign_objects_to_cells_kernel<<<blocks, 256>>>(
        d_aabbs, numObjects, d_cellCounters.Get(), d_objectIndices.Get(),
        m_gridMin, m_gridMax, m_cellSize, m_gridResolution);
    CUDA_CHECK(cudaGetLastError());
}

void ChCudaBroadphase::DetectCollisions(const GPU_AABB* d_aabbs, GPU_ContactManifold* d_manifolds, int* d_manifoldCount) {
    // Reset manifold count
    CUDA_CHECK(cudaMemset(d_manifoldCount, 0, sizeof(int)));

    // Launch broadphase collision kernel
    int blocks = (m_numCells + 255) / 256;
    broadphase_collision_kernel<<<blocks, 256>>>(
        d_aabbs, d_cellStarts.Get(), d_cellCounts.Get(), d_objectIndices.Get(),
        d_manifolds, d_manifoldCount, m_gridResolution);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace collision
}  // namespace chrono