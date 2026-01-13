#pragma once

#include "ChCudaCollisionTypes.h"
#include "ChCudaDeviceManager.h"
#include <vector>

namespace chrono {
namespace collision {
namespace cuda {

class ChCudaBroadphase {
public:
    ChCudaBroadphase(ChCudaDeviceManager* deviceManager);
    ~ChCudaBroadphase();

    // Initialize with scene bounds
    void Initialize(float3 sceneMin, float3 sceneMax, float cellSize = 1.0f);

    // Build spatial grid from AABBs
    void BuildGrid(const GPU_AABB* d_aabbs, int numObjects);

    // Perform broadphase collision detection
    void DetectCollisions(const GPU_AABB* d_aabbs, GPU_ContactManifold* d_manifolds, int* d_manifoldCount);

    // Get grid parameters
    int GetGridResolution() const { return m_gridResolution; }
    float3 GetGridMin() const { return m_gridMin; }
    float3 GetGridMax() const { return m_gridMax; }
    float GetCellSize() const { return m_cellSize; }

private:
    ChCudaDeviceManager* m_deviceManager;

    // Grid parameters
    float3 m_gridMin;
    float3 m_gridMax;
    float m_cellSize;
    int m_gridResolution;
    int m_numCells;

    // Device memory
    int* d_cellCounts;
    int* d_cellStarts;
    int* d_cellCounters;  // Temporary counters for assignment
    int* d_objectIndices;
    int m_maxObjectIndices;

    // Helper functions
    void AllocateDeviceMemory();
    void FreeDeviceMemory();
};

}  // namespace cuda
}  // namespace collision
}  // namespace chrono