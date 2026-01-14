#include "ChCudaBroadphase.h"
#include <algorithm>
#include <cmath>

namespace chrono {
namespace collision {
namespace cuda {

ChCudaBroadphase::ChCudaBroadphase(ChCudaDeviceManager* deviceManager)
    : m_deviceManager(deviceManager),
      m_gridMin(make_float3(0, 0, 0)),
      m_gridMax(make_float3(0, 0, 0)),
      m_cellSize(1.0f),
      m_gridResolution(GRID_RESOLUTION),
      m_maxObjectIndices(0) {
    m_numCells = m_gridResolution * m_gridResolution * m_gridResolution;
}

ChCudaBroadphase::~ChCudaBroadphase() {
    FreeDeviceMemory();
}

void ChCudaBroadphase::Initialize(float3 sceneMin, float3 sceneMax, float cellSize) {
    m_gridMin = sceneMin;
    m_gridMax = sceneMax;
    m_cellSize = cellSize;

    // Calculate grid resolution based on scene size
    float3 sceneSize = make_float3(
        sceneMax.x - sceneMin.x,
        sceneMax.y - sceneMin.y,
        sceneMax.z - sceneMin.z);

    m_gridResolution = std::max(1, std::min(GRID_RESOLUTION,
        (int)ceilf(std::max(std::max(sceneSize.x, sceneSize.y), sceneSize.z) / cellSize)));

    m_numCells = m_gridResolution * m_gridResolution * m_gridResolution;

    AllocateDeviceMemory();
}

}  // namespace cuda
}  // namespace collision
}  // namespace chrono