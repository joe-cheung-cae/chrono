#pragma once

#include "ChCudaCollisionTypes.h"
#include "ChCudaDeviceManager.h"
#include <vector>

namespace chrono {
namespace collision {
namespace cuda {

// GPU-friendly shape data structures
struct GPU_Sphere {
    float3 position;
    float radius;
};

struct GPU_Box {
    float3 position;
    float3 halfExtents;
    float4 orientation;  // quaternion
};

struct GPU_Capsule {
    float3 position;
    float3 direction;  // normalized axis direction
    float radius;
    float height;      // cylinder height (excluding caps)
};

struct GPU_ConvexHull {
    float3 position;
    float4 orientation;
    int vertexOffset;   // offset in global vertex array
    int vertexCount;
    int faceOffset;     // offset in global face array
    int faceCount;
};

// Shape data union for efficient storage
struct GPU_ShapeData {
    union {
        GPU_Sphere sphere;
        GPU_Box box;
        GPU_Capsule capsule;
        GPU_ConvexHull convexHull;
    };
};

class ChCudaNarrowphase {
public:
    ChCudaNarrowphase(ChCudaDeviceManager* deviceManager);
    ~ChCudaNarrowphase();

    // Initialize with shape data
    void Initialize(const GPU_ShapeData* d_shapeData, int numShapes);

    // Process manifolds and generate contact points
    void ProcessManifolds(const GPU_ContactManifold* d_manifolds,
                         int manifoldCount,
                         const GPU_AABB* d_aabbs,
                         GPU_ContactPoint* d_contacts,
                         int* d_contactCount);

    // Get maximum number of contacts per manifold
    int GetMaxContactsPerManifold() const { return m_maxContactsPerManifold; }

private:
    ChCudaDeviceManager* m_deviceManager;

    // Shape data
    const GPU_ShapeData* d_shapeData;
    int m_numShapes;

    // Convex hull data (if needed)
    float3* d_convexVertices;
    int* d_convexFaces;
    int m_maxConvexVertices;
    int m_maxConvexFaces;

    // Configuration
    int m_maxContactsPerManifold;

    // Helper functions
    void AllocateDeviceMemory();
    void FreeDeviceMemory();
};

}  // namespace cuda
}  // namespace collision
}  // namespace chrono