#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

namespace chrono {
namespace collision {
namespace cuda {

// GPU-friendly AABB structure
struct GPU_AABB {
    float3 min;
    int objectId;
    float3 max;
    int shapeType;
    int collisionGroup;
    int collisionMask;
};

// Compact contact point structure
struct GPU_ContactPoint {
    float3 positionOnA;
    float3 positionOnB;
    float3 normal;
    float distance;
    int shapeIndexA;
    int shapeIndexB;
};

// Contact manifold structure
struct GPU_ContactManifold {
    int objectIdA;
    int objectIdB;
    int contactCount;
    int contactOffset;  // Offset in global contact array
};

// Grid cell structure for spatial hashing
struct GPU_GridCell {
    int startIndex;  // Start index in object list
    int endIndex;    // End index in object list
};

// Constants
const int MAX_OBJECTS = 100000;
const int MAX_MANIFOLDS = 500000;
const int MAX_CONTACTS = 1000000;
const int GRID_RESOLUTION = 32;  // 32x32x32 grid

}  // namespace cuda
}  // namespace collision
}  // namespace chrono