#include "ChCudaNarrowphase.h"
#include "ChCudaCollisionTypes.h"
#include "ChCudaUtils.h"
#include <cooperative_groups.h>
#include <algorithm>

namespace chrono {
namespace collision {
namespace cuda {

namespace cg = cooperative_groups;

// Shape type constants (matching ChCollisionShape::Type)
const int SHAPE_SPHERE = 0;
const int SHAPE_BOX = 2;
const int SHAPE_CAPSULE = 11;
const int SHAPE_CONVEXHULL = 6;

// Utility functions for collision detection

// Sphere-sphere collision
__device__ bool sphere_sphere_collision(
    const GPU_Sphere& sphereA, const GPU_Sphere& sphereB,
    float3& contactPointA, float3& contactPointB, float3& normal, float& distance) {

    float3 delta = make_float3(sphereB.position.x - sphereA.position.x,
                              sphereB.position.y - sphereA.position.y,
                              sphereB.position.z - sphereA.position.z);
    float dist = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

    if (dist >= sphereA.radius + sphereB.radius) {
        return false;  // No collision
    }

    // Collision detected
    if (dist > 1e-6f) {
        normal = make_float3(delta.x / dist, delta.y / dist, delta.z / dist);
    } else {
        normal = make_float3(1.0f, 0.0f, 0.0f);
    }
    distance = dist - (sphereA.radius + sphereB.radius);

    // Contact points
    contactPointA = make_float3(sphereA.position.x + normal.x * sphereA.radius,
                               sphereA.position.y + normal.y * sphereA.radius,
                               sphereA.position.z + normal.z * sphereA.radius);
    contactPointB = make_float3(sphereB.position.x - normal.x * sphereB.radius,
                               sphereB.position.y - normal.y * sphereB.radius,
                               sphereB.position.z - normal.z * sphereB.radius);

    return true;
}

// Box-box collision (SAT - Separating Axis Theorem)
__device__ bool box_box_collision(
    const GPU_Box& boxA, const GPU_Box& boxB,
    float3& contactPointA, float3& contactPointB, float3& normal, float& distance) {

    // For simplicity, implement AABB-AABB collision first
    // TODO: Implement full OBB-OBB collision with rotations

    float3 minA = make_float3(boxA.position.x - boxA.halfExtents.x,
                             boxA.position.y - boxA.halfExtents.y,
                             boxA.position.z - boxA.halfExtents.z);
    float3 maxA = make_float3(boxA.position.x + boxA.halfExtents.x,
                             boxA.position.y + boxA.halfExtents.y,
                             boxA.position.z + boxA.halfExtents.z);
    float3 minB = make_float3(boxB.position.x - boxB.halfExtents.x,
                             boxB.position.y - boxB.halfExtents.y,
                             boxB.position.z - boxB.halfExtents.z);
    float3 maxB = make_float3(boxB.position.x + boxB.halfExtents.x,
                             boxB.position.y + boxB.halfExtents.y,
                             boxB.position.z + boxB.halfExtents.z);

    // Check for overlap
    if (maxA.x < minB.x || minA.x > maxB.x ||
        maxA.y < minB.y || minA.y > maxB.y ||
        maxA.z < minB.z || minA.z > maxB.z) {
        return false;
    }

    // Find penetration depths
    float dx1 = maxB.x - minA.x;
    float dx2 = maxA.x - minB.x;
    float dy1 = maxB.y - minA.y;
    float dy2 = maxA.y - minB.y;
    float dz1 = maxB.z - minA.z;
    float dz2 = maxA.z - minB.z;

    float min_dx = fminf(dx1, dx2);
    float min_dy = fminf(dy1, dy2);
    float min_dz = fminf(dz1, dz2);
    float minPenetration = fminf(fminf(min_dx, min_dy), min_dz);

    // Determine normal based on minimum penetration
    if (minPenetration == dx1) {
        normal = make_float3(-1.0f, 0.0f, 0.0f);
        distance = -dx1;
    } else if (minPenetration == dx2) {
        normal = make_float3(1.0f, 0.0f, 0.0f);
        distance = -dx2;
    } else if (minPenetration == dy1) {
        normal = make_float3(0.0f, -1.0f, 0.0f);
        distance = -dy1;
    } else if (minPenetration == dy2) {
        normal = make_float3(0.0f, 1.0f, 0.0f);
        distance = -dy2;
    } else if (minPenetration == dz1) {
        normal = make_float3(0.0f, 0.0f, -1.0f);
        distance = -dz1;
    } else {
        normal = make_float3(0.0f, 0.0f, 1.0f);
        distance = -dz2;
    }

    // Simple contact point calculation (center of overlap)
    contactPointA = boxA.position;
    contactPointB = boxB.position;

    return true;
}

// Capsule-capsule collision
__device__ bool capsule_capsule_collision(
    const GPU_Capsule& capA, const GPU_Capsule& capB,
    float3& contactPointA, float3& contactPointB, float3& normal, float& distance) {

    // Compute capsule segment endpoints
    float3 halfHeightA = make_float3(capA.direction.x * capA.height * 0.5f,
                                    capA.direction.y * capA.height * 0.5f,
                                    capA.direction.z * capA.height * 0.5f);
    float3 a1 = make_float3(capA.position.x - halfHeightA.x,
                           capA.position.y - halfHeightA.y,
                           capA.position.z - halfHeightA.z);
    float3 a2 = make_float3(capA.position.x + halfHeightA.x,
                           capA.position.y + halfHeightA.y,
                           capA.position.z + halfHeightA.z);

    float3 halfHeightB = make_float3(capB.direction.x * capB.height * 0.5f,
                                    capB.direction.y * capB.height * 0.5f,
                                    capB.direction.z * capB.height * 0.5f);
    float3 b1 = make_float3(capB.position.x - halfHeightB.x,
                           capB.position.y - halfHeightB.y,
                           capB.position.z - halfHeightB.z);
    float3 b2 = make_float3(capB.position.x + halfHeightB.x,
                           capB.position.y + halfHeightB.y,
                           capB.position.z + halfHeightB.z);

    // Find closest points on the two line segments
    float3 d1 = make_float3(a2.x - a1.x, a2.y - a1.y, a2.z - a1.z);
    float3 d2 = make_float3(b2.x - b1.x, b2.y - b1.y, b2.z - b1.z);
    float3 r = make_float3(a1.x - b1.x, a1.y - b1.y, a1.z - b1.z);

    float a = d1.x*d1.x + d1.y*d1.y + d1.z*d1.z;
    float b = d1.x*d2.x + d1.y*d2.y + d1.z*d2.z;
    float c = d1.x*r.x + d1.y*r.y + d1.z*r.z;
    float d = d2.x*d2.x + d2.y*d2.y + d2.z*d2.z;
    float e = d2.x*r.x + d2.y*r.y + d2.z*r.z;

    float denom = a*d - b*b;

    float t, s;
    if (denom < 1e-6f) {
        // Lines are parallel
        t = 0.0f;
        s = (b > c ? 1.0f : 0.0f);
    } else {
        t = (b*e - c*d) / denom;
        s = (a*e - b*c) / denom;
        t = fmaxf(0.0f, fminf(1.0f, t));
        s = fmaxf(0.0f, fminf(1.0f, s));
    }

    // Closest points
    float3 p1 = make_float3(a1.x + d1.x*t, a1.y + d1.y*t, a1.z + d1.z*t);
    float3 p2 = make_float3(b1.x + d2.x*s, b1.y + d2.y*s, b1.z + d2.z*s);

    // Distance between closest points
    float3 delta = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    float dist = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);

    float combinedRadius = capA.radius + capB.radius;
    if (dist >= combinedRadius) {
        return false;  // No collision
    }

    // Collision detected
    if (dist > 1e-6f) {
        normal = make_float3(delta.x / dist, delta.y / dist, delta.z / dist);
    } else {
        normal = make_float3(1.0f, 0.0f, 0.0f);
    }
    distance = dist - combinedRadius;

    // Contact points
    contactPointA = make_float3(p1.x + normal.x * capA.radius,
                               p1.y + normal.y * capA.radius,
                               p1.z + normal.z * capA.radius);
    contactPointB = make_float3(p2.x - normal.x * capB.radius,
                               p2.y - normal.y * capB.radius,
                               p2.z - normal.z * capB.radius);

    return true;
}

// Convex hull collision (simplified GJK implementation)
__device__ bool convex_convex_collision(
    const GPU_ConvexHull& hullA, const GPU_ConvexHull& hullB,
    const float3* vertices, const int* faces,
    float3& contactPointA, float3& contactPointB, float3& normal, float& distance) {

    // Simplified convex hull collision - check if AABBs overlap first
    // TODO: Implement full GJK algorithm

    // Get vertex ranges
    const float3* vertsA = vertices + hullA.vertexOffset;
    const float3* vertsB = vertices + hullB.vertexOffset;

    // Compute AABBs
    float3 minA = vertsA[0], maxA = vertsA[0];
    for (int i = 1; i < hullA.vertexCount; ++i) {
        minA.x = fminf(minA.x, vertsA[i].x);
        minA.y = fminf(minA.y, vertsA[i].y);
        minA.z = fminf(minA.z, vertsA[i].z);
        maxA.x = fmaxf(maxA.x, vertsA[i].x);
        maxA.y = fmaxf(maxA.y, vertsA[i].y);
        maxA.z = fmaxf(maxA.z, vertsA[i].z);
    }

    float3 minB = vertsB[0], maxB = vertsB[0];
    for (int i = 1; i < hullB.vertexCount; ++i) {
        minB.x = fminf(minB.x, vertsB[i].x);
        minB.y = fminf(minB.y, vertsB[i].y);
        minB.z = fminf(minB.z, vertsB[i].z);
        maxB.x = fmaxf(maxB.x, vertsB[i].x);
        maxB.y = fmaxf(maxB.y, vertsB[i].y);
        maxB.z = fmaxf(maxB.z, vertsB[i].z);
    }

    // AABB overlap check
    if (maxA.x < minB.x || minA.x > maxB.x ||
        maxA.y < minB.y || minA.y > maxB.y ||
        maxA.z < minB.z || minA.z > maxB.z) {
        return false;
    }

    // Simple collision response - center points
    contactPointA = hullA.position;
    contactPointB = hullB.position;
    normal = make_float3(0.0f, 1.0f, 0.0f);  // Default normal
    distance = -0.1f;  // Small penetration

    return true;
}

// Main narrow phase kernel
__global__ void narrowphase_collision_kernel(
    const GPU_ContactManifold* __restrict__ manifolds,
    int manifoldCount,
    const GPU_ShapeData* __restrict__ shapeData,
    const GPU_AABB* __restrict__ aabbs,
    const float3* __restrict__ convexVertices,
    const int* __restrict__ convexFaces,
    GPU_ContactPoint* __restrict__ contacts,
    int* __restrict__ contactCount,
    int maxContactsPerManifold) {

    int manifoldIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (manifoldIdx >= manifoldCount) return;

    const GPU_ContactManifold& manifold = manifolds[manifoldIdx];
    int objA = manifold.objectIdA;
    int objB = manifold.objectIdB;

    const GPU_AABB& aabbA = aabbs[objA];
    const GPU_AABB& aabbB = aabbs[objB];

    int shapeTypeA = aabbA.shapeType;
    int shapeTypeB = aabbB.shapeType;

    const GPU_ShapeData& shapeA = shapeData[objA];
    const GPU_ShapeData& shapeB = shapeData[objB];

    GPU_ContactPoint contact;
    bool collision = false;

    // Dispatch based on shape types
    if (shapeTypeA == SHAPE_SPHERE && shapeTypeB == SHAPE_SPHERE) {
        collision = sphere_sphere_collision(
            shapeA.sphere, shapeB.sphere,
            contact.positionOnA, contact.positionOnB, contact.normal, contact.distance);

    } else if (shapeTypeA == SHAPE_BOX && shapeTypeB == SHAPE_BOX) {
        collision = box_box_collision(
            shapeA.box, shapeB.box,
            contact.positionOnA, contact.positionOnB, contact.normal, contact.distance);

    } else if (shapeTypeA == SHAPE_CAPSULE && shapeTypeB == SHAPE_CAPSULE) {
        collision = capsule_capsule_collision(
            shapeA.capsule, shapeB.capsule,
            contact.positionOnA, contact.positionOnB, contact.normal, contact.distance);

    } else if (shapeTypeA == SHAPE_CONVEXHULL && shapeTypeB == SHAPE_CONVEXHULL) {
        collision = convex_convex_collision(
            shapeA.convexHull, shapeB.convexHull,
            convexVertices, convexFaces,
            contact.positionOnA, contact.positionOnB, contact.normal, contact.distance);
    }
    // TODO: Add mixed shape collisions (sphere-box, etc.)

    if (collision) {
        contact.shapeIndexA = objA;
        contact.shapeIndexB = objB;

        // Add contact to global array
        int contactIdx = atomicAdd(contactCount, 1);
        if (contactIdx < MAX_CONTACTS) {
            contacts[contactIdx] = contact;
        }
    }
}

ChCudaNarrowphase::ChCudaNarrowphase(ChCudaDeviceManager* deviceManager)
    : m_deviceManager(deviceManager),
      d_shapeData(nullptr),
      m_numShapes(0),
      d_convexVertices(nullptr),
      d_convexFaces(nullptr),
      m_maxConvexVertices(0),
      m_maxConvexFaces(0),
      m_maxContactsPerManifold(4) {

    AllocateDeviceMemory();
}

ChCudaNarrowphase::~ChCudaNarrowphase() {
    FreeDeviceMemory();
}

void ChCudaNarrowphase::Initialize(const GPU_ShapeData* d_shapeData, int numShapes) {
    this->d_shapeData = d_shapeData;
    this->m_numShapes = numShapes;
}

void ChCudaNarrowphase::ProcessManifolds(const GPU_ContactManifold* d_manifolds,
                                        int manifoldCount,
                                        const GPU_AABB* d_aabbs,
                                        GPU_ContactPoint* d_contacts,
                                        int* d_contactCount) {
    // Reset contact count
    CUDA_CHECK(cudaMemset(d_contactCount, 0, sizeof(int)));

    // Launch narrow phase kernel
    int blocks = (manifoldCount + 255) / 256;
    narrowphase_collision_kernel<<<blocks, 256>>>(
        d_manifolds, manifoldCount, d_shapeData, d_aabbs,
        d_convexVertices, d_convexFaces,
        d_contacts, d_contactCount, m_maxContactsPerManifold);
    CUDA_CHECK(cudaGetLastError());
}

void ChCudaNarrowphase::AllocateDeviceMemory() {
    // Allocate convex hull data
    m_maxConvexVertices = 10000;  // TODO: Make configurable
    m_maxConvexFaces = 5000;

    CUDA_CHECK(cudaMalloc(&d_convexVertices, m_maxConvexVertices * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_convexFaces, m_maxConvexFaces * sizeof(int)));
}

void ChCudaNarrowphase::FreeDeviceMemory() {
    if (d_convexVertices) cudaFree(d_convexVertices);
    if (d_convexFaces) cudaFree(d_convexFaces);

    d_convexVertices = nullptr;
    d_convexFaces = nullptr;
}

}  // namespace cuda
}  // namespace collision
}  // namespace chrono