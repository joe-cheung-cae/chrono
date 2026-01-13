#include "ChCudaNarrowphase.h"
#include "ChCudaBroadphase.h"
#include "ChCudaCollisionTypes.h"
#include "ChCudaDeviceManager.h"
#include <iostream>
#include <vector>
#include <chrono>

namespace chrono {
namespace collision {
namespace cuda {

void TestSphereSphereCollision() {
    std::cout << "Testing sphere-sphere collision..." << std::endl;

    ChCudaDeviceManager& deviceManager = ChCudaDeviceManager::GetInstance();
    deviceManager.Initialize();
    ChCudaNarrowphase narrowphase(&deviceManager);

    // Create test shapes
    std::vector<GPU_ShapeData> shapes(2);
    shapes[0].sphere = {make_float3(0.0f, 0.0f, 0.0f), 1.0f};
    shapes[1].sphere = {make_float3(1.5f, 0.0f, 0.0f), 1.0f};

    std::vector<GPU_AABB> aabbs(2);
    aabbs[0] = {make_float3(-1.0f, -1.0f, -1.0f), 0, make_float3(1.0f, 1.0f, 1.0f), 0, 0, 0};
    aabbs[1] = {make_float3(0.5f, -1.0f, -1.0f), 1, make_float3(2.5f, 1.0f, 1.0f), 0, 0, 0};

    // Create manifold
    GPU_ContactManifold manifold = {0, 1, 0, 0};

    // Allocate device memory
    GPU_ShapeData* d_shapes;
    GPU_AABB* d_aabbs;
    GPU_ContactManifold* d_manifolds;
    GPU_ContactPoint* d_contacts;
    int* d_contactCount;

    cudaMalloc(&d_shapes, shapes.size() * sizeof(GPU_ShapeData));
    cudaMalloc(&d_aabbs, aabbs.size() * sizeof(GPU_AABB));
    cudaMalloc(&d_manifolds, sizeof(GPU_ContactManifold));
    cudaMalloc(&d_contacts, MAX_CONTACTS * sizeof(GPU_ContactPoint));
    cudaMalloc(&d_contactCount, sizeof(int));

    cudaMemcpy(d_shapes, shapes.data(), shapes.size() * sizeof(GPU_ShapeData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aabbs, aabbs.data(), aabbs.size() * sizeof(GPU_AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_manifolds, &manifold, sizeof(GPU_ContactManifold), cudaMemcpyHostToDevice);

    narrowphase.Initialize(d_shapes, shapes.size());
    narrowphase.ProcessManifolds(d_manifolds, 1, d_aabbs, d_contacts, d_contactCount);

    // Read results
    int contactCount;
    cudaMemcpy(&contactCount, d_contactCount, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<GPU_ContactPoint> contacts(contactCount);
    cudaMemcpy(contacts.data(), d_contacts, contactCount * sizeof(GPU_ContactPoint), cudaMemcpyDeviceToHost);

    std::cout << "Found " << contactCount << " contacts" << std::endl;
    for (const auto& contact : contacts) {
        std::cout << "Contact: normal=(" << contact.normal.x << "," << contact.normal.y << "," << contact.normal.z
                  << "), distance=" << contact.distance << std::endl;
    }

    // Cleanup
    cudaFree(d_shapes);
    cudaFree(d_aabbs);
    cudaFree(d_manifolds);
    cudaFree(d_contacts);
    cudaFree(d_contactCount);
}

void TestBoxBoxCollision() {
    std::cout << "Testing box-box collision..." << std::endl;

    ChCudaDeviceManager& deviceManager = ChCudaDeviceManager::GetInstance();
    deviceManager.Initialize();
    ChCudaNarrowphase narrowphase(&deviceManager);

    // Create test shapes (overlapping boxes)
    std::vector<GPU_ShapeData> shapes(2);
    shapes[0].box = {make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), make_float4(1.0f, 0.0f, 0.0f, 0.0f)};
    shapes[1].box = {make_float3(1.5f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), make_float4(1.0f, 0.0f, 0.0f, 0.0f)};

    std::vector<GPU_AABB> aabbs(2);
    aabbs[0] = {make_float3(-1.0f, -1.0f, -1.0f), 0, make_float3(1.0f, 1.0f, 1.0f), 2, 0, 0};
    aabbs[1] = {make_float3(0.5f, -1.0f, -1.0f), 1, make_float3(2.5f, 1.0f, 1.0f), 2, 0, 0};

    // Create manifold
    GPU_ContactManifold manifold = {0, 1, 0, 0};

    // Allocate device memory and run test (similar to sphere test)
    GPU_ShapeData* d_shapes;
    GPU_AABB* d_aabbs;
    GPU_ContactManifold* d_manifolds;
    GPU_ContactPoint* d_contacts;
    int* d_contactCount;

    cudaMalloc(&d_shapes, shapes.size() * sizeof(GPU_ShapeData));
    cudaMalloc(&d_aabbs, aabbs.size() * sizeof(GPU_AABB));
    cudaMalloc(&d_manifolds, sizeof(GPU_ContactManifold));
    cudaMalloc(&d_contacts, MAX_CONTACTS * sizeof(GPU_ContactPoint));
    cudaMalloc(&d_contactCount, sizeof(int));

    cudaMemcpy(d_shapes, shapes.data(), shapes.size() * sizeof(GPU_ShapeData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aabbs, aabbs.data(), aabbs.size() * sizeof(GPU_AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_manifolds, &manifold, sizeof(GPU_ContactManifold), cudaMemcpyHostToDevice);

    narrowphase.Initialize(d_shapes, shapes.size());
    narrowphase.ProcessManifolds(d_manifolds, 1, d_aabbs, d_contacts, d_contactCount);

    // Read results
    int contactCount;
    cudaMemcpy(&contactCount, d_contactCount, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<GPU_ContactPoint> contacts(contactCount);
    cudaMemcpy(contacts.data(), d_contacts, contactCount * sizeof(GPU_ContactPoint), cudaMemcpyDeviceToHost);

    std::cout << "Found " << contactCount << " contacts" << std::endl;

    // Cleanup
    cudaFree(d_shapes);
    cudaFree(d_aabbs);
    cudaFree(d_manifolds);
    cudaFree(d_contacts);
    cudaFree(d_contactCount);
}

}  // namespace cuda
}  // namespace collision
}  // namespace chrono

int main() {
    std::cout << "CUDA Narrow Phase Collision Tests" << std::endl;

    chrono::collision::cuda::TestSphereSphereCollision();
    chrono::collision::cuda::TestBoxBoxCollision();

    std::cout << "Tests completed." << std::endl;
    return 0;
}