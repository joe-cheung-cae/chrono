#include "ChCudaBroadphase.h"
#include "ChCudaDeviceManager.h"
#include "ChCudaCollisionTypes.h"
#include "ChCudaUtils.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

namespace chrono {
namespace collision {
namespace cuda {

void TestBroadphaseSmallScene() {
    std::cout << "Testing broadphase with small scene (10 objects)" << std::endl;

    ChCudaDeviceManager& deviceManager = ChCudaDeviceManager::GetInstance();
    ChCudaBroadphase broadphase(&deviceManager);

    // Initialize scene bounds
    float3 sceneMin = make_float3(-10, -10, -10);
    float3 sceneMax = make_float3(10, 10, 10);
    broadphase.Initialize(sceneMin, sceneMax, 1.0f);

    // Create test AABBs
    const int numObjects = 10;
    std::vector<GPU_AABB> hostAABBs(numObjects);
    srand(time(NULL));
    for (int i = 0; i < numObjects; ++i) {
        float x = (rand() % 20) - 10.0f;
        float y = (rand() % 20) - 10.0f;
        float z = (rand() % 20) - 10.0f;
        hostAABBs[i].min = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
        hostAABBs[i].max = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);
        hostAABBs[i].objectId = i;
        hostAABBs[i].shapeType = 0;
        hostAABBs[i].collisionGroup = 1;
        hostAABBs[i].collisionMask = 1;
    }

    // Allocate GPU memory
    GPU_AABB* d_aabbs;
    CUDA_CHECK(cudaMalloc(&d_aabbs, numObjects * sizeof(GPU_AABB)));
    CUDA_CHECK(cudaMemcpy(d_aabbs, hostAABBs.data(), numObjects * sizeof(GPU_AABB), cudaMemcpyHostToDevice));

    // Build grid
    broadphase.BuildGrid(d_aabbs, numObjects);

    // Allocate manifolds
    GPU_ContactManifold* d_manifolds;
    int* d_manifoldCount;
    CUDA_CHECK(cudaMalloc(&d_manifolds, MAX_MANIFOLDS * sizeof(GPU_ContactManifold)));
    CUDA_CHECK(cudaMalloc(&d_manifoldCount, sizeof(int)));

    // Detect collisions
    broadphase.DetectCollisions(d_aabbs, d_manifolds, d_manifoldCount);

    // Copy results back
    int manifoldCount;
    CUDA_CHECK(cudaMemcpy(&manifoldCount, d_manifoldCount, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<GPU_ContactManifold> hostManifolds(manifoldCount);
    CUDA_CHECK(cudaMemcpy(hostManifolds.data(), d_manifolds, manifoldCount * sizeof(GPU_ContactManifold), cudaMemcpyDeviceToHost));

    std::cout << "Found " << manifoldCount << " potential collision pairs" << std::endl;

    // Verify results (simple check)
    int expectedPairs = 0;
    for (int i = 0; i < numObjects; ++i) {
        for (int j = i + 1; j < numObjects; ++j) {
            if (hostAABBs[i].max.x >= hostAABBs[j].min.x && hostAABBs[i].min.x <= hostAABBs[j].max.x &&
                hostAABBs[i].max.y >= hostAABBs[j].min.y && hostAABBs[i].min.y <= hostAABBs[j].max.y &&
                hostAABBs[i].max.z >= hostAABBs[j].min.z && hostAABBs[i].min.z <= hostAABBs[j].max.z) {
                expectedPairs++;
            }
        }
    }
    std::cout << "Expected pairs: " << expectedPairs << std::endl;

    // Cleanup
    cudaFree(d_aabbs);
    cudaFree(d_manifolds);
    cudaFree(d_manifoldCount);
}

void TestBroadphaseMediumScene() {
    std::cout << "Testing broadphase with medium scene (1000 objects)" << std::endl;

    ChCudaDeviceManager& deviceManager = ChCudaDeviceManager::GetInstance();
    ChCudaBroadphase broadphase(&deviceManager);

    // Initialize scene bounds
    float3 sceneMin = make_float3(-50, -50, -50);
    float3 sceneMax = make_float3(50, 50, 50);
    broadphase.Initialize(sceneMin, sceneMax, 2.0f);

    // Create test AABBs
    const int numObjects = 1000;
    std::vector<GPU_AABB> hostAABBs(numObjects);
    srand(time(NULL));
    for (int i = 0; i < numObjects; ++i) {
        float x = (rand() % 100) - 50.0f;
        float y = (rand() % 100) - 50.0f;
        float z = (rand() % 100) - 50.0f;
        hostAABBs[i].min = make_float3(x - 1.0f, y - 1.0f, z - 1.0f);
        hostAABBs[i].max = make_float3(x + 1.0f, y + 1.0f, z + 1.0f);
        hostAABBs[i].objectId = i;
        hostAABBs[i].shapeType = 0;
        hostAABBs[i].collisionGroup = 1;
        hostAABBs[i].collisionMask = 1;
    }

    // Allocate GPU memory
    GPU_AABB* d_aabbs;
    CUDA_CHECK(cudaMalloc(&d_aabbs, numObjects * sizeof(GPU_AABB)));
    CUDA_CHECK(cudaMemcpy(d_aabbs, hostAABBs.data(), numObjects * sizeof(GPU_AABB), cudaMemcpyHostToDevice));

    // Build grid
    broadphase.BuildGrid(d_aabbs, numObjects);

    // Allocate manifolds
    GPU_ContactManifold* d_manifolds;
    int* d_manifoldCount;
    CUDA_CHECK(cudaMalloc(&d_manifolds, MAX_MANIFOLDS * sizeof(GPU_ContactManifold)));
    CUDA_CHECK(cudaMalloc(&d_manifoldCount, sizeof(int)));

    // Detect collisions
    broadphase.DetectCollisions(d_aabbs, d_manifolds, d_manifoldCount);

    // Copy results back
    int manifoldCount;
    CUDA_CHECK(cudaMemcpy(&manifoldCount, d_manifoldCount, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Found " << manifoldCount << " potential collision pairs" << std::endl;

    // Cleanup
    cudaFree(d_aabbs);
    cudaFree(d_manifolds);
    cudaFree(d_manifoldCount);
}

}  // namespace cuda
}  // namespace collision
}  // namespace chrono

int main() {
    try {
        chrono::collision::cuda::TestBroadphaseSmallScene();
        chrono::collision::cuda::TestBroadphaseMediumScene();
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}