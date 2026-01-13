#include "ChCudaUtils.h"
#include <iostream>

namespace chrono {
namespace collision {

int ChCudaDeviceUtils::GetDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void ChCudaDeviceUtils::SetDevice(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

cudaDeviceProp ChCudaDeviceUtils::GetDeviceProperties(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

bool ChCudaDeviceUtils::IsCudaAvailable() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int ChCudaDeviceUtils::GetCurrentDevice() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void ChCudaDeviceUtils::ResetDevice() {
    CUDA_CHECK(cudaDeviceReset());
}

size_t ChCudaDeviceUtils::GetTotalMemory(int device) {
    if (device == -1) {
        device = GetCurrentDevice();
    }
    cudaDeviceProp prop = GetDeviceProperties(device);
    return prop.totalGlobalMem;
}

size_t ChCudaDeviceUtils::GetFreeMemory(int device) {
    if (device == -1) {
        device = GetCurrentDevice();
    }
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

void ChCudaDeviceUtils::PrintDeviceInfo(int device) {
    if (device == -1) {
        device = GetCurrentDevice();
    }

    cudaDeviceProp prop = GetDeviceProperties(device);

    std::cout << "CUDA Device " << device << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;

    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    std::cout << "  Free memory: " << free / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Total memory: " << total / (1024 * 1024) << " MB" << std::endl;
}

} // namespace collision
} // namespace chrono