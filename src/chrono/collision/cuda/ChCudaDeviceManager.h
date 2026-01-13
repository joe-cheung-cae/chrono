#ifndef CH_CUDA_DEVICE_MANAGER_H
#define CH_CUDA_DEVICE_MANAGER_H

#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace chrono {
namespace collision {

/// Singleton class for managing CUDA device resources
class ChCudaDeviceManager {
public:
    /// Get the singleton instance
    static ChCudaDeviceManager& GetInstance();

    /// Initialize the device manager
    void Initialize();

    /// Shutdown the device manager
    void Shutdown();

    /// Get the current device ID
    int GetCurrentDevice() const;

    /// Set the current device
    void SetDevice(int device);

    /// Get device properties
    const cudaDeviceProp& GetDeviceProperties(int device = -1) const;

    /// Get number of available devices
    int GetDeviceCount() const;

    /// Check if CUDA is available
    bool IsCudaAvailable() const;

    /// Allocate device memory
    template<typename T>
    T* AllocateDeviceMemory(size_t count) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        return ptr;
    }

    /// Free device memory
    template<typename T>
    void FreeDeviceMemory(T* ptr) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    /// Copy data from host to device
    template<typename T>
    void CopyToDevice(T* dst, const T* src, size_t count) {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    /// Copy data from device to host
    template<typename T>
    void CopyToHost(T* dst, const T* src, size_t count) {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /// Copy data from device to device
    template<typename T>
    void CopyDeviceToDevice(T* dst, const T* src, size_t count) {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    /// Allocate pinned host memory
    template<typename T>
    T* AllocatePinnedMemory(size_t count) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
        return ptr;
    }

    /// Free pinned host memory
    template<typename T>
    void FreePinnedMemory(T* ptr) {
        if (ptr) {
            CUDA_CHECK(cudaFreeHost(ptr));
        }
    }

    /// Get memory information
    void GetMemoryInfo(size_t& free, size_t& total) const;

    /// Print memory usage
    void PrintMemoryUsage() const;

private:
    ChCudaDeviceManager();
    ~ChCudaDeviceManager();

    // Disable copy and assignment
    ChCudaDeviceManager(const ChCudaDeviceManager&) = delete;
    ChCudaDeviceManager& operator=(const ChCudaDeviceManager&) = delete;

    bool m_initialized;
    int m_currentDevice;
    int m_deviceCount;
    std::vector<cudaDeviceProp> m_deviceProperties;
};

} // namespace collision
} // namespace chrono

#endif // CH_CUDA_DEVICE_MANAGER_H