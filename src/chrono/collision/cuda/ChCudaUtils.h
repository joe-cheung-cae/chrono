#ifndef CH_CUDA_UTILS_H
#define CH_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <string>

namespace chrono {
namespace collision {

/// CUDA error checking utility
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

/// CUDA kernel error checking utility
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA sync error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

/// Utility class for CUDA device management
class ChCudaDeviceUtils {
public:
    /// Get the number of available CUDA devices
    static int GetDeviceCount();

    /// Set the current CUDA device
    static void SetDevice(int device);

    /// Get device properties
    static cudaDeviceProp GetDeviceProperties(int device);

    /// Check if CUDA is available
    static bool IsCudaAvailable();

    /// Get the current device
    static int GetCurrentDevice();

    /// Reset the current device
    static void ResetDevice();

    /// Get total GPU memory in bytes
    static size_t GetTotalMemory(int device = -1);

    /// Get free GPU memory in bytes
    static size_t GetFreeMemory(int device = -1);

    /// Print device information
    static void PrintDeviceInfo(int device = -1);
};

} // namespace collision
} // namespace chrono

#endif // CH_CUDA_UTILS_H