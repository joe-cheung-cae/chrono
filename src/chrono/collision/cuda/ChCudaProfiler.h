#ifndef CH_CUDA_PROFILER_H
#define CH_CUDA_PROFILER_H

#include "ChCudaUtils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <iostream>

namespace chrono {
namespace collision {

/// Simple CUDA profiling utility
class ChCudaProfiler {
public:
    /// Start timing
    static void StartTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    /// Stop timing and return elapsed time in milliseconds
    static float StopTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return milliseconds;
    }

    /// Print memory usage
    static void PrintMemoryUsage(const std::string& label = "") {
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        std::cout << (label.empty() ? "CUDA Memory" : label) << ": "
                  << free / (1024.0 * 1024.0) << " MB free / "
                  << total / (1024.0 * 1024.0) << " MB total" << std::endl;
    }

    /// Get current device properties
    static cudaDeviceProp GetCurrentDeviceProperties() {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        return prop;
    }

    /// Print device info
    static void PrintDeviceInfo() {
        auto prop = GetCurrentDeviceProperties();
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    }
};

/// Scoped timer for automatic timing
class ChCudaScopedTimer {
public:
    ChCudaScopedTimer(const std::string& label) : m_label(label) {
        ChCudaProfiler::StartTimer(m_start, m_stop);
    }

    ~ChCudaScopedTimer() {
        float time = ChCudaProfiler::StopTimer(m_start, m_stop);
        std::cout << m_label << ": " << time << " ms" << std::endl;
    }

private:
    std::string m_label;
    cudaEvent_t m_start, m_stop;
};

} // namespace collision
} // namespace chrono

#endif // CH_CUDA_PROFILER_H