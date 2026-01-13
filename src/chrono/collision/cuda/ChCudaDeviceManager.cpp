#include "ChCudaDeviceManager.h"
#include "ChCudaUtils.h"
#include <iostream>
#include <algorithm>

namespace chrono {
namespace collision {

ChCudaDeviceManager& ChCudaDeviceManager::GetInstance() {
    static ChCudaDeviceManager instance;
    return instance;
}

ChCudaDeviceManager::ChCudaDeviceManager()
    : m_initialized(false), m_currentDevice(0), m_deviceCount(0) {
}

ChCudaDeviceManager::~ChCudaDeviceManager() {
    Shutdown();
}

void ChCudaDeviceManager::Initialize() {
    if (m_initialized) {
        return;
    }

    try {
        m_deviceCount = ChCudaDeviceUtils::GetDeviceCount();
        if (m_deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return;
        }

        m_deviceProperties.resize(m_deviceCount);
        for (int i = 0; i < m_deviceCount; ++i) {
            m_deviceProperties[i] = ChCudaDeviceUtils::GetDeviceProperties(i);
        }

        // Set the device with the most multiprocessors as default
        int bestDevice = 0;
        int maxMultiprocessors = m_deviceProperties[0].multiProcessorCount;
        for (int i = 1; i < m_deviceCount; ++i) {
            if (m_deviceProperties[i].multiProcessorCount > maxMultiprocessors) {
                maxMultiprocessors = m_deviceProperties[i].multiProcessorCount;
                bestDevice = i;
            }
        }

        SetDevice(bestDevice);
        m_initialized = true;

        std::cout << "CUDA Device Manager initialized with device " << m_currentDevice << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA Device Manager: " << e.what() << std::endl;
        m_deviceCount = 0;
    }
}

void ChCudaDeviceManager::Shutdown() {
    if (!m_initialized) {
        return;
    }

    try {
        CUDA_CHECK(cudaDeviceReset());
    } catch (const std::exception& e) {
        std::cerr << "Error during CUDA device reset: " << e.what() << std::endl;
    }

    m_initialized = false;
    m_deviceCount = 0;
    m_deviceProperties.clear();
}

int ChCudaDeviceManager::GetCurrentDevice() const {
    return m_currentDevice;
}

void ChCudaDeviceManager::SetDevice(int device) {
    if (device < 0 || device >= m_deviceCount) {
        throw std::runtime_error("Invalid CUDA device ID: " + std::to_string(device));
    }

    CUDA_CHECK(cudaSetDevice(device));
    m_currentDevice = device;
}

const cudaDeviceProp& ChCudaDeviceManager::GetDeviceProperties(int device) const {
    if (device == -1) {
        device = m_currentDevice;
    }

    if (device < 0 || device >= m_deviceCount) {
        throw std::runtime_error("Invalid CUDA device ID: " + std::to_string(device));
    }

    return m_deviceProperties[device];
}

int ChCudaDeviceManager::GetDeviceCount() const {
    return m_deviceCount;
}

bool ChCudaDeviceManager::IsCudaAvailable() const {
    return m_deviceCount > 0;
}

void ChCudaDeviceManager::GetMemoryInfo(size_t& free, size_t& total) const {
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
}

void ChCudaDeviceManager::PrintMemoryUsage() const {
    if (!m_initialized) {
        std::cout << "CUDA Device Manager not initialized" << std::endl;
        return;
    }

    size_t free, total;
    GetMemoryInfo(free, total);

    std::cout << "CUDA Memory Usage:" << std::endl;
    std::cout << "  Total: " << total / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Free:  " << free / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Used:  " << (total - free) / (1024 * 1024) << " MB" << std::endl;
}

} // namespace collision
} // namespace chrono