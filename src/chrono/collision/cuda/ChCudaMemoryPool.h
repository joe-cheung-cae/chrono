#ifndef CH_CUDA_MEMORY_POOL_H
#define CH_CUDA_MEMORY_POOL_H

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace chrono {
namespace collision {
namespace cuda {

/// Memory block information
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool inUse;
    cudaEvent_t readyEvent;  // For async operations

    MemoryBlock() : ptr(nullptr), size(0), inUse(false) {
        cudaEventCreate(&readyEvent);
    }

    ~MemoryBlock() {
        if (readyEvent) cudaEventDestroy(readyEvent);
    }
};

/// GPU memory pool for efficient allocation/deallocation
class ChCudaMemoryPool {
public:
    /// Get singleton instance
    static ChCudaMemoryPool& GetInstance();

    /// Initialize the memory pool
    void Initialize(size_t initialPoolSize = 256 * 1024 * 1024);  // 256MB default

    /// Shutdown the memory pool
    void Shutdown();

    /// Allocate device memory from pool
    void* AllocateDevice(size_t size);

    /// Free device memory back to pool
    void FreeDevice(void* ptr);

    /// Allocate pinned host memory
    void* AllocatePinned(size_t size);

    /// Free pinned host memory
    void FreePinned(void* ptr);

    /// Asynchronous memory copy with event tracking
    cudaError_t AsyncCopyToDevice(void* dst, const void* src, size_t count, cudaStream_t stream = 0);
    cudaError_t AsyncCopyToHost(void* dst, const void* src, size_t count, cudaStream_t stream = 0);

    /// Asynchronous memory copy with explicit event management
    cudaError_t AsyncCopyToDeviceWithEvent(void* dst, const void* src, size_t count, cudaStream_t stream, cudaEvent_t* event);
    cudaError_t AsyncCopyToHostWithEvent(void* dst, const void* src, size_t count, cudaStream_t stream, cudaEvent_t* event);

    /// Wait for all pending transfers to complete
    void WaitForAllTransfers();

    /// Get memory usage statistics
    void GetMemoryStats(size_t& totalAllocated, size_t& totalUsed, size_t& poolSize);

    /// Get detailed memory usage statistics
    void GetDetailedMemoryStats(size_t& totalAllocated, size_t& totalUsed, size_t& poolSize,
                               size_t& pinnedMemory, size_t& freeBlocks, size_t& largestFreeBlock);

    /// Print memory usage report
    void PrintMemoryReport();

    /// Print detailed memory usage report
    void PrintDetailedMemoryReport();

    /// Defragment the pool (compact free blocks)
    void Defragment();

    /// Advanced defragmentation with compaction
    void AdvancedDefragment();

    /// Check for out-of-memory conditions
    bool CheckOOM(size_t requestedSize);

    /// Handle OOM by attempting recovery (defragment, free unused)
    bool HandleOOM(size_t requestedSize);

    /// Set memory allocation strategy
    void SetAllocationStrategy(int strategy);

    /// Get current allocation strategy
    int GetAllocationStrategy() const;

private:
    ChCudaMemoryPool();
    ~ChCudaMemoryPool();

    // Disable copy and assignment
    ChCudaMemoryPool(const ChCudaMemoryPool&) = delete;
    ChCudaMemoryPool& operator=(const ChCudaMemoryPool&) = delete;

    /// Find best fit block for allocation
    MemoryBlock* FindBestFit(size_t size);

    /// Find first fit block for allocation
    MemoryBlock* FindFirstFit(size_t size);

    /// Find worst fit block for allocation
    MemoryBlock* FindWorstFit(size_t size);

    /// Split a block for allocation
    MemoryBlock* SplitBlock(MemoryBlock* block, size_t size);

    /// Merge adjacent free blocks
    void MergeFreeBlocks();

    /// Expand pool if needed
    bool ExpandPool(size_t minSize);

    std::vector<std::unique_ptr<MemoryBlock>> m_blocks;
    std::unordered_map<void*, MemoryBlock*> m_ptrToBlock;
    std::vector<void*> m_pinnedAllocations;

    size_t m_totalPoolSize;
    size_t m_usedSize;
    bool m_initialized;
    int m_allocationStrategy;  // 0: Best Fit, 1: First Fit, 2: Worst Fit

    std::mutex m_mutex;  // Thread safety
    std::vector<cudaEvent_t> m_pendingEvents;  // Track pending async operations
};

/// RAII wrapper for device memory
template<typename T>
class ChCudaDeviceBuffer {
public:
    ChCudaDeviceBuffer() : m_ptr(nullptr), m_size(0) {}

    explicit ChCudaDeviceBuffer(size_t count) : m_size(count) {
        m_ptr = static_cast<T*>(ChCudaMemoryPool::GetInstance().AllocateDevice(count * sizeof(T)));
    }

    ~ChCudaDeviceBuffer() {
        if (m_ptr) {
            ChCudaMemoryPool::GetInstance().FreeDevice(m_ptr);
        }
    }

    // Disable copy
    ChCudaDeviceBuffer(const ChCudaDeviceBuffer&) = delete;
    ChCudaDeviceBuffer& operator=(const ChCudaDeviceBuffer&) = delete;

    // Move semantics
    ChCudaDeviceBuffer(ChCudaDeviceBuffer&& other) noexcept
        : m_ptr(other.m_ptr), m_size(other.m_size) {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }

    ChCudaDeviceBuffer& operator=(ChCudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (m_ptr) ChCudaMemoryPool::GetInstance().FreeDevice(m_ptr);
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    /// Resize buffer
    void Resize(size_t newSize) {
        if (m_ptr) ChCudaMemoryPool::GetInstance().FreeDevice(m_ptr);
        m_size = newSize;
        m_ptr = static_cast<T*>(ChCudaMemoryPool::GetInstance().AllocateDevice(newSize * sizeof(T)));
    }

    /// Get pointer
    T* Get() const { return m_ptr; }

    /// Get size
    size_t Size() const { return m_size; }

    /// Check if valid
    bool IsValid() const { return m_ptr != nullptr; }

private:
    T* m_ptr;
    size_t m_size;
};

/// RAII wrapper for pinned host memory
template<typename T>
class ChCudaPinnedBuffer {
public:
    ChCudaPinnedBuffer() : m_ptr(nullptr), m_size(0) {}

    explicit ChCudaPinnedBuffer(size_t count) : m_size(count) {
        m_ptr = static_cast<T*>(ChCudaMemoryPool::GetInstance().AllocatePinned(count * sizeof(T)));
    }

    ~ChCudaPinnedBuffer() {
        if (m_ptr) {
            ChCudaMemoryPool::GetInstance().FreePinned(m_ptr);
        }
    }

    // Disable copy
    ChCudaPinnedBuffer(const ChCudaPinnedBuffer&) = delete;
    ChCudaPinnedBuffer& operator=(const ChCudaPinnedBuffer&) = delete;

    // Move semantics
    ChCudaPinnedBuffer(ChCudaPinnedBuffer&& other) noexcept
        : m_ptr(other.m_ptr), m_size(other.m_size) {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }

    ChCudaPinnedBuffer& operator=(ChCudaPinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (m_ptr) ChCudaMemoryPool::GetInstance().FreePinned(m_ptr);
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    /// Resize buffer
    void Resize(size_t newSize) {
        if (m_ptr) ChCudaMemoryPool::GetInstance().FreePinned(m_ptr);
        m_size = newSize;
        m_ptr = static_cast<T*>(ChCudaMemoryPool::GetInstance().AllocatePinned(newSize * sizeof(T)));
    }

    /// Get pointer
    T* Get() const { return m_ptr; }

    /// Get size
    size_t Size() const { return m_size; }

    /// Check if valid
    bool IsValid() const { return m_ptr != nullptr; }

private:
    T* m_ptr;
    size_t m_size;
};

} // namespace cuda
} // namespace collision
} // namespace chrono

#endif // CH_CUDA_MEMORY_POOL_H