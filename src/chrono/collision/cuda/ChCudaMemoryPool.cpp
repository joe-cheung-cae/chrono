#include "ChCudaMemoryPool.h"
#include "ChCudaUtils.h"
#include <algorithm>
#include <iostream>
#include <sstream>

namespace chrono {
namespace collision {
namespace cuda {

ChCudaMemoryPool& ChCudaMemoryPool::GetInstance() {
    static ChCudaMemoryPool instance;
    return instance;
}

ChCudaMemoryPool::ChCudaMemoryPool()
    : m_totalPoolSize(0), m_usedSize(0), m_initialized(false), m_allocationStrategy(0) {
}

ChCudaMemoryPool::~ChCudaMemoryPool() {
    Shutdown();
}

void ChCudaMemoryPool::Initialize(size_t initialPoolSize) {
    if (m_initialized) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    try {
        // Allocate initial pool
        if (!ExpandPool(initialPoolSize)) {
            throw std::runtime_error("Failed to allocate initial memory pool");
        }

        m_initialized = true;
        std::cout << "CUDA Memory Pool initialized with " << initialPoolSize / (1024 * 1024) << " MB" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA Memory Pool: " << e.what() << std::endl;
        Shutdown();
        throw;
    }
}

void ChCudaMemoryPool::Shutdown() {
    if (!m_initialized) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    // Free all blocks
    for (auto& block : m_blocks) {
        if (block->ptr) {
            cudaFree(block->ptr);
        }
    }
    m_blocks.clear();
    m_ptrToBlock.clear();

    // Free pinned allocations
    for (void* ptr : m_pinnedAllocations) {
        cudaFreeHost(ptr);
    }
    m_pinnedAllocations.clear();

    m_totalPoolSize = 0;
    m_usedSize = 0;
    m_initialized = false;
}

void* ChCudaMemoryPool::AllocateDevice(size_t size) {
    if (!m_initialized) {
        throw std::runtime_error("Memory pool not initialized");
    }

    if (size == 0) return nullptr;

    std::lock_guard<std::mutex> lock(m_mutex);

    // Check for OOM
    if (CheckOOM(size)) {
        if (!HandleOOM(size)) {
            throw std::runtime_error("Out of memory: requested " + std::to_string(size) + " bytes");
        }
    }

    // Find best fit block
    MemoryBlock* block = FindBestFit(size);
    if (!block) {
        // Try to expand pool
        if (!ExpandPool(size)) {
            throw std::runtime_error("Failed to expand memory pool for " + std::to_string(size) + " bytes");
        }
        block = FindBestFit(size);
        if (!block) {
            throw std::runtime_error("No suitable memory block found after expansion");
        }
    }

    // Split block if necessary
    block = SplitBlock(block, size);
    block->inUse = true;
    m_usedSize += block->size;

    return block->ptr;
}

void ChCudaMemoryPool::FreeDevice(void* ptr) {
    if (!ptr || !m_initialized) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    auto it = m_ptrToBlock.find(ptr);
    if (it == m_ptrToBlock.end()) {
        // Not from our pool, use direct free
        cudaFree(ptr);
        return;
    }

    MemoryBlock* block = it->second;
    block->inUse = false;
    m_usedSize -= block->size;

    // Try to merge free blocks
    MergeFreeBlocks();
}

void* ChCudaMemoryPool::AllocatePinned(size_t size) {
    if (size == 0) return nullptr;

    std::lock_guard<std::mutex> lock(m_mutex);

    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    m_pinnedAllocations.push_back(ptr);

    return ptr;
}

void ChCudaMemoryPool::FreePinned(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    auto it = std::find(m_pinnedAllocations.begin(), m_pinnedAllocations.end(), ptr);
    if (it != m_pinnedAllocations.end()) {
        cudaFreeHost(ptr);
        m_pinnedAllocations.erase(it);
    } else {
        // Not from our pool
        cudaFreeHost(ptr);
    }
}

cudaError_t ChCudaMemoryPool::AsyncCopyToDevice(void* dst, const void* src, size_t count, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);

    // Record event for tracking
    if (err == cudaSuccess) {
        auto it = m_ptrToBlock.find(dst);
        if (it != m_ptrToBlock.end()) {
            cudaEventRecord(it->second->readyEvent, stream);
        }
    }

    return err;
}

cudaError_t ChCudaMemoryPool::AsyncCopyToHost(void* dst, const void* src, size_t count, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);

    // Record event for tracking
    if (err == cudaSuccess) {
        auto it = m_ptrToBlock.find(const_cast<void*>(src));
        if (it != m_ptrToBlock.end()) {
            cudaEventRecord(it->second->readyEvent, stream);
        }
    }

    return err;
}

cudaError_t ChCudaMemoryPool::AsyncCopyToDeviceWithEvent(void* dst, const void* src, size_t count, cudaStream_t stream, cudaEvent_t* event) {
    cudaError_t err = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);

    if (err == cudaSuccess && event) {
        cudaEventRecord(*event, stream);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pendingEvents.push_back(*event);
    }

    return err;
}

cudaError_t ChCudaMemoryPool::AsyncCopyToHostWithEvent(void* dst, const void* src, size_t count, cudaStream_t stream, cudaEvent_t* event) {
    cudaError_t err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);

    if (err == cudaSuccess && event) {
        cudaEventRecord(*event, stream);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pendingEvents.push_back(*event);
    }

    return err;
}

void ChCudaMemoryPool::WaitForAllTransfers() {
    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto event : m_pendingEvents) {
        if (event) {
            cudaEventSynchronize(event);
            cudaEventDestroy(event);
        }
    }
    m_pendingEvents.clear();
}

void ChCudaMemoryPool::GetMemoryStats(size_t& totalAllocated, size_t& totalUsed, size_t& poolSize) {
    std::lock_guard<std::mutex> lock(m_mutex);
    totalAllocated = m_totalPoolSize;
    totalUsed = m_usedSize;
    poolSize = m_blocks.size();
}

void ChCudaMemoryPool::GetDetailedMemoryStats(size_t& totalAllocated, size_t& totalUsed, size_t& poolSize,
                                              size_t& pinnedMemory, size_t& freeBlocks, size_t& largestFreeBlock) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    totalAllocated = m_totalPoolSize;
    totalUsed = m_usedSize;
    poolSize = m_blocks.size();
    
    // Calculate pinned memory usage - we need to track this separately
    // For now, we'll use a simple approximation
    pinnedMemory = m_pinnedAllocations.size() * 1024 * 1024; // Approximate 1MB per pinned allocation
    
    // Calculate free blocks and largest free block
    freeBlocks = 0;
    largestFreeBlock = 0;
    for (const auto& block : m_blocks) {
        if (!block->inUse) {
            freeBlocks++;
            largestFreeBlock = std::max(largestFreeBlock, block->size);
        }
    }
}

void ChCudaMemoryPool::PrintMemoryReport() {
    std::lock_guard<std::mutex> lock(m_mutex);

    size_t totalAllocated, totalUsed, poolSize;
    GetMemoryStats(totalAllocated, totalUsed, poolSize);

    std::cout << "CUDA Memory Pool Report:" << std::endl;
    std::cout << "  Total allocated: " << totalAllocated / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Currently used:  " << totalUsed / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Free:           " << (totalAllocated - totalUsed) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Utilization:    " << (totalAllocated > 0 ? (totalUsed * 100.0 / totalAllocated) : 0) << "%" << std::endl;
    std::cout << "  Pool blocks:    " << poolSize << std::endl;
    std::cout << "  Pinned buffers: " << m_pinnedAllocations.size() << std::endl;

    // Show fragmentation info
    size_t freeBlocks = 0;
    size_t largestFree = 0;
    for (const auto& block : m_blocks) {
        if (!block->inUse) {
            freeBlocks++;
            largestFree = std::max(largestFree, block->size);
        }
    }
    std::cout << "  Free blocks:    " << freeBlocks << std::endl;
    std::cout << "  Largest free:   " << largestFree / (1024 * 1024) << " MB" << std::endl;
}

void ChCudaMemoryPool::PrintDetailedMemoryReport() {
    std::lock_guard<std::mutex> lock(m_mutex);

    size_t totalAllocated, totalUsed, poolSize, pinnedMemory, freeBlocks, largestFreeBlock;
    GetDetailedMemoryStats(totalAllocated, totalUsed, poolSize, pinnedMemory, freeBlocks, largestFreeBlock);

    std::cout << "CUDA Memory Pool Detailed Report:" << std::endl;
    std::cout << "  Total allocated: " << totalAllocated / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Currently used:  " << totalUsed / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Free:           " << (totalAllocated - totalUsed) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Utilization:    " << (totalAllocated > 0 ? (totalUsed * 100.0 / totalAllocated) : 0) << "%" << std::endl;
    std::cout << "  Pinned memory:  " << pinnedMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Pool blocks:    " << poolSize << std::endl;
    std::cout << "  Pinned buffers: " << m_pinnedAllocations.size() << std::endl;
    std::cout << "  Free blocks:    " << freeBlocks << std::endl;
    std::cout << "  Largest free:   " << largestFreeBlock / (1024 * 1024) << " MB" << std::endl;
    
    // Fragmentation analysis
    double fragmentation = 0.0;
    if (freeBlocks > 1) {
        fragmentation = 1.0 - (static_cast<double>(largestFreeBlock) / (totalAllocated - totalUsed));
    }
    std::cout << "  Fragmentation:  " << fragmentation * 100.0 << "%" << std::endl;
    
    // Allocation strategy info
    const char* strategyNames[] = {"Best Fit", "First Fit", "Worst Fit"};
    std::cout << "  Allocation strategy: " << strategyNames[m_allocationStrategy] << std::endl;
}

void ChCudaMemoryPool::Defragment() {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Simple defragmentation: merge all free blocks
    MergeFreeBlocks();

    // Could implement more sophisticated defragmentation here
    // For now, just ensure contiguous free space at end
}

void ChCudaMemoryPool::AdvancedDefragment() {
    std::lock_guard<std::mutex> lock(m_mutex);

    // First, merge all adjacent free blocks
    MergeFreeBlocks();

    // If we have many free blocks, try to compact memory
    if (m_blocks.size() > 10 && m_usedSize < m_totalPoolSize / 2) {
        // Create a new large block and copy all used data to it
        void* newPool = nullptr;
        size_t newPoolSize = m_totalPoolSize;
        
        try {
            CUDA_CHECK(cudaMalloc(&newPool, newPoolSize));
            
            // Copy all used blocks to the new pool
            char* currentPos = static_cast<char*>(newPool);
            std::vector<std::pair<void*, MemoryBlock*>> usedBlocks;
            
            for (auto& block : m_blocks) {
                if (block->inUse) {
                    // Copy data to new location
                    CUDA_CHECK(cudaMemcpy(currentPos, block->ptr, block->size, cudaMemcpyDeviceToDevice));
                    usedBlocks.emplace_back(block->ptr, block.get());
                    currentPos += block->size;
                }
            }
            
            // Update block pointers and free old memory
            currentPos = static_cast<char*>(newPool);
            for (auto& pair : usedBlocks) {
                pair.second->ptr = currentPos;
                currentPos += pair.second->size;
            }
            
            // Free old blocks
            for (auto& block : m_blocks) {
                if (block->ptr && block->ptr != newPool) {
                    cudaFree(block->ptr);
                }
            }
            
            // Create new single free block
            m_blocks.clear();
            m_ptrToBlock.clear();
            
            auto newBlock = std::make_unique<MemoryBlock>();
            newBlock->ptr = newPool;
            newBlock->size = newPoolSize;
            newBlock->inUse = false;
            
            m_blocks.push_back(std::move(newBlock));
            m_ptrToBlock[newPool] = m_blocks.back().get();
            
            // Re-add used blocks
            currentPos = static_cast<char*>(newPool);
            for (auto& pair : usedBlocks) {
                auto block = std::make_unique<MemoryBlock>();
                block->ptr = currentPos;
                block->size = pair.second->size;
                block->inUse = true;
                
                m_blocks.push_back(std::move(block));
                m_ptrToBlock[currentPos] = m_blocks.back().get();
                currentPos += block->size;
            }
            
        } catch (const std::exception&) {
            // Cleanup if failed
            if (newPool) cudaFree(newPool);
            // Fall back to simple defragmentation
            MergeFreeBlocks();
        }
    }
}

bool ChCudaMemoryPool::CheckOOM(size_t requestedSize) {
    return (m_usedSize + requestedSize > m_totalPoolSize);
}

bool ChCudaMemoryPool::HandleOOM(size_t requestedSize) {
    // Try defragmentation first
    Defragment();

    // Check if we have enough space now
    if (m_usedSize + requestedSize <= m_totalPoolSize) {
        return true;
    }

    // Try to expand pool
    return ExpandPool(requestedSize);
}

MemoryBlock* ChCudaMemoryPool::FindBestFit(size_t size) {
    switch (m_allocationStrategy) {
        case 1: // First Fit
            return FindFirstFit(size);
        case 2: // Worst Fit
            return FindWorstFit(size);
        case 0: // Best Fit (default)
        default:
            break;
    }

    MemoryBlock* bestFit = nullptr;
    size_t smallestWaste = SIZE_MAX;

    for (auto& block : m_blocks) {
        if (!block->inUse && block->size >= size) {
            size_t waste = block->size - size;
            if (waste < smallestWaste) {
                smallestWaste = waste;
                bestFit = block.get();
            }
        }
    }

    return bestFit;
}

MemoryBlock* ChCudaMemoryPool::SplitBlock(MemoryBlock* block, size_t size) {
    if (block->size <= size + 1024) {  // Minimum split size
        return block;
    }

    // Create new block for remainder
    auto newBlock = std::make_unique<MemoryBlock>();
    newBlock->ptr = static_cast<char*>(block->ptr) + size;
    newBlock->size = block->size - size;
    newBlock->inUse = false;

    // Shrink original block
    block->size = size;

    // Insert new block after current block
    auto it = std::find_if(m_blocks.begin(), m_blocks.end(),
                          [block](const std::unique_ptr<MemoryBlock>& b) { return b.get() == block; });
    if (it != m_blocks.end()) {
        ++it;
        m_ptrToBlock[newBlock->ptr] = newBlock.get();
        m_blocks.insert(it, std::move(newBlock));
    }

    return block;
}

void ChCudaMemoryPool::MergeFreeBlocks() {
    for (size_t i = 0; i < m_blocks.size() - 1; ++i) {
        if (!m_blocks[i]->inUse && !m_blocks[i + 1]->inUse) {
            // Merge i+1 into i
            m_blocks[i]->size += m_blocks[i + 1]->size;
            m_ptrToBlock.erase(m_blocks[i + 1]->ptr);
            m_blocks.erase(m_blocks.begin() + i + 1);
            --i;  // Check this position again
        }
    }
}

bool ChCudaMemoryPool::ExpandPool(size_t minSize) {
    // Calculate expansion size (at least minSize, but prefer larger chunks)
    size_t expandSize = std::max(minSize, m_totalPoolSize / 4);  // Grow by at least 25%
    expandSize = std::max(expandSize, size_t(64 * 1024 * 1024));  // Minimum 64MB expansion

    try {
        void* newPtr = nullptr;
        CUDA_CHECK(cudaMalloc(&newPtr, expandSize));

        auto newBlock = std::make_unique<MemoryBlock>();
        newBlock->ptr = newPtr;
        newBlock->size = expandSize;
        newBlock->inUse = false;

        m_blocks.push_back(std::move(newBlock));
        m_ptrToBlock[newPtr] = m_blocks.back().get();
        m_totalPoolSize += expandSize;

        return true;

    } catch (const std::exception&) {
        return false;
    }
}

void ChCudaMemoryPool::SetAllocationStrategy(int strategy) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (strategy >= 0 && strategy <= 2) {
        m_allocationStrategy = strategy;
    }
}

int ChCudaMemoryPool::GetAllocationStrategy() const {
    return m_allocationStrategy;
}

MemoryBlock* ChCudaMemoryPool::FindFirstFit(size_t size) {
    for (auto& block : m_blocks) {
        if (!block->inUse && block->size >= size) {
            return block.get();
        }
    }
    return nullptr;
}

MemoryBlock* ChCudaMemoryPool::FindWorstFit(size_t size) {
    MemoryBlock* worstFit = nullptr;
    size_t largestWaste = 0;

    for (auto& block : m_blocks) {
        if (!block->inUse && block->size >= size) {
            size_t waste = block->size - size;
            if (waste > largestWaste) {
                largestWaste = waste;
                worstFit = block.get();
            }
        }
    }

    return worstFit;
}

} // namespace cuda
} // namespace collision
} // namespace chrono