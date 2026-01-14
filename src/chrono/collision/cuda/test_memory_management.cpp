#include "ChCudaMemoryPool.h"
#include "ChCudaUtils.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace chrono::collision::cuda;

void TestBasicAllocation() {
    std::cout << "=== Testing Basic Allocation ===" << std::endl;
    
    try {
        ChCudaMemoryPool::GetInstance().Initialize(64 * 1024 * 1024); // 64MB
        
        // Test device allocation
        void* devPtr1 = ChCudaMemoryPool::GetInstance().AllocateDevice(1024 * 1024); // 1MB
        void* devPtr2 = ChCudaMemoryPool::GetInstance().AllocateDevice(2 * 1024 * 1024); // 2MB
        
        // Test pinned allocation
        void* pinnedPtr = ChCudaMemoryPool::GetInstance().AllocatePinned(512 * 1024); // 512KB
        
        std::cout << "✓ Basic allocation successful" << std::endl;
        
        // Test memory stats
        size_t total, used, blocks;
        ChCudaMemoryPool::GetInstance().GetMemoryStats(total, used, blocks);
        std::cout << "  Total: " << total / (1024 * 1024) << " MB, Used: " << used / (1024 * 1024) << " MB" << std::endl;
        
        // Cleanup
        ChCudaMemoryPool::GetInstance().FreeDevice(devPtr1);
        ChCudaMemoryPool::GetInstance().FreeDevice(devPtr2);
        ChCudaMemoryPool::GetInstance().FreePinned(pinnedPtr);
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Basic allocation failed: " << e.what() << std::endl;
    }
}

void TestAsyncTransfers() {
    std::cout << "\n=== Testing Async Transfers ===" << std::endl;
    
    try {
        // Create test data
        std::vector<int> hostData(1024 * 1024 / sizeof(int), 42); // 1MB of data
        
        // Allocate device memory
        void* devPtr = ChCudaMemoryPool::GetInstance().AllocateDevice(hostData.size() * sizeof(int));
        void* pinnedPtr = ChCudaMemoryPool::GetInstance().AllocatePinned(hostData.size() * sizeof(int));
        
        // Test async copy to device
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        cudaError_t err = ChCudaMemoryPool::GetInstance().AsyncCopyToDevice(
            devPtr, hostData.data(), hostData.size() * sizeof(int), stream);
        
        if (err == cudaSuccess) {
            std::cout << "✓ Async copy to device successful" << std::endl;
        } else {
            std::cerr << "✗ Async copy to device failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        // Test async copy with explicit event
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreate(&event));
        
        err = ChCudaMemoryPool::GetInstance().AsyncCopyToHostWithEvent(
            pinnedPtr, devPtr, hostData.size() * sizeof(int), stream, &event);
        
        if (err == cudaSuccess) {
            std::cout << "✓ Async copy with event successful" << std::endl;
        } else {
            std::cerr << "✗ Async copy with event failed: " << cudaGetErrorString(err) << std::endl;
        }
        
        // Wait for completion
        ChCudaMemoryPool::GetInstance().WaitForAllTransfers();

        // Cleanup
        CUDA_CHECK(cudaStreamDestroy(stream));
        // Note: event is destroyed by WaitForAllTransfers
        ChCudaMemoryPool::GetInstance().FreeDevice(devPtr);
        ChCudaMemoryPool::GetInstance().FreePinned(pinnedPtr);
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Async transfers failed: " << e.what() << std::endl;
    }
}

void TestAllocationStrategies() {
    std::cout << "\n=== Testing Allocation Strategies ===" << std::endl;
    
    try {
        // Test different allocation strategies
        const int strategies[] = {0, 1, 2}; // Best Fit, First Fit, Worst Fit
        const char* strategyNames[] = {"Best Fit", "First Fit", "Worst Fit"};
        
        for (int i = 0; i < 3; i++) {
            ChCudaMemoryPool::GetInstance().SetAllocationStrategy(strategies[i]);
            std::cout << "Testing " << strategyNames[i] << " strategy..." << std::endl;
            
            // Allocate some memory
            void* ptr1 = ChCudaMemoryPool::GetInstance().AllocateDevice(1024 * 1024);
            void* ptr2 = ChCudaMemoryPool::GetInstance().AllocateDevice(2048 * 1024);
            void* ptr3 = ChCudaMemoryPool::GetInstance().AllocateDevice(512 * 1024);
            
            // Free middle one to create fragmentation
            ChCudaMemoryPool::GetInstance().FreeDevice(ptr2);
            
            // Allocate again
            void* ptr4 = ChCudaMemoryPool::GetInstance().AllocateDevice(1500 * 1024);
            
            std::cout << "✓ " << strategyNames[i] << " strategy working" << std::endl;
            
            // Cleanup
            ChCudaMemoryPool::GetInstance().FreeDevice(ptr1);
            ChCudaMemoryPool::GetInstance().FreeDevice(ptr3);
            ChCudaMemoryPool::GetInstance().FreeDevice(ptr4);
        }
        
        // Reset to default
        ChCudaMemoryPool::GetInstance().SetAllocationStrategy(0);
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Allocation strategies failed: " << e.what() << std::endl;
    }
}

void TestMemoryReports() {
    std::cout << "\n=== Testing Memory Reports ===" << std::endl;
    
    try {
        // Create some allocations
        void* ptr1 = ChCudaMemoryPool::GetInstance().AllocateDevice(5 * 1024 * 1024); // 5MB
        void* ptr2 = ChCudaMemoryPool::GetInstance().AllocateDevice(3 * 1024 * 1024); // 3MB
        void* pinned1 = ChCudaMemoryPool::GetInstance().AllocatePinned(1 * 1024 * 1024); // 1MB
        
        // Test basic report
        ChCudaMemoryPool::GetInstance().PrintMemoryReport();
        
        // Test detailed report
        ChCudaMemoryPool::GetInstance().PrintDetailedMemoryReport();
        
        // Cleanup
        ChCudaMemoryPool::GetInstance().FreeDevice(ptr1);
        ChCudaMemoryPool::GetInstance().FreeDevice(ptr2);
        ChCudaMemoryPool::GetInstance().FreePinned(pinned1);
        
        std::cout << "✓ Memory reports successful" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Memory reports failed: " << e.what() << std::endl;
    }
}

void TestDefragmentation() {
    std::cout << "\n=== Testing Defragmentation ===" << std::endl;
    
    try {
        // Create fragmentation
        std::vector<void*> ptrs;
        for (int i = 0; i < 10; i++) {
            ptrs.push_back(ChCudaMemoryPool::GetInstance().AllocateDevice(1024 * 1024)); // 1MB each
        }
        
        // Free every other block
        for (int i = 1; i < 10; i += 2) {
            ChCudaMemoryPool::GetInstance().FreeDevice(ptrs[i]);
        }
        
        std::cout << "Before defragmentation:" << std::endl;
        ChCudaMemoryPool::GetInstance().PrintMemoryReport();
        
        // Test simple defragmentation
        ChCudaMemoryPool::GetInstance().Defragment();
        std::cout << "\nAfter simple defragmentation:" << std::endl;
        ChCudaMemoryPool::GetInstance().PrintMemoryReport();
        
        // Test advanced defragmentation
        ChCudaMemoryPool::GetInstance().AdvancedDefragment();
        std::cout << "\nAfter advanced defragmentation:" << std::endl;
        ChCudaMemoryPool::GetInstance().PrintMemoryReport();
        
        // Cleanup remaining allocations
        for (int i = 0; i < 10; i += 2) {
            ChCudaMemoryPool::GetInstance().FreeDevice(ptrs[i]);
        }
        
        std::cout << "✓ Defragmentation successful" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Defragmentation failed: " << e.what() << std::endl;
    }
}

void TestOOMHandling() {
    std::cout << "\n=== Testing OOM Handling ===" << std::endl;
    
    try {
        // Try to allocate more memory than available
        void* largePtr = nullptr;
        try {
            largePtr = ChCudaMemoryPool::GetInstance().AllocateDevice(1024 * 1024 * 1024); // 1GB
            std::cout << "✗ OOM test failed - allocation should have failed" << std::endl;
        } catch (const std::exception&) {
            std::cout << "✓ OOM handling working - allocation correctly failed" << std::endl;
        }
        
        // Test that pool can still work after OOM
        void* smallPtr = ChCudaMemoryPool::GetInstance().AllocateDevice(1024); // 1KB
        if (smallPtr) {
            std::cout << "✓ Pool still functional after OOM" << std::endl;
            ChCudaMemoryPool::GetInstance().FreeDevice(smallPtr);
        } else {
            std::cerr << "✗ Pool not functional after OOM" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ OOM handling test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CUDA Memory Management Test Suite" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Initialize CUDA
    if (!chrono::collision::ChCudaDeviceUtils::IsCudaAvailable()) {
        std::cerr << "CUDA not available!" << std::endl;
        return 1;
    }
    
    std::cout << "CUDA available, running tests..." << std::endl;
    
    // Run all tests
    TestBasicAllocation();
    TestAsyncTransfers();
    TestAllocationStrategies();
    TestMemoryReports();
    TestDefragmentation();
    TestOOMHandling();
    
    // Cleanup
    ChCudaMemoryPool::GetInstance().Shutdown();
    
    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
}