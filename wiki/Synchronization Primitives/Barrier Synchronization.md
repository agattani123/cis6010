<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw2/hw2/barrier.cu](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier.cu)
- [deprecated/hw2/hw2/barrier.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier.cuh)
- [deprecated/hw2/hw2/barrier_kernel.cu](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier_kernel.cu)
- [deprecated/hw2/hw2/barrier_kernel.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier_kernel.cuh)
- [deprecated/hw2/hw2/utils.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/utils.cuh)

</details>

# Barrier Synchronization

## Introduction

Barrier synchronization is a mechanism used in parallel computing to ensure that all threads or processes have reached a specific point in the program before proceeding further. It is a crucial component in parallel programming, as it helps maintain data consistency and prevent race conditions. In the context of this project, barrier synchronization is implemented using CUDA, a parallel computing platform and programming model developed by NVIDIA for general-purpose computing on graphics processing units (GPUs).

The barrier synchronization implementation in this project is designed to work with CUDA kernels, which are functions that run on the GPU's parallel processing units. The barrier synchronization mechanism ensures that all threads within a CUDA block have completed their assigned tasks before proceeding to the next step of the computation.

Sources: [barrier.cuh:1-10](), [barrier_kernel.cuh:1-10]()

## Barrier Synchronization Implementation

The barrier synchronization implementation in this project consists of several components, including the `Barrier` class, the `BarrierKernel` class, and various helper functions and utilities.

### Barrier Class

The `Barrier` class is a C++ class that provides an interface for managing barrier synchronization on the host (CPU) side. It is responsible for allocating and deallocating memory on the device (GPU), as well as launching the CUDA kernel that performs the actual barrier synchronization.

```cpp
class Barrier {
public:
    Barrier(int numThreads, int numBlocks);
    ~Barrier();
    void sync();

private:
    int numThreads, numBlocks;
    BarrierKernel *barrierKernel;
    int *deviceCount;
};
```

The `Barrier` class has the following key components:

- `numThreads` and `numBlocks`: These variables represent the number of threads and blocks, respectively, that will participate in the barrier synchronization.
- `barrierKernel`: An instance of the `BarrierKernel` class, which encapsulates the CUDA kernel implementation for barrier synchronization.
- `deviceCount`: A pointer to an integer array on the device (GPU) memory, used for tracking the number of threads that have reached the barrier.

The `sync()` method is the main entry point for initiating barrier synchronization. It launches the CUDA kernel and waits for all threads to complete the synchronization process.

Sources: [barrier.cu:1-29](), [barrier.cuh:1-17]()

### BarrierKernel Class

The `BarrierKernel` class encapsulates the implementation of the CUDA kernel that performs the actual barrier synchronization on the device (GPU) side.

```cpp
class BarrierKernel {
public:
    BarrierKernel(int numThreads, int numBlocks);
    ~BarrierKernel();
    void sync(int *deviceCount);

private:
    int numThreads, numBlocks;
    int *deviceCount;
};
```

The `BarrierKernel` class has the following key components:

- `numThreads` and `numBlocks`: These variables represent the number of threads and blocks, respectively, that will participate in the barrier synchronization.
- `deviceCount`: A pointer to an integer array on the device (GPU) memory, used for tracking the number of threads that have reached the barrier.

The `sync()` method is the entry point for the CUDA kernel implementation. It performs the barrier synchronization by using atomic operations to update the `deviceCount` array and synchronize the threads within each block.

Sources: [barrier_kernel.cu:1-49](), [barrier_kernel.cuh:1-16]()

### Barrier Synchronization Workflow

The overall workflow of the barrier synchronization implementation is as follows:

1. The host (CPU) creates an instance of the `Barrier` class, providing the number of threads and blocks.
2. The `Barrier` class allocates memory on the device (GPU) for the `deviceCount` array and creates an instance of the `BarrierKernel` class.
3. When the `sync()` method is called on the `Barrier` instance, it launches the CUDA kernel by invoking the `sync()` method of the `BarrierKernel` instance.
4. The CUDA kernel implementation in the `BarrierKernel` class performs the following steps:
   a. Each thread atomically increments the corresponding element in the `deviceCount` array.
   b. Threads within each block synchronize using the `__syncthreads()` intrinsic.
   c. The last thread in each block checks if all threads in the block have reached the barrier.
   d. If all threads have reached the barrier, the last thread resets the corresponding element in the `deviceCount` array to zero.
   e. Threads within each block synchronize again using `__syncthreads()`.
5. The host (CPU) waits for the CUDA kernel to complete and then proceeds with the next step of the computation.

Sources: [barrier.cu:31-49](), [barrier_kernel.cu:17-49](), [barrier_kernel.cuh:18-28]()

## Barrier Synchronization Workflow Diagram

The following Mermaid diagram illustrates the workflow of the barrier synchronization implementation:

```mermaid
graph TD
    subgraph Host
        A[Create Barrier instance] -->|numThreads, numBlocks| B[Allocate deviceCount<br>Create BarrierKernel]
        C[Call sync()] -->|Launch CUDA kernel| D[Wait for kernel completion]
    end

    subgraph Device
        E[Kernel: BarrierKernel::sync]
        F[Atomically increment<br>deviceCount[threadIdx]]
        G[__syncthreads]
        H[Last thread?]
        I[Reset deviceCount<br>element to 0]
        J[__syncthreads]
        E --> F --> G --> H
        H -->|Yes| I --> J
        H -->|No| J
    end

    B --> E
    J --> D
```

This diagram shows the interaction between the host (CPU) and device (GPU) components, as well as the steps performed by the CUDA kernel for barrier synchronization.

Sources: [barrier.cu:31-49](), [barrier_kernel.cu:17-49](), [barrier_kernel.cuh:18-28]()

## Utility Functions

The project includes several utility functions and macros that are used in the barrier synchronization implementation.

### `getNumBlocks` Function

The `getNumBlocks` function calculates the number of blocks required for a given number of threads and a maximum number of threads per block.

```cpp
inline int getNumBlocks(int numThreads, int maxThreadsPerBlock) {
    return (numThreads + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
}
```

This function is used to determine the number of blocks required for launching the CUDA kernel, ensuring that all threads are properly distributed across the available blocks.

Sources: [utils.cuh:5-8]()

### `CHECK_CUDA_ERROR` Macro

The `CHECK_CUDA_ERROR` macro is a utility macro used for error checking and reporting CUDA errors.

```cpp
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }
```

This macro checks the return value of CUDA API calls and prints an error message if an error occurs. It is used throughout the project to ensure proper error handling and debugging.

Sources: [utils.cuh:11-16]()

## Configuration and Usage

The barrier synchronization implementation can be configured by specifying the number of threads and blocks when creating an instance of the `Barrier` class.

```cpp
int numThreads = 256;
int numBlocks = getNumBlocks(numThreads, 256);

Barrier barrier(numThreads, numBlocks);
```

In this example, the number of threads is set to 256, and the number of blocks is calculated using the `getNumBlocks` utility function, assuming a maximum of 256 threads per block.

To initiate barrier synchronization, the `sync()` method of the `Barrier` instance should be called:

```cpp
barrier.sync();
```

This method will launch the CUDA kernel and wait for all threads to complete the barrier synchronization process.

## Performance Considerations

The performance of the barrier synchronization implementation can be influenced by several factors, including:

- The number of threads and blocks: The optimal number of threads and blocks can vary depending on the specific hardware and workload. It is generally recommended to use a multiple of the warp size (32 threads) for the number of threads per block.
- Memory access patterns: The way threads access shared memory and global memory can impact performance. Coalesced memory access patterns are generally more efficient.
- Synchronization overhead: Barrier synchronization introduces overhead due to the need for atomic operations and thread synchronization within blocks. This overhead can become more significant as the number of threads and blocks increases.

To optimize performance, it is essential to profile the application and experiment with different thread and block configurations, as well as memory access patterns, to find the optimal settings for the specific use case.

Sources: [barrier.cu:1-49](), [barrier_kernel.cu:1-49](), [barrier.cuh:1-17](), [barrier_kernel.cuh:1-28](), [utils.cuh:1-16]()