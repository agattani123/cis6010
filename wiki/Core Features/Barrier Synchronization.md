<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw2/hw2/barrier.cu](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier.cu)
- [deprecated/hw2/hw2/barrier.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier.cuh)
- [deprecated/hw2/hw2/lock.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/lock.cuh)
- [deprecated/hw2/hw2/timer.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/timer.cuh)
- [deprecated/hw2/hw2/utils.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/utils.cuh)

</details>

# Barrier Synchronization

## Introduction

Barrier synchronization is a mechanism used in parallel computing to ensure that all threads or processes have reached a specific point in the program before any of them can proceed further. This is particularly important in scenarios where threads or processes need to coordinate their activities or share data, preventing race conditions and ensuring data consistency.

In the context of this project, barrier synchronization is implemented using different approaches, including a spin barrier and a two-level barrier. These barriers are used in a kernel function `rotateRows` to synchronize the execution of threads across multiple blocks and warps, ensuring that all threads have completed their tasks before proceeding to the next iteration.

Sources: [barrier.cu](), [barrier.cuh]()

## Barrier Interface

The project defines an abstract interface `IBarrier` for barrier synchronization. This interface serves as a base class for implementing different types of barriers.

```cpp
class IBarrier {
protected:
    const unsigned m_expected;
    unsigned arrived;
    bool sense;

public:
    __device__ IBarrier(const unsigned count) : m_expected(count) {
        arrived = 0;
        sense = true;
    }
    __device__ virtual void wait() = 0;
};
```

The `IBarrier` class has the following members:

- `m_expected`: The expected number of threads or processes that should arrive at the barrier.
- `arrived`: The number of threads or processes that have arrived at the barrier so far.
- `sense`: A flag used for alternating between different barrier instances.
- `wait()`: A pure virtual function that must be implemented by derived classes to define the barrier synchronization logic.

Sources: [barrier.cu:18-31]()

## Spin Barrier

The `SpinBarrier` class is a concrete implementation of the `IBarrier` interface. It uses a spin-lock mechanism to synchronize threads at the barrier.

```cpp
class SpinBarrier : public IBarrier, public WarpLevelLock {
public:
    __device__ SpinBarrier(const unsigned count) : IBarrier(count) {}

    __device__ virtual void wait() {
        // TODO: PART 5
    }
};
```

The `SpinBarrier` class inherits from `IBarrier` and `WarpLevelLock`. The `WarpLevelLock` class (defined in `lock.cuh`) provides a warp-level lock implementation for synchronization within a warp.

The `wait()` function is currently unimplemented and marked as "TODO: PART 5", suggesting that it needs to be completed as part of the project's requirements.

Sources: [barrier.cu:34-42]()

## Two-Level Barrier

The `TwoLevelBarrier` class is another concrete implementation of the `IBarrier` interface. It extends the `SpinBarrier` class and likely implements a two-level barrier synchronization mechanism.

```cpp
class TwoLevelBarrier : public SpinBarrier {
public:
    __device__ TwoLevelBarrier(const unsigned count) : SpinBarrier(count) {}

    __device__ virtual void wait() {
        // TODO: PART 6
    }
};
```

Similar to the `SpinBarrier` class, the `wait()` function is currently unimplemented and marked as "TODO: PART 6", indicating that it needs to be completed as part of the project's requirements.

Sources: [barrier.cu:44-51]()

## Barrier Initialization and Destruction

The project provides global device variables `d_SpinBar` and `d_2LBar` to hold instances of the `SpinBarrier` and `TwoLevelBarrier` classes, respectively.

```cpp
__device__ SpinBarrier* d_SpinBar = NULL;
__device__ TwoLevelBarrier* d_2LBar = NULL;
```

These instances are initialized and destroyed using the following kernel functions:

```cpp
__global__ void initBarriers() {
    assert(blockIdx.x == 0 && threadIdx.x == 0);
    d_SpinBar = new SpinBarrier(NUM_WARPS);
    d_2LBar = new TwoLevelBarrier(NUM_BLOCKS);
}

__global__ void destroyBarriers() {
    assert(blockIdx.x == 0 && threadIdx.x == 0);
    delete d_SpinBar;
    delete d_2LBar;
}
```

The `initBarriers` kernel function initializes the `d_SpinBar` and `d_2LBar` instances with the number of warps (`NUM_WARPS`) and blocks (`NUM_BLOCKS`), respectively. The `destroyBarriers` kernel function deallocates the memory used by these instances.

Sources: [barrier.cu:53-62](), [barrier.cu:64-69]()

## Barrier Usage

The barrier synchronization mechanisms are used in the `rotateRows` kernel function, which performs a row rotation operation on a 2D square array.

```cpp
__global__ void rotateRows(const BarrierFlavor flavor, int* array, const int arrayDim, const int sourceRow) {
    // ... (implementation omitted for brevity) ...

    if (flavor == SPIN_BARRIER) {
        d_SpinBar->wait();
    }
    else if (flavor == TWO_LEVEL_BARRIER) {
        d_2LBar->wait();
    }
    // ... (implementation omitted for brevity) ...
}
```

The `rotateRows` kernel function takes a `BarrierFlavor` parameter, which determines the type of barrier synchronization to be used. If the `SPIN_BARRIER` flavor is specified, the `wait()` function of the `d_SpinBar` instance is called. If the `TWO_LEVEL_BARRIER` flavor is specified, the `wait()` function of the `d_2LBar` instance is called.

Sources: [barrier.cu:73-95]()

## Barrier Testing

The project includes a `barrierTest` function that performs a test of the barrier synchronization mechanisms by launching the `rotateRows` kernel with different barrier flavors.

```cpp
void barrierTest(const BarrierFlavor flavor) {
    // ... (implementation omitted for brevity) ...

    initBarriers<<<1, 1>>>();
    // ... (implementation omitted for brevity) ...

    if (flavor == KERNEL_LAUNCH_BARRIER) {
        // TODO: PART 4
    } else {
        rotateRows<<<NUM_BLOCKS, WARPS_PER_BLOCK * WARP_SIZE>>>(flavor, d_array, numThreads, -1);
    }

    // ... (implementation omitted for brevity) ...

    destroyBarriers<<<1, 1>>>();
    // ... (implementation omitted for brevity) ...
}
```

The `barrierTest` function initializes the barriers, launches the `rotateRows` kernel with the specified barrier flavor, and then destroys the barriers. It also performs memory allocation, data initialization, and result verification.

Sources: [barrier.cu:97-184]()

## Conclusion

The barrier synchronization implementation in this project provides a flexible mechanism for synchronizing threads across multiple blocks and warps. The `IBarrier` interface serves as a base for implementing different barrier synchronization strategies, such as the `SpinBarrier` and `TwoLevelBarrier` classes. These barriers are used in the `rotateRows` kernel function to ensure proper coordination and data consistency during the row rotation operation.

While the implementation details of the `wait()` functions for the `SpinBarrier` and `TwoLevelBarrier` classes are not provided in the given source files, the overall structure and usage of the barrier synchronization mechanisms are clearly defined.