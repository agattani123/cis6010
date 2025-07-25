<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw2/hw2/lock.cu](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/lock.cu)
- [deprecated/hw2/hw2/lock.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/lock.cuh)
- [deprecated/hw2/hw2/barrier.cu](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier.cu)
- [deprecated/hw2/hw2/barrier.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/barrier.cuh)
- [deprecated/hw2/hw2/timer.cuh](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/timer.cuh)

</details>

# Lock Implementation

## Introduction

The Lock Implementation is a critical component of the project, providing synchronization mechanisms to ensure thread safety and data integrity in parallel computations on the GPU. It introduces the `WarpLevelLock` class, which allows for efficient locking at the warp level, enabling coordinated access to shared resources among threads within a warp.

The implementation includes various lock-related kernels, such as `initLock`, `destroyLock`, `incrementCounterWarpsOnly`, and `incrementCounterAllThreads`, which demonstrate the usage and performance of the locking mechanism in different scenarios.

Sources: [lock.cu](), [lock.cuh]()

## WarpLevelLock Class

The `WarpLevelLock` class is the core of the Lock Implementation, providing methods for acquiring and releasing locks at the warp level.

### Class Members

#### `theLock`

```cpp
volatile unsigned theLock;
```

A volatile unsigned integer variable representing the lock state. It can have two values:

- `LOCK_FREE` (0): The lock is available.
- `LOCK_HELD` (1): The lock is currently held by a warp.

Sources: [lock.cuh:9]()

### Constructor

```cpp
__device__ WarpLevelLock() {
    theLock = LOCK_FREE;
}
```

The constructor initializes the `theLock` variable to `LOCK_FREE`, indicating that the lock is initially available.

Sources: [lock.cuh:12-14]()

### Methods

#### `waitFor`

```cpp
__device__ void waitFor(unsigned int until) {
    clock_t start = clock64();
    clock_t now;
    for (;;) {
        now = clock64();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= until) {
            break;
        }
    }
}
```

This method is a helper function that introduces a delay by spinning for a specified number of clock cycles. It is used for implementing backoff strategies in the locking mechanism.

Sources: [lock.cuh:16-24]()

#### `lock`

```cpp
__device__ void lock() {
    // TODO: PART 1
}
```

This method is responsible for acquiring the lock. The implementation is not provided in the given source files and is marked as a TODO for Part 1 of the project.

Sources: [lock.cuh:26-28]()

#### `backoffLock`

```cpp
__device__ void backoffLock() {
    // TODO: PART 2
    unsigned int period = 100;
}
```

This method is intended to implement a backoff strategy for acquiring the lock. The implementation is not provided in the given source files and is marked as a TODO for Part 2 of the project. It includes a placeholder `period` variable, which might be used for controlling the backoff behavior.

Sources: [lock.cuh:30-33]()

#### `unlock`

```cpp
__device__ void unlock() {
    // TODO: PART 1
}
```

This method is responsible for releasing the lock. The implementation is not provided in the given source files and is marked as a TODO for Part 1 of the project.

Sources: [lock.cuh:35-37]()

## Lock-related Kernels

The project includes several CUDA kernels that demonstrate the usage and performance of the locking mechanism in different scenarios.

### `initLock`

```cpp
__global__ void initLock() {
    assert(blockIdx.x == 0 && threadIdx.x == 0);
    d_WLLock = new WarpLevelLock();
}
```

This kernel initializes the `d_WLLock` device variable, which is a pointer to a `WarpLevelLock` instance. It is launched with a single thread block and a single thread, ensuring that only one thread creates the lock instance.

Sources: [lock.cu:14-17]()

### `destroyLock`

```cpp
__global__ void destroyLock() {
    assert(blockIdx.x == 0 && threadIdx.x == 0);
    delete d_WLLock;
}
```

This kernel deallocates the memory occupied by the `d_WLLock` instance. Similar to `initLock`, it is launched with a single thread block and a single thread to ensure thread safety during deallocation.

Sources: [lock.cu:19-22]()

### `incrementCounterWarpsOnly`

```cpp
__global__ void incrementCounterWarpsOnly() {
    if (threadIdx.x % warpSize != 0) { return; }

    for (unsigned i = 0; i < ITERATIONS; i++) {
        d_WLLock->lock();
        d_counter++;
        d_WLLock->unlock();
    }
}
```

This kernel demonstrates the usage of the `WarpLevelLock` by incrementing a shared counter `d_counter` in a synchronized manner. Only one thread per warp executes the loop, acquiring the lock, incrementing the counter, and releasing the lock for a specified number of iterations.

Sources: [lock.cu:24-32]()

### `incrementCounterAllThreads`

```cpp
__global__ void incrementCounterAllThreads() {
    for (unsigned i = 0; i < ITERATIONS; i++) {
        d_counter++;
    }
}
```

This kernel is intended to demonstrate the usage of the locking mechanism when all threads increment the shared counter `d_counter`. However, the implementation is not provided in the given source files and is marked as a TODO for Part 3 of the project.

Sources: [lock.cu:34-38]()

## Lock Testing

The `lockTest` function is responsible for testing the performance and correctness of the locking mechanism in different scenarios.

```cpp
void lockTest(const LockFlavor flavor) {
    cudaError_t cudaStatus;
    CudaTimer timer;

    // ALLOCATE DEVICE MEMORY
    timer.start();

    initLock<<<1, 1>>>();
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);

    printf("Setup device memory:  %3.1f ms \n", timer.stop());

    // LAUNCH KERNELS

    timer.start();
    d_counter = 0;
    if (flavor == LOCK_PER_WARP) {
        incrementCounterWarpsOnly<<<NUM_BLOCKS, WARPS_PER_BLOCK * WARP_SIZE>>>();
    } else if (flavor == LOCK_PER_THREAD) {
        incrementCounterAllThreads<<<NUM_BLOCKS, WARPS_PER_BLOCK * WARP_SIZE>>>();
    } else {
        assert(false);
    }

    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);

    const float elapsed = timer.stop();
    printf("Lock kernel time:  %3.1f ms \n", elapsed);

    // CHECK COUNTER VALUE IS CORRECT
    unsigned expected = 0;
    if (flavor == LOCK_PER_WARP) {
        expected = NUM_BLOCKS * WARPS_PER_BLOCK * ITERATIONS;
    } else if (flavor == LOCK_PER_THREAD) {
        expected = NUM_BLOCKS * WARPS_PER_BLOCK * WARP_SIZE * ITERATIONS;
    }
    if (d_counter != expected) {
        printf("Expected counter value %u BUT GOT %u INSTEAD :-(\n", expected, d_counter);
    } else {
        printf("Counter has expected value of %u\n", expected);
    }

    printf("Increments/ms: %3.1f\n", expected / elapsed);

    // CLEANUP

    destroyLock<<<1, 1>>>();
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);
}
```

The `lockTest` function takes a `LockFlavor` enum value, which determines whether the test should use the `incrementCounterWarpsOnly` kernel (LOCK_PER_WARP) or the `incrementCounterAllThreads` kernel (LOCK_PER_THREAD).

The function performs the following steps:

1. Allocates device memory and initializes the lock using the `initLock` kernel.
2. Launches the appropriate kernel (`incrementCounterWarpsOnly` or `incrementCounterAllThreads`) based on the provided `LockFlavor`.
3. Checks for any CUDA errors and synchronizes the device.
4. Calculates the expected value of the counter based on the number of blocks, warps per block, warp size, and iterations.
5. Compares the actual counter value with the expected value and prints the result.
6. Prints the number of increments per millisecond.
7. Cleans up by deallocating the lock using the `destroyLock` kernel.

The `lockTest` function is called twice in the `main` function, once with `LOCK_PER_WARP` and once with `LOCK_PER_THREAD`, to test both scenarios.

Sources: [lock.cu:40-93](), [lock.cu:111-112]()

## Main Function

The `main` function is the entry point of the program and performs the following tasks:

1. Checks if kernel timeout is enabled on the GPU device.
2. Calls the `lockTest` function with `LOCK_PER_WARP` to test the locking mechanism with one thread per warp.
3. Calls the `lockTest` function with `LOCK_PER_THREAD` to test the locking mechanism with all threads.
4. Calls the `barrierTest` function with different barrier implementations (`KERNEL_LAUNCH_BARRIER`, `SPIN_BARRIER`, and `TWO_LEVEL_BARRIER`).
5. Resets the CUDA device before exiting.

Sources: [lock.cu:95-125]()

## Conclusion

The Lock Implementation provides a `WarpLevelLock` class and related kernels to enable synchronized access to shared resources among threads within a warp. It demonstrates the usage and performance of the locking mechanism in different scenarios, where either one thread per warp or all threads increment a shared counter. The implementation includes placeholders for the actual lock acquisition and release logic, which are marked as TODOs for future development.

Sources: [lock.cu](), [lock.cuh]()