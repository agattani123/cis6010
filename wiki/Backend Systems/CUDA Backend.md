<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw0/helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/deprecated/hw0/helper_cuda.h)
- [deprecated/hw1/src/helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/deprecated/hw1/src/helper_cuda.h)
- [deprecated/hw2/hw2/helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2/helper_cuda.h)
- [deprecated/transpose/transpose/helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/deprecated/transpose/transpose/helper_cuda.h)
- [src/cuda/helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)
</details>

# CUDA Backend

## Introduction

The CUDA Backend is a set of helper functions and utilities for initializing and managing CUDA devices, handling errors, and providing common functionality for CUDA-based applications. It serves as a foundation for various CUDA-related projects, offering a consistent and streamlined approach to working with CUDA.

The CUDA Backend primarily consists of the `helper_cuda.h` header file, which includes functions for device initialization, error checking, and GPU architecture detection. It also provides utility functions for converting data types and handling command-line arguments related to CUDA devices.

## Error Handling

The CUDA Backend provides a comprehensive error handling mechanism through the `_cudaGetErrorEnum` function and the `check` template function. These functions allow for the retrieval of error messages from various CUDA APIs, such as CUDA Runtime, CUDA Driver, cuBLAS, cuFFT, cuSPARSE, cuSOLVER, cuRAND, and NPP.

The `check` template function is used to check the result of a CUDA operation and print an error message with the corresponding error code and description if an error occurs. It also performs a device reset and exits the application in case of an error.

```cpp
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}
```

The `checkCudaErrors` and `getLastCudaError` macros are provided for convenient error checking in CUDA host calls and retrieving the last CUDA error, respectively.

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

## Device Initialization

The CUDA Backend offers several functions for initializing and selecting CUDA devices:

### `gpuDeviceInit`

This function initializes a CUDA device with the specified device ID. It checks for the availability of CUDA devices, validates the provided device ID, and sets the current device for CUDA operations.

```cpp
inline int gpuDeviceInit(int devID)
{
    // ...
    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
    return devID;
}
```

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

### `gpuGetMaxGflopsDeviceId`

This function returns the ID of the CUDA device with the maximum GFLOPS (Giga Floating-Point Operations per Second) performance. It iterates through all available devices, calculates the compute performance based on the device's architecture and clock rate, and selects the device with the highest performance.

```cpp
inline int gpuGetMaxGflopsDeviceId()
{
    // ...
    while (current_device < device_count)
    {
        // ...
        unsigned long long compute_perf  = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if (compute_perf  > max_compute_perf)
        {
            // ...
            max_compute_perf  = compute_perf;
            max_perf_device   = current_device;
        }
        // ...
    }
    return max_perf_device;
}
```

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

### `findCudaDevice`

This function is responsible for selecting the CUDA device to be used in the application. It checks for command-line arguments specifying a device ID and, if not provided, selects the device with the highest GFLOPS performance using `gpuGetMaxGflopsDeviceId`.

```cpp
inline int findCudaDevice(int argc, const char **argv)
{
    // ...
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");
        devID = gpuDeviceInit(devID);
    }
    else
    {
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(devID));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    return devID;
}
```

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

## GPU Architecture Detection

The CUDA Backend provides functionality to detect the GPU architecture and determine the number of cores per Streaming Multiprocessor (SM) based on the SM version.

### `_ConvertSMVer2Cores`

This function takes the major and minor version numbers of the SM and returns the corresponding number of cores per SM for that GPU architecture.

```cpp
inline int _ConvertSMVer2Cores(int major, int minor)
{
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        // ...
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
```

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

### `checkCudaCapabilities`

This function checks if the current CUDA device supports the specified compute capability (major and minor version numbers). It prints a message indicating whether the device meets the required capability or not.

```cpp
inline bool checkCudaCapabilities(int major_version, int minor_version)
{
    cudaDeviceProp deviceProp;
    // ...
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if ((deviceProp.major > major_version) ||
        (deviceProp.major == major_version && deviceProp.minor >= minor_version))
    {
        printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
        return true;
    }
    else
    {
        printf("  No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}
```

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

## Utility Functions

The CUDA Backend provides several utility functions for common tasks, such as data type conversion and handling command-line arguments.

### `ftoi`

This function converts a floating-point value to an integer using rounding.

```cpp
inline int ftoi(float value)
{
    return (value >= 0 ? (int)(value + 0.5) : (int)(value - 0.5));
}
```

Sources: [helper_cuda.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_cuda.h)

### `checkCmdLineFlag` and `getCmdLineArgumentInt`

These functions are used to check for command-line arguments and retrieve integer values from them, respectively. They are used in the `findCudaDevice` function to handle command-line arguments specifying the CUDA device ID.

```cpp
// Defined in helper_string.h
inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    // ...
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    // ...
}
```

Sources: [helper_string.h](https://github.com/agattani123/cis6010/blob/main/src/cuda/helper_string.h)

## Conclusion

The CUDA Backend provides a comprehensive set of helper functions and utilities for working with CUDA devices, handling errors, detecting GPU architectures, and performing common tasks. It serves as a foundation for CUDA-based applications, offering a consistent and streamlined approach to managing CUDA resources and ensuring proper error handling.