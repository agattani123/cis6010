<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw1/hw1.sln](https://github.com/agattani123/cis6010/blob/main/deprecated/hw1/hw1.sln)
- [deprecated/hw2/hw2.sln](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2.sln)
- [deprecated/transpose/transpose.sln](https://github.com/agattani123/cis6010/blob/main/deprecated/transpose/transpose.sln)
- [gemm/Makefile](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile)
- [gemm/cugemm.cu](https://github.com/agattani123/cis6010/blob/main/gemm/cugemm.cu)

</details>

# Local Development Setup

## Introduction

The provided source files appear to be related to a project involving CUDA programming and matrix operations on GPUs. The "Local Development Setup" section likely covers the steps and configurations required to build and run the project on a local development environment. This may include instructions for setting up the necessary tools, dependencies, and build processes.

Sources: [gemm/Makefile]()

## Build Process

The project includes a `Makefile` that defines various targets for building the project. The main targets are:

### 1. `all` (Default Target)

This target builds three different binaries:

- `cugemm.bin`: An optimized binary for the CUDA code.
- `cugemm-debug.bin`: A debug binary without optimizations, enabling easier debugging.
- `cugemm-profile.bin`: An optimized binary with line number information for profiling purposes.

```makefile
all: cugemm.bin cugemm-debug.bin cugemm-profile.bin
```

Sources: [gemm/Makefile:1-3]()

### 2. `cugemm.bin`

This target compiles the `cugemm.cu` source file using the NVIDIA CUDA compiler (`nvcc`). It enables C++17 support and generates code for the compute capability 7.5 (Turing architecture) with specific optimization flags.

```makefile
cugemm.bin: $(SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@
```

Sources: [gemm/Makefile:6-8]()

### 3. `cugemm-debug.bin`

This target compiles the `cugemm.cu` source file with debug symbols and without optimizations. It also includes additional flags for generating PTX (Parallel Thread Execution) code and preserving source code information.

```makefile
cugemm-debug.bin: $(SOURCE_FILE)
	nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@
```

Sources: [gemm/Makefile:11-13]()

### 4. `cugemm-profile.bin`

This target compiles the `cugemm.cu` source file with optimizations and includes line number information for profiling purposes. It also generates PTX code and preserves source code information.

```makefile
cugemm-profile.bin: $(SOURCE_FILE)
	nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@
```

Sources: [gemm/Makefile:16-18]()

### 5. `profile`

This target runs the NVIDIA profiling tool (`ncu`) on the `cugemm-profile.bin` binary. It generates a profiling report named `my-profile` and includes various profiling options, such as setting the problem size, number of repetitions, and algorithm to profile.

```makefile
profile: cugemm-profile.bin
	sudo /usr/local/cuda-11.8/bin/ncu --export my-profile --set full ./cugemm-profile.bin --size=1024 --reps=1 --algo=1 --validate=false
```

Sources: [gemm/Makefile:21-23]()

### 6. `clean`

This target removes all the generated binary files from the build directory.

```makefile
clean:
	rm -f cugemm*.bin
```

Sources: [gemm/Makefile:25-26]()

## Build Dependencies

The build process relies on the following dependencies:

- NVIDIA CUDA Toolkit (version 11.8 based on the `profile` target)
- CUDA Libraries (e.g., `cuBLAS` for BLAS operations)

Sources: [gemm/Makefile:8, 13, 18, 22]()

## Project Structure

The project appears to contain the following files:

- `cugemm.cu`: The main CUDA source file containing the implementation of matrix operations.
- `hw1.sln`, `hw2.sln`, `transpose.sln`: Visual Studio solution files, potentially related to previous homework assignments or projects.

Sources: [gemm/Makefile:5](), [deprecated/hw1/hw1.sln](), [deprecated/hw2/hw2.sln](), [deprecated/transpose/transpose.sln]()

## Conclusion

The "Local Development Setup" section covers the build process for the project, including various targets for compiling optimized, debug, and profiling binaries. It also includes instructions for running the profiling tool and cleaning the build directory. The build process relies on the NVIDIA CUDA Toolkit and CUDA Libraries, specifically `cuBLAS`. The project structure includes the main CUDA source file (`cugemm.cu`) and potentially some legacy solution files from previous assignments or projects.