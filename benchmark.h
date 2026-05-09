#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <time.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU Benchmarking macros
struct timespec start_cpu, end_cpu;

#define START_TIMER() clock_gettime(CLOCK_MONOTONIC, &start_cpu)

#define STOP_TIMER(name) \
    clock_gettime(CLOCK_MONOTONIC, &end_cpu); \
    double time_taken = (end_cpu.tv_sec - start_cpu.tv_sec) * 1e3 + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6; \
    printf("[%s] Time: %f ms\n", name, time_taken)

#ifdef __cplusplus
}
#endif

// GPU Benchmarking macros (only defined if compiling with NVCC)
#ifdef __CUDACC__
#define START_CUDA_TIMER(start_event, stop_event) \
    cudaEventCreate(&start_event); \
    cudaEventCreate(&stop_event); \
    cudaEventRecord(start_event)

#define STOP_CUDA_TIMER(start_event, stop_event, name) \
    cudaEventRecord(stop_event); \
    cudaEventSynchronize(stop_event); \
    float ms = 0.0f; \
    cudaEventElapsedTime(&ms, start_event, stop_event); \
    printf("[%s] Time: %f ms\n", name, ms); \
    cudaEventDestroy(start_event); \
    cudaEventDestroy(stop_event)

#define CUDA_CHECK_ERROR() \
    { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        } \
    }
#endif

#endif // BENCHMARK_H
