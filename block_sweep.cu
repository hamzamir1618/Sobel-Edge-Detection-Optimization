#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "benchmark.h"
#include "image_io.h"

__global__ void sweep_naive_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float Gx = 0.0f;
        float Gy = 0.0f;
        int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int ky[3][3] = {{-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1}};

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int px = x + j;
                int py = y + i;
                if (px < 0) px = 0; else if (px >= width) px = width - 1;
                if (py < 0) py = 0; else if (py >= height) py = height - 1;

                float val = in[py * width + px];
                Gx += val * kx[i + 1][j + 1];
                Gy += val * ky[i + 1][j + 1];
            }
        }
        float mag = sqrtf(Gx * Gx + Gy * Gy);
        if (mag > 255.0f) mag = 255.0f;
        out[y * width + x] = mag;
    }
}

template<int TILE_W, int TILE_H>
__global__ void sweep_tiled_kernel(const float* in, float* out, int width, int height) {
    const int HALO = 1;
    const int SHARED_W = TILE_W + 2 * HALO;
    const int SHARED_H = TILE_H + 2 * HALO;
    __shared__ float tile[SHARED_H][SHARED_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_W;
    int by = blockIdx.y * TILE_H;
    int x = bx + tx;
    int y = by + ty;

    for (int i = ty; i < SHARED_H; i += TILE_H) {
        for (int j = tx; j < SHARED_W; j += TILE_W) {
            int px = bx + j - HALO;
            int py = by + i - HALO;
            if (px < 0) px = 0; else if (px >= width) px = width - 1;
            if (py < 0) py = 0; else if (py >= height) py = height - 1;
            tile[i][j] = in[py * width + px];
        }
    }
    __syncthreads();

    if (x < width && y < height) {
        float Gx = 0.0f;
        float Gy = 0.0f;
        int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int ky[3][3] = {{-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1}};

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                float val = tile[ty + HALO + i][tx + HALO + j];
                Gx += val * kx[i + 1][j + 1];
                Gy += val * ky[i + 1][j + 1];
            }
        }
        float mag = sqrtf(Gx * Gx + Gy * Gy);
        if (mag > 255.0f) mag = 255.0f;
        out[y * width + x] = mag;
    }
}

void extract_basename(const char* path, char* basename) {
    const char* last_slash = strrchr(path, '/');
    const char* start = last_slash ? last_slash + 1 : path;
    const char* last_dot = strrchr(start, '.');
    if (last_dot) {
        strncpy(basename, start, last_dot - start);
        basename[last_dot - start] = '\0';
    } else {
        strcpy(basename, start);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <naive|tiled>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const char* kernel_type = argv[2];
    
    int is_naive = (strcmp(kernel_type, "naive") == 0);
    int is_tiled = (strcmp(kernel_type, "tiled") == 0);
    
    if (!is_naive && !is_tiled) {
        printf("Invalid kernel type. Use 'naive' or 'tiled'.\n");
        return 1;
    }

    char basename[256];
    extract_basename(input_file, basename);

    // system("mkdir -p results"); // Usually done by Makefile or script, but safe to assume it exists

    char csv_filename[512];
    snprintf(csv_filename, sizeof(csv_filename), "results/block_sweep_%s.csv", basename);

    FILE* csv = fopen(csv_filename, "a");
    if (!csv) {
        printf("Could not open %s for writing. Creating results dir...\n", csv_filename);
        system("mkdir -p results");
        csv = fopen(csv_filename, "a");
        if (!csv) return 1;
    }

    int width, height;
    float* h_in = read_image(input_file, &width, &height);
    if (!h_in) return 1;

    int size = width * height * sizeof(float);
    float* h_out = (float*)malloc(size);
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    int block_sizes[] = {8, 16, 32};
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) {
        fprintf(csv, "Kernel,BlockSize,H2D_Time(ms),Kernel_Time(ms),D2H_Time(ms),Total_Time(ms),Occupancy(%%)\n");
    }

    cudaEvent_t start, stop, k_start, k_stop, d2h_start, d2h_stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&k_start); cudaEventCreate(&k_stop);
    cudaEventCreate(&d2h_start); cudaEventCreate(&d2h_stop);

    for (int i = 0; i < 3; i++) {
        int bs = block_sizes[i];
        dim3 block(bs, bs);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        float h2d_time=0, kernel_time=0, d2h_time=0;
        
        cudaEventRecord(start);
        cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&h2d_time, start, stop);

        int numBlocks = 0;
        float occupancy = 0.0f;

        cudaEventRecord(k_start);
        if (is_naive) {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, sweep_naive_kernel, bs * bs, 0);
            occupancy = (numBlocks * bs * bs * 100.0f) / prop.maxThreadsPerMultiProcessor;
            sweep_naive_kernel<<<grid, block>>>(d_in, d_out, width, height);
        } else {
            if (bs == 8) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, sweep_tiled_kernel<8, 8>, bs * bs, 0);
                occupancy = (numBlocks * bs * bs * 100.0f) / prop.maxThreadsPerMultiProcessor;
                sweep_tiled_kernel<8, 8><<<grid, block>>>(d_in, d_out, width, height);
            } else if (bs == 16) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, sweep_tiled_kernel<16, 16>, bs * bs, 0);
                occupancy = (numBlocks * bs * bs * 100.0f) / prop.maxThreadsPerMultiProcessor;
                sweep_tiled_kernel<16, 16><<<grid, block>>>(d_in, d_out, width, height);
            } else if (bs == 32) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, sweep_tiled_kernel<32, 32>, bs * bs, 0);
                occupancy = (numBlocks * bs * bs * 100.0f) / prop.maxThreadsPerMultiProcessor;
                sweep_tiled_kernel<32, 32><<<grid, block>>>(d_in, d_out, width, height);
            }
        }
        cudaEventRecord(k_stop);
        cudaEventSynchronize(k_stop);
        cudaEventElapsedTime(&kernel_time, k_start, k_stop);
        CUDA_CHECK_ERROR();

        cudaEventRecord(d2h_start);
        cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(d2h_stop);
        cudaEventSynchronize(d2h_stop);
        cudaEventElapsedTime(&d2h_time, d2h_start, d2h_stop);

        float total_time = h2d_time + kernel_time + d2h_time;

        printf("Kernel: %s, Block: %dx%d, H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms, Occupancy: %.1f%%\n",
               kernel_type, bs, bs, h2d_time, kernel_time, d2h_time, total_time, occupancy);
               
        fprintf(csv, "%s,%dx%d,%.3f,%.3f,%.3f,%.3f,%.1f\n",
                kernel_type, bs, bs, h2d_time, kernel_time, d2h_time, total_time, occupancy);
    }

    fclose(csv);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
