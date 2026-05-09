#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "benchmark.h"
#include "image_io.h"

#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif
#define HALO 1
#define SHARED_W (TILE_W + 2 * HALO)
#define SHARED_H (TILE_H + 2 * HALO)

__global__ void sobel_tiled_kernel(const float* in, float* out, int width, int height) {
    __shared__ float tile[SHARED_H][SHARED_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_W;
    int by = blockIdx.y * TILE_H;

    int x = bx + tx;
    int y = by + ty;

    // Load data into shared memory with halo elements
    // A simple strided loop approach allows threads to cooperatively load the (TILE_H+2)x(TILE_W+2) shared block.
    for (int i = ty; i < SHARED_H; i += TILE_H) {
        for (int j = tx; j < SHARED_W; j += TILE_W) {
            int px = bx + j - HALO;
            int py = by + i - HALO;
            
            // Clamp to border
            if (px < 0) px = 0;
            else if (px >= width) px = width - 1;
            
            if (py < 0) py = 0;
            else if (py >= height) py = height - 1;

            tile[i][j] = in[py * width + px];
        }
    }

    __syncthreads();

    // Compute the Sobel filter if within bounds
    if (x < width && y < height) {
        float Gx = 0.0f;
        float Gy = 0.0f;

        int kx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        int ky[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                // Read from shared memory
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

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    int width, height;
    float* h_in = read_image(input_file, &width, &height);
    if (!h_in) return 1;

    printf("Image Dimensions: %d x %d\n", width, height);

    int size = width * height * sizeof(float);
    float* h_out = (float*)malloc(size);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaEvent_t start, stop;

    // 1. Host-to-Device transfer
    START_CUDA_TIMER(start, stop);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    STOP_CUDA_TIMER(start, stop, "Host-to-Device Transfer");

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 2. Kernel execution
    START_CUDA_TIMER(start, stop);
    sobel_tiled_kernel<<<grid, block>>>(d_in, d_out, width, height);
    STOP_CUDA_TIMER(start, stop, "Kernel Execution");
    CUDA_CHECK_ERROR();

    // 3. Device-to-Host transfer
    START_CUDA_TIMER(start, stop);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    STOP_CUDA_TIMER(start, stop, "Device-to-Host Transfer");

    write_image(output_file, h_out, width, height);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
