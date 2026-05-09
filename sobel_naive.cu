#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "benchmark.h"
#include "image_io.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

__global__ void sobel_naive_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

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
                int px = x + j;
                int py = y + i;

                // Clamp
                if (px < 0) px = 0;
                if (px >= width) px = width - 1;
                if (py < 0) py = 0;
                if (py >= height) py = height - 1;

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

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 2. Kernel execution
    START_CUDA_TIMER(start, stop);
    sobel_naive_kernel<<<grid, block>>>(d_in, d_out, width, height);
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
