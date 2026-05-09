#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "benchmark.h"
#include "image_io.h"

void compute_sobel_cpu(const float* in, float* out, int width, int height) {
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sumX = 0.0f;
            float sumY = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = x + kx;
                    int py = y + ky;

                    // Clamp to edge
                    if (px < 0) px = 0;
                    if (px >= width) px = width - 1;
                    if (py < 0) py = 0;
                    if (py >= height) py = height - 1;

                    float val = in[py * width + px];
                    sumX += val * Gx[ky + 1][kx + 1];
                    sumY += val * Gy[ky + 1][kx + 1];
                }
            }
            float mag = sqrtf(sumX * sumX + sumY * sumY);
            if (mag > 255.0f) mag = 255.0f;
            out[y * width + x] = mag;
        }
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

    float* h_out = (float*)malloc(width * height * sizeof(float));

    START_TIMER();
    compute_sobel_cpu(h_in, h_out, width, height);
    STOP_TIMER("Sobel CPU");

    write_image(output_file, h_out, width, height);

    free(h_in);
    free(h_out);
    return 0;
}
