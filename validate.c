#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "image_io.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <reference_image> <target_image>\n", argv[0]);
        return 1;
    }

    const char* ref_file = argv[1];
    const char* target_file = argv[2];

    int w1, h1, w2, h2;
    float* ref = read_image(ref_file, &w1, &h1);
    if (!ref) {
        printf("FAIL: Could not read reference image %s\n", ref_file);
        return 1;
    }

    float* tgt = read_image(target_file, &w2, &h2);
    if (!tgt) {
        printf("FAIL: Could not read target image %s\n", target_file);
        free(ref);
        return 1;
    }

    if (w1 != w2 || h1 != h2) {
        printf("FAIL: Dimensions do not match (%dx%d vs %dx%d)\n", w1, h1, w2, h2);
        free(ref);
        free(tgt);
        return 1;
    }

    int n = w1 * h1;
    float max_err = 0.0f;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - tgt[i]);
        if (diff > max_err) {
            max_err = diff;
        }
        sum_err += diff;
        sum_sq_err += (double)(diff * diff);
    }

    double mae = sum_err / n;
    double mse = sum_sq_err / n;
    double psnr = 0.0;
    
    if (mse > 0.0) {
        psnr = 10.0 * log10((255.0 * 255.0) / mse);
    } else {
        psnr = 999.9; // Arbitrary high value for exact match
    }

    printf("Validation Results (%s vs %s):\n", ref_file, target_file);
    printf("Max Error: %.4f\n", max_err);
    printf("Mean Absolute Error: %.4f\n", mae);
    
    if (mse > 0.0) {
        printf("PSNR: %.2f dB\n", psnr);
    } else {
        printf("PSNR: Infinity (Exact Match)\n");
    }

    free(ref);
    free(tgt);

    if (max_err <= 1.0f) {
        printf("STATUS: PASS\n");
        return 0;
    } else {
        printf("STATUS: FAIL (max error %.4f > 1.0)\n", max_err);
        return 1;
    }
}
