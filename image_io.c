#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

// Helper to check extension
static int ends_with(const char* str, const char* suffix) {
    if (!str || !suffix) return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr) return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

float* read_pgm(const char* filename, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) return NULL;
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Only P5 PGM format is supported.\n");
        fclose(fp);
        return NULL;
    }
    
    // Skip comments
    int c = getc(fp);
    while (c == '#' || c == '\n' || c == '\r' || c == ' ') {
        if (c == '#') {
            while (getc(fp) != '\n' && !feof(fp));
        }
        c = getc(fp);
    }
    ungetc(c, fp);
    
    int maxval;
    if (fscanf(fp, "%d %d %d", width, height, &maxval) != 3) {
        fclose(fp);
        return NULL;
    }
    fgetc(fp); // skip single whitespace/newline character after maxval
    
    int num_pixels = (*width) * (*height);
    unsigned char* temp = (unsigned char*)malloc(num_pixels);
    fread(temp, 1, num_pixels, fp);
    fclose(fp);
    
    float* data = (float*)malloc(num_pixels * sizeof(float));
    for (int i = 0; i < num_pixels; i++) {
        data[i] = (float)temp[i] * 255.0f / maxval;
    }
    free(temp);
    return data;
}

void write_pgm(const char* filename, const float* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    int num_pixels = width * height;
    unsigned char* temp = (unsigned char*)malloc(num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        float val = data[i];
        if (val > 255.0f) val = 255.0f;
        if (val < 0.0f) val = 0.0f;
        temp[i] = (unsigned char)val;
    }
    fwrite(temp, 1, num_pixels, fp);
    free(temp);
    fclose(fp);
}

float* read_tiff(const char* filename, int* width, int* height) {
    TIFF* tif = TIFFOpen(filename, "r");
    if (!tif) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }
    
    uint32_t w, h;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    *width = (int)w;
    *height = (int)h;
    
    uint16_t bpp, spp;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    
    if (bpp != 8 || spp != 1) {
        fprintf(stderr, "Error: Only 8-bit grayscale TIFF is supported (bpp=%d, spp=%d).\n", bpp, spp);
        TIFFClose(tif);
        return NULL;
    }
    
    int num_pixels = w * h;
    float* data = (float*)malloc(num_pixels * sizeof(float));
    unsigned char* buf = (unsigned char*)_TIFFmalloc(TIFFScanlineSize(tif));
    
    for (uint32_t row = 0; row < h; row++) {
        TIFFReadScanline(tif, buf, row, 0);
        for (uint32_t col = 0; col < w; col++) {
            data[row * w + col] = (float)buf[col];
        }
    }
    
    _TIFFfree(buf);
    TIFFClose(tif);
    return data;
}

void write_tiff(const char* filename, const float* data, int width, int height) {
    TIFF* tif = TIFFOpen(filename, "w");
    if (!tif) return;
    
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    
    unsigned char* buf = (unsigned char*)_TIFFmalloc(TIFFScanlineSize(tif));
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float val = data[row * width + col];
            if (val > 255.0f) val = 255.0f;
            if (val < 0.0f) val = 0.0f;
            buf[col] = (unsigned char)val;
        }
        TIFFWriteScanline(tif, buf, row, 0);
    }
    
    _TIFFfree(buf);
    TIFFClose(tif);
}

float* read_image(const char* filename, int* width, int* height) {
    if (ends_with(filename, ".pgm") || ends_with(filename, ".PGM")) {
        return read_pgm(filename, width, height);
    } else if (ends_with(filename, ".tiff") || ends_with(filename, ".tif") || ends_with(filename, ".TIFF") || ends_with(filename, ".TIF")) {
        return read_tiff(filename, width, height);
    }
    fprintf(stderr, "Unsupported file format for reading: %s\n", filename);
    return NULL;
}

void write_image(const char* filename, const float* data, int width, int height) {
    if (ends_with(filename, ".pgm") || ends_with(filename, ".PGM")) {
        write_pgm(filename, data, width, height);
    } else if (ends_with(filename, ".tiff") || ends_with(filename, ".tif") || ends_with(filename, ".TIFF") || ends_with(filename, ".TIF")) {
        write_tiff(filename, data, width, height);
    } else {
        fprintf(stderr, "Unsupported file format for writing: %s\n", filename);
    }
}
