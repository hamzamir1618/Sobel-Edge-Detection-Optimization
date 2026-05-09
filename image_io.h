#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#ifdef __cplusplus
extern "C" {
#endif

// Reads an image (PGM or TIFF based on extension) and returns a flat float array.
// Allocates memory that should be freed by the caller.
// pixel values are normalized to [0.0, 1.0].
float* read_image(const char* filename, int* width, int* height);

// Writes a flat float array to an image (PGM or TIFF based on extension).
// Data should be in range [0.0, 1.0].
void write_image(const char* filename, const float* data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_IO_H
