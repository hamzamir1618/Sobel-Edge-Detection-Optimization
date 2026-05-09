CC = gcc
NVCC = nvcc

CFLAGS = -O2 -Wall
NVCCFLAGS = -arch=sm_75 -O2

LIBS = -ltiff -lm

TARGETS = sobel_serial sobel_naive sobel_tiled block_sweep validate

all: prep $(TARGETS)

serial: sobel_serial
naive: sobel_naive
tiled: sobel_tiled
sweep: block_sweep

prep:
	mkdir -p results

image_io.o: image_io.c image_io.h
	$(CC) $(CFLAGS) -c image_io.c -o image_io.o

sobel_serial: sobel_serial.c image_io.o benchmark.h
	$(CC) $(CFLAGS) sobel_serial.c image_io.o -o sobel_serial $(LIBS)

sobel_naive: sobel_naive.cu image_io.o benchmark.h
	$(NVCC) $(NVCCFLAGS) sobel_naive.cu image_io.o -o sobel_naive $(LIBS)

sobel_tiled: sobel_tiled.cu image_io.o benchmark.h
	$(NVCC) $(NVCCFLAGS) sobel_tiled.cu image_io.o -o sobel_tiled $(LIBS)

block_sweep: block_sweep.cu image_io.o benchmark.h
	$(NVCC) $(NVCCFLAGS) block_sweep.cu image_io.o -o block_sweep $(LIBS)

validate: validate.c image_io.o
	$(CC) $(CFLAGS) validate.c image_io.o -o validate $(LIBS)

clean:
	rm -f *.o $(TARGETS)
