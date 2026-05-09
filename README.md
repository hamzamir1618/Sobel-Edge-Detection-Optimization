# CUDA Sobel Edge Detection

## Project Overview
This project explores the performance characteristics of accelerating the Sobel Edge Detection filter using GPU parallelization. It compares a standard CPU serial implementation against two distinct GPU CUDA implementations:
1. **Naive CUDA:** Maps one thread per pixel, fetching surrounding convolution data directly from global memory.
2. **Shared-Memory Tiled CUDA:** Threads cooperatively load `(TILE_W + 2) x (TILE_H + 2)` pixel blocks (including boundary halos) into ultra-fast L1 `__shared__` memory, eliminating redundant global fetches during the convolution pass.

**Citation:** Based on the architectural principles outlined in *"Implementation of Sobel Filter using CUDA"* by Akshita Akasapu et al.

**Authors:**
- M. Basim Kafeel (22i-0775)
- Hamza Ahmad Mir (22i-1003)

## System Requirements
- **OS:** Ubuntu 20.04+ or WSL2 on Windows
- **CUDA Toolkit:** 11.0+ (with `nvcc` compiler)
- **Compiler:** GCC 9+
- **Libraries:** `libtiff-dev`
- **Hardware:** NVIDIA GPU with compute capability 7.5+ (e.g., Turing RTX series)
- **VRAM:** At least 4GB recommended for 4K (`4096x4096`) image benchmarks

## Installation & Setup

### 1. Installing CUDA Toolkit (Ubuntu)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda
```
Add `nvcc` to your environment path by appending this to your `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 2. Installing libtiff
```bash
sudo apt-get install libtiff-dev
```

### 3. Verifying the Install
Confirm your compiler and GPU driver are active:
```bash
nvcc --version
nvidia-smi
```

### 4. Clone and Navigate
```bash
git clone <your_repo_url>
cd sobel-cuda-project
```

## Building the Project
Use the provided `Makefile` to compile the targets:
- `make all`         — builds everything
- `make serial`      — builds CPU baseline only (`sobel_serial`)
- `make naive`       — builds naive CUDA kernel only (`sobel_naive`)
- `make tiled`       — builds tiled shared-memory kernel only (`sobel_tiled`)
- `make sweep`       — builds block size sweep utility (`block_sweep`)
- `make clean`       — removes all build artifacts

## Running Each Binary

### 1. `sobel_serial`
Executes the standard CPU convolution loop.
**Usage:** `./sobel_serial <input_image> <output_image>`
**Example:** `./sobel_serial images/lena_1024.pgm output/lena_serial.pgm`

### 2. `sobel_naive`
Executes the naive CUDA kernel. Prints accurate timings for Host-to-Device transfer, Kernel Execution, and Device-to-Host transfer separately to isolate PCI-e bottleneck overhead.
**Usage:** `./sobel_naive <input_image> <output_image>`
**Example:** `./sobel_naive images/lena_1024.pgm output/lena_naive.pgm`

### 3. `sobel_tiled`
Executes the optimized shared-memory kernel. 
*Note: Because `__shared__` memory sizes must be known at compile time, tile dimensions are strictly parameterized via compiler macros (`TILE_W` and `TILE_H`) rather than runtime CLI arguments.*
**Usage:** `./sobel_tiled <input_image> <output_image>`
**Example:** `./sobel_tiled images/lena_1024.pgm output/lena_tiled.pgm`
*(To test custom sizes manually, compile via: `make NVCCFLAGS="-arch=sm_75 -O2 -DTILE_W=32 -DTILE_H=32"`)*

### 4. `block_sweep`
Utility to benchmark theoretical and actual occupancy across 8x8, 16x16, and 32x32 block configurations.
**Usage:** `./block_sweep <input_image> <naive|tiled>`
**Example:** `./block_sweep images/lena_1024.pgm tiled`
**Output:** Automatically appends metrics to `results/block_sweep_lena_1024.csv`

## Validating Correctness
We use `validate.c` to cross-reference GPU outputs against the CPU baseline to verify absolute accuracy.
**Command:**
```bash
./validate output/lena_serial.pgm output/lena_naive.pgm
./validate output/lena_serial.pgm output/lena_tiled.pgm
```
**Expected Output:** 
The script will pass if the Maximum Pixel Error $\le 1.0$. This tolerance accommodates standard `float32` vs 8-bit quantization rounding differences. Because the GPU and CPU math is nearly identical, you can expect the **PSNR to be Infinity (Exact Match)** or at least $> 100 \text{ dB}$.

## Running the Full Benchmark
Run the automated benchmarking shell script:
```bash
chmod +x benchmark.sh
./benchmark.sh
```
**Step-by-step Execution:**
1. Dynamically generates synthetic PGM test images at `256²`, `512²`, `1024²`, `2048²`, and `4096²` if they don't already exist.
2. Runs the `block_sweep` utility on `1024²` to dynamically calculate the optimal block topology.
3. Automatically triggers `make clean` and recompiles `sobel_naive` and `sobel_tiled` using the optimal parameters discovered in step 2.
4. Sweeps through all resolutions, collecting CPU times and GPU Total times (including PCI-e overhead).
5. Cross-validates all outputs cleanly.
6. Saves aggregated Speedup and Efficiency calculations to `results/benchmark_summary.csv`.

**Expected runtime:** ~10-15 minutes for the full suite depending on GPU class.

## Generating Test Images
If you want to create real-world custom image sizes instead of relying on the shell script's byte noise, you can use this Python Pillow one-liner:
```bash
python3 -c "from PIL import Image; import numpy as np; Image.fromarray(np.random.randint(0,255,(1024,1024),dtype=np.uint8)).save('images/test_1024.pgm')"
```

## Results Table
*(Placeholder values to be populated post-benchmark execution)*

| Resolution | Serial (ms) | Naive GPU (ms) | Tiled GPU (ms) | Speedup (Naive) | Speedup (Tiled) | Efficiency |
|------------|-------------|----------------|----------------|-----------------|-----------------|------------|
| 256×256    | TBD         | TBD            | TBD            | TBD             | TBD             | TBD        |
| 512×512    | TBD         | TBD            | TBD            | TBD             | TBD             | TBD        |
| 1024×1024  | TBD         | TBD            | TBD            | TBD             | TBD             | TBD        |
| 2048×2048  | TBD         | TBD            | TBD            | TBD             | TBD             | TBD        |
| 4096×4096  | TBD         | TBD            | TBD            | TBD             | TBD             | TBD        |

## Troubleshooting

- **`nvcc: command not found`** 
  *Cause:* The CUDA binaries are not in your system path.
  *Fix:* Run `export PATH=/usr/local/cuda/bin:$PATH`.
- **`libcuda.so not found`** 
  *Cause:* The CUDA runtime bindings are misconfigured in the linker cache.
  *Fix:* Run `sudo ldconfig /usr/local/cuda/lib64`.
- **`libtiff.so not found`** 
  *Cause:* Missing standard TIFF libraries.
  *Fix:* Run `sudo apt-get install libtiff-dev` followed by `sudo ldconfig`.
- **Out of memory on GPU** 
  *Cause:* Attempting to allocate `4096x4096` floats exceeds your GPU's VRAM capacity.
  *Fix:* Edit `benchmark.sh` and remove `4096` from the `SIZES` variable string to cap the test footprint at 2K.
- **Mismatched output vs serial** 
  *Cause:* Floating-point convolution rounding variations across `fma` architectures. 
  *Fix:* Ensure you are relying on `validate.c` which features an epsilon/rounding tolerance `max_error <= 1.0` to handle quantization drops mathematically.

## Project Structure
```text
.
├── Makefile              # Centralized build and compilation rules
├── benchmark.h           # High-resolution CPU (clock_gettime) and GPU (cudaEvent) timing macros
├── benchmark.sh          # Full automated suite triggering generation, sweeps, and CSV logging
├── block_sweep.cu        # Utility using templates to natively benchmark 8x8, 16x16, 32x32 topologies
├── image_io.c            # Implementation for reading/writing strict [0, 255] float arrays to PGM/TIFF
├── image_io.h            # IO headers interfacing with libtiff
├── sobel_naive.cu        # CUDA Kernel: Global memory direct fetching baseline
├── sobel_serial.c        # CPU Kernel: O(N^2) baseline relying on strict nested loops
├── sobel_tiled.cu        # CUDA Kernel: Shared memory overlapping tile optimization
└── validate.c            # Verification suite computing PSNR and absolute bounds errors
```
