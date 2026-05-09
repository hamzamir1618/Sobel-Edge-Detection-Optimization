#!/bin/bash

mkdir -p results

# 1. Compile initial sweep utility
echo "Compiling..."
make all

# 2. Generate dummy PGM images if they don't exist
SIZES="256 512 1024 2048 4096"
for size in $SIZES; do
  if [ ! -f "img_${size}.pgm" ]; then
    echo "Generating dummy image of size ${size}x${size}..."
    echo -e "P5\n${size} ${size}\n255" > img_${size}.pgm
    dd if=/dev/urandom bs=1 count=$((size*size)) >> img_${size}.pgm 2>/dev/null
  fi
done

# 3. Sweep for best block size
echo "Running block_sweep on 1024x1024 to find optimal block sizes..."
rm -f results/block_sweep_img_1024.csv
./block_sweep img_1024.pgm naive > /dev/null
./block_sweep img_1024.pgm tiled > /dev/null

# Extract from CSV using awk (skipping header)
# CSV Format: Kernel,BlockSize,H2D_Time(ms),Kernel_Time(ms),D2H_Time(ms),Total_Time(ms),Occupancy(%)
BEST_NAIVE=$(tail -n +2 results/block_sweep_img_1024.csv | grep "^naive" | sort -t',' -k6 -n | head -n 1 | awk -F',' '{print $2}' | cut -d'x' -f1)
BEST_TILED=$(tail -n +2 results/block_sweep_img_1024.csv | grep "^tiled" | sort -t',' -k6 -n | head -n 1 | awk -F',' '{print $2}' | cut -d'x' -f1)

if [ -z "$BEST_NAIVE" ]; then BEST_NAIVE=8; fi
if [ -z "$BEST_TILED" ]; then BEST_TILED=16; fi

echo "Optimal Naive Block Size: ${BEST_NAIVE}x${BEST_NAIVE}"
echo "Optimal Tiled Block Size: ${BEST_TILED}x${BEST_TILED}"

# 4. Recompile with optimal block sizes
echo "Recompiling algorithms with optimal block sizes..."
make clean > /dev/null
make NVCCFLAGS="-arch=sm_75 -O2 -DTILE_W=$BEST_TILED -DTILE_H=$BEST_TILED -DBLOCK_SIZE=$BEST_NAIVE" > /dev/null

# Number of GPU cores for efficiency calc
# Using 4352 (RTX 2080 Ti for Turing sm_75 architecture) as a safe baseline default
GPU_CORES=${GPU_CORES:-4352}

# 5. Run benchmarks
CSV="results/benchmark_summary.csv"
echo "Resolution,Serial_Time(ms),Naive_Total(ms),Tiled_Total(ms),Speedup_Naive,Speedup_Tiled,Efficiency_Naive,Efficiency_Tiled" > $CSV

echo -e "\nRunning Benchmarks (GPU Cores used for Efficiency: $GPU_CORES)..."
printf "%-12s | %-10s | %-10s | %-10s | %-10s | %-10s | %-15s | %-15s\n" "Resolution" "Serial(ms)" "Naive(ms)" "Tiled(ms)" "Spd_Tiled" "Eff_Tiled" "Naive_Valid" "Tiled_Valid"
echo "----------------------------------------------------------------------------------------------------------------"

for size in $SIZES; do
    IN="img_${size}.pgm"
    OUT_S="out_serial_${size}.pgm"
    OUT_N="out_naive_${size}.pgm"
    OUT_T="out_tiled_${size}.pgm"

    # Run serial
    S_OUT=$(./sobel_serial $IN $OUT_S)
    T_SERIAL=$(echo "$S_OUT" | grep "Time:" | awk '{print $3}')

    # Run naive
    N_OUT=$(./sobel_naive $IN $OUT_N)
    T_H2D=$(echo "$N_OUT" | grep "Host-to-Device" | awk '{print $3}')
    T_KER=$(echo "$N_OUT" | grep "Kernel Execution" | awk '{print $3}')
    T_D2H=$(echo "$N_OUT" | grep "Device-to-Host" | awk '{print $3}')
    T_NAIVE=$(echo "$T_H2D + $T_KER + $T_D2H" | bc)
    
    # Validate Naive
    VAL_N=$(./validate $OUT_S $OUT_N | grep "STATUS")

    # Run tiled
    T_OUT=$(./sobel_tiled $IN $OUT_T)
    T_H2D_T=$(echo "$T_OUT" | grep "Host-to-Device" | awk '{print $3}')
    T_KER_T=$(echo "$T_OUT" | grep "Kernel Execution" | awk '{print $3}')
    T_D2H_T=$(echo "$T_OUT" | grep "Device-to-Host" | awk '{print $3}')
    T_TILED=$(echo "$T_H2D_T + $T_KER_T + $T_D2H_T" | bc)
    
    # Validate Tiled
    VAL_T=$(./validate $OUT_S $OUT_T | grep "STATUS")

    # Computations (using awk for floating point math since some systems limit bc scale features)
    SPD_NAIVE=$(awk "BEGIN {print $T_SERIAL / $T_NAIVE}")
    SPD_TILED=$(awk "BEGIN {print $T_SERIAL / $T_TILED}")
    
    EFF_NAIVE=$(awk "BEGIN {print $SPD_NAIVE / $GPU_CORES}")
    EFF_TILED=$(awk "BEGIN {print $SPD_TILED / $GPU_CORES}")

    printf "%-12s | %-10.2f | %-10.2f | %-10.2f | %-10.2f | %-10.6f | %s | %s\n" "${size}x${size}" "$T_SERIAL" "$T_NAIVE" "$T_TILED" "$SPD_TILED" "$EFF_TILED" "${VAL_N}" "${VAL_T}"
    
    echo "${size}x${size},$T_SERIAL,$T_NAIVE,$T_TILED,$SPD_NAIVE,$SPD_TILED,$EFF_NAIVE,$EFF_TILED" >> $CSV
    
    # Cleanup dummy outputs to save space
    rm -f $OUT_S $OUT_N $OUT_T
done

echo -e "\nBenchmark complete. Results saved to $CSV"
