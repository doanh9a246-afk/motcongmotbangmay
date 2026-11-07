#!/usr/bin/env bash
# Chay benchmark: tuan tu, OpenMP (nhieu so luong), CUDA naive & tiled.
# Ket qua ghi vao results/results.csv, sau do ve do thi bang scripts/plot_results.py
#
#   bash scripts/benchmark.sh
#   NS="1024 4096 16384" THREADS="1 2 4 8" STEPS=5 bash scripts/benchmark.sh
#   CUDA_NS="16384 65536 131072" bash scripts/benchmark.sh   # N lon chi chay tren GPU
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-results/results.csv}
CPU=./bin/nbody_cpu
CUDA=./bin/nbody_cuda
NCPU=$(nproc 2>/dev/null || echo 4)
NS=${NS:-"1024 2048 4096 8192 16384"}
THREADS=${THREADS:-"1 2 4 8 $NCPU"}
STEPS=${STEPS:-5}
CUDA_NS=${CUDA_NS:-"32768 65536 131072"}
BLOCKS=${BLOCKS:-"64 128 256 512"}

mkdir -p "$(dirname "$OUT")"
echo "impl,mode,n,steps,par,time_s,ginter_per_s,gflops" > "$OUT"
res() { grep '^RESULT' | cut -d, -f2- >> "$OUT"; }

have_cuda=0
if [ -x "$CUDA" ] || [ -x "$CUDA.exe" ]; then have_cuda=1; fi

for n in $NS; do
    echo ">> N = $n"
    "$CPU" -n "$n" -s "$STEPS" -m serial -noenergy | res
    for t in $(echo $THREADS | tr ' ' '\n' | sort -n | uniq); do
        "$CPU" -n "$n" -s "$STEPS" -m omp -t "$t" -noenergy | res
    done
    if [ $have_cuda = 1 ]; then
        for k in naive tiled; do "$CUDA" -n "$n" -s "$STEPS" -k $k -noenergy | res; done
    fi
done

if [ $have_cuda = 1 ]; then
    for n in $CUDA_NS; do
        echo ">> (GPU) N = $n"
        for k in naive tiled; do "$CUDA" -n "$n" -s "$STEPS" -k $k -noenergy | res; done
    done
    # Anh huong cua kich thuoc block / tile
    nb=$(echo $CUDA_NS | awk '{print $NF}')
    for b in $BLOCKS; do "$CUDA" -n "$nb" -s "$STEPS" -k tiled -b "$b" -noenergy | res; done
fi

echo "Da ghi $OUT"
column -s, -t "$OUT" 2>/dev/null || cat "$OUT"
