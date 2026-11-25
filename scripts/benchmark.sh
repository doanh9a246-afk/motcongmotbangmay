#!/usr/bin/env bash
# Chay benchmark: tuan tu, OpenMP (nhieu so luong), CUDA naive & tiled.
# Ket qua ghi vao results/results.csv (+ results/results.machine.txt ghi cau hinh may),
# sau do ve do thi bang scripts/plot_results.py
#
#   bash scripts/benchmark.sh
#   NS="1024 4096 16384" THREADS="1 2 4 8" STEPS=5 bash scripts/benchmark.sh
#
# Quy uoc cua nhom (xem PHAN_CONG.md): CPU do tren may nhieu loi that, GPU do tren Colab T4:
#   OUT=results/cpu_laptop.csv SKIP_GPU=1 bash scripts/benchmark.sh      # tren laptop/PC
#   OUT=results/gpu_t4.csv     SKIP_CPU=1 bash scripts/benchmark.sh      # tren Colab
#   python scripts/plot_results.py results/cpu_laptop.csv results/gpu_t4.csv
#
# Bien moi truong:
#   NS        N cho thi nghiem day du (tuan tu + OpenMP moi so luong [+ CUDA])
#   THREADS   cac so luong OpenMP
#   STEPS     so buoc moi lan chay
#   REPS      so lan lap moi cau hinh, lay lan nhanh nhat (giam nhieu do turbo/nhiet)
#   BIG_NS    N lon chi do tuan tu + OpenMP toi da (de so voi GPU o N lon), BIG_STEPS buoc
#   CUDA_NS   N lon chi chay tren GPU;  BLOCKS  cac kich thuoc block cho thi nghiem tile
#   SKIP_CPU / SKIP_GPU = 1 de bo qua phan CPU / GPU
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-results/results.csv}
CPU=./bin/nbody_cpu
CUDA=./bin/nbody_cuda
NCPU=$(nproc 2>/dev/null || echo 4)
NS=${NS:-"1024 2048 4096 8192 16384"}
THREADS=${THREADS:-"1 2 4 8 $NCPU"}
STEPS=${STEPS:-5}
REPS=${REPS:-3}
BIG_NS=${BIG_NS:-"32768 65536 131072"}
BIG_STEPS=${BIG_STEPS:-1}
CUDA_NS=${CUDA_NS:-"32768 65536 131072"}
BLOCKS=${BLOCKS:-"64 128 256 512"}
SKIP_CPU=${SKIP_CPU:-0}
SKIP_GPU=${SKIP_GPU:-0}

have_cpu=0; have_cuda=0
if [ "$SKIP_CPU" != 1 ] && { [ -x "$CPU" ] || [ -x "$CPU.exe" ]; }; then have_cpu=1; fi
if [ "$SKIP_GPU" != 1 ] && { [ -x "$CUDA" ] || [ -x "$CUDA.exe" ]; }; then have_cuda=1; fi
if [ $have_cpu = 0 ] && [ $have_cuda = 0 ]; then
    echo "Khong co chuong trinh nao de chay (chua build? hay SKIP_CPU/SKIP_GPU?)" >&2; exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "impl,mode,n,steps,par,time_s,ginter_per_s,gflops" > "$OUT"

# Chay mot cau hinh REPS lan, ghi lai dong RESULT co thoi gian nho nhat
run() {
    local best="" line t bt
    for _ in $(seq "$REPS"); do
        line=$("$@" | grep '^RESULT' | cut -d, -f2-)
        t=$(echo "$line" | cut -d, -f6)
        if [ -z "$best" ] || awk -v a="$t" -v b="$bt" 'BEGIN{exit !(a<b)}'; then best=$line; bt=$t; fi
    done
    echo "$best" >> "$OUT"
    echo "   $best"
}

# ---- Ghi cau hinh may canh file ket qua ----
MACHINE=${OUT%.csv}.machine.txt
{
    echo "Ngay do   : $(date '+%Y-%m-%d %H:%M')"
    echo "He dieu hanh: $(uname -sr)"
    # Windows (Git Bash/MSYS): /proc/cpuinfo bao sai so loi voi CPU lai P/E -> hoi Windows
    if command -v powershell >/dev/null 2>&1; then
        echo "CPU       : $(powershell -NoProfile -c '(Get-CimInstance Win32_Processor).Name' | tr -d '\r')"
        echo "So loi    : $(powershell -NoProfile -c '(Get-CimInstance Win32_Processor).NumberOfCores' | tr -d '\r')"
    elif [ -r /proc/cpuinfo ]; then
        echo "CPU       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
        echo "So loi    : $(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{if(c&&s)print c*s}')"
    fi
    echo "So luong  : $NCPU"
    echo "Compiler  : $(${CC:-gcc} --version 2>/dev/null | head -1)"
    if [ $have_cuda = 1 ]; then
        command -v nvidia-smi >/dev/null && echo "GPU       : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
        command -v nvcc >/dev/null && echo "nvcc      : $(nvcc --version | tail -1)"
    fi
    echo "Tham so   : NS=\"$NS\" THREADS=\"$THREADS\" STEPS=$STEPS REPS=$REPS BIG_NS=\"$BIG_NS\" BIG_STEPS=$BIG_STEPS"
    [ $have_cuda = 1 ] && echo "            CUDA_NS=\"$CUDA_NS\" BLOCKS=\"$BLOCKS\""
} > "$MACHINE"
cat "$MACHINE"

THREADS_SORTED=$(echo $THREADS | tr ' ' '\n' | sort -n | uniq)
TMAX=$(echo "$THREADS_SORTED" | tail -1)

for n in $NS; do
    echo ">> N = $n"
    if [ $have_cpu = 1 ]; then
        run "$CPU" -n "$n" -s "$STEPS" -m serial -noenergy
        for t in $THREADS_SORTED; do run "$CPU" -n "$n" -s "$STEPS" -m omp -t "$t" -noenergy; done
    fi
    if [ $have_cuda = 1 ]; then
        for k in naive tiled; do run "$CUDA" -n "$n" -s "$STEPS" -k $k -noenergy; done
    fi
done

if [ $have_cpu = 1 ]; then
    for n in $BIG_NS; do
        echo ">> (CPU, N lon) N = $n"
        run "$CPU" -n "$n" -s "$BIG_STEPS" -m serial -noenergy
        run "$CPU" -n "$n" -s "$BIG_STEPS" -m omp -t "$TMAX" -noenergy
    done
fi

if [ $have_cuda = 1 ]; then
    for n in $CUDA_NS; do
        echo ">> (GPU) N = $n"
        for k in naive tiled; do run "$CUDA" -n "$n" -s "$STEPS" -k $k -noenergy; done
    done
    # Anh huong cua kich thuoc block / tile (bo qua 256 vi da do o tren)
    nb=$(echo $CUDA_NS | awk '{print $NF}')
    echo ">> (GPU) block size, N = $nb"
    for b in $BLOCKS; do
        [ "$b" = 256 ] && continue
        run "$CUDA" -n "$nb" -s "$STEPS" -k tiled -b "$b" -noenergy
    done
fi

echo "Da ghi $OUT va $MACHINE"
column -s, -t "$OUT" 2>/dev/null || cat "$OUT"
