# Song song hóa mô phỏng N-body (lực hấp dẫn) bằng OpenMP và CUDA

**Nhóm:** Nguyễn Quốc Doanh (nhóm trưởng) · Vũ Văn Dương · Nguyễn Tùng Dương · Trần Tiến Đạt
**Phân công:** xem [PHAN_CONG.md](PHAN_CONG.md)

Tài liệu tham khảo:
- L. Nyland, M. Harris, J. Prins — *Fast N-Body Simulation with CUDA*, GPU Gems 3, Chương 31.
  https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
- OpenMP Specifications — https://www.openmp.org/specifications/

## 1. Bài toán

N vật thể hút nhau theo định luật vạn vật hấp dẫn. Gia tốc của vật thể *i*:

```
a_i = G · Σ_j  m_j · r_ij / (|r_ij|² + ε²)^(3/2)
```

- Thuật toán all-pairs: mỗi bước tính N² tương tác → độ phức tạp **O(N²)**.
- **ε (softening)** tránh lực vô cùng khi hai vật quá gần, đồng thời cho phép bỏ điều kiện `j ≠ i` (khi `r = 0` đóng góp bằng 0) → vòng lặp không rẽ nhánh, dễ vector hóa / chạy trên GPU.
- Tích phân: Euler đối xứng (symplectic Euler) `v += a·dt; p += v·dt`.
- Khởi tạo: cụm sao **Plummer** trong đơn vị N-body (G = 1, tổng khối lượng = 1), dùng bộ sinh số ngẫu nhiên tất định nên mọi phiên bản bắt đầu từ cùng một trạng thái.
- Kiểm tra đúng đắn: (1) so sánh vị trí giữa các phiên bản (`-verify`), (2) độ lệch năng lượng toàn phần `|ΔE/E₀|`.

## 2. Các phiên bản

| Phiên bản | File | Ý tưởng song song |
|---|---|---|
| Tuần tự | `src/nbody_cpu_kernels.h` (`nb_acc_serial`) | Hai vòng lặp lồng nhau, 1 luồng |
| OpenMP | `src/nbody_cpu_kernels.h` (`nb_acc_omp`) | `#pragma omp parallel for schedule(static)` trên vòng *i*; vòng *j* dùng `#pragma omp simd reduction`. Mỗi luồng chỉ ghi `acc[i]` của mình → không có race condition, không cần `critical`/`atomic`. |
| CUDA naive | `src/nbody_cuda.cu` (`accNaive`) | 1 thread = 1 vật thể, đọc N vị trí trực tiếp từ global memory |
| CUDA tiled | `src/nbody_cuda.cu` (`accTiled`) | Thuật toán GPU Gems 3: chia thành tile kích thước p = blockDim; mỗi block nạp tile vào **shared memory**, `__syncthreads()`, rồi mỗi thread tính p tương tác → giảm số lần đọc global memory p lần |

Hiệu năng tính theo quy ước của GPU Gems 3: **20 FLOP / tương tác**.

## 3. Biên dịch

Yêu cầu: `gcc` có OpenMP; CUDA Toolkit (`nvcc`) cho bản GPU.

```bash
make            # bin/nbody_cpu   (tuần tự + OpenMP)
make cuda       # bin/nbody_cuda  (cần GPU NVIDIA + nvcc)
make ARCH=sm_75 cuda   # nếu nvcc cũ không hỗ trợ -arch=native (T4 = sm_75)
```

Không có `make` (Windows/MinGW):

```bash
gcc -O3 -march=native -ffast-math -fopenmp src/nbody_cpu.c -o bin/nbody_cpu.exe -lm
```

**Không có GPU NVIDIA?** Mở [notebooks/nbody_colab.ipynb](notebooks/nbody_colab.ipynb) trên Google Colab (Runtime → GPU T4) — notebook clone repo, biên dịch, kiểm tra, benchmark và vẽ đồ thị.

## 4. Chạy

```bash
./bin/nbody_cpu  -n 8192 -s 10 -m serial          # tuần tự
./bin/nbody_cpu  -n 8192 -s 10 -m omp -t 8        # OpenMP 8 luồng
./bin/nbody_cpu  -n 2048 -s 5 -verify             # kiểm tra OpenMP == tuần tự
./bin/nbody_cuda -n 65536 -s 10 -k tiled -b 256   # CUDA tiled
./bin/nbody_cuda -n 4096 -s 5 -k naive -verify    # kiểm tra CUDA so với CPU
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `-n` | số vật thể | 4096 |
| `-s` | số bước | 10 |
| `-dt`, `-eps` | bước thời gian, softening | 0.001, 0.01 |
| `-m` (CPU) | `serial` \| `omp` | `omp` |
| `-t` (CPU) | số luồng OpenMP | tất cả lõi |
| `-k` (CUDA) | `naive` \| `tiled` | `tiled` |
| `-b` (CUDA) | threads/block = kích thước tile | 256 |
| `-verify` | so sánh với bản tham chiếu | tắt |
| `-o file.csv -every k` | ghi quỹ đạo mỗi k bước | tắt |
| `-noenergy` | bỏ tính năng lượng (O(N²) trên CPU) | tắt |

Mỗi lần chạy in một dòng `RESULT,impl,mode,n,steps,threads|block,time_s,Ginteractions/s,GFLOP/s` để script đọc.

## 5. Benchmark & đồ thị

```bash
bash scripts/benchmark.sh          # -> results/results.csv
python scripts/plot_results.py     # -> results/*.png + bảng Markdown cho báo cáo
```

Có thể chỉnh qua biến môi trường: `NS`, `THREADS`, `STEPS`, `CUDA_NS`, `BLOCKS`.

Trực quan hóa:

```bash
./bin/nbody_cpu -n 1024 -s 2000 -dt 0.002 -o results/traj.csv -every 20 -noenergy
python scripts/visualize.py results/traj.csv --gif
```

## 6. Kết quả thử nghiệm sơ bộ (CPU)

Laptop 16 luồng, MinGW gcc 12.1, 5 bước. Số liệu laptop dao động mạnh (turbo, nhân P/E) — **chỉ để tham khảo**; số liệu chính thức cho báo cáo sẽ đo lại theo [PHAN_CONG.md](PHAN_CONG.md).

| N | Tuần tự (ms/bước) | OpenMP 16 luồng (ms/bước) | Speedup |
|---|---|---|---|
| 4096 | 25.96 | 7.86 | 3.3× |
| 8192 | 77.46 | 15.31 | 5.1× |

Kiểm tra: `-verify` cho sai số 0 (OpenMP khớp tuần tự từng bit); `|ΔE/E₀| ≈ 1e-6` sau 10 bước, `≈ 3e-5` sau 1000 bước (N = 2048).

## 7. Lưu ý kỹ thuật

- Trên MinGW gcc 12.1 (bản Brecht Sanders), libgomp làm chương trình thoát (mã 127) với **mọi vùng song song chỉ có 1 luồng**. Code đã né lỗi này: dùng `num_threads(...)` thay vì `omp_set_num_threads`, và chạy nhánh tuần tự khi chỉ có 1 luồng. Trên Linux/Colab không bị.
- Thời gian đo **không** gồm khởi tạo, ghi file, tính năng lượng. Bản CUDA in riêng thời gian sao chép H2D/D2H; có 1 lần warm-up kernel trước khi đo.
- Bản CUDA chưa được biên dịch trên máy phát triển (không có GPU NVIDIA) — cần chạy thử trên Colab trước tiên (việc của Đạt, xem phân công).

## 8. Cấu trúc thư mục

```
src/nbody_common.h        khởi tạo Plummer, năng lượng, tham số, đo thời gian, ghi CSV
src/nbody_cpu_kernels.h   nhân tính toán tuần tự + OpenMP (cũng là tham chiếu cho CUDA)
src/nbody_cpu.c           chương trình CPU
src/nbody_cuda.cu         chương trình CUDA (naive + tiled)
scripts/                  benchmark.sh, plot_results.py, visualize.py
notebooks/                nbody_colab.ipynb
```
