# Chương 5. So sánh tổng hợp và kết luận

*Người viết: Nguyễn Quốc Doanh*

## 5.1. Môi trường đo

| | CPU | GPU |
|---|---|---|
| Phần cứng | Intel Core i5-12500H (4P + 8E = 12 lõi, 16 luồng) | NVIDIA Tesla T4 (40 SM, Turing) — Google Colab |
| Trình biên dịch | MinGW gcc 12.1, `-O3 -march=native -ffast-math -fopenmp` | nvcc 13.0, `-O3 -arch=sm_75 -use_fast_math` |
| Cách đo | Mỗi cấu hình chạy 3 lần, lấy lần nhanh nhất; không tính khởi tạo / năng lượng / ghi file | Như CPU; có 1 lần warm-up kernel; thời gian H2D/D2H tính riêng |
| File số liệu | `results/cpu_laptop.csv` (+ `.machine.txt`) | `results/gpu_t4.csv` (+ `.machine.txt`) |

## 5.2. Bảng tổng hợp

| N | Tuần tự (ms/bước) | OpenMP 16 luồng (ms) | Speedup OpenMP | CUDA naive (ms) | CUDA tiled (ms) | Tiled / tuần tự | Tiled / OpenMP |
|---|---|---|---|---|---|---|---|
| 1024 | 0.611 | 0.383 | 1.6× | 0.063 | 0.041 | 14.9× | 9.3× |
| 2048 | 2.464 | 1.171 | 2.1× | 0.125 | 0.078 | 31.5× | 15.0× |
| 4096 | 10.006 | 2.844 | 3.5× | 0.255 | 0.150 | 66.6× | 18.9× |
| 8192 | 39.446 | 9.065 | 4.4× | 0.498 | 0.298 | 132.3× | 30.4× |
| 16384 | 158.032 | 40.023 | 3.9× | 1.443 | 1.148 | 137.6× | 34.9× |
| 32768 | 693.889 | 146.524 | 4.7× | 5.737 | 4.891 | 141.9× | 30.0× |
| 65536 | 2742.200 | 629.848 | 4.4× | 26.209 | 21.873 | 125.4× | 28.8× |
| 131072 | 11693.782 | 2431.867 | 4.8× | 103.057 | 88.551 | 132.1× | 27.5× |

*Bảng 5.1 — Thời gian mỗi bước và speedup. Nguồn: `results/summary.md`, sinh bằng `python scripts/plot_results.py results/cpu_laptop.csv results/gpu_t4.csv`. CPU: N ≤ 16384 đo 5 bước, N ≥ 32768 đo 1 bước; GPU: 5 bước, block 256. Mỗi cấu hình lấy lần nhanh nhất trong 3 lần.*

| Phiên bản (N = 16384) | GFLOP/s | So với tuần tự |
|---|---|---|
| Tuần tự | 34 | 1× |
| OpenMP 16 luồng | 134 | 3.9× |
| CUDA naive | 3 721 | 110× |
| CUDA tiled | 4 676 | 138× |

*Bảng 5.2 — Hiệu năng tính theo quy ước 20 FLOP/tương tác của GPU Gems 3.*

Hình 5.1 — `results/time_vs_n.png`: thời gian / bước theo N (log–log).
Hình 5.2 — `results/gpu_speedup.png`: speedup của các phiên bản so với tuần tự.

## 5.3. Nhận xét

**Độ phức tạp O(N²).** Trên Hình 5.1 (thang log–log), mỗi lần N tăng gấp đôi thì thời gian tuần tự tăng khoảng 4 lần (ví dụ 39.4 → 158.0 → 693.9 ms), tức độ dốc ≈ 2, đúng với độ phức tạp O(N²) của thuật toán all-pairs. Đường OpenMP và hai đường CUDA cũng có độ dốc ≈ 2 khi N đủ lớn; các phiên bản song song chỉ dịch đường thẳng xuống dưới, không làm thay đổi bậc của bài toán.

**OpenMP.** Với 16 luồng, OpenMP nhanh hơn tuần tự 4.4–4.8× khi N ≥ 8192 trong lần đo tổng hợp này; thí nghiệm riêng ở Chương 3 đạt tối đa 5.91× (N = 8192). Con số này thấp hơn nhiều so với 16× lý tưởng vì i5-12500H chỉ có 4 lõi P (có siêu phân luồng) và 8 lõi E chậm hơn, đồng thời xung nhịp turbo giảm khi nhiều lõi cùng chạy (phân tích chi tiết ở mục 3.5). Với N nhỏ (1024–2048), chi phí tạo và đồng bộ luồng chiếm tỉ lệ lớn nên speedup chỉ 1.6–2.1×.

**GPU so với CPU.** CUDA tiled trên Tesla T4 nhanh hơn bản tuần tự 125–142× và nhanh hơn OpenMP 16 luồng 28–35× khi N ≥ 8192. Với N nhỏ, lợi thế giảm còn 15× (N = 1024): lưới chỉ có 4 block 256 luồng, không đủ để lấp đầy 40 SM của T4, nên phần lớn GPU không có việc để làm. Đây là lý do đường CUDA trên Hình 5.1 thoải hơn (độ dốc < 2) ở vùng N ≤ 8192.

**Tiled so với naive.** Kernel tiled nhanh hơn naive 1.5–1.7× khi N ≤ 8192, vì mỗi vị trí được nạp từ bộ nhớ toàn cục một lần cho cả block rồi dùng lại 256 lần trong shared memory. Khi N ≥ 16384, khoảng cách chỉ còn khoảng 1.2×: bộ nhớ đệm L1/L2 của Turing đã giữ lại phần lớn dữ liệu mà kernel naive đọc lặp, nên lợi ích của shared memory giảm (chi tiết ở Chương 4).

**Hiệu năng tuyệt đối.** Kernel tiled đạt cao nhất 4.68 TFLOP/s (N = 16384), khoảng 58% đỉnh lý thuyết FP32 của T4 (~8.1 TFLOP/s) và gấp khoảng 19 lần con số ~250 GFLOP/s của GeForce 8800 GTX trong GPU Gems 3. Khi N ≥ 32768, hiệu năng giảm nhẹ về khoảng 3.9 TFLOP/s; nhiều khả năng do T4 (giới hạn công suất 70 W) hạ xung khi kernel chạy liên tục nhiều trăm mili giây.

**Kích thước block và chi phí truyền dữ liệu.** Với N = 131072, block 64, 128, 256 và 512 cho hiệu năng chênh nhau dưới 4%, block 256 nhanh nhất (`results/block_size.png`, Chương 4). Thời gian sao chép H2D + D2H chỉ bằng 0.3% thời gian tính 10 bước ở N = 131072 (2.5 ms so với 840 ms), nhưng chiếm khoảng 6% ở N = 4096. Lý do là dữ liệu truyền tăng theo O(N) còn khối lượng tính tăng theo O(N²); hơn nữa dữ liệu được giữ trên GPU suốt mô phỏng và chỉ sao chép ở đầu và cuối.

**Lưu ý về phép so sánh.** CPU và GPU được đo trên hai máy khác nhau (laptop và Google Colab), CPU dùng `-ffast-math` còn GPU dùng `-use_fast_math` (hàm `rsqrtf`). Vì vậy các tỉ số CPU–GPU ở trên phản ánh bậc độ lớn chứ không phải phép so sánh trên cùng một hệ thống.

## 5.4. Kết luận

Nhóm đã cài đặt bài toán N-body all-pairs theo bốn phiên bản (tuần tự, OpenMP, CUDA naive, CUDA tiled). Các phiên bản song song trên CPU cho kết quả khớp với bản tuần tự (`-verify`), và năng lượng toàn phần được bảo toàn tốt với bước thời gian và hệ số làm mềm đã chọn (Chương 2). Trên CPU 12 lõi lai, OpenMP tăng tốc 4.8–5.9×, trong đó vector hóa SIMD đóng góp khoảng 2.2×. Trên Tesla T4, kernel tiled nhanh hơn bản tuần tự khoảng 132× và nhanh hơn OpenMP 16 luồng khoảng 28× ở N = 131072, đạt tới 4.7 TFLOP/s. Bài học chính là một bài toán có cường độ tính toán cao và các phép tính độc lập như N-body rất phù hợp với GPU, nhưng để đạt hiệu năng cao vẫn cần tối ưu truy cập bộ nhớ (shared memory, SIMD) và cho phần cứng đủ việc (N đủ lớn).

## 5.5. Hướng phát triển

- **Thuật toán Barnes–Hut** (cây octree, O(N log N)) hoặc **Fast Multipole Method** (O(N)) để mô phỏng hàng triệu vật thể.
- **Tận dụng định luật 3 Newton** (chỉ tính N²/2 cặp) trên CPU với mảng gia tốc riêng cho từng luồng.
- **Nhiều GPU / nhiều máy** bằng MPI: chia các vật thể cho từng tiến trình, trao đổi vị trí mỗi bước (mô hình ring).
- **Tích phân bậc cao hơn** (leapfrog / Hermite bậc 4) và bước thời gian thích nghi để giảm sai số năng lượng.
- **Kết hợp CPU + GPU** (OpenMP offload, chồng lấp truyền dữ liệu với tính toán bằng CUDA streams).
