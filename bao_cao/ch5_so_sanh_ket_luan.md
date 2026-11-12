# Chương 5. So sánh tổng hợp và kết luận

*Người viết: Nguyễn Quốc Doanh — khung, sẽ điền khi có đủ số liệu GPU (`results/gpu_t4.csv`)*

## 5.1. Môi trường đo

| | CPU | GPU |
|---|---|---|
| Phần cứng | Intel Core i5-12500H (4P + 8E = 12 lõi, 16 luồng) | NVIDIA Tesla T4 (40 SM, Turing) — Google Colab |
| Trình biên dịch | MinGW gcc 12.1, `-O3 -march=native -ffast-math -fopenmp` | nvcc, `-O3 -arch=sm_75 -use_fast_math` |
| Cách đo | Mỗi cấu hình chạy 3 lần, lấy lần nhanh nhất; không tính khởi tạo / năng lượng / ghi file | Như CPU; có 1 lần warm-up kernel; thời gian H2D/D2H tính riêng |
| File số liệu | `results/cpu_laptop.csv` (+ `.machine.txt`) | `results/gpu_t4.csv` (+ `.machine.txt`) |

## 5.2. Bảng tổng hợp

<!-- Dán bảng từ results/summary.md sau khi chạy:
     python scripts/plot_results.py results/cpu_laptop.csv results/gpu_t4.csv -->

*(chờ số liệu GPU)*

Hình 5.1 — `results/time_vs_n.png`: thời gian / bước theo N (log–log).
Hình 5.2 — `results/gpu_speedup.png`: speedup của các phiên bản so với tuần tự.

## 5.3. Nhận xét

<!-- Các ý cần viết sau khi có số liệu:
  - Độ dốc ~2 trên đồ thị log–log xác nhận độ phức tạp O(N²) cho mọi phiên bản (với N đủ lớn).
  - OpenMP: speedup tối đa đo được bao nhiêu với 16 luồng? Vì sao thấp hơn 16 (lõi E chậm hơn lõi P,
    hyper-threading chỉ có 4 lõi P, turbo giảm khi nhiều lõi chạy) -> tham chiếu Ch.3.
  - GPU: với N nhỏ (< ~8192), GPU chưa đủ block để lấp 40 SM, nên lợi thế còn nhỏ; với N lớn GPU vượt CPU bao nhiêu lần.
  - Tiled vs naive: nhanh hơn bao nhiêu lần, vì sao (tái sử dụng dữ liệu trong shared memory) -> tham chiếu Ch.4.
  - So sánh GFLOP/s đạt được với số liệu GPU Gems 3 (GeForce 8800 GTX: ~250 GFLOP/s) và với đỉnh lý thuyết của T4 (~8.1 TFLOP/s FP32).
  - Chi phí truyền H2D/D2H rất nhỏ so với thời gian tính khi N lớn vì tính O(N²), truyền O(N).
-->

## 5.4. Kết luận

<!-- Tóm tắt 4–5 câu: cả 4 phiên bản đúng; mức tăng tốc OpenMP; mức tăng tốc GPU; bài học chính
     (bài toán cường độ tính toán cao + độc lập dữ liệu -> GPU vượt trội; tối ưu bộ nhớ quan trọng). -->

## 5.5. Hướng phát triển

- **Thuật toán Barnes–Hut** (cây octree, O(N log N)) hoặc **Fast Multipole Method** (O(N)) để mô phỏng hàng triệu vật thể.
- **Tận dụng định luật 3 Newton** (chỉ tính N²/2 cặp) trên CPU với mảng gia tốc riêng cho từng luồng.
- **Nhiều GPU / nhiều máy** bằng MPI: chia các vật thể cho từng tiến trình, trao đổi vị trí mỗi bước (mô hình ring).
- **Tích phân bậc cao hơn** (leapfrog / Hermite bậc 4) và bước thời gian thích nghi để giảm sai số năng lượng.
- **Kết hợp CPU + GPU** (OpenMP offload, chồng lấp truyền dữ liệu với tính toán bằng CUDA streams).
