# Phân công nhiệm vụ

**Đề tài:** Song song hóa mô phỏng N-body (lực hấp dẫn) bằng OpenMP và CUDA
**Nhóm trưởng:** Nguyễn Quốc Doanh

Nguyên tắc chia việc: mỗi người phụ trách **1 phần code + 1 nhóm thí nghiệm + 1 chương báo cáo + phần slide tương ứng**, khối lượng tương đương nhau (~25% mỗi người). Nhóm trưởng gánh thêm việc điều phối nên phần code riêng nhẹ hơn một chút.

## Tổng quan

| Thành viên | Vai trò | Phần code | Thí nghiệm | Chương báo cáo |
|---|---|---|---|---|
| **Nguyễn Quốc Doanh** | Nhóm trưởng, tích hợp | `nbody_common.h`, `Makefile`, `benchmark.sh`, `plot_results.py` | Benchmark tổng hợp: tuần tự vs OpenMP vs CUDA | Ch.1 Giới thiệu, Ch.5 So sánh & kết luận |
| **Vũ Văn Dương** | Thuật toán & bản tuần tự | `nb_acc_serial`, `nb_integrate`, khởi tạo Plummer, `visualize.py` | Tính đúng đắn: bảo toàn năng lượng theo `dt`, `eps`; ảnh/GIF quỹ đạo | Ch.2 Cơ sở lý thuyết & thuật toán tuần tự |
| **Nguyễn Tùng Dương** | Song song OpenMP | `nb_acc_omp`, `nbody_cpu.c` | Speedup/hiệu suất theo số luồng; so sánh `schedule`; có/không `simd` | Ch.3 Song song hóa bằng OpenMP |
| **Trần Tiến Đạt** | Song song CUDA | `nbody_cuda.cu` (naive + tiled), notebook Colab | Naive vs tiled; ảnh hưởng block size; thời gian truyền dữ liệu | Ch.4 Song song hóa bằng CUDA |

## Chi tiết

### Nguyễn Quốc Doanh — nhóm trưởng
- Quản lý repo GitHub, review và merge code của các thành viên, giữ cho `make` luôn build được.
- Phụ trách `src/nbody_common.h` (tham số dòng lệnh, đo thời gian, ghi CSV), `Makefile`, `scripts/benchmark.sh`, `scripts/plot_results.py`.
- Chạy **benchmark tổng hợp** trên cùng một máy (Colab GPU) với N = 1024 → 131072; vẽ đồ thị thời gian theo N và speedup so với tuần tự.
- Báo cáo: **Ch.1** (đặt vấn đề, mục tiêu, ứng dụng của N-body) và **Ch.5** (so sánh 4 phiên bản, nhận xét, hướng phát triển: Barnes-Hut O(N log N), multi-GPU, MPI).
- Ghép báo cáo + slide hoàn chỉnh, định dạng, tài liệu tham khảo; chuẩn bị phần mở đầu/kết luận khi thuyết trình.

### Vũ Văn Dương — thuật toán & bản tuần tự
- Phụ trách hàm tuần tự `nb_acc_serial`, `nb_integrate`, hàm khởi tạo Plummer và hàm tính năng lượng.
- Thí nghiệm **tính đúng đắn**: chạy dài (1000–5000 bước), đo `|ΔE/E₀|` khi thay đổi `dt` (0.0005 → 0.01) và `eps` (0.001 → 0.1); vẽ đồ thị.
- Tạo ảnh / GIF quỹ đạo bằng `scripts/visualize.py` để đưa vào báo cáo và slide.
- Báo cáo **Ch.2**: định luật hấp dẫn, softening, tích phân symplectic Euler, mô hình Plummer, độ phức tạp O(N²), mã giả thuật toán tuần tự, phân tích phần nào song song hóa được (định luật Amdahl).

### Nguyễn Tùng Dương — OpenMP
- Phụ trách `nb_acc_omp` và chương trình `src/nbody_cpu.c` (chế độ `-verify`).
- Thí nghiệm: speedup & hiệu suất với 1, 2, 4, 8, … luồng cho N = 2048 → 16384; so sánh `schedule(static | dynamic | guided)`; so sánh có/không `#pragma omp simd`; thử biến thể dùng định luật 3 Newton (tính N²/2 cặp) và giải thích vì sao khó song song hơn (race condition → cần mảng riêng mỗi luồng rồi cộng dồn).
- Báo cáo **Ch.3**: mô hình fork-join, cách chia vòng lặp, vì sao không có race condition, phân tích kết quả đo (vì sao speedup < lý tưởng: băng thông bộ nhớ, nhân P/E, turbo).

### Trần Tiến Đạt — CUDA
- Phụ trách `src/nbody_cuda.cu` và `notebooks/nbody_colab.ipynb`. ~~Việc đầu tiên: biên dịch và chạy `-verify` trên Colab~~ — **xong (09/11): PASS cả naive và tiled trên T4.**
- Thí nghiệm: kernel naive vs tiled (shared memory) với N = 4096 → 131072; block size 64/128/256/512; tỉ lệ thời gian sao chép H2D/D2H so với thời gian tính; (nâng cao) thử `#pragma unroll`, profile bằng Nsight Compute.
- Báo cáo **Ch.4**: kiến trúc GPU, ánh xạ 1 thread = 1 vật thể, thuật toán tile của GPU Gems 3 (hình minh họa tile), shared memory và `__syncthreads()`, phân tích kết quả, so sánh với số liệu trong GPU Gems 3.

## Kế hoạch 4 tuần

| Tuần | Doanh | V.V. Dương | N.T. Dương | Đạt |
|---|---|---|---|---|
| 1 | Lập repo, phân công, đọc GPU Gems 3 cùng nhóm | Đọc lý thuyết, kiểm tra bản tuần tự | Đọc OpenMP spec, kiểm tra bản OpenMP | Chạy bản CUDA trên Colab, sửa lỗi |
| 2 | Hoàn thiện script benchmark, review code | Thí nghiệm năng lượng (dt, eps) | Thí nghiệm số luồng, schedule, simd | Thí nghiệm naive/tiled, block size |
| 3 | Benchmark tổng hợp, vẽ đồ thị | Viết Ch.2, làm GIF | Viết Ch.3 | Viết Ch.4 |
| 4 | Viết Ch.1, Ch.5, ghép báo cáo + slide | Slide Ch.2 | Slide Ch.3 | Slide Ch.4 |

Mỗi người tự thuyết trình phần của mình (~5 phút/người).

## Quy ước làm việc
- Mỗi người làm trên nhánh riêng (`git checkout -b <ten>/<viec>`), tạo Pull Request; nhóm trưởng review rồi merge vào `main`.
- Số liệu GPU đo trên **Colab T4**. Số liệu CPU/OpenMP đo trên **máy nhiều lõi thật** (laptop/PC), không dùng Colab vì Colab chỉ có 2 vCPU. Luôn ghi rõ cấu hình máy (CPU, số lõi/luồng, trình biên dịch) cạnh mỗi bảng số liệu.
- Khi so sánh GPU với CPU trong Ch.5, dùng số liệu CPU tốt nhất (OpenMP trên máy nhiều lõi), không dùng CPU của Colab.
- Không dán token/mật khẩu vào notebook, code hay ảnh chụp gửi nhóm. Repo đã public nên clone không cần token.
- Kết quả thô (`results/*.csv`) gửi kèm để nhóm trưởng tổng hợp.
