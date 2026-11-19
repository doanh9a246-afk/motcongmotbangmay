# Nguyễn Quốc Doanh: Mở đầu, So sánh tổng hợp, Kết luận

**Slide:** 2–5 (mở bài, ~2 phút) và 25–29 (kết bài, ~3 phút)
**Báo cáo:** Mục 1.1–1.4, 3.1 (môi trường đo), 3.5 (so sánh tổng hợp), Kết luận
**Code/script:** `src/nbody_common.h`, `Makefile`, `scripts/benchmark.sh`, `scripts/plot_results.py`

## Kịch bản nói

### Mở bài (~2 phút)
| Slide | Nói gì | Thời gian |
|---|---|---|
| 2 Nội dung | Giới thiệu nhóm. Bài trình bày có 5 phần, mỗi người trình bày phần mình làm. | 20 s |
| 3–4 Tính cấp thiết | N vật thể thì có N² tương tác. Với N = 10⁵, một bước cần 2·10¹¹ FLOP. Một lõi CPU (~34 GFLOP/s) chạy 10⁴ bước mất ~16 giờ. Xung nhịp không tăng nữa, muốn nhanh hơn phải song song. N-body là bài toán mẫu vì các tương tác độc lập với nhau. | 60 s |
| 5 Mục tiêu | Bốn phiên bản cùng đầu vào (Plummer, seed 42): tuần tự làm mốc, OpenMP, CUDA naive, CUDA tiled. Kiểm tra đúng, rồi đo hiệu năng. → *"Mời bạn Vũ Văn Dương trình bày cơ sở lý thuyết."* | 40 s |

### Kết bài (~3 phút)
| Slide | Nói gì | Thời gian |
|---|---|---|
| 26 So sánh tổng hợp | Trên đồ thị log–log, mỗi đường có độ dốc ≈ 2, xác nhận O(N²). Song song hóa chỉ **dịch đường xuống**, không đổi bậc. Ở N = 131072: OpenMP 4.8×, CUDA tiled 132× so với tuần tự (28× so với OpenMP). Với N nhỏ, GPU không đủ việc cho 40 SM. CPU và GPU đo trên hai máy, nên chỉ so được bậc độ lớn. | 75 s |
| 27 Kết luận | Bốn phiên bản cho kết quả khớp nhau. OpenMP 4.8–5.9×, SIMD ×2.2, dynamic tốt hơn trên CPU lai. CUDA tiled đạt 4.7 TFLOP/s. Kết quả khớp lý thuyết (Gustafson, roofline). **Bài học: cách truy cập bộ nhớ quan trọng không kém số phép tính.** | 60 s |
| 28 Hướng phát triển | Barnes–Hut O(N log N), nhiều GPU + MPI, tối ưu kernel, tích phân bậc cao. | 30 s |
| 29 | Cảm ơn, mời thầy/cô đặt câu hỏi. | 5 s |

## Số liệu phải thuộc

| | Giá trị |
|---|---|
| Máy đo tổng hợp CPU | Intel i5-12500H (4P + 8E, 16 luồng), gcc 12.1 `-O3 -march=native -ffast-math -fopenmp` |
| Máy đo GPU | Tesla T4 trên Colab: 40 SM, 2560 nhân, 8.1 TFLOP/s, 320 GB/s, nvcc 13.0 |
| Cách đo | Mỗi cấu hình chạy 3 lần, lấy lần nhanh nhất; không tính thời gian khởi tạo, tính năng lượng, ghi file |
| Tuần tự, N = 131072 | 11 694 ms/bước |
| OpenMP 16 luồng, N = 131072 | 2 432 ms/bước → 4.8× |
| CUDA tiled, N = 131072 | 88.6 ms/bước → 132×, 27.5× so với OpenMP |
| N = 16384 (GFLOP/s) | tuần tự 34, OpenMP 134, naive 3721, tiled 4676 |
| N = 1024 | tiled chỉ 14.9× (lưới có 4 khối, không lấp đủ 40 SM) |

## Câu hỏi có thể gặp

**H: Vì sao thí nghiệm OpenMP riêng đạt 5.9× mà bảng tổng hợp chỉ 4.8×?**
Hai phép đo chạy trên **hai máy khác nhau**. Thí nghiệm riêng dùng i7-13620H (6 lõi P + 4 lõi E). Bảng tổng hợp dùng i5-12500H (4 lõi P + 8 lõi E), ít lõi P hơn nên speedup thấp hơn.

**H: Ở N = 16384, OpenMP chỉ đạt 3.9×, thấp hơn N = 8192 (4.4×). Vì sao?**
Nên trả lời thật: mỗi cấu hình chỉ đo 3 lần trên laptop, nên kết quả chịu ảnh hưởng của xung nhịp turbo và nhiệt độ. Xu hướng chung (speedup tăng theo N rồi bão hòa ở ~4.5–4.8×) vẫn rõ. Đừng bịa ra một nguyên nhân cụ thể.

**H: Đồ thị log–log cho biết điều gì?**
Nếu t ∝ N² thì log t = 2·log N + hằng số, nên đồ thị là đường thẳng có độ dốc 2. Ví dụ: tuần tự 39.4 → 158 → 694 ms, mỗi lần N gấp đôi thì thời gian gấp ~4. Song song hóa làm giảm hằng số, không giảm bậc.

**H: Vì sao GPU kém hiệu quả khi N nhỏ?**
Với N = 1024 và B = 256, lưới chỉ có 4 khối, trong khi T4 có 40 SM, nên phần lớn GPU đứng yên. Thêm vào đó, chi phí khởi chạy kernel là cố định. Đây là ý của định luật Gustafson: GPU chỉ phát huy khi bài toán đủ lớn.

**H: So sánh CPU và GPU trên hai máy khác nhau có công bằng không?**
Không hoàn toàn, và báo cáo có ghi rõ. Colab chỉ có 2 vCPU, còn laptop không có GPU NVIDIA. Ngoài ra CPU dùng `-ffast-math`, GPU dùng `-use_fast_math`. Vì vậy tỷ số chỉ nói lên bậc độ lớn (hàng chục đến hàng trăm lần), không phải con số chính xác.

**H: Barnes–Hut khác gì thuật toán của nhóm?**
Barnes–Hut chia không gian thành cây octree. Một nhóm vật thể ở xa được gộp thành một "vật thể" đặt tại khối tâm của nhóm, nên độ phức tạp còn O(N log N). Đổi lại, kết quả là xấp xỉ (điều chỉnh bằng tham số θ), và cây khó song song trên GPU hơn vì truy cập bộ nhớ không đều.

**H: MPI thì chia việc thế nào?**
Mỗi tiến trình giữ N/P vật thể. Mỗi bước, các tiến trình trao đổi vị trí theo vòng (ring): sau P−1 lần chuyển, mỗi tiến trình đã thấy hết vị trí. Phần tính lực của mỗi tiến trình vẫn có thể dùng OpenMP hoặc CUDA.

**H: Nhóm đo thời gian thế nào cho công bằng?**
Mỗi cấu hình chạy 3 lần, lấy lần nhanh nhất. Không tính thời gian khởi tạo, tính năng lượng, ghi file. Có một lần chạy khởi động (warm-up) trước khi bấm giờ. Trên CPU dùng `omp_get_wtime` hoặc `QueryPerformanceCounter` (`nb_wtime` trong [nbody_common.h:55](../../src/nbody_common.h#L55)); trên GPU dùng `cudaEvent`. Mỗi lần chạy in một dòng `RESULT,...` để `benchmark.sh` thu lại.

## Code cần biết
- [nbody_common.h](../../src/nbody_common.h): khởi tạo Plummer (`nb_init_plummer`), năng lượng (`nb_energy`), đọc tham số (`nb_parse`), in dòng `RESULT` (`nb_report`).
- [Makefile](../../Makefile): `make` (CPU), `make cuda`, `make docs` (biên dịch báo cáo và slide).
- `scripts/benchmark.sh` chạy các cấu hình và ghi CSV; `scripts/plot_results.py` vẽ `time_vs_n.png` và `gpu_speedup.png`, đồng thời tạo `results/summary.md`.
