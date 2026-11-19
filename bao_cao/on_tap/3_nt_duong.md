# Nguyễn Tùng Dương: Song song hóa bằng OpenMP

**Slide:** 6–8 (~2.5 phút)
**Báo cáo:** Mục 2.4 (mô hình OpenMP), 2.6 (chiến lược song song hóa), 3.3 (cài đặt và thực nghiệm OpenMP)
**Code:** `nb_acc_omp`, `nb_acc_omp_custom`, `nb_acc_omp_newton3` ([nbody_cpu_kernels.h](../../src/nbody_cpu_kernels.h)), [nbody_cpu.c](../../src/nbody_cpu.c)

## Kịch bản nói

| Slide | Nói gì | Thời gian |
|---|---|---|
| 6 Cách song song hóa | Chỉ vào đoạn code: `parallel for` chia vòng i cho các luồng (fork–join, barrier ngầm cuối vùng, theo OpenMP spec); `simd reduction` vector hóa vòng j, 8 float/lệnh. Mỗi luồng chỉ ghi `acc[i]` của mình nên không cần `critical`/`atomic`. | 45 s |
| 7 Speedup | Máy i7-13620H (6P + 4E). 1–4 luồng gần tuyến tính; tối đa 5.9× ở 16 luồng, N = 8192. Không đạt 16× vì lõi E chậm, 2 luồng chung 1 lõi P, xung nhịp giảm: **giới hạn do phần cứng.** | 50 s |
| 8 SIMD, lập lịch, Newton 3 | SIMD ×2.2. `dynamic` nhanh hơn `static` 13–33% nhờ cân bằng tải giữa lõi P và E. Newton 3 tính một nửa mà lại chậm hơn 1.2–2.3× (ghi phân tán, mất SIMD, mảng đệm 4 MB). → *"Mời bạn Trần Tiến Đạt."* | 55 s |

## Số liệu phải thuộc

| | Giá trị |
|---|---|
| Máy | i7-13620H: 6 lõi P (2 luồng/lõi) + 4 lõi E = 16 luồng; MinGW-W64 GCC 16.2, `-O3 -march=native -ffast-math -fopenmp` |
| Tuần tự | ≈ 34–35 GFLOP/s |
| Tốt nhất | 5.91× (N = 8192, 16 luồng, 201 GFLOP/s); N = 16384: 5.58× |
| 4 luồng | 3.44–3.92×, E = 86–98% |
| 16 luồng, N = 2048 | chỉ 3.51× (E = 22%): bài toán quá nhỏ, 0.68 ms/bước |
| SIMD | ×1.88–2.50; ở N = 16384: dynamic 232 GFLOP/s, static 205 |
| Newton 3 (16 luồng, N = 16384) | 41.6 ms so với all-pairs 25.5 ms → chậm hơn 1.63× |
| Mảng đệm Newton 3 | 16 luồng × 16384 × 16 byte ≈ 4.2 MB |

## Câu hỏi có thể gặp

**H: Vì sao không có race condition? Không cần `critical` hay `atomic` à?**
Mỗi chỉ số i chỉ thuộc về một luồng. Luồng đó chỉ **ghi** `acc[i]`. Mảng `pos` thì mọi luồng chỉ **đọc**. Hai luồng không bao giờ ghi vào cùng một ô nhớ, nên không cần đồng bộ.

**H: Có false sharing không? `acc[i]` và `acc[i+1]` có thể nằm chung một dòng cache.**
Chỉ xảy ra ở chỗ tiếp giáp giữa hai đoạn của hai luồng. Hơn nữa, mỗi `acc[i]` chỉ được ghi **một lần** sau khi tính xong N tương tác (giá trị cộng dồn nằm trong thanh ghi `ax, ay, az`). Vì vậy ảnh hưởng không đáng kể.

**H: static, dynamic, guided khác nhau thế nào? Vì sao dynamic thắng dù mọi i có khối lượng như nhau?**
static chia trước N/16 vòng cho mỗi luồng, gần như không tốn chi phí lập lịch. dynamic,64 đặt các gói 64 vòng vào hàng đợi chung, luồng nào rảnh thì lấy gói tiếp theo. guided cũng vậy nhưng gói lớn trước, nhỏ dần.
Khối lượng mỗi i bằng nhau, nhưng **tốc độ các lõi không bằng nhau**. Với static, luồng trên lõi P xong trước rồi đứng chờ lõi E ở barrier. dynamic để lõi P làm phần nhiều hơn.

**H: Chương trình mặc định dùng static. Sao không đổi sang dynamic?**
`nb_acc_omp` mặc định static vì đó là lựa chọn lý thuyết cho khối lượng đều và cho kết quả tất định. Chế độ khác chọn bằng tham số `-sched`. Kết quả đo cho thấy trên CPU lai nên dùng dynamic; đây là một kết luận của nhóm.

**H: Vì sao 16 luồng mà chỉ nhanh 5.9×?**
1. Chỉ có 6 lõi P. 6 luồng thứ hai chạy hyper-threading, dùng chung đơn vị tính với luồng thứ nhất trên cùng lõi P.
2. 4 lõi E chậm hơn (tối đa 3.6 GHz so với 4.9 GHz).
3. Khi nhiều lõi cùng chạy AVX2, xung nhịp giảm xuống khoảng 2.8–3.2 GHz.
4. Với N nhỏ, chi phí tạo luồng và đồng bộ ở barrier chiếm tỷ lệ lớn.

Phần tuần tự (cập nhật O(N)) rất nhỏ, nên Amdahl **không** phải nguyên nhân chính.

**H: Hiệu suất 102% ở N = 8192 với 1–2 luồng là sao?**
Đó là nhiễu đo. Bản tuần tự và bản OpenMP 1 luồng được đo ở hai lần chạy khác nhau, nên xung nhịp turbo lúc đo có thể chênh vài phần trăm. Không phải siêu tuyến tính thật.

**H: `#pragma omp simd reduction(+:ax,ay,az)` làm gì?**
Nó yêu cầu trình biên dịch vector hóa vòng j. Mỗi "làn" (lane) SIMD giữ một tổng riêng; `reduction` báo cho trình biên dịch cộng các tổng riêng đó lại ở cuối. Kết quả là một lệnh AVX2 xử lý 8 tương tác cùng lúc (lệnh `vrsqrtps`, `vfmadd...`).

**H: Làm sao chắc bản "tắt SIMD" thật sự là vô hướng?**
Với `-O3 -ffast-math`, GCC tự vector hóa dù không có pragma. Vì vậy hàm `nb_body_acc_nosimd` được gắn `__attribute__((optimize("no-tree-vectorize")))`. Kiểm tra bằng `-fopt-info-vec` cho thấy hàm này không còn vòng lặp nào được vector hóa.

**H: Biến thể Newton 3 cài đặt thế nào? Sao không dùng `atomic`?**
Mỗi luồng có mảng đệm `my_acc` gồm N phần tử. Luồng cộng vào `my_acc[i]` và trừ vào `my_acc[j]`. Sau đó một vòng `omp for` cộng p mảng đệm lại thành `acc`.
Dùng `atomic` cho mỗi lần ghi (~N²/2 lần ghi) sẽ chậm hơn nhiều vì các luồng tranh nhau cùng dòng cache.

**H: Vì sao phải chạy khởi động (warm-up) trước khi đo?**
Lần đầu vào vùng song song, runtime phải tạo nhóm luồng và bộ đệm còn trống. Không loại phần này ra thì kết quả đo bị sai, nhất là khi N nhỏ.

## Code cần biết
- [nbody_cpu_kernels.h:106-142](../../src/nbody_cpu_kernels.h#L106-L142): `nb_acc_omp_custom` có 3 chế độ schedule × có/không SIMD; `nb_acc_omp` mặc định static + SIMD.
- [nbody_cpu_kernels.h:152](../../src/nbody_cpu_kernels.h#L152) `nb_acc_omp_newton3`: mảng đệm mỗi luồng và bước cộng dồn.
- [nbody_cpu_kernels.h:311](../../src/nbody_cpu_kernels.h#L311) `nb_run_cpu_ex`: warm-up, bấm giờ từng bước.
- Lệnh: `./bin/nbody_cpu -n 8192 -s 5 -m omp -t 16`, `./bin/nbody_cpu -n 2048 -s 5 -verify`
