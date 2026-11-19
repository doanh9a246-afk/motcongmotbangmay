# Vũ Văn Dương: Bài toán N-body, lý thuyết song song, kiểm chứng

**Slide:** 3–5 (~2.5 phút)
**Báo cáo:** Mục 1.5 (cơ sở lý thuyết N-body, thuật toán tuần tự), 2.1 (kiến trúc), 2.2 (độ đo hiệu năng), 2.3 (phân tích khả năng song song), 3.2 (kiểm chứng tính đúng đắn)
**Code:** `nb_init_plummer`, `nb_energy` ([nbody_common.h](../../src/nbody_common.h)); `nb_body_acc`, `nb_acc_serial`, `nb_integrate` ([nbody_cpu_kernels.h](../../src/nbody_cpu_kernels.h)); `scripts/visualize.py`

## Kịch bản nói

| Slide | Nói gì | Thời gian |
|---|---|---|
| 3 Bài toán và thuật toán tuần tự | Công thức gia tốc có softening ε: lực luôn hữu hạn; khi j = i đóng góp bằng 0 nên **vòng lặp không cần `if`** (theo GPU Gems 3). Euler đối xứng: cập nhật v trước rồi dùng v mới cập nhật r. Mỗi bước 2 pha: tính gia tốc O(N²), cập nhật O(N), nên cần song song hóa pha 1. | 55 s |
| 4 Khả năng song song và độ đo | Mỗi a_i chỉ đọc `pos`, chỉ ghi `acc[i]`, nên các i độc lập, không xung đột ghi. Speedup S = T₁/Tₚ, hiệu suất E = S/p. Amdahl: bài toán cố định thì speedup bị chặn. Gustafson: N lớn lên thì phần O(N) co lại, speedup tăng. | 50 s |
| 5 Kiểm chứng | `-verify`: OpenMP khớp từng bit, CUDA sai khác ~10⁻⁸. Đồ thị: Δt > 0.002 thì sai số năng lượng tăng nhanh. Chọn Δt = 0.002, ε = 0.01, cho \|ΔE/E₀\| ≈ 2.7·10⁻⁶. → *"Mời bạn Nguyễn Tùng Dương."* | 45 s |

Các nội dung không còn trên slide (phân loại Flynn, CPU lai, roofline) vẫn có trong báo cáo. Hãy ôn để trả lời nếu bị hỏi.

## Số liệu phải thuộc

| | Giá trị |
|---|---|
| Thí nghiệm năng lượng | N = 2048, 2000 bước, seed 42, i5-12450HX, 12 luồng |
| Δt = 0.0005 / 0.001 / 0.002 / 0.005 / 0.01 (ε = 0.01) | 1.7e-5 / 1.75e-6 / **2.7e-6** / 9.3e-5 / 1.26e-3 |
| ε = 0.001 / 0.01 / 0.1 (Δt = 0.002) | 4.8e-3 / **2.7e-6** / 6.8e-6 |
| Amdahl P = 0.99, p = 16 | S = 1 / (0.01 + 0.99/16) ≈ 13.9; giới hạn 100 |
| Quỹ đạo | N = 1024, 2000 bước, lưu mỗi 20 bước → 101 khung hình GIF |
| AVX2 | thanh ghi 256 bit = 8 float |

## Câu hỏi có thể gặp

**H: Softening ε để làm gì? Có làm sai vật lý không?**
Khi hai vật rất gần nhau, 1/r² tiến tới vô cùng, gia tốc tăng đột biến và bước Δt hữu hạn không theo kịp. ε làm lực luôn hữu hạn. Có làm thay đổi mô hình ở cự ly ngắn (khoảng cách cỡ ε), vì vậy thế năng cũng phải tính theo công thức đã làm mềm U_ε cho nhất quán. Đây là cách làm chuẩn trong mô phỏng cụm sao, và GPU Gems 3 (tài liệu [1]) cũng dùng đúng cách này.

**H: Vì sao không cần điều kiện `j ≠ i`?**
Khi j = i thì r = 0. Nhờ ε² > 0 nên mẫu số khác 0 và đóng góp là 0·(…) = 0. Bỏ được `if` giúp CPU vector hóa vòng lặp và GPU không bị phân kỳ warp.

**H: Euler đối xứng khác Euler thường thế nào? Vì sao chọn nó?**
Euler thường dùng v cũ để cập nhật r. Euler đối xứng cập nhật v trước rồi dùng **v mới** để cập nhật r. Cả hai đều bậc 1, nhưng Euler đối xứng là phương pháp symplectic: giữ được cấu trúc của hệ Hamilton, nên năng lượng **dao động quanh** giá trị đúng thay vì trôi dần. Chi phí tính toán như Euler thường.

**H: Vì sao Δt = 0.0005 (nhỏ hơn) lại cho sai số lớn hơn Δt = 0.001?**
Có hai lý do:
1. Với tích phân symplectic, năng lượng dao động, nên sai số đo tại một thời điểm cuối phụ thuộc vào pha dao động lúc đó.
2. Thí nghiệm giữ cố định 2000 bước, nên tổng thời gian vật lý T = 2000·Δt khác nhau giữa các lần chạy (1 đến 20 đơn vị). Đây là hạn chế đã ghi trong báo cáo. Một khảo sát chặt chẽ hơn phải giữ T cố định.

**H: Vì sao năng lượng tính bằng `double` mà mô phỏng dùng `float`?**
Thế năng là tổng của ~N²/2 số hạng. Cộng bằng `float` sẽ tích lũy sai số làm tròn lớn hơn chính độ lệch cần đo (~10⁻⁶). Năng lượng chỉ tính 2 lần (đầu và cuối) nên dùng `double` không tốn đáng kể.

**H: Mô hình Plummer là gì? Vì sao dùng nó?**
Đây là phân bố mật độ của một cụm sao ở trạng thái cân bằng: ρ(r) ∝ (1 + r²/a²)^(−5/2). Bán kính được lấy mẫu bằng biến đổi nghịch đảo r = a/√(u^(−2/3) − 1), vận tốc bằng lấy mẫu loại bỏ. Mô hình này cho dữ liệu đầu vào thực tế, cân bằng (cụm không tự sụp hay tan rã) và **tất định** nhờ seed 42, nên mọi phiên bản bắt đầu từ cùng một trạng thái.

**H: Amdahl và Gustafson có mâu thuẫn không?**
Không mâu thuẫn, mà trả lời hai câu hỏi khác nhau. Amdahl: giữ bài toán **cố định**, tăng số luồng thì speedup bị chặn bởi 1/(1−P). Gustafson: giữ thời gian cố định, cho **bài toán lớn lên** theo số luồng thì speedup gần tuyến tính. N-body hợp với Gustafson vì phần song song O(N²) lớn nhanh hơn phần tuần tự O(N).

**H: Có thể song song theo các bước thời gian không?**
Không. Bước n+1 cần toàn bộ vị trí của bước n. Song song chỉ làm được **bên trong** mỗi bước.

**H: Vì sao OpenMP khớp bản tuần tự từng bit, còn CUDA thì không?**
Bản tuần tự và bản OpenMP gọi **cùng một hàm** `nb_body_acc`. Với mỗi i, một luồng cộng theo đúng thứ tự j như bản tuần tự, nên kết quả trùng từng bit. GPU dùng `rsqrtf` (xấp xỉ phần cứng) và trình biên dịch khác (có FMA), nên sai khác ở mức làm tròn `float`, ≈ 10⁻⁸.

**H: GPU thuộc loại nào trong phân loại Flynn?**
NVIDIA gọi là SIMT. Về bản chất, mỗi warp 32 luồng chạy cùng một lệnh, gần với SIMD. Khác ở chỗ mỗi luồng có thanh ghi và địa chỉ riêng, và có thể rẽ nhánh (nhưng các nhánh khi đó phải chạy lần lượt).

## Code cần biết
- [nbody_cpu_kernels.h:44](../../src/nbody_cpu_kernels.h#L44) `nb_body_acc`: vòng j có `#pragma omp simd`, không có `if (j != i)`.
- [nbody_cpu_kernels.h:94](../../src/nbody_cpu_kernels.h#L94) `nb_acc_serial`, [nbody_cpu_kernels.h:299](../../src/nbody_cpu_kernels.h#L299) `nb_integrate`: v += a·dt rồi p += v·dt.
- [nbody_common.h:97](../../src/nbody_common.h#L97) `nb_init_plummer`, [nbody_common.h:150](../../src/nbody_common.h#L150) `nb_energy`: tính bằng `double`, vòng j > i.
- Lệnh kiểm tra: `./bin/nbody_cpu -n 2048 -s 5 -verify`
