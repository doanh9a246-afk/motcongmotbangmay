# Tài liệu ôn thuyết trình

Mỗi người đọc **file của mình** và **phần "Ai cũng phải trả lời được"** ở cuối trang này.

| Người | File | Slide trình bày | Mục trong báo cáo |
|---|---|---|---|
| Nguyễn Quốc Doanh | [1_doanh.md](1_doanh.md) | 2–5 (mở đầu), 25–29 (so sánh, kết luận) | 1.1–1.4, 3.1, 3.5, Kết luận |
| Vũ Văn Dương | [2_vv_duong.md](2_vv_duong.md) | 6–12 | 1.5, 2.1–2.3, 3.2 |
| Nguyễn Tùng Dương | [3_nt_duong.md](3_nt_duong.md) | 13–18 | 2.4, 2.6, 3.3 |
| Trần Tiến Đạt | [4_dat.md](4_dat.md) | 19–24 | 2.2.4 (roofline), 2.5, 3.4 |

Số slide tính theo [slide_nhom.pdf](../slide_nhom.pdf). Tên người nói in ở góc dưới bên trái mỗi slide. Mỗi người **≤ 5 phút**.

Cách dùng mỗi file:
1. **Kịch bản nói:** từng slide nói gì và mất bao lâu. Tập nói theo đúng thời gian, đừng đọc chữ trên slide.
2. **Số liệu phải thuộc:** những con số thầy/cô hay hỏi lại.
3. **Câu hỏi có thể gặp:** tự trả lời thành tiếng trước khi đọc đáp án.
4. **Code cần biết:** mở file trong `src/` và đọc đúng các dòng được chỉ. Thầy/cô hay hỏi "chỉ cho tôi đoạn code đó".

---

## Ai cũng phải trả lời được

Thầy/cô có thể hỏi **bất kỳ ai** những câu này, không chỉ người phụ trách phần đó.

**1. Bài toán N-body là gì? Vì sao độ phức tạp là O(N²)?**
Có N vật thể hút nhau theo định luật hấp dẫn. Mỗi bước phải tính lực giữa mọi cặp vật thể, nên có N² tương tác. Sau đó cập nhật vận tốc và vị trí, phần này chỉ O(N).

**2. Vì sao bài toán này dễ song song hóa?**
Trong một bước, gia tốc của vật i chỉ **đọc** vị trí ở đầu bước và chỉ **ghi** vào `acc[i]` của riêng nó. Vì vậy các vật thể độc lập với nhau, không có xung đột ghi. Các *bước thời gian* thì phải chạy lần lượt.

**3. Bốn phiên bản là gì, kết quả chính ra sao?**

| Phiên bản | Kết quả ở N = 131072 |
|---|---|
| Tuần tự | 11.7 s/bước, ≈ 34 GFLOP/s |
| OpenMP 16 luồng | 4.8× so với tuần tự (tối đa 5.9× trong thí nghiệm riêng) |
| CUDA naive | ≈ 113× |
| CUDA tiled | 132× so với tuần tự, 28× so với OpenMP; đỉnh 4.7 TFLOP/s ở N = 16384 |

**4. Làm sao biết kết quả đúng?**
- `-verify`: chạy thêm một bản tham chiếu từ cùng trạng thái đầu, rồi so **vị trí** sau mô phỏng. OpenMP khớp bản tuần tự từng bit. CUDA (so với OpenMP) sai khác ≈ 10⁻⁸, do làm tròn `float` và `rsqrtf`.
- Năng lượng: |ΔE/E₀| ≈ 2.7·10⁻⁶ sau 2000 bước (Δt = 0.002, ε = 0.01).

**5. "20 FLOP/tương tác" ở đâu ra?**
Đây là quy ước của GPU Gems 3. Phép trừ 3 + bình phương khoảng cách cộng ε² 6 + `rsqrt` 2 + lập phương 2 + nhân khối lượng 1 + cộng dồn gia tốc 6 = 20. Có thể xem chú thích trong [nbody_cuda.cu:31-45](../../src/nbody_cuda.cu#L31-L45).

**6. Vì sao so sánh CPU với GPU trên hai máy khác nhau?**
Colab chỉ có 2 vCPU nên không đo CPU ở đó được, còn laptop của nhóm không có GPU NVIDIA. Vì vậy tỷ số CPU/GPU chỉ cho biết **bậc độ lớn**. Báo cáo có ghi rõ điểm này.

**7. Ai làm gì trong nhóm?**
- Doanh: tích hợp, benchmark tổng hợp, Mở đầu và Kết luận, ghép báo cáo/slide.
- V.V. Dương: thuật toán tuần tự, Plummer, thí nghiệm năng lượng, lý thuyết song song.
- N.T. Dương: OpenMP.
- Đạt: CUDA.
