# Trần Tiến Đạt: Song song hóa bằng CUDA

**Slide:** 19–24 (~5 phút)
**Báo cáo:** Mục 2.2.4 (roofline), 2.5 (mô hình CUDA, kiến trúc GPU), 3.4 (cài đặt và thực nghiệm CUDA)
**Code:** [nbody_cuda.cu](../../src/nbody_cuda.cu), [notebooks/nbody_colab.ipynb](../../notebooks/nbody_colab.ipynb)

## Kịch bản nói

| Slide | Nói gì | Thời gian |
|---|---|---|
| 20 Mô hình CUDA | Lưới → khối → luồng; mỗi warp 32 luồng chạy cùng một lệnh. Khối được chạy trên các SM. Luồng trong cùng khối dùng chung shared memory và đồng bộ bằng `__syncthreads()`. Phân cấp bộ nhớ: thanh ghi, shared, L1/L2, global. T4 có 40 SM, 8.1 TFLOP/s, 320 GB/s. | 50 s |
| 21 Roofline | Mỗi tương tác 20 FLOP, đọc 16 byte, nên AI = 1.25, thấp hơn nhiều so với điểm cân bằng 25.3 của T4. Đọc thẳng từ global memory thì bị **giới hạn bởi bộ nhớ**. Nhưng mỗi vị trí được dùng lại N lần: nếu B luồng nạp chung thì AI tăng B lần (≈ 320). | 50 s |
| 22 Kernel tiled | Cả khối nạp một tile B vị trí vào shared memory, mỗi luồng nạp một phần tử. Sau đó mỗi luồng dùng lại cả B phần tử. Hai lần `__syncthreads()`: lần 1 chờ nạp xong rồi mới đọc; lần 2 chờ dùng xong rồi mới nạp tile mới. | 60 s |
| 23 Kết quả T4 | Tiled đạt 4.7 TFLOP/s (≈ 58% đỉnh) ở N = 16384. Nhanh hơn naive 1.5–1.7× khi N ≤ 8192, chỉ còn ~1.2× khi N lớn vì bộ đệm L1/L2 đã giúp naive. Kích thước khối 64–512 chênh nhau dưới 4%; B = 256 tốt nhất. | 60 s |
| 24 Truyền dữ liệu | Chi phí H2D + D2H giảm từ 5.7% xuống 0.29% vì truyền là O(N) còn tính là O(N²). Điều kiện: dữ liệu nằm trên GPU suốt mô phỏng. → *"Mời bạn Doanh trình bày so sánh tổng hợp và kết luận."* | 40 s |

## Số liệu phải thuộc

| | Giá trị |
|---|---|
| T4 | Turing, 40 SM × 64 = 2560 nhân, 8.1 TFLOP/s FP32, 16 GB GDDR6, 320 GB/s, 70 W, shared ≤ 64 KB/SM, L2 4 MB |
| Môi trường | Colab, CUDA 13.0, `nvcc -O3 -arch=sm_75 -use_fast_math` |
| Tiled/naive | 1.54–1.70× (N ≤ 8192); 1.16–1.26× (N ≥ 16384) |
| Đỉnh | tiled 4676 GFLOP/s (N = 16384) = 57.7% đỉnh; N ≥ 65536 khoảng 3880–4090 GFLOP/s |
| Kích thước khối (N = 131072) | 64: 3815, 128: 3823, **256: 3880**, 512: 3727 GFLOP/s |
| H2D+D2H / kernel | N = 4096: 5.72% → N = 131072: 0.29% |
| GPU Gems 3 | 8800 GTX ≈ 250 GFLOP/s; T4 cao hơn ≈ 18.7× |
| Shared memory mỗi khối | B × 16 byte = 4 KB với B = 256 |

## Câu hỏi có thể gặp

**H: Ánh xạ luồng vào vật thể thế nào? Có bao nhiêu khối?**
Mỗi luồng ứng với một vật thể đích, `i = blockIdx.x * blockDim.x + threadIdx.x`. Số khối là `grid = (N + B - 1) / B`, tức làm tròn lên. Vì luồng cuối có thể vượt N, phải kiểm tra `i < N`.

**H: Vì sao kernel naive dùng `if (i >= n) return;` mà kernel tiled lại không thoát sớm?**
Trong kernel tiled, **mọi** luồng của khối phải tham gia nạp tile và gặp `__syncthreads()`. Nếu một luồng thoát sớm, các luồng còn lại sẽ chờ nó ở barrier mãi (treo hoặc hành vi không xác định). Vì vậy luồng thừa vẫn chạy hết vòng, nạp phần tử khối lượng 0, và chỉ không ghi kết quả (`if (i < n) acc[i] = ...`).

**H: Bỏ một trong hai `__syncthreads()` thì sao?**
- Bỏ lần 1: có luồng đọc `shPos[k]` khi luồng khác chưa nạp xong, nên đọc phải dữ liệu cũ hoặc rác (lỗi RAW).
- Bỏ lần 2: warp chạy nhanh bắt đầu nạp tile mới, ghi đè lên dữ liệu mà warp chậm còn đang dùng (lỗi WAR).

Cả hai trường hợp đều cho kết quả sai, và lỗi có thể chỉ thỉnh thoảng mới xuất hiện.

**H: Vì sao dùng `float4`?**
`x, y, z` là vị trí, `w` là khối lượng. Một bản ghi 16 byte được đọc bằng một lệnh. Các luồng liền kề đọc các địa chỉ liền kề nên truy cập được gộp (coalesced).

**H: Vì sao tiled nhanh hơn naive, nhưng khi N lớn thì chỉ nhanh hơn 1.2×?**
Theo lý thuyết, tiled giảm số lần đọc global memory đi B lần. Trên thực tế, GPU Turing có bộ đệm L1/L2. Với naive, mọi luồng trong warp cùng đọc `pos[j]` (cùng một địa chỉ, được phát cho cả warp) và dữ liệu đó thường đã nằm trong L1/L2, nên naive không thiếu băng thông đến mức mô hình lý tưởng dự đoán. Tiled lại phải trả thêm chi phí nạp tile và đồng bộ.

**H: Vì sao chỉ đạt 58% đỉnh lý thuyết?**
1. `rsqrtf` chạy trên đơn vị SFU, có thông lượng thấp hơn đơn vị FP32 thường.
2. Hai `__syncthreads()` mỗi tile.
3. Chi phí điều khiển vòng lặp.
4. Occupancy bị giới hạn bởi thanh ghi và shared memory.
5. Mỗi luồng chỉ tính một vật thể (chưa làm thread coarsening).
6. T4 giới hạn 70 W nên có thể hạ xung khi chạy lâu.

Ngoài ra, quy ước 20 FLOP/tương tác chỉ là ước lượng.

**H: Kernel có dùng `#pragma unroll` không?**
Có, vòng trong của kernel tiled dùng `#pragma unroll 8` ([nbody_cuda.cu:72](../../src/nbody_cuda.cu#L72)). Hướng phát triển là thử các mức unroll khác và thread coarsening.

**H: Chọn kích thước khối thế nào?**
Khối nhỏ (64, 128) thì mỗi tile được dùng lại ít lần hơn. Khối lớn (512) tốn nhiều tài nguyên mỗi khối, nên mỗi SM chạy được ít khối cùng lúc hơn (occupancy giảm). B = 256 là điểm cân bằng, nhưng 4 cấu hình chỉ chênh nhau dưới 4%. Giá trị này không phải tối ưu cho mọi GPU.

**H: Đo thời gian GPU thế nào? Có tính thời gian chép dữ liệu không?**
Dùng `cudaEventRecord` và `cudaEventElapsedTime`. Thời gian kernel tính riêng, thời gian H2D/D2H đo riêng. Trước khi đo có một lần warm-up kernel. Dữ liệu chỉ chép lên GPU một lần ở đầu và chép về ở cuối; mỗi bước chỉ gọi 2 kernel (`accTiled`/`accNaive` rồi `integrateBodies`).

**H: Warp có bị phân kỳ không?**
Gần như không. Vòng lặp không có `if (j != i)` (nhờ softening). Phần tử đệm có khối lượng 0 nên cũng không cần `if`. Chỉ có điều kiện `i < n` ở khối cuối.

**H: Vì sao kết quả GPU khác CPU khoảng 10⁻⁸?**
`rsqrtf` là phép tính xấp xỉ của phần cứng, `-use_fast_math` cho phép thêm các phép xấp xỉ, và trình biên dịch gộp phép nhân–cộng thành FMA, nên thứ tự làm tròn khác CPU. `-verify` so vị trí GPU với bản OpenMP; ngưỡng đạt là 10⁻³ và mức sai khác thực tế là ~10⁻⁸.

**H: So với GPU Gems 3 thì sao?**
GPU Gems 3 đạt ~250 GFLOP/s trên GeForce 8800 GTX (2007). T4 đạt ~4.7 TFLOP/s, cao hơn ~18.7×, chủ yếu nhờ phần cứng mới hơn (nhiều nhân hơn). Thuật toán tiling là cùng một thuật toán.

## Code cần biết
- [nbody_cuda.cu:31](../../src/nbody_cuda.cu#L31) `bodyBodyInteraction`: chú thích đếm 20 FLOP.
- [nbody_cuda.cu:48](../../src/nbody_cuda.cu#L48) `accNaive`, [nbody_cuda.cu:59](../../src/nbody_cuda.cu#L59) `accTiled` (hai `__syncthreads()` ở dòng 71 và 74).
- [nbody_cuda.cu:92](../../src/nbody_cuda.cu#L92) `launchAcc`: `<<<grid, block, block*sizeof(float4)>>>` cấp shared memory động.
- [nbody_cuda.cu:137-187](../../src/nbody_cuda.cu#L137-L187): đo bằng `cudaEvent`, H2D/D2H.
- Lệnh: `./bin/nbody_cuda -n 65536 -s 10 -k tiled -b 256`, `./bin/nbody_cuda -n 4096 -s 5 -k naive -verify`
