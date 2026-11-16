# CHƯƠNG 4: SONG SONG HÓA BÀI TOÁN N-BODY TRÊN GPU VỚI CUDA

## 4.1. Kiến trúc GPU NVIDIA Tesla T4

Bài toán được thực nghiệm trên GPU **NVIDIA Tesla T4**, sử dụng kiến trúc Turing thông qua nền tảng Google Colab. Các đặc điểm phần cứng có ảnh hưởng trực tiếp đến quá trình thiết kế và tối ưu hóa CUDA Kernel bao gồm:

- **Streaming Multiprocessor (SM):** GPU Tesla T4 sở hữu 40 SM, mỗi SM có 64 nhân CUDA FP32, tổng cộng 2.560 nhân CUDA.
- **Hiệu năng lý thuyết đỉnh:** Khoảng 8,1 TFLOP/s đối với phép tính FP32 tại xung nhịp Boost 1,59 GHz.
- **Mô hình thực thi SIMT và Warp:** Các luồng được tổ chức thành từng Warp gồm 32 luồng, thực thi cùng một lệnh tại một thời điểm. Khi một Warp phải chờ dữ liệu từ bộ nhớ, Warp Scheduler có thể chuyển sang thực thi Warp khác để che giấu độ trễ truy cập bộ nhớ.
- **Hệ thống phân cấp bộ nhớ:**
  - **Global Memory (VRAM):** Dung lượng 16 GB GDDR6, băng thông lý thuyết 320 GB/s nhưng có độ trễ truy cập tương đối cao.
  - **Shared Memory/L1 Cache:** Bộ nhớ nằm trên chip của từng SM, có độ trễ thấp và băng thông cao. Shared Memory được chia sẻ giữa các luồng trong cùng một Thread Block, với dung lượng tối đa 64 KB Shared Memory mỗi SM trên kiến trúc Turing.
  - **Registers:** Thanh ghi riêng của từng luồng, có tốc độ truy cập nhanh nhất trong hệ thống phân cấp bộ nhớ GPU.

Những đặc điểm trên là cơ sở để lựa chọn chiến lược phân chia công việc và tối ưu hóa truy cập bộ nhớ cho bài toán N-body.

## 4.2. Ánh xạ bài toán và tổ chức Grid/Block

Bài toán N-body mô phỏng tương tác hấp dẫn giữa N vật thể. Mỗi vật thể chịu tác động từ tất cả các vật thể còn lại, do đó có độ phức tạp tính toán là O(N²).

Để khai thác khả năng xử lý song song của GPU, chương trình sử dụng phương pháp phân rã dữ liệu (Data Parallelism), trong đó mỗi luồng CUDA chịu trách nhiệm tính toán gia tốc cho một vật thể đích (Target Body).

**Tổ chức phân cấp luồng (Thread Hierarchy):**

Tập hợp N vật thể được chia thành các Thread Block một chiều (1D Thread Block). Mỗi Block có B luồng, với các kích thước được khảo sát gồm:

B ∈ {64, 128, 256, 512}

Số lượng Block trong Grid được xác định bằng công thức làm tròn lên nhằm đảm bảo bao phủ toàn bộ N vật thể:

\[
gridDim.x = \left\lceil \frac{N}{B} \right\rceil
= \frac{N+B-1}{B}
\]

Mỗi luồng xác định chỉ số toàn cục của vật thể mà mình phụ trách thông qua công thức:

\[
i = blockIdx.x \times blockDim.x + threadIdx.x
\]

Trong đó:
- blockIdx.x là chỉ số của Thread Block.
- blockDim.x là số luồng trong mỗi Block.
- threadIdx.x là chỉ số của luồng bên trong Block.
- i là chỉ số vật thể đích mà luồng chịu trách nhiệm tính toán.

**Đảm bảo tính độc lập giữa các luồng:**

Mỗi luồng chỉ đọc dữ liệu của toàn bộ N vật thể nguồn để tính gia tốc cho vật thể đích tương ứng, sau đó ghi kết quả vào phần tử thứ i của mảng gia tốc. Vì mỗi luồng có một vị trí ghi kết quả riêng biệt nên không xảy ra xung đột ghi dữ liệu (Race Condition). Do đó, chương trình không cần sử dụng các phép toán nguyên tử như atomicAdd.

## 4.3. Kernel Naive và nút thắt cổ chai bộ nhớ toàn cục

### 4.3.1. Nguyên lý hoạt động của Naive Kernel

Naive Kernel là phương pháp triển khai trực tiếp bài toán N-body trên GPU. Trong cách tiếp cận này, mỗi luồng CUDA chịu trách nhiệm tính gia tốc cho một vật thể đích i.

Đầu tiên, luồng tải tọa độ và khối lượng của vật thể đích từ Global Memory vào Register. Sau đó, luồng thực hiện vòng lặp qua toàn bộ N vật thể nguồn j, lần lượt đọc dữ liệu từ Global Memory để tính toán lực hấp dẫn.

Quá trình tính toán bao gồm các bước:
1. Đọc tọa độ và khối lượng của vật thể nguồn.
2. Tính khoảng cách theo ba trục tọa độ x, y, z.
3. Tính bình phương khoảng cách và cộng thêm hệ số epsilon² nhằm tránh chia cho giá trị bằng 0.
4. Tính nghịch đảo khoảng cách bằng hàm rsqrtf().
5. Xác định hệ số tương tác và cộng dồn các thành phần gia tốc theo ba trục.
6. Ghi kết quả gia tốc vào vị trí tương ứng trong mảng đầu ra.

### 4.3.2. Phân tích nút thắt cổ chai bộ nhớ (Memory-Bound)

Mặc dù có cấu trúc đơn giản và dễ triển khai, Naive Kernel gặp hạn chế đáng kể về hiệu năng do truy cập Global Memory liên tục.

**Thứ nhất, dư thừa truy cập bộ nhớ.**

Với mỗi cặp tương tác (i, j), luồng i phải đọc 16 bytes dữ liệu từ Global Memory, tương ứng với cấu trúc float4 gồm tọa độ x, y, z và khối lượng m.

Tổng số lần đọc dữ liệu của toàn bộ hệ thống là N², tương đương lưu lượng:

\[
16N^2 \text{ bytes}
\]

Trong khi đó, dữ liệu của toàn bộ N vật thể chỉ chiếm:

\[
16N \text{ bytes}
\]

Điều này cho thấy cùng một dữ liệu vật thể nguồn phải được đọc lặp lại bởi nhiều luồng khác nhau, tạo ra lượng truy cập bộ nhớ dư thừa lớn.

**Thứ hai, cường độ tính toán trên truy cập bộ nhớ thấp (Arithmetic Intensity).**

Mỗi tương tác giữa hai vật thể tiêu tốn khoảng 20 phép toán dấu phẩy động (FLOPs) và cần 16 bytes dữ liệu từ Global Memory. Do đó:

\[
AI = \frac{20}{16}=1,25 \text{ FLOP/byte}
\]

Trong khi đó, để khai thác mức hiệu năng lý thuyết 8.100 GFLOP/s của Tesla T4 với băng thông bộ nhớ 320 GB/s, thuật toán cần đạt mức Arithmetic Intensity:

\[
AI \geq \frac{8100}{320}
\approx 25,3 \text{ FLOP/byte}
\]

Như vậy, Arithmetic Intensity của Naive Kernel thấp hơn đáng kể so với mức cần thiết để khai thác tối đa năng lực tính toán của GPU. Vì vậy, các nhân CUDA có thể phải chờ dữ liệu từ bộ nhớ trong phần lớn thời gian thực thi.

## 4.4. Kernel Tiled sử dụng Shared Memory (GPU Gems 3, Chương 31)

Để khắc phục hạn chế về truy cập Global Memory của Naive Kernel, chương trình áp dụng kỹ thuật Tiling sử dụng Shared Memory, dựa trên phương pháp được trình bày trong GPU Gems 3, Chương 31.

Ý tưởng chính của kỹ thuật này là tận dụng khả năng chia sẻ dữ liệu giữa các luồng trong cùng một Thread Block. Thay vì để từng luồng liên tục đọc dữ liệu vật thể nguồn từ Global Memory, các luồng trong Block phối hợp nạp một nhóm dữ liệu vào Shared Memory để tái sử dụng trong quá trình tính toán.

### 4.4.1. Cơ chế hoạt động của Tiling

Vòng lặp duyệt N vật thể nguồn được chia thành các Tile có kích thước B, tương ứng với số luồng trong mỗi Thread Block. Số Tile cần xử lý được xác định bởi:

\[
numTiles = \left\lceil \frac{N}{B} \right\rceil
\]

Quá trình xử lý được thực hiện qua các bước:

**Bước 1: Chia dữ liệu thành các Tile.**

Toàn bộ N vật thể nguồn được chia thành các nhóm dữ liệu liên tiếp, mỗi nhóm chứa tối đa B vật thể.

**Bước 2: Hợp tác nạp dữ liệu vào Shared Memory.**

Tại mỗi Tile, B luồng trong Block phối hợp nạp B vật thể liên tiếp từ Global Memory vào mảng Shared Memory:

\[
\_\_shared\_\_ float4\ shPos[B]
\]

Do các luồng liền kề đọc các địa chỉ bộ nhớ liền kề nên quá trình truy cập được tối ưu thông qua cơ chế Memory Coalescing.

**Bước 3: Tái sử dụng dữ liệu trong Shared Memory.**

Sau khi dữ liệu được nạp thành công, tất cả B luồng trong Block cùng truy cập tập dữ liệu shPos để tính toán tương tác giữa vật thể đích mà mình phụ trách với các vật thể nguồn trong Tile hiện tại.

Nhờ đó, mỗi dữ liệu vật thể nguồn được nhiều luồng trong cùng Block tái sử dụng thay vì phải đọc lại từ Global Memory nhiều lần.

Về mặt lý thuyết, lượng truy cập Global Memory đối với dữ liệu vật thể nguồn giảm từ N² xuống khoảng:

\[
\frac{N^2}{B}
\]

Với B = 256, số lần đọc dữ liệu từ Global Memory được giảm khoảng 256 lần so với cách tiếp cận trực tiếp, xét theo mô hình tải dữ liệu của thuật toán.

### 4.4.2. Vai trò của hai lệnh đồng bộ hóa __syncthreads()

Trong quá trình triển khai Tiled Kernel, lệnh __syncthreads() được sử dụng hai lần trong mỗi vòng lặp xử lý Tile nhằm đảm bảo tính nhất quán của dữ liệu giữa các luồng trong cùng Block.

**Lần đồng bộ thứ nhất – Sau khi nạp dữ liệu vào Shared Memory:**

Sau khi các luồng hoàn thành việc nạp dữ liệu vào shPos, lệnh __syncthreads() được gọi để đảm bảo toàn bộ B luồng đã ghi dữ liệu thành công trước khi bất kỳ luồng nào bắt đầu đọc và sử dụng dữ liệu trong Shared Memory.

Nếu thiếu lệnh đồng bộ này, một số Warp có thể bắt đầu đọc dữ liệu khi các Warp khác chưa hoàn thành quá trình ghi, dẫn đến lỗi Read After Write (RAW).

**Lần đồng bộ thứ hai – Sau khi hoàn thành tính toán trên Tile:**

Sau khi các luồng sử dụng xong dữ liệu của Tile hiện tại, lệnh __syncthreads() tiếp tục được gọi để đảm bảo toàn bộ luồng đã hoàn thành việc đọc dữ liệu trước khi chuyển sang Tile tiếp theo.

Lệnh này ngăn chặn tình trạng một số Warp ghi đè dữ liệu mới lên Shared Memory trong khi các Warp khác vẫn đang sử dụng dữ liệu cũ, gây ra lỗi Write After Read (WAR).

Như vậy, hai lệnh đồng bộ hóa đóng vai trò quan trọng trong việc đảm bảo tính chính xác của chương trình song song, đồng thời duy trì sự nhất quán dữ liệu trong quá trình tái sử dụng Shared Memory.

## 4.5. Kết quả thực nghiệm trên NVIDIA Tesla T4

### 4.5.1. Cấu hình hệ thống thực nghiệm

Các thử nghiệm được tiến hành trên nền tảng Google Colab với cấu hình phần cứng và phần mềm như sau:

| Thành phần | Thông số |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Kiến trúc GPU | Turing |
| Dung lượng VRAM | 16 GB GDDR6 |
| Driver | 580.82.07 |
| CUDA Toolkit | CUDA 13.0 |
| Trình biên dịch | nvcc |
| Cờ tối ưu hóa | -O3 |
| Độ chính xác | FP32 |
| CPU | Intel Xeon CPU @ 2.00GHz |
| Cấu hình CPU | 1 lõi / 2 luồng |
| Hệ điều hành | Linux 6.6.159+ |
| Trình biên dịch C/C++ | gcc 13.3.0 |

Các tham số benchmark được thiết lập gồm:
- STEPS = 5.
- REPS = 3.
- Số lượng vật thể khảo sát: N ∈ {1.024, 2.048, 4.096, 8.192, 16.384, 32.768, 65.536, 131.072}.
- Kích thước Block khảo sát: B ∈ {64, 128, 256, 512}.

### 4.5.2. So sánh hiệu năng giữa Naive Kernel và Tiled Kernel

Thí nghiệm đầu tiên so sánh thời gian thực thi và hiệu năng của hai phương pháp Naive và Tiled, sử dụng Block Size = 256 và STEPS = 5.

| Số vật thể (N) | Naive (ms) | Tiled (ms) | Tăng tốc | Naive (GFLOP/s) | Tiled (GFLOP/s) |
|---:|---:|---:|---:|---:|---:|
| 1.024 | 0,315 | 0,205 | 1,54x | 332,54 | 511,52 |
| 2.048 | 0,625 | 0,391 | 1,60x | 670,93 | 1.073,74 |
| 4.096 | 1,277 | 0,751 | 1,70x | 1.313,48 | 2.233,39 |
| 8.192 | 2,489 | 1,491 | 1,67x | 2.696,09 | 4.501,10 |
| 16.384 | 7,214 | 5,741 | 1,26x | 3.720,98 | 4.675,87 |
| 32.768 | 28,685 | 24,457 | 1,17x | 3.743,21 | 4.390,29 |
| 65.536 | 131,047 | 109,366 | 1,20x | 3.277,41 | 3.927,16 |
| 131.072 | 515,285 | 442,754 | 1,16x | 3.334,05 | 3.880,23 |

Kết quả cho thấy Tiled Kernel có thời gian thực thi thấp hơn Naive Kernel trên toàn bộ các kích thước bài toán được khảo sát. Mức tăng tốc dao động từ 1,16x đến 1,70x.

Đáng chú ý, mức tăng tốc cao nhất đạt 1,70x tại N = 4.096 và 1,67x tại N = 8.192. Tại N = 16.384, Tiled Kernel đạt hiệu năng cao nhất với 4.675,87 GFLOP/s.

### 4.5.3. Ảnh hưởng của kích thước Block đến hiệu năng

Để đánh giá ảnh hưởng của kích thước Thread Block, thí nghiệm được thực hiện tại N = 131.072 với bốn cấu hình B = 64, 128, 256 và 512.

| Block Size (B) | Thời gian (ms) | Thông lượng (GInter/s) | Hiệu năng (GFLOP/s) |
|---:|---:|---:|---:|
| 64 | 450,288 | 190,77 | 3.815,31 |
| 128 | 449,339 | 191,17 | 3.823,36 |
| 256 | 442,754 | 194,01 | 3.880,23 |
| 512 | 460,908 | 186,37 | 3.727,40 |

Kết quả thực nghiệm cho thấy Block Size = 256 đạt hiệu năng tốt nhất, với thời gian thực thi thấp nhất là 442,754 ms và hiệu năng đạt 3.880,23 GFLOP/s.

Các cấu hình B = 64 và B = 128 có thời gian thực thi tương đối gần nhau nhưng vẫn chậm hơn cấu hình B = 256. Trong khi đó, B = 512 cho kết quả thấp nhất về hiệu năng, với thời gian thực thi lên tới 460,908 ms.

Điều này cho thấy việc tăng kích thước Block không đồng nghĩa với việc cải thiện hiệu năng. Block quá lớn có thể làm tăng mức sử dụng tài nguyên trên mỗi Block và ảnh hưởng đến khả năng duy trì số lượng Warp hoạt động đồng thời (Occupancy) trên mỗi SM.

Vì vậy, trong phạm vi thực nghiệm, B = 256 là cấu hình phù hợp nhất cho bài toán N-body trên GPU Tesla T4.

### 4.5.4. Đánh giá chi phí truyền dữ liệu Host-to-Device và Device-to-Host

Ngoài thời gian thực thi Kernel, chi phí truyền dữ liệu giữa CPU và GPU cũng là một yếu tố cần xem xét khi đánh giá hiệu năng tổng thể.

Thí nghiệm được thực hiện với STEPS = 10 và Block Size = 256, nhằm đo thời gian truyền dữ liệu theo hai chiều:
- **H2D (Host-to-Device):** Thời gian truyền dữ liệu từ bộ nhớ CPU sang bộ nhớ GPU.
- **D2H (Device-to-Host):** Thời gian truyền dữ liệu từ bộ nhớ GPU về bộ nhớ CPU.

Kết quả thu được như sau:

| Số vật thể (N) | H2D (ms) | D2H (ms) | Tổng truyền dữ liệu (ms) | Kernel (ms) | Tổng thời gian (ms) | Tỷ lệ truyền dữ liệu (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 4.096 | 0,090 | 0,066 | 0,156 | 2,569 | 2,725 | 5,72% |
| 16.384 | 0,288 | 0,178 | 0,466 | 15,663 | 16,129 | 2,89% |
| 65.536 | 0,780 | 0,684 | 1,464 | 212,138 | 213,602 | 0,69% |
| 131.072 | 1,230 | 1,231 | 2,461 | 839,962 | 842,423 | 0,29% |

Kết quả cho thấy khi số lượng vật thể tăng, tỷ lệ thời gian truyền dữ liệu trên tổng thời gian thực thi giảm đáng kể. Cụ thể, tỷ lệ này giảm từ 5,72% tại N = 4.096 xuống còn 0,29% tại N = 131.072.

Tại quy mô 131.072 vật thể, tổng thời gian truyền dữ liệu chỉ là 2,461 ms, trong khi thời gian tính toán Kernel lên tới 839,962 ms.

Điều này cho thấy chi phí truyền dữ liệu có xu hướng trở nên ít đáng kể hơn khi quy mô bài toán tăng, do độ phức tạp truyền dữ liệu là O(N), trong khi độ phức tạp tính toán của bài toán N-body là O(N²).

Do đó, việc duy trì dữ liệu trên VRAM trong suốt quá trình mô phỏng nhiều bước thời gian là một hướng triển khai phù hợp nhằm hạn chế chi phí truyền dữ liệu giữa CPU và GPU.

## 4.6. Nhận xét và đánh giá hiệu năng

### 4.6.1. Hiệu quả của kỹ thuật Shared Memory Tiling

Kết quả thực nghiệm cho thấy Tiled Kernel luôn đạt hiệu năng cao hơn Naive Kernel trên toàn bộ các kích thước bài toán được khảo sát, với mức tăng tốc từ 1,16x đến 1,70x.

Hiệu quả tối ưu thể hiện rõ nhất tại N = 4.096 và N = 8.192, khi Tiled Kernel lần lượt đạt mức tăng tốc 1,70x và 1,67x. Tại N = 8.192, hiệu năng tăng từ 2.696,09 GFLOP/s lên 4.501,10 GFLOP/s.

Nguyên nhân chủ yếu đến từ khả năng tái sử dụng dữ liệu thông qua Shared Memory. Với cấu hình B = 256, mỗi Tile được nạp một lần từ Global Memory và được 256 luồng trong Block cùng sử dụng. Nhờ đó, số lần truy cập Global Memory được giảm đáng kể so với phương pháp Naive.

Tuy nhiên, khi quy mô bài toán tăng lên N ≥ 32.768, khoảng cách hiệu năng giữa hai phương pháp thu hẹp, mức tăng tốc chỉ còn khoảng 1,16x–1,20x. Một nguyên nhân được xác định là bộ nhớ đệm L2 Cache dung lượng 4 MB trên Tesla T4 có thể giữ lại một phần dữ liệu vật thể nguồn cho Naive Kernel, giúp giảm chi phí truy cập Global Memory.

Mặc dù vậy, Tiled Kernel vẫn duy trì ưu thế về hiệu năng ở các quy mô lớn, cho thấy tính hiệu quả của phương pháp tối ưu hóa bằng Shared Memory.

### 4.6.2. Lựa chọn Block Size tối ưu

Qua khảo sát bốn cấu hình Block Size tại N = 131.072, cấu hình B = 256 cho kết quả tốt nhất với thời gian thực thi 442,754 ms và hiệu năng 3.880,23 GFLOP/s.

Cấu hình B = 64 và B = 128 có hiệu năng thấp hơn một chút do khả năng tái sử dụng dữ liệu trong mỗi Tile thấp hơn. Ngược lại, B = 512 có thời gian thực thi cao nhất, đạt 460,908 ms.

Nguyên nhân có thể liên quan đến mức tiêu thụ tài nguyên của Block, số lượng Warp cần đồng bộ hóa và khả năng duy trì Occupancy trên các SM. Với B = 512, mỗi Block gồm 16 Warp, làm tăng mức độ phụ thuộc vào quá trình đồng bộ hóa giữa các Warp.

Kết quả thực nghiệm cho thấy việc lựa chọn kích thước Block cần cân bằng giữa khả năng tái sử dụng dữ liệu, tài nguyên phần cứng và chi phí đồng bộ hóa. Trong phạm vi thử nghiệm, B = 256 là lựa chọn tối ưu.

### 4.6.3. So sánh hiệu năng với GPU Gems 3 và mức hiệu năng lý thuyết của Tesla T4

So với kết quả được trình bày trong GPU Gems 3, Chương 31, thuật toán N-body trên NVIDIA GeForce 8800 GTX đạt khoảng 250 GFLOP/s.

Trong khi đó, thực nghiệm trên Tesla T4 đạt hiệu năng đỉnh 4.675,87 GFLOP/s tại N = 16.384. Ở các quy mô lớn, hiệu năng vẫn duy trì ở mức khoảng 3.880–4.091 GFLOP/s theo đánh giá trong tài liệu thực nghiệm.

Như vậy, hiệu năng đạt được trên Tesla T4 cao hơn khoảng 15,5–18,7 lần so với mốc tham chiếu trên GeForce 8800 GTX. Sự khác biệt này đến từ việc Tesla T4 sở hữu số lượng nhân CUDA lớn hơn đáng kể, cùng với những cải tiến về kiến trúc GPU qua các thế hệ.

So với mức hiệu năng lý thuyết FP32 khoảng 8.100 GFLOP/s của Tesla T4, mức hiệu năng đỉnh thực nghiệm 4.675,87 GFLOP/s tương đương:

\[
\frac{4675,87}{8100}\times 100
\approx 57,7\%
\]

Ở các quy mô lớn, hiệu năng đạt được tương đương khoảng 47,9–50,5% mức lý thuyết.

Khoảng cách giữa hiệu năng thực tế và lý thuyết có thể được lý giải bởi một số yếu tố:

- **Chi phí tính toán của hàm rsqrtf():** Hàm này sử dụng các đơn vị xử lý hàm đặc biệt (Special Function Units – SFU), có thông lượng khác với các đơn vị thực hiện phép toán FP32 thông thường.
- **Chi phí đồng bộ hóa:** Hai lệnh __syncthreads() được thực hiện trong mỗi Tile, tạo ra chi phí đồng bộ hóa giữa các luồng.
- **Lệnh điều khiển:** Các vòng lặp và lệnh điều khiển làm tăng số lượng thao tác ngoài tính toán số học chính.
- **Chưa áp dụng các kỹ thuật tối ưu hóa nâng cao:** Chương trình chưa sử dụng mở vòng lặp (#pragma unroll) và kỹ thuật Thread Coarsening (tính toán nhiều vật thể trên mỗi luồng) để tối ưu hóa việc sử dụng thanh ghi.

Do đó, mặc dù chưa đạt mức hiệu năng lý thuyết tối đa, kết quả thực nghiệm cho thấy Tiled Kernel đã khai thác tương đối hiệu quả năng lực xử lý song song của GPU Tesla T4.

### 4.6.4. Tác động của chi phí truyền dữ liệu H2D/D2H

Kết quả thực nghiệm cho thấy chi phí truyền dữ liệu giữa CPU và GPU chiếm tỷ trọng ngày càng nhỏ khi quy mô bài toán tăng.

Nguyên nhân xuất phát từ sự khác biệt về độ phức tạp giữa hai quá trình:

- Truyền dữ liệu qua bus PCIe có độ phức tạp O(N).
- Tính toán tương tác trong bài toán N-body có độ phức tạp O(N²).

Khi N tăng từ 4.096 lên 131.072, tỷ lệ thời gian truyền dữ liệu trên tổng thời gian giảm từ 5,72% xuống còn 0,29%.

Điều này cho thấy GPU phát huy hiệu quả rõ rệt đối với các bài toán mô phỏng có quy mô lớn, đặc biệt trong trường hợp dữ liệu được giữ cố định trên VRAM xuyên suốt nhiều bước thời gian. Việc hạn chế truyền dữ liệu qua lại giữa CPU và GPU giúp giảm chi phí phụ trợ và tập trung tài nguyên cho quá trình tính toán song song.

### 4.6.5. Kết luận chương

Chương 4 đã trình bày quá trình triển khai và tối ưu hóa bài toán N-body trên GPU NVIDIA Tesla T4 bằng nền tảng CUDA. Từ phương pháp Naive Kernel ban đầu, kỹ thuật Shared Memory Tiling được áp dụng nhằm giảm lượng truy cập Global Memory và tăng khả năng tái sử dụng dữ liệu giữa các luồng trong cùng Thread Block.

Kết quả thực nghiệm cho thấy Tiled Kernel luôn nhanh hơn Naive Kernel trên toàn bộ các quy mô bài toán được khảo sát, với mức tăng tốc từ 1,16x đến 1,70x. Cấu hình Block Size = 256 cho hiệu năng tốt nhất tại N = 131.072, đạt 3.880,23 GFLOP/s. Đồng thời, chi phí truyền dữ liệu H2D/D2H chỉ chiếm 0,29% tổng thời gian tại quy mô 131.072 vật thể.

Những kết quả này khẳng định hiệu quả của kỹ thuật Shared Memory Tiling trong việc tối ưu hóa bài toán N-body trên GPU. Qua đó, chương đã làm rõ vai trò của tổ chức luồng, phân cấp bộ nhớ và lựa chọn tham số thực thi trong việc khai thác hiệu năng của kiến trúc GPU hiện đại.