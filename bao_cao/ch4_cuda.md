*Người viết: Trần Tiến Đạt*

# Chương 4: Song song hóa bài toán N-body trên GPU với CUDA

## 4.1. Kiến trúc GPU NVIDIA Tesla T4

Bài toán được thực nghiệm trên GPU **NVIDIA Tesla T4** (kiến trúc Turing) thông qua nền tảng Google Colab. Các thông số phần cứng cốt lõi ảnh hưởng trực tiếp đến việc thiết kế kernel CUDA bao gồm:

* **Streaming Multiprocessor (SM):** T4 sở hữu 40 SM, mỗi SM chứa 64 nhân CUDA FP32 (tổng cộng 2.560 nhân CUDA).
* **Hiệu năng lý thuyết đỉnh (Peak FP32):** ~8.1 TFLOP/s (8.100 GFLOP/s) ở xung nhịp Boost 1.59 GHz.
* **Mô hình thực thi SIMT & Warp:** Các luồng (thread) được nhóm thành từng **Warp** gồm **32 luồng** thực thi cùng một lệnh tại một thời điểm. Khi một warp bị chặn do chờ dữ liệu từ bộ nhớ (memory latency), bộ lập lịch (Warp Scheduler) của SM sẽ chuyển ngữ cảnh tức thì sang warp khác sẵn sàng chạy để che giấu độ trễ.
* **Hệ thống phân cấp bộ nhớ:**
  * **Global Memory (VRAM):** Dung lượng 16 GB GDDR6, băng thông lý thuyết **320 GB/s**, nhưng có độ trễ truy cập rất cao (hàng trăm chu kỳ xung nhịp).
  * **Shared Memory / L1 Cache:** Bộ nhớ nằm ngay trên chip (on-chip) của từng SM với độ trễ cực thấp và băng thông rất cao, được chia sẻ chung giữa các luồng trong cùng một **Thread Block** (tối đa 64 KB Shared Memory mỗi SM trên kiến trúc Turing).
  * **Registers:** Thanh ghi riêng của từng luồng, tốc độ truy cập nhanh nhất.

---

## 4.2. Ánh xạ bài toán và tổ chức Grid / Block

Bài toán N-body tính lực hấp dẫn đôi một giữa $N$ vật thể với độ phức tạp $O(N^2)$. Trên GPU, ta áp dụng chiến lược phân rã dữ liệu (data parallelism) theo hướng **1 luồng phụ trách 1 vật thể đích (target body)**:

* **Tổ chức phân cấp luồng (Thread Hierarchy):**
  * Tập $N$ vật thể được chia thành các khối **1D Thread Block**, mỗi block có kích thước $B$ luồng (`blockDim.x = B`, với các giá trị khảo sát $B \in \{64, 128, 256, 512\}$).
  * Số lượng block trong lưới (**1D Grid**) được tính bằng công thức làm tròn lên để bao phủ toàn bộ $N$ vật thể:
    $$\text{gridDim.x} = \left\lceil \frac{N}{B} \right\rceil = \frac{N + B - 1}{B}$$
* **Ánh xạ chỉ số toàn cục:** Mỗi luồng xác định chỉ số vật thể $i$ mà nó chịu trách nhiệm tính gia tốc thông qua công thức:
  $$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
* **Tính chất không xung đột ghi (Race-free):** Vì mỗi luồng $i$ chỉ đọc dữ liệu của toàn bộ $N$ vật thể nguồn và ghi kết quả gia tốc $(a_x, a_y, a_z)$ vào đúng phần tử thứ $i$ của chính nó, các luồng hoàn toàn độc lập khi ghi dữ liệu và không cần dùng các phép toán nguyên tử (`atomicAdd`).

---

## 4.3. Kernel Naive và nút thắt cổ chai bộ nhớ toàn cục

### Mã giả thuật toán (Naive Kernel)

Trong cách tiếp cận trực tiếp (Naive), mỗi luồng tải tọa độ vật thể $i$ của chính nó vào thanh ghi, sau đó duyệt vòng lặp qua tất cả $N$ vật thể $j$ trực tiếp từ **Global Memory**:

```cuda
__global__ void compute_forces_naive(const float4* pos, float3* acc, int N, float eps2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 myPos = pos[i]; // Đọc từ Global Memory vào Register
    float3 a = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < N; j++) {
        float4 otherPos = pos[j]; // Đọc trực tiếp từ Global Memory mỗi vòng lặp
        float dx = otherPos.x - myPos.x;
        float dy = otherPos.y - myPos.y;
        float dz = otherPos.z - myPos.z;
        float distSqr = dx*dx + dy*dy + dz*dz + eps2;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float s = otherPos.w * invDist3; // w lưu khối lượng m_j
        a.x += dx * s;
        a.y += dy * s;
        a.z += dz * s;
    }
    acc[i] = a;
}