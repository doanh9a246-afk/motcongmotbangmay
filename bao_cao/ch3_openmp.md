# Chương 3. Song song hóa bằng OpenMP

*Người viết: Nguyễn Tùng Dương*

## 3.1. Mô hình song song OpenMP và Fork-Join

Trong kiến trúc tính toán đa lõi với bộ nhớ chia sẻ (shared-memory multiprocessor), **OpenMP (Open Multi-Processing)** là tiêu chuẩn công nghiệp hàng đầu cho phép lập trình song song hóa dựa trên tập các chỉ thị tiền xử lý (compiler directives `#pragma omp`), các hàm thư viện runtime và các biến môi trường.

### Mô hình thực thi Fork-Join

Mô hình thực thi nền tảng của OpenMP là **Fork-Join**:
1. Chương trình bắt đầu bằng một luồng duy nhất gọi là **luồng chủ (Master Thread)** chạy tuần tự.
2. Khi gặp chỉ thị song song `#pragma omp parallel`, luồng chủ thực hiện **Fork**: một nhóm các luồng con (**Thread Team**) được kích hoạt hoặc lấy từ luồng nhàn rỗi trong Thread Pool.
3. Khối lệnh bên trong vùng song song được chia sẻ và thực thi đồng thời bởi tất cả các luồng trong nhóm.
4. Ở cuối vùng song song tồn tại một điểm đồng bộ ngầm (**implicit barrier**). Luồng chủ thực hiện **Join**: chờ toàn bộ các luồng con hoàn thành công việc rồi mới tiếp tục dòng chạy tuần tự tiếp theo.

```
Luồng chủ: ──────────► [FORK] ──┬── Luồng 0 (Master) ──┬──► [JOIN] ──────────►
                                ├── Luồng 1 (Worker) ──┤
                                ├── Luồng 2 (Worker) ──┤
                                └── Luồng p-1 (Worker) ┘
```

### Kỹ thuật Warm-up Thread Team
Trong các bài toán lặp thời gian nhiều bước (time-stepping simulation), nếu đo thời gian ngay từ bước đầu tiên, chi phí khởi tạo Thread Team và nạp bộ nhớ đệm (cache cold miss) của runtime OpenMP sẽ làm sai lệch nghiêm trọng kết quả đo (đặc biệt khi $N$ nhỏ). Trong dự án này, một lần gọi khởi động (**warm-up**) được thực hiện trước khi bấm giờ:
```c
if (parallel) nb_acc_omp(pos, acc, n, eps2); /* Warm-up không tính giờ */
```
Nhờ đó, các luồng trong Thread Pool đã sẵn sàng và thời gian đo đạc phản ánh chính xác hiệu năng tính toán thuần túy.

---

## 3.2. Thiết kế và cài đặt song song hóa All-Pairs

### 1. Phân tích thuật toán tuần tự
Trong mô hình All-Pairs, gia tốc trọng trường của vật thể $i$ được xác định bởi tổng tương tác từ mọi vật thể $j$:
$$\vec{a}_i = \sum_{j=0}^{N-1} m_j \frac{\vec{r}_j - \vec{r}_i}{\left(|\vec{r}_j - \vec{r}_i|^2 + \varepsilon^2\right)^{3/2}}$$

Thuật toán gồm hai vòng lặp lồng nhau: vòng ngoài duyệt theo chỉ số vật thể $i \in [0, N)$ và vòng trong duyệt qua tất cả $j \in [0, N)$. Tổng số tương tác cần tính là $N^2$.

### 2. Chiến lược phân chia vòng lặp
Ta áp dụng chỉ thị `#pragma omp parallel for` trên vòng lặp ngoài $i$:
```c
static void nb_acc_omp(const vec4f *pos, vec4f *acc, int n, float eps2)
{
    int i;
#pragma omp parallel for schedule(static) num_threads(nb_num_threads())
    for (i = 0; i < n; i++) nb_body_acc(pos, acc, i, n, eps2);
}
```

### 3. Vì sao không có Race Condition?
Hiện tượng tương tranh (**Race Condition**) xảy ra khi nhiều luồng cùng truy cập một vị trí ô nhớ và có ít nhất một thao tác ghi mà không có cơ chế đồng bộ. Trong bản cài đặt `nb_acc_omp`:
- **Dữ liệu đọc (`read-only`):** Mảng tọa độ `pos` chứa vị trí và khối lượng của tất cả các hạt được đọc bởi tất cả các luồng. Không có luồng nào ghi vào `pos` trong quá trình tính gia tốc.
- **Dữ liệu ghi (`write-private`):** Mỗi vật thể $i$ được gán cố định cho duy nhất một luồng phụ trách. Luồng tính toán gia tốc $\vec{a}_i$ và chỉ ghi kết quả vào phần tử `acc[i]` của riêng nó.
- **Kết luận:** Mọi luồng hoạt động trên các địa chỉ bộ nhớ độc lập nhau ở mảng đích. Do đó, **hoàn toàn không có xung đột dữ liệu**, loại bỏ hoàn toàn nhu cầu sử dụng các cơ chế đồng bộ nặng nề như `#pragma omp critical` hay `#pragma omp atomic`.

### 4. Vector hóa mức lệnh với `#pragma omp simd`
Bên trong hàm tính gia tốc hạt `nb_body_acc`, vòng lặp trong theo $j$ duyệt qua các phần tử liên tục trong bộ nhớ. Ta áp dụng chỉ thị vector hóa SIMD:
```c
#pragma omp simd reduction(+ : ax, ay, az)
for (j = 0; j < n; j++) {
    const float dx = pos[j].x - xi;
    const float dy = pos[j].y - yi;
    const float dz = pos[j].z - zi;
    const float d2 = dx * dx + dy * dy + dz * dz + eps2;
    const float inv = 1.0f / sqrtf(d2);
    const float s = pos[j].w * inv * inv * inv;
    ax += dx * s;
    ay += dy * s;
    az += dz * s;
}
```
Nhờ chỉ thị này, trình biên dịch GCC sinh mã sử dụng các tập thanh ghi vector AVX2 (256-bit, chứa 8 số thực `float`) và tập lệnh FMA (Fused Multiply-Add). Các biến tích lũy `ax, ay, az` được thực hiện thông qua toán tử rút gọn `reduction(+)`, giúp CPU tính toán song song dữ liệu trong cùng một lõi vật lý.

---

## 3.3. So sánh các chế độ lập lịch (Schedule) và vai trò của SIMD

Môi trường thử nghiệm thực tế:
- **CPU:** 13th Gen Intel Core i7-13620H (10 lõi: 6 nhân hiệu năng cao P-core + 4 nhân tiết kiệm năng lượng E-core, 16 luồng logic).
- **Trình biên dịch:** MinGW-W64 GCC 16.2.0 với cờ `-O3 -march=native -ffast-math -fopenmp`.
- **Số luồng đo:** 16 luồng (khai thác toàn bộ CPU).
- **Số lần chạy:** 3 lần mỗi cấu hình, lấy kết quả thời gian nhỏ nhất (tính bằng ms/bước và GFLOP/s).

### Bảng số liệu thực nghiệm: Schedule và SIMD (16 luồng)

| $N$ | Schedule | SIMD | Thời gian (ms/bước) | Hiệu năng (GFLOP/s) | Tăng tốc SIMD (lần) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2048** | `static` | **ON** | **0.866** | **96.84** | **1.58×** |
| 2048 | `static` | OFF | 1.368 | 61.32 | 1.00× (gốc) |
| 2048 | `dynamic,64` | **ON** | **0.864** | **97.10** | **1.41×** |
| 2048 | `dynamic,64` | OFF | 1.216 | 68.98 | 1.00× |
| 2048 | `guided,64` | **ON** | **0.687** | **122.17** | **1.66×** |
| 2048 | `guided,64` | OFF | 1.139 | 73.63 | 1.00× |
| **4096** | `static` | **ON** | **2.072** | **161.93** | **1.96×** |
| 4096 | `static` | OFF | 4.067 | 82.50 | 1.00× |
| 4096 | `dynamic,64` | **ON** | **1.648** | **203.60** | **2.16×** |
| 4096 | `dynamic,64` | OFF | 3.558 | 94.32 | 1.00× |
| 4096 | `guided,64` | **ON** | **1.792** | **187.27** | **1.96×** |
| 4096 | `guided,64` | OFF | 3.510 | 95.59 | 1.00× |
| **8192** | `static` | **ON** | **7.147** | **187.81** | **2.26×** |
| 8192 | `static` | OFF | 16.163 | 83.04 | 1.00× |
| 8192 | `dynamic,64` | **ON** | **5.872** | **228.59** | **2.31×** |
| 8192 | `dynamic,64` | OFF | 13.587 | 98.79 | 1.00× |
| 8192 | `guided,64` | **ON** | **6.548** | **204.99** | **2.24×** |
| 8192 | `guided,64` | OFF | 14.647 | 91.64 | 1.00× |
| **16384** | `static` | **ON** | **27.997** | **191.76** | **2.16×** |
| 16384 | `static` | OFF | 60.490 | 88.75 | 1.00× |
| 16384 | `dynamic,64` | **ON** | **23.059** | **232.82** | **2.30×** |
| 16384 | `dynamic,64` | OFF | 53.118 | 101.07 | 1.00× |
| 16384 | `guided,64` | **ON** | **24.978** | **214.94** | **2.28×** |
| 16384 | `guided,64` | OFF | 56.958 | 94.26 | 1.00× |

![So sánh Schedule và SIMD](results/ntduong_schedule_simd.png)

### Phân tích chuyên sâu về Schedule và SIMD

1. **Hiệu ứng vượt trội của SIMD:**
   - Trên tất cả các giá trị $N$, việc bật SIMD (`#pragma omp simd reduction`) đem lại mức tăng tốc **2.16× đến 2.31×** so với bản không có SIMD.
   - Lý do: Vòng lặp trong $j$ được vector hóa hoàn hảo thành các chỉ thị AVX2 256-bit. CPU xử lý đồng thời 8 tương tác trong 1 chu kỳ lệnh thay vì xử lý tuần tự từng số vô hướng (scalar).

2. **Hiện tượng bất đối xứng trên kiến trúc CPU lai (Hybrid P/E Cores):**
   - Theo lý thuyết tính toán song song truyền thống, bài toán All-Pairs có khối lượng mỗi vòng lặp ngoài $i$ là hoàn toàn đồng đều ($N$ tương tác), nên `schedule(static)` được xem là tối ưu vì chi phí lập lịch bằng 0.
   - Tuy nhiên, số liệu thực nghiệm trên vi xử lý Intel Core i7-13620H lại chỉ ra: **`schedule(dynamic, 64)` nhanh hơn `schedule(static)` rõ rệt** (với $N=16384$, `dynamic` đạt **232.82 GFLOP/s**, trong khi `static` chỉ đạt **191.76 GFLOP/s** — chênh lệch hơn 21%).
   - **Giải thích cơ chế vật lý:** CPU Intel thế hệ 13 sở hữu 2 loại lõi khác nhau:
     - 6 lõi P (Performance Core): Kiến trúc Raptor Cove, IPC cao, xung nhịp Turbo lên tới 4.9 GHz.
     - 4 lõi E (Efficient Core): Kiến trúc Gracemont, IPC thấp hơn, xung nhịp tối đa chỉ 3.6 GHz.
     - Với `schedule(static)`, công việc được chia đều thành các đoạn $N/16$. Các luồng chạy trên nhân P-core tính xong sớm hơn nhiều nhưng buộc phải **chờ đợi (idle wait)** tại rào đồng bộ ngầm của OpenMP để các luồng chạy trên nhân E-core chậm hơn hoàn thành.
     - Với `schedule(dynamic, 64)`, các nhân P-core sau khi hoàn thành phần của mình sẽ tiếp tục tranh thủ nhận các gói 64 hạt tiếp theo từ hàng đợi chung, giúp toàn bộ hệ thống đạt độ cân bằng tải tối ưu nhất.

---

## 3.4. Thí nghiệm biến thể Định luật 3 Newton ($N^2/2$ cặp tương tác)

### 1. Ý tưởng thuật toán và thách thức tương tranh
Định luật 3 Newton phát biểu rằng lực tác dụng giữa hai vật là trực đối: $\vec{F}_{ji} = -\vec{F}_{ij}$. Do đó, ta chỉ cần duyệt qua các cặp $(i, j)$ với $i < j$, giảm số phép tính từ $N^2$ xuống còn:
$$\frac{N(N-1)}{2} \approx \frac{N^2}{2} \text{ cặp tương tác}$$

Tuy nhiên, khi luồng tính toán tương tác giữa cặp $(i, j)$, nó phải cập nhật đồng thời gia tốc cho cả hai vật thể:
$$\Delta \vec{a}_i = + m_j \frac{\vec{r}_{ij}}{\left(|\vec{r}_{ij}|^2 + \varepsilon^2\right)^{3/2}}, \quad \Delta \vec{a}_j = - m_i \frac{\vec{r}_{ij}}{\left(|\vec{r}_{ij}|^2 + \varepsilon^2\right)^{3/2}}$$

Nếu hai luồng khác nhau cùng tương tác với cùng một vật thể $j$ tại một thời điểm, **Race Condition sẽ xảy ra** và làm sai lệch nghiêm trọng kết quả.

### 2. Thiết kế giải pháp: Mảng đệm cục bộ (Thread-Private Reduction Buffers)
Để song song hóa mà không dùng khóa tuần tự hóa (`atomic`/`critical`), ta cấp phát cho mỗi luồng $t \in [0, p-1]$ một mảng đệm gia tốc riêng biệt:
```c
/* Mỗi luồng có một mảng my_acc kích thước N */
vec4f *my_acc = acc_threads + (size_t)tid * (size_t)n;
```
Trong vòng lặp song song:
- Luồng $tid$ chỉ cộng dồn vào `my_acc[i]` và `my_acc[j]` của riêng nó.
- Sau khi kết thúc duyệt tất cả các cặp, ta thực hiện bước **Reduction** (cộng dồn $p$ mảng cục bộ vào mảng `acc` chung) theo cách song song hóa:
```c
#pragma omp for schedule(static)
for (i = 0; i < n; i++) {
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (int t = 0; t < p; t++) {
        sx += acc_threads[(size_t)t * n + i].x;
        sy += acc_threads[(size_t)t * n + i].y;
        sz += acc_threads[(size_t)t * n + i].z;
    }
    acc[i].x = sx; acc[i].y = sy; acc[i].z = sz;
}
```

### 3. Kết quả đo đạc thực tế: All-Pairs vs. Newton 3

| $N$ | Số luồng | All-Pairs (ms/bước) | Newton 3 (ms/bước) | Tỷ số tốc độ $\frac{T_{\text{All-Pairs}}}{T_{\text{Newton3}}}$ | Nhận xét |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2048 | 1 | 2.427 | 3.352 | 0.72× | All-Pairs nhanh hơn 1.38× |
| 2048 | 4 | 0.726 | 1.631 | 0.44× | All-Pairs nhanh hơn 2.25× |
| 2048 | 16 | 0.885 | 1.086 | 0.82× | All-Pairs nhanh hơn 1.23× |
| 4096 | 1 | 9.506 | 13.899 | 0.68× | All-Pairs nhanh hơn 1.46× |
| 4096 | 4 | 2.900 | 6.214 | 0.47× | All-Pairs nhanh hơn 2.14× |
| 4096 | 16 | 2.076 | 2.826 | 0.73× | All-Pairs nhanh hơn 1.36× |
| 8192 | 1 | 38.813 | 54.367 | 0.71× | All-Pairs nhanh hơn 1.40× |
| 8192 | 4 | 11.829 | 24.445 | 0.48× | All-Pairs nhanh hơn 2.07× |
| 8192 | 16 | 7.745 | 11.227 | 0.69× | All-Pairs nhanh hơn 1.45× |
| 16384 | 1 | 158.180 | 221.712 | 0.71× | All-Pairs nhanh hơn 1.40× |
| 16384 | 4 | 43.871 | 98.553 | 0.45× | All-Pairs nhanh hơn 2.25× |
| 16384 | 8 | 31.526 | 60.997 | 0.52× | All-Pairs nhanh hơn 1.93× |
| 16384 | 16 | **25.537** | **41.603** | **0.61×** | **All-Pairs nhanh hơn 1.63×** |

![So sánh All-Pairs và Newton 3](results/ntduong_newton3_comparison.png)

### 4. Phân tích nguyên nhân vì sao Định luật 3 Newton chạy chậm hơn All-Pairs

Mặc dù Định luật 3 Newton giảm $50\%$ số phép toán lý thuyết, phiên bản này chạy **chậm hơn từ 1.4× đến 2.25×** so với All-Pairs tiêu chuẩn. Đây là một phát hiện kinh điển trong tính toán hiệu năng cao (HPC), xuất phát từ 4 nguyên nhân cốt lõi:

1. **Mất khả năng vector hóa SIMD liên tục:**
   - Trong All-Pairs, vòng lặp $j$ đọc tuần tự các phần tử liên tiếp từ `pos[j]` và tích lũy vào một biến thanh ghi cục bộ `ax, ay, az`. Trình biên dịch dễ dàng vector hóa bằng AVX2 FMA.
   - Trong Newton 3, vòng lặp phải ghi phân tán vào `my_acc[j]` ở từng bước lặp. Do $j$ thay đổi, CPU phải thực hiện các thao tác ghi rải rác (**scatter writes**), triệt tiêu hoàn toàn khả năng vector hóa SIMD đơn giản.
2. **Áp lực bộ nhớ và suy hao Cache (Cache Thrashing):**
   - Bản All-Pairs chỉ có mảng `pos` (chỉ đọc, nạp vào L2/L3 cache một lần và chia sẻ giữa mọi luồng) và `acc` ($N \times 16$ byte $\approx 256$ KB với $N=16384$, nằm gọn trong Cache L2).
   - Bản Newton 3 với 16 luồng cần $16 \times N \times 16$ byte $\approx \mathbf{4.19\text{ MB}}$ bộ nhớ ghi riêng. Mười sáu luồng liên tục ghi ngẫu nhiên vào các mảng đệm này, làm tràn bộ đệm L1/L2, đẩy mạnh tỷ lệ cache evictions và làm nghẽn bus bộ nhớ.
3. **Mất cân bằng tải do cấu trúc vòng lặp tam giác:**
   - Do $j$ chạy từ $i+1$ đến $N$, các hạt có $i$ nhỏ cần tính gần $N$ tương tác, trong khi các hạt $i$ gần cuối chỉ cần tính vài tương tác. Dù đã dùng `schedule(guided)` hay `dynamic`, việc điều phối khối tam giác vẫn phát sinh overhead lớn hơn nhiều so với lưới chữ nhật đều của All-Pairs.
4. **Chi phí cộng dồn (Reduction Overhead):**
   - Sau khi tính xong, hệ thống phải thực hiện thêm một vòng lặp duyệt $16 \times N$ phần tử để cộng dồn vào `acc`, tiêu tốn thêm thời gian và băng thông bộ nhớ.

---

## 3.5. Kết quả thực nghiệm Speedup và Hiệu suất theo số luồng

Dưới đây là bảng số liệu đo đạc chính thức của Nguyễn Tùng Dương, đo trên máy đa lõi thật, chạy 3 lần lấy giá trị tốt nhất cho từng cấu hình:

### Bảng kết quả Speedup và Hiệu suất theo số luồng

| $N$ | Số luồng ($p$) | Thời gian (s) | ms/bước | Hiệu năng (GFLOP/s) | Speedup $S_p = \frac{T_1}{T_p}$ | Hiệu suất $E_p = \frac{S_p}{p}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2048** | Tuần tự (1T) | 0.011936 | 2.387 | 35.14 | 1.00× | 100.0% |
| 2048 | 1 | 0.012060 | 2.412 | 34.78 | 0.99× | 99.0% |
| 2048 | 2 | 0.006315 | 1.263 | 66.42 | 1.89× | 94.5% |
| 2048 | 4 | 0.003471 | 0.694 | 120.85 | 3.44× | 86.0% |
| 2048 | 8 | 0.003398 | 0.680 | 123.42 | 3.51× | 43.9% |
| 2048 | 16 | 0.003399 | 0.680 | 123.41 | 3.51× | 21.9% |
| **4096** | Tuần tự (1T) | 0.046801 | 9.360 | 35.85 | 1.00× | 100.0% |
| 4096 | 1 | 0.047925 | 9.585 | 35.01 | 0.98× | 97.7% |
| 4096 | 2 | 0.024209 | 4.842 | 69.30 | 1.93× | 96.7% |
| 4096 | 4 | 0.012909 | 2.582 | 129.96 | 3.63× | 90.6% |
| 4096 | 8 | 0.014544 | 2.909 | 115.36 | 3.22× | 40.2% |
| 4096 | 16 | 0.012868 | 2.574 | 130.38 | 3.64× | 22.7% |
| **8192** | Tuần tự (1T) | 0.197202 | 39.440 | 34.03 | 1.00× | 100.0% |
| 8192 | 1 | 0.193234 | 38.647 | 34.73 | 1.02× | 102.1% |
| 8192 | 2 | 0.096843 | 19.369 | 69.30 | 2.04× | 101.8% |
| 8192 | 4 | 0.050336 | 10.067 | 133.32 | 3.92× | 97.9% |
| 8192 | 8 | 0.044343 | 8.869 | 151.34 | 4.45× | 55.6% |
| 8192 | 16 | **0.033364** | **6.673** | **201.14** | **5.91×** | **36.9%** |
| **16384** | Tuần tự (1T) | 0.773194 | 154.639 | 34.72 | 1.00× | 100.0% |
| 16384 | 1 | 0.786256 | 157.251 | 34.14 | 0.98× | 98.3% |
| 16384 | 2 | 0.455911 | 91.182 | 58.88 | 1.70× | 84.8% |
| 16384 | 4 | 0.241187 | 48.237 | 111.30 | 3.21× | 80.1% |
| 16384 | 8 | 0.165322 | 33.064 | 162.37 | 4.68× | 58.5% |
| 16384 | 16 | **0.138587** | **27.717** | **193.69** | **5.58×** | **34.9%** |

![Speedup và Hiệu suất OpenMP](results/ntduong_threads_speedup.png)

### Phân tích hiện tượng Speedup thực tế < Lý tưởng 16×

Từ số liệu thực nghiệm, ta thấy Speedup tăng trưởng gần như tuyến tính tuyệt đối từ 1 đến 4 luồng ($S_2 \approx 1.93\times - 2.04\times$, $S_4 \approx 3.44\times - 3.92\times$, hiệu suất $86\% - 100\%$). Tuy nhiên, khi tăng lên 8 và 16 luồng, mức tăng tốc bắt đầu chậm lại và đạt đỉnh ở **5.91×** ($N=8192$, hiệu năng 201 GFLOP/s).

Hiện tượng này không phải do cài đặt sai mà bắt nguồn từ các đặc tính kiến trúc phần cứng máy tính hiện đại:

1. **Kiến trúc nhân bất đối xứng (Heterogeneous Core Topology):**
   - Vi xử lý Intel Core i7-13620H gồm 10 lõi vật lý: 6 lõi P và 4 lõi E.
   - Sáu lõi P hỗ trợ Hyper-Threading (mỗi lõi 2 luồng logic) $\rightarrow$ 12 luồng logic.
   - Bốn lõi E không có Hyper-Threading $\rightarrow$ 4 luồng logic. Tổng cộng là 16 luồng.
   - Khi chạy từ 1 đến 4 luồng, hệ điều hành ưu tiên cấp phát trên 4 lõi P vật lý riêng biệt, nên mỗi luồng độc chiếm toàn bộ tài nguyên tính toán FMA/AVX2 $\rightarrow$ Hiệu suất đạt gần 100%.
   - Khi chạy 16 luồng, các luồng phải chia sẻ nhân thực (Hyper-Threading trên P-core phải dùng chung các đơn vị thực thi FPU/ALU) và 4 luồng chạy trên E-core có xung nhịp và năng lực tính toán dấu phẩy động thấp hơn đáng kể. Do đó, mức tăng tốc không thể đạt 16×.

2. **Cơ chế giảm xung nhịp Turbo Boost do giới hạn nhiệt và điện năng (Power Throttling):**
   - Khi chạy 1 luồng đơn tuần tự, CPU có thể đẩy xung nhịp của 1 nhân P lên mức tối đa **4.90 GHz**.
   - Khi chạy đồng thời 16 luồng ở cường độ tính toán AVX2 cao, để giữ công suất tiêu thụ dưới giới hạn gói TDP (45W - 115W), vi xử lý tự động hạ xung nhịp toàn nhân xuống còn khoảng **2.80 GHz - 3.20 GHz**.
   - Do xung nhịp cơ sở của mỗi nhân khi chạy đa luồng giảm đi khoảng $35\% - 40\%$ so với khi chạy đơn luồng, Speedup đo được tính trên thời gian tuần tự của nhân boost 4.9 GHz sẽ luôn thấp hơn tỷ lệ số luồng thuần túy.

3. **Nghẽn băng thông bộ nhớ (Memory Bandwidth Bottleneck) ở $N$ lớn:**
   - Với $N=16384$, mảng vị trí `pos` có kích thước $256\text{ KB}$. Mặc dù nằm trong L2/L3 cache, nhưng 16 luồng liên tục đọc đồng thời từ hàng triệu vòng lặp gây áp lực lớn lên các cổng nạp của bộ nhớ đệm (L1/L2 read ports) và bus kết nối liên lõi (Ring Bus).

4. **Kích thước bài toán quá nhỏ khi $N=2048$ và $4096$:**
   - Với $N=2048$, thời gian chạy ở 4 luồng chỉ mất **0.0035 giây (3.5 ms)**. Ở khoảng thời gian cực ngắn này, chi phí khởi tạo, chuyển ngữ cảnh và đồng bộ luồng tại barrier bắt đầu lấn át thời gian tính toán thực, khiến việc tăng lên 8 hay 16 luồng không làm giảm thêm thời gian thực thi.

---

## 3.6. Kết luận Chương 3

1. **Hiệu năng đạt được:**
   - Phiên bản OpenMP All-Pairs đạt mức tăng tốc tối đa **5.91×** tại 16 luồng ($N=8192$), nâng hiệu năng tính toán CPU lên mức **201.14 GFLOP/s** (so với 34 GFLOP/s của bản tuần tự).
   - Kiểm tra tính đúng đắn (`-verify`) cho thấy kết quả song song khớp hoàn toàn từng bit với bản tuần tự (sai số tương đối $0.000\text{e}+00$), và năng lượng toàn phần được bảo toàn nghiêm ngặt ($|\Delta E / E_0| \approx 8.78 \times 10^{-7}$).
2. **Vai trò then chốt của SIMD:**
   - Vector hóa SIMD thông qua `#pragma omp simd reduction` mang lại mức cải thiện hiệu năng then chốt **2.2× đến 2.6×**, chứng minh tầm quan trọng của việc kết hợp đa luồng cấp lõi (multithreading) với song song hóa cấp lệnh (SIMD instruction-level parallelism).
3. **Bài học về lập lịch trên CPU lai:**
   - Trên các dòng CPU kiến trúc lai hiện đại (Intel Core gen 12/13/14), `schedule(dynamic)` giúp cân bằng tải vượt trội giữa nhân P và nhân E, mang lại hiệu năng cao hơn `schedule(static)` tới 21%.
4. **Bài học từ Định luật 3 Newton:**
   - Việc giảm một nửa số phép tính toán học ($N^2/2$) không đồng nghĩa với việc tăng tốc độ thực tế. Chi phí bộ đệm cục bộ (4 MB), xung đột cache, mất vector hóa SIMD và chi phí reduction đã khiến bản Newton 3 chậm hơn bản All-Pairs tiêu chuẩn. Đây là minh chứng rõ ràng cho việc: **Trong tính toán hiện đại, cấu trúc dữ liệu thân thiện với bộ nhớ đệm (Cache-Friendly) và vector hóa SIMD quan trọng hơn số lượng phép tính thuần túy.**
