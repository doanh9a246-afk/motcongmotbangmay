# Chương 1. Giới thiệu

*Người viết: Nguyễn Quốc Doanh — bản nháp*

## 1.1. Đặt vấn đề

Bài toán N-body mô phỏng chuyển động của N vật thể tương tác với nhau qua một lực tầm xa, điển hình là lực hấp dẫn Newton. Mỗi vật thể chịu lực hút từ tất cả N − 1 vật thể còn lại, nên để cập nhật trạng thái hệ sau một bước thời gian theo cách trực tiếp (*all-pairs*) cần tính N² tương tác. Với N = 10⁵ vật thể, một bước đã cần khoảng 10¹⁰ tương tác, tương đương 2·10¹¹ phép tính dấu phẩy động (theo quy ước 20 FLOP/tương tác của GPU Gems 3). Một mô phỏng có ý nghĩa vật lý thường cần hàng nghìn đến hàng triệu bước, nên trên một lõi CPU thời gian tính có thể lên tới hàng ngày hoặc hàng tuần.

Mặt khác, các tương tác trong cùng một bước **hoàn toàn độc lập** với nhau: gia tốc của vật thể *i* chỉ phụ thuộc vào vị trí của các vật thể ở đầu bước, không phụ thuộc gia tốc của vật thể khác. Vì vậy N-body là bài toán *song song dữ liệu* (data-parallel) điển hình, có cường độ tính toán cao (O(N²) phép tính trên O(N) dữ liệu). Đây là loại bài toán phù hợp với cả CPU đa lõi lẫn GPU.

## 1.2. Ứng dụng của mô phỏng N-body

- **Vật lý thiên văn:** sự tiến hóa của cụm sao, va chạm thiên hà, hình thành cấu trúc vũ trụ quy mô lớn (các mô phỏng như Millennium Simulation dùng tới hàng tỷ hạt).
- **Động lực học phân tử:** lực Coulomb và Lennard-Jones giữa các nguyên tử có cùng dạng "mỗi hạt tương tác với mọi hạt khác".
- **Cơ học thiên thể:** quỹ đạo hành tinh, tiểu hành tinh, tàu vũ trụ.
- **Đồ họa máy tính và trò chơi:** hệ hạt (particle system), mô phỏng chất lỏng theo phương pháp SPH.
- **Bài toán chuẩn (benchmark) cho tính toán hiệu năng cao:** N-body all-pairs là ví dụ kinh điển để đánh giá sức mạnh tính toán của phần cứng song song (GPU Gems 3, Chương 31).

## 1.3. Mục tiêu đề tài

1. Cài đặt thuật toán N-body all-pairs với tích phân Euler đối xứng (symplectic Euler), khởi tạo theo mô hình cụm sao Plummer.
2. Xây dựng **bốn phiên bản** có cùng đầu vào và đầu ra:
   - tuần tự (1 luồng CPU) làm mốc so sánh;
   - song song trên CPU đa lõi bằng **OpenMP**;
   - song song trên GPU bằng **CUDA** theo hai cách: *naive* (đọc trực tiếp từ global memory) và *tiled* (dùng shared memory theo GPU Gems 3).
3. Kiểm tra tính đúng đắn: các phiên bản cho cùng kết quả (sai số làm tròn), năng lượng toàn phần được bảo toàn.
4. Đo và phân tích hiệu năng: thời gian theo N, speedup và hiệu suất theo số luồng, ảnh hưởng của kích thước block/tile, so sánh CPU với GPU.

## 1.4. Phạm vi và giới hạn

- Chỉ xét thuật toán trực tiếp O(N²); các thuật toán xấp xỉ như Barnes–Hut O(N log N) hay Fast Multipole Method chỉ được nêu ở phần hướng phát triển.
- Dùng số thực độ chính xác đơn (`float`), giống GPU Gems 3.
- Phần cứng thử nghiệm:
  - CPU: laptop Intel Core i5-12500H (12 lõi gồm 4 lõi P và 8 lõi E, 16 luồng), MinGW gcc 12.1;
  - GPU: NVIDIA Tesla T4 trên Google Colab (40 SM, kiến trúc Turing, sm_75).
- Số liệu CPU và GPU được đo trên hai máy khác nhau. Vì vậy phép so sánh CPU–GPU ở Chương 5 mang tính tham khảo về bậc độ lớn, không phải so sánh trên cùng một hệ thống.

## 1.5. Bố cục báo cáo

- **Chương 1** — Giới thiệu (chương này).
- **Chương 2** — Cơ sở lý thuyết và thuật toán tuần tự: định luật hấp dẫn, softening, tích phân symplectic Euler, mô hình Plummer, phân tích độ phức tạp và định luật Amdahl.
- **Chương 3** — Song song hóa bằng OpenMP: mô hình fork–join, chia vòng lặp, vector hóa SIMD, kết quả đo speedup và hiệu suất.
- **Chương 4** — Song song hóa bằng CUDA: kiến trúc GPU, kernel naive và tiled, shared memory, ảnh hưởng của block size, chi phí truyền dữ liệu.
- **Chương 5** — So sánh tổng hợp, kết luận và hướng phát triển.
