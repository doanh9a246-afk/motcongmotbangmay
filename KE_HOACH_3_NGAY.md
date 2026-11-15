# Kế hoạch nước rút 3 ngày (15/11 → 17/11, nộp 18/11)

> Đọc hết file này một lần (5 phút). Sau đó chỉ cần nhìn **bảng tiếp sức** bên dưới: mình đang ở bước nào, đang chờ ai, xong thì báo ai.

## Quy tắc tiếp sức

1. Mỗi bước có **một người chịu trách nhiệm**, một **hạn giờ**, và một điều kiện **"Xong khi"** rõ ràng.
2. Xong bước nào thì **nhắn ngay vào nhóm**: `✅ Bước X xong — @người tiếp theo đến lượt bạn` kèm link PR.
3. **Quá hạn 1 tiếng** mà chưa thấy báo xong: người đứng ở bước sau **được quyền giục**. Quá hạn 3 tiếng thì Doanh kích hoạt **phương án dự phòng** của bước đó.
4. Bị kẹt quá 30 phút thì hỏi nhóm luôn, không ngồi im.
5. Không push thẳng `main`. Mỗi việc làm trên nhánh riêng, mở PR, chờ CI xanh và 1 approve.

## Hiện trạng (sáng 15/11)

| PR | Của ai | Nội dung | Vấn đề |
|---|---|---|---|
| #2 | Doanh | CI tự động | CI xanh, **cần 1 người approve** (Doanh không tự approve được) |
| #3 | V.V. Dương | Ch.2 + năng lượng | **Trùng 100% với #4** → đóng |
| #4 | V.V. Dương | Ch.2 + năng lượng | Chờ CI (sau khi #2 merge) |
| #1 | N.T. Dương | Ch.3 + đo OpenMP | Sửa code chung (`nbody_common.h`, `nbody_cpu.c`, `nbody_cpu_kernels.h`) → cần review kỹ, rebase |
| — | Đạt | **Chưa có gì**: số liệu GPU, Ch.4 | **Đang chặn Ch.5 và slide kết luận** |

---

## Bảng tiếp sức

| Bước | Người | Hạn | Chờ bước | Xong thì báo |
|---|---|---|---|---|
| 1 | **V.V. Dương** | 15/11 10:00 | — | Doanh |
| 2 | **Doanh** | 15/11 12:00 | 1 | N.T. Dương, V.V. Dương |
| 3 | **Đạt** ⚠️ | 15/11 22:00 | — (làm song song ngay) | Doanh |
| 4 | **N.T. Dương** | 15/11 18:00 | 2 | Doanh |
| 5 | **Doanh** | 15/11 21:00 | 4 | cả nhóm |
| 6 | **V.V. Dương** | 16/11 12:00 | — | Doanh |
| 7 | **N.T. Dương** | 16/11 12:00 | — | Doanh |
| 8 | **Doanh** | 16/11 14:00 | 3 | Đạt |
| 9 | **Đạt** ⚠️ | 16/11 20:00 | 3 | V.V. Dương, Doanh |
| 10 | **V.V. Dương** | 16/11 23:00 | 9 | Đạt |
| 11 | **Doanh** | 16/11 23:00 | 8 | cả nhóm |
| 12 | **Đạt** | 17/11 12:00 | 9, 10 | Doanh |
| 13 | **Doanh** | 17/11 18:00 | 5, 11, 12 | cả nhóm |
| 14 | **Cả nhóm** | 17/11 20:00–22:00 | 13 | — |
| 15 | **Doanh** | 18/11 sáng | 14 | — |

Thứ tự ưu tiên khi có xung đột: **Bước 3 (số GPU của Đạt) > Bước 9 (Ch.4) > mọi việc khác.**

---

## Chi tiết từng bước

### Bước 1 — V.V. Dương: approve PR #2, đóng PR #3 (15/11, 10:00)
- Vào PR #2 → tab *Files changed* → **Review changes** → **Approve**.
- Vào PR #3 → comment `Trùng với #4, đóng` → **Close pull request**.
- **Xong khi:** PR #2 có dấu approve xanh, PR #3 ở trạng thái Closed.
- Nhắn: `✅ Bước 1 xong — @Doanh merge #2 đi`.

### Bước 2 — Doanh: merge CI và PR #4 (15/11, 12:00)
- Merge PR #2 (CI) vào `main`.
- Vào PR #4 → **Update branch** để CI chạy → đọc lướt Ch.2 → approve → merge khi CI xanh.
- Comment vào PR #1: nhờ N.T. Dương rebase lên `main` (xem Bước 4).
- **Xong khi:** `main` có `.github/workflows/ci.yml` và `bao_cao/ch2_*.md`.
- Nhắn: `✅ Bước 2 xong — @N.T.Dương rebase PR #1`.

### Bước 3 — Đạt: lấy số liệu GPU trên Colab ⚠️ VIỆC GẤP NHẤT (15/11, 22:00)
Làm **ngay từ sáng**, không cần chờ ai.
1. Mở `notebooks/nbody_colab.ipynb` trên Colab → **Runtime → Change runtime type → T4 GPU**.
2. Chạy lần lượt các ô từ trên xuống:
   - Ô `nvidia-smi`: phải thấy **Tesla T4**. Không thấy thì đổi runtime lại.
   - Ô clone + `make all`.
   - Ô **Kiểm tra tính đúng**: cả 3 dòng phải **PASS**.
   - Ô **Benchmark GPU** (`OUT=results/gpu_t4.csv SKIP_CPU=1 ... benchmark.sh`) mất khoảng 10–20 phút. Ô này đo luôn naive/tiled, N lớn và **block size 64/128/256/512**.
3. Thêm một ô mới để đo **thời gian sao chép H2D/D2H**, dùng cho Ch.4:
   ```
   !for n in 4096 16384 65536 131072; do echo "== N=$n"; ./bin/nbody_cuda -n $n -s 10 -k tiled -noenergy | grep -E "RESULT|H2D"; done | tee results/gpu_t4_h2d.txt
   ```
4. Tải về 3 file: `results/gpu_t4.csv`, `results/gpu_t4.machine.txt`, `results/gpu_t4_h2d.txt`. Tải luôn các ảnh `.png` mà notebook vẽ ra.
5. Trên máy mình, chép 3 file vào thư mục `results/` của repo rồi chạy:
   ```
   git checkout main && git pull
   git checkout -b dat/gpu-results
   git add results/gpu_t4.csv results/gpu_t4.machine.txt results/gpu_t4_h2d.txt
   git commit -m "GPU results on Colab T4 (naive, tiled, block size, H2D/D2H)"
   git push -u origin dat/gpu-results
   ```
   Lên GitHub mở PR. **Không** dán token vào notebook.
- **Xong khi:** PR có 3 file trên và CI xanh.
- Nhắn: `✅ Bước 3 xong — @Doanh có số GPU rồi`.
- **Dự phòng:** đến **16/11 10:00** vẫn chưa có PR thì Doanh tự chạy notebook (khoảng 30 phút). Đạt chuyển sang làm Bước 9 luôn.

### Bước 4 — N.T. Dương: rebase PR #1 cho CI chạy (15/11, 18:00)
```
git checkout ntduong/openMP
git fetch origin
git rebase origin/main
make            # phải build được
./bin/nbody_cpu -n 2048 -s 5 -verify    # phải PASS
git push --force-with-lease
```
- Nếu CI đỏ: mở tab *Checks* → đọc lỗi → sửa → push lại.
- Trả lời các comment review của Doanh trên PR #1.
- **Xong khi:** PR #1 có CI xanh cả 3 mục `cpu`, `cuda`, `scripts`.
- Nhắn: `✅ Bước 4 xong — @Doanh review/merge #1`.

### Bước 5 — Doanh: review và merge PR #1 (15/11, 21:00)
- Soát phần sửa code chung: không làm đổi kết quả `-verify`, không đổi định dạng dòng `RESULT` (vì `benchmark.sh` và `plot_results.py` đang đọc dòng này).
- Approve → merge.
- **Xong khi:** `main` có `bao_cao/ch3_openmp.md`.

### Bước 6 — V.V. Dương: slide Ch.2 (16/11, 12:00)
- Làm **5–6 slide** trong file Google Slides chung (Doanh gửi link ở nhóm), mục "Ch.2".
- Nội dung: bài toán và định luật hấp dẫn + softening → tích phân symplectic Euler → mô hình Plummer → O(N²) + Amdahl → đồ thị `|ΔE/E₀|` theo `dt` và `eps` → GIF quỹ đạo.
- Ảnh lấy từ `results/vvduong_energy_*.png` và `results/vvduong_energy_orbit.gif`.
- Tự bấm giờ: **≤ 5 phút**.

### Bước 7 — N.T. Dương: slide Ch.3 (16/11, 12:00)
- Làm **5–6 slide**, mục "Ch.3": fork-join → chia vòng lặp, vì sao không có race → đồ thị speedup/hiệu suất theo số luồng → schedule và simd → Newton 3 (vì sao khó song song) → vì sao speedup < lý tưởng (băng thông bộ nhớ, lõi P/E, turbo).
- Ảnh lấy từ `results/ntduong_*.png`. Tự bấm giờ ≤ 5 phút.

### Bước 8 — Doanh: merge số GPU, vẽ đồ thị tổng hợp (16/11, 14:00)
```
git checkout main && git pull
python scripts/plot_results.py results/cpu_laptop.csv results/gpu_t4.csv
```
- Kiểm tra `results/summary.md` đã có cột CUDA naive/tiled, và có `results/block_size.png`.
- Commit đồ thị mới qua PR.
- Nhắn: `✅ Bước 8 xong — @Đạt đồ thị block size đây, dùng cho Ch.4`.

### Bước 9 — Đạt: viết Ch.4 (16/11, 20:00) ⚠️
- File `bao_cao/ch4_cuda.md`. Mở đầu bằng dòng `*Người viết: Trần Tiến Đạt*` (giống Ch.1).
- Các mục bắt buộc:
  1. **Kiến trúc GPU T4:** SM, warp, shared memory, thông số chính.
  2. **Ánh xạ** 1 thread = 1 vật thể, cách chia grid/block.
  3. **Kernel naive:** mã giả và lý do bị giới hạn bởi bộ nhớ toàn cục.
  4. **Kernel tiled** (GPU Gems 3 ch.31): hình minh họa tile, shared memory, vì sao cần `__syncthreads()` hai lần.
  5. **Kết quả:** bảng naive vs tiled theo N; đồ thị block size; bảng tỉ lệ H2D/D2H so với thời gian tính (từ `gpu_t4_h2d.txt`).
  6. **Nhận xét:** tiled nhanh hơn naive bao nhiêu lần và vì sao; block size nào tốt nhất; GFLOP/s đạt được so với 8800 GTX (~250 GFLOP/s) trong GPU Gems 3 và so với đỉnh lý thuyết của T4 (~8.1 TFLOP/s).
- Bỏ phần nâng cao (Nsight, `#pragma unroll`) nếu không kịp.
- Mở PR từ nhánh `dat/ch4`.
- **Dự phòng:** đến 16/11 18:00 chưa có bản nháp thì **N.T. Dương viết mục 1–2** (lý thuyết GPU), Đạt tập trung vào mục 3–6.

### Bước 10 — V.V. Dương: đọc chéo Ch.4 (16/11, 23:00)
- Đọc PR Ch.4, comment chỗ sai, thiếu số liệu hoặc thiếu nguồn. Approve khi ổn.

### Bước 11 — Doanh: hoàn thiện Ch.1 và Ch.5 (16/11, 23:00)
- Dán bảng từ `results/summary.md` vào mục 5.2.
- Viết mục 5.3 Nhận xét và 5.4 Kết luận theo các ý đã ghi sẵn trong file.
- Bỏ chữ "bản nháp" ở Ch.1.

### Bước 12 — Đạt: sửa Ch.4 theo review và làm slide Ch.4 (17/11, 12:00)
- Sửa theo comment → merge.
- **5–6 slide**, mục "Ch.4": kiến trúc → naive → tiled (hình tile) → naive vs tiled → block size → H2D/D2H. Tự bấm giờ ≤ 5 phút.

### Bước 13 — Doanh: ghép báo cáo và slide (17/11, 18:00)
- Báo cáo: trang bìa, mục lục, Ch.1 → Ch.5, tài liệu tham khảo (GPU Gems 3 ch.31, OpenMP spec, CUDA Programming Guide…). Thống nhất định dạng, đánh số hình/bảng. Xuất PDF.
- Slide: mở đầu → Ch.2 → Ch.3 → Ch.4 → Ch.5 kết luận. Thống nhất font và màu.
- Gửi PDF và link slide vào nhóm.

### Bước 14 — Cả nhóm: tổng duyệt (17/11, 20:00–22:00)
- Gọi online, mỗi người trình bày đúng phần mình, **bấm giờ ≤ 5 phút/người**.
- Mỗi người chuẩn bị sẵn 2 câu hỏi thầy/cô có thể hỏi về phần mình, kèm câu trả lời.
- Ghi lại lỗi cần sửa; Doanh sửa xong trước 23:30.

### Bước 15 — Doanh: nộp bài (18/11 sáng)
- Nộp PDF báo cáo, slide, link repo GitHub.

---

## Checklist cuối cùng (Doanh tick trước khi nộp)

- [ ] `main` build được, CI xanh
- [ ] Có đủ `bao_cao/ch1` → `ch5`
- [ ] `results/` có `cpu_laptop.csv`, `gpu_t4.csv` (+ `.machine.txt`), đồ thị đã cập nhật
- [ ] Mỗi bảng số liệu có ghi cấu hình máy
- [ ] Không có token/mật khẩu trong repo hoặc notebook
- [ ] Slide đủ 5 phần, mỗi người ≤ 5 phút
- [ ] PDF báo cáo đã xuất và đã đọc lại một lượt
