"""
plot_ntduong.py - Ve do thi chat luong cao tu ket qua thi nghiem OpenMP cua Nguyen Tung Duong
Dung Pillow voi supersampling 2x de net ve va chu sac net, ro rang.
Sinh ra:
1. results/ntduong_threads_speedup.png (Speedup & Hieu suat theo so luong)
2. results/ntduong_schedule_simd.png (So sanh static, dynamic, guided va SIMD ON/OFF)
3. results/ntduong_newton3_comparison.png (So sanh All-Pairs vs Dinh luat 3 Newton)
"""

import csv
import math
import os
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/segoeuib.ttf"

def get_font(size, bold=False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, size * 2)  # 2x for supersampling
    except Exception:
        return ImageFont.load_default()

def draw_aa_line(draw, p1, p2, color, width=2):
    draw.line([p1, p2], fill=color, width=width * 2)

def draw_aa_circle(draw, center, radius, fill_color, outline_color=None, width=1):
    x, y = center
    r = radius * 2
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill_color, outline=outline_color, width=width * 2)

# ==============================================================================
# DO THI 1: Speedup & Hieu suat theo so luong
# ==============================================================================
def plot_threads_speedup():
    csv_path = "results/ntduong_threads.csv"
    if not os.path.exists(csv_path):
        return

    data = {} # n -> list of (thread, time_s, gflops)
    serial = {} # n -> time_s
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r["n"])
            t = int(r["par"])
            sec = float(r["time_s"])
            gflops = float(r["gflops"])
            if r["mode"] == "serial":
                serial[n] = sec
            elif r["mode"] == "omp":
                if n not in data:
                    data[n] = []
                data[n].append((t, sec, gflops))

    # Tao anh 2x (1200 x 550 => 2400 x 1100)
    w, h = 1200, 550
    scale = 2
    img = Image.new("RGBA", (w * scale, h * scale), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_axis_title = get_font(14, bold=True)
    f_axis = get_font(11)
    f_legend = get_font(11)
    f_sub = get_font(11, bold=True)

    # Title
    title = "Hieu nang OpenMP: Speedup va Hieu suat theo so luong (Intel Core i7-13620H, 16 luong)"
    draw.text((w * scale // 2, 25 * scale), title, font=f_title, fill=(30, 41, 59), anchor="mm")

    colors = {
        2048: (239, 68, 68),   # Do
        4096: (245, 158, 11),  # Cam
        8192: (16, 185, 129),  # Xanh la
        16384: (59, 130, 246)  # Xanh duong
    }

    # Hai panel: Panel 1 (Speedup, x: 80..540), Panel 2 (Hieu suat, x: 640..1100)
    # y: 80..460
    panels = [
        {"x0": 80, "x1": 550, "y0": 80, "y1": 460, "type": "speedup", "ymax": 16, "title": "Speedup = T_serial / T_omp"},
        {"x0": 670, "x1": 1140, "y0": 80, "y1": 460, "type": "efficiency", "ymax": 1.1, "title": "Hieu suat = Speedup / p"}
    ]

    threads_all = [1, 2, 4, 8, 16]

    for p in panels:
        px0, px1 = p["x0"] * scale, p["x1"] * scale
        py0, py1 = p["y0"] * scale, p["y1"] * scale

        # Tieu de panel
        draw.text(((px0 + px1) // 2, (p["y0"] - 15) * scale), p["title"], font=f_sub, fill=(51, 65, 85), anchor="mm")

        # Khung truc
        draw_aa_line(draw, (px0, py1), (px1, py1), (203, 213, 225), 1)
        draw_aa_line(draw, (px0, py0), (px0, py1), (203, 213, 225), 1)

        # Luoi ngang va nhan truc Y
        if p["type"] == "speedup":
            y_ticks = [0, 2, 4, 6, 8, 10, 12, 14, 16]
            for yt in y_ticks:
                y_pos = py1 - int((yt / 16.0) * (py1 - py0))
                draw_aa_line(draw, (px0, y_pos), (px1, y_pos), (241, 245, 249), 1)
                draw.text((px0 - 10 * scale, y_pos), f"{yt}x", font=f_axis, fill=(100, 116, 139), anchor="rm")
        else:
            y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            for yt in y_ticks:
                y_pos = py1 - int((yt / 1.0) * (py1 - py0))
                draw_aa_line(draw, (px0, y_pos), (px1, y_pos), (241, 245, 249), 1)
                draw.text((px0 - 10 * scale, y_pos), f"{int(yt*100)}%", font=f_axis, fill=(100, 116, 139), anchor="rm")

        # Nhan truc X
        for idx, t in enumerate(threads_all):
            x_pos = px0 + int((idx / (len(threads_all) - 1)) * (px1 - px0))
            draw_aa_line(draw, (x_pos, py1), (x_pos, py1 + 4 * scale), (148, 163, 184), 1)
            draw.text((x_pos, py1 + 15 * scale), str(t), font=f_axis, fill=(71, 85, 105), anchor="mm")

        # Truc X label
        draw.text(((px0 + px1) // 2, py1 + 35 * scale), "So luong OpenMP (threads)", font=f_axis_title, fill=(71, 85, 105), anchor="mm")

        # Duong ly tuong
        if p["type"] == "speedup":
            # Duong cheo ly tuong
            x_pts = []
            for idx, t in enumerate(threads_all):
                x_pos = px0 + int((idx / (len(threads_all) - 1)) * (px1 - px0))
                y_pos = py1 - int((t / 16.0) * (py1 - py0))
                x_pts.append((x_pos, y_pos))
            for k in range(len(x_pts) - 1):
                draw_aa_line(draw, x_pts[k], x_pts[k+1], (148, 163, 184), 1)
            draw.text((x_pts[-1][0] - 20 * scale, x_pts[-1][1] - 12 * scale), "Ly tuong (p)", font=f_axis, fill=(100, 116, 139), anchor="rm")
        else:
            # 100% efficiency line
            y_100 = py1 - int((1.0 / 1.0) * (py1 - py0))
            draw_aa_line(draw, (px0, y_100), (px1, y_100), (148, 163, 184), 1)
            draw.text((px1 - 10 * scale, y_100 - 12 * scale), "100%", font=f_axis, fill=(100, 116, 139), anchor="rm")

        # Ve cac duong du lieu
        for n in sorted(data.keys()):
            pts = []
            for t, sec, _ in sorted(data[n]):
                idx = threads_all.index(t)
                x_pos = px0 + int((idx / (len(threads_all) - 1)) * (px1 - px0))
                sp = serial[n] / sec
                if p["type"] == "speedup":
                    y_pos = py1 - int((sp / 16.0) * (py1 - py0))
                else:
                    eff = sp / t
                    y_pos = py1 - int((min(eff, 1.1) / 1.0) * (py1 - py0))
                pts.append((x_pos, y_pos))

            col = colors[n]
            for k in range(len(pts) - 1):
                draw_aa_line(draw, pts[k], pts[k+1], col, 2)
            for pt in pts:
                draw_aa_circle(draw, pt, 4, (255, 255, 255), col, 2)

    # Legend o giua ben duoi
    leg_x = 420 * scale
    leg_y = 515 * scale
    for idx, n in enumerate(sorted(data.keys())):
        cx = leg_x + idx * 110 * scale
        col = colors[n]
        draw_aa_line(draw, (cx, leg_y), (cx + 25 * scale, leg_y), col, 2)
        draw_aa_circle(draw, (cx + 12 * scale, leg_y), 4, (255, 255, 255), col, 2)
        draw.text((cx + 32 * scale, leg_y), f"N = {n}", font=f_legend, fill=(30, 41, 59), anchor="lm")

    # Resize ve 1200x550 voi Lanczos
    res = img.resize((w, h), Image.Resampling.LANCZOS)
    res.save("results/ntduong_threads_speedup.png", dpi=(150, 150))
    print("Da luu results/ntduong_threads_speedup.png")


# ==============================================================================
# DO THI 2: So sanh Schedule (static, dynamic, guided) & SIMD
# ==============================================================================
def plot_schedule_simd():
    csv_path = "results/ntduong_schedule_simd.csv"
    if not os.path.exists(csv_path):
        return

    data = {} # (n, sched, simd) -> gflops
    ns = [2048, 4096, 8192, 16384]
    schedules = ["static", "dynamic", "guided"]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r["n"])
            sched = r["sched"]
            simd = int(r["simd"])
            gflops = float(r["gflops"])
            data[(n, sched, simd)] = gflops

    w, h = 1000, 520
    scale = 2
    img = Image.new("RGBA", (w * scale, h * scale), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_axis_title = get_font(13, bold=True)
    f_axis = get_font(11)
    f_legend = get_font(11)
    f_bar_val = get_font(9)

    title = "So sanh Hieu nang cac Schedule va Vector hoa SIMD tren OpenMP (16 luong)"
    draw.text((w * scale // 2, 25 * scale), title, font=f_title, fill=(30, 41, 59), anchor="mm")

    # Khung bieu do cot
    x0, x1 = 80 * scale, 950 * scale
    y0, y1 = 70 * scale, 430 * scale

    draw_aa_line(draw, (x0, y1), (x1, y1), (203, 213, 225), 1)
    draw_aa_line(draw, (x0, y0), (x0, y1), (203, 213, 225), 1)

    max_gflops = 260.0
    for yt in range(0, 261, 50):
        y_pos = y1 - int((yt / max_gflops) * (y1 - y0))
        draw_aa_line(draw, (x0, y_pos), (x1, y_pos), (241, 245, 249), 1)
        draw.text((x0 - 10 * scale, y_pos), f"{yt}", font=f_axis, fill=(100, 116, 139), anchor="rm")

    draw.text((x0 - 45 * scale, (y0 + y1) // 2), "Hieu nang (GFLOP/s)", font=f_axis_title, fill=(71, 85, 105), anchor="mm")
    draw.text(((x0 + x1) // 2, y1 + 35 * scale), "So vat the N", font=f_axis_title, fill=(71, 85, 105), anchor="mm")

    # 6 bien the: (sched, simd)
    variants = [
        ("static", 1, (37, 99, 235), "static + SIMD"),       # Xanh dam
        ("dynamic", 1, (16, 185, 129), "dynamic + SIMD"),    # Xanh la
        ("guided", 1, (14, 165, 233), "guided + SIMD"),      # Xanh lam
        ("static", 0, (239, 68, 68), "static (No SIMD)"),    # Do
        ("dynamic", 0, (245, 158, 11), "dynamic (No SIMD)"), # Cam
        ("guided", 0, (168, 85, 247), "guided (No SIMD)")     # Tim
    ]

    group_width = (x1 - x0) / len(ns)
    bar_width = (group_width * 0.75) / len(variants)

    for i, n in enumerate(ns):
        gx = x0 + i * group_width + group_width * 0.125
        # Nhan N o giua group
        draw.text((gx + (len(variants) * bar_width) / 2, y1 + 15 * scale), f"N = {n}", font=f_axis, fill=(30, 41, 59), anchor="mm")

        for j, (sched, simd, col, _) in enumerate(variants):
            bx = gx + j * bar_width
            gf = data.get((n, sched, simd), 0.0)
            by = y1 - int((gf / max_gflops) * (y1 - y0))
            draw.rectangle([bx, by, bx + bar_width - 2 * scale, y1], fill=col)
            # In gia tri tren cot
            draw.text((bx + bar_width / 2, by - 6 * scale), f"{int(gf)}", font=f_bar_val, fill=(71, 85, 105), anchor="mm")

    # Legend
    leg_x = 100 * scale
    leg_y = 485 * scale
    for idx, (_, _, col, label) in enumerate(variants):
        lx = leg_x + idx * 140 * scale
        draw.rectangle([lx, leg_y - 6 * scale, lx + 14 * scale, leg_y + 8 * scale], fill=col)
        draw.text((lx + 20 * scale, leg_y), label, font=f_legend, fill=(30, 41, 59), anchor="lm")

    res = img.resize((w, h), Image.Resampling.LANCZOS)
    res.save("results/ntduong_schedule_simd.png", dpi=(150, 150))
    print("Da luu results/ntduong_schedule_simd.png")


# ==============================================================================
# DO THI 3: So sanh Newton 3 vs All-Pairs
# ==============================================================================
def plot_newton3_comparison():
    csv_path = "results/ntduong_newton3.csv"
    if not os.path.exists(csv_path):
        return

    data = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append({
                "n": int(r["n"]),
                "threads": int(r["threads"]),
                "ms_omp": float(r["ms_omp"]),
                "ms_n3": float(r["ms_newton3"]),
                "gf_omp": float(r["gflops_omp"]),
                "gf_n3": float(r["gflops_newton3"]),
                "ratio": float(r["ratio_omp_over_n3"])
            })

    w, h = 1100, 520
    scale = 2
    img = Image.new("RGBA", (w * scale, h * scale), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_axis_title = get_font(13, bold=True)
    f_axis = get_font(11)
    f_legend = get_font(11)
    f_sub = get_font(12, bold=True)

    title = "So sanh OpenMP All-Pairs (N^2) va Dinh luat 3 Newton (N^2/2 cap) theo So luong"
    draw.text((w * scale // 2, 25 * scale), title, font=f_title, fill=(30, 41, 59), anchor="mm")

    # Ve cho 2 truong hop dien hinh: Panel Trai (N = 8192), Panel Phai (N = 16384)
    panels = [
        {"n": 8192, "x0": 80, "x1": 530, "y0": 80, "y1": 430, "ymax": 60.0},
        {"n": 16384, "x0": 630, "x1": 1050, "y0": 80, "y1": 430, "ymax": 240.0}
    ]

    threads_all = [1, 2, 4, 8, 16]

    for p in panels:
        px0, px1 = p["x0"] * scale, p["x1"] * scale
        py0, py1 = p["y0"] * scale, p["y1"] * scale
        n_val = p["n"]
        ymax = p["ymax"]

        draw.text(((px0 + px1) // 2, (p["y0"] - 15) * scale), f"Thoi gian tinh / buoc: N = {n_val}", font=f_sub, fill=(30, 41, 59), anchor="mm")

        draw_aa_line(draw, (px0, py1), (px1, py1), (203, 213, 225), 1)
        draw_aa_line(draw, (px0, py0), (px0, py1), (203, 213, 225), 1)

        # Trục Y
        steps_cnt = 5
        for s_idx in range(steps_cnt + 1):
            val = (ymax / steps_cnt) * s_idx
            y_pos = py1 - int((val / ymax) * (py1 - py0))
            draw_aa_line(draw, (px0, y_pos), (px1, y_pos), (241, 245, 249), 1)
            draw.text((px0 - 10 * scale, y_pos), f"{val:.1f} ms", font=f_axis, fill=(100, 116, 139), anchor="rm")

        # Nhan truc X
        for idx, t in enumerate(threads_all):
            x_pos = px0 + int((idx / (len(threads_all) - 1)) * (px1 - px0))
            draw_aa_line(draw, (x_pos, py1), (x_pos, py1 + 4 * scale), (148, 163, 184), 1)
            draw.text((x_pos, py1 + 15 * scale), f"{t}T", font=f_axis, fill=(71, 85, 105), anchor="mm")

        draw.text(((px0 + px1) // 2, py1 + 35 * scale), "So luong (threads)", font=f_axis_title, fill=(71, 85, 105), anchor="mm")

        # Du lieu cho N
        rows_n = [r for r in data if r["n"] == n_val]
        pts_omp = []
        pts_n3 = []
        for r in sorted(rows_n, key=lambda x: x["threads"]):
            idx = threads_all.index(r["threads"])
            x_pos = px0 + int((idx / (len(threads_all) - 1)) * (px1 - px0))
            y_omp = py1 - int((min(r["ms_omp"], ymax) / ymax) * (py1 - py0))
            y_n3  = py1 - int((min(r["ms_n3"], ymax) / ymax) * (py1 - py0))
            pts_omp.append((x_pos, y_omp))
            pts_n3.append((x_pos, y_n3))

        # Ve duong All-Pairs (Xanh duong)
        col_omp = (37, 99, 235)
        for k in range(len(pts_omp) - 1):
            draw_aa_line(draw, pts_omp[k], pts_omp[k+1], col_omp, 2)
        for pt in pts_omp:
            draw_aa_circle(draw, pt, 4, (255, 255, 255), col_omp, 2)

        # Ve duong Newton 3 (Do cam)
        col_n3 = (225, 29, 72)
        for k in range(len(pts_n3) - 1):
            draw_aa_line(draw, pts_n3[k], pts_n3[k+1], col_n3, 2)
        for pt in pts_n3:
            draw_aa_circle(draw, pt, 4, (255, 255, 255), col_n3, 2)

    # Legend
    leg_x = 360 * scale
    leg_y = 485 * scale
    # All-pairs
    draw_aa_line(draw, (leg_x, leg_y), (leg_x + 30 * scale, leg_y), (37, 99, 235), 2)
    draw_aa_circle(draw, (leg_x + 15 * scale, leg_y), 4, (255, 255, 255), (37, 99, 235), 2)
    draw.text((leg_x + 38 * scale, leg_y), "All-Pairs OpenMP (N^2 tuong tac, SIMD reduction)", font=f_legend, fill=(30, 41, 59), anchor="lm")

    # Newton 3
    leg2_x = leg_x + 350 * scale
    draw_aa_line(draw, (leg2_x, leg_y), (leg2_x + 30 * scale, leg_y), (225, 29, 72), 2)
    draw_aa_circle(draw, (leg2_x + 15 * scale, leg_y), 4, (255, 255, 255), (225, 29, 72), 2)
    draw.text((leg2_x + 38 * scale, leg_y), "Dinh luat 3 Newton (N^2/2 cap, mang cuc bo moi luong)", font=f_legend, fill=(30, 41, 59), anchor="lm")

    res = img.resize((w, h), Image.Resampling.LANCZOS)
    res.save("results/ntduong_newton3_comparison.png", dpi=(150, 150))
    print("Da luu results/ntduong_newton3_comparison.png")

if __name__ == "__main__":
    plot_threads_speedup()
    plot_schedule_simd()
    plot_newton3_comparison()
    print("HOAN THANH VE DO THI!")
