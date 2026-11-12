"""Ve do thi tu cac file CSV do scripts/benchmark.sh tao ra.

    python scripts/plot_results.py [file1.csv file2.csv ...]

Mac dinh doc results/results.csv. Co the ghep nhieu file, vi du so lieu CPU do tren
laptop va so lieu GPU do tren Colab (theo quy uoc trong PHAN_CONG.md):

    python scripts/plot_results.py results/cpu_laptop.csv results/gpu_t4.csv

Neu cung mot cau hinh (impl, mode, N, so luong/block) xuat hien nhieu lan, lay lan nhanh nhat.
Do thi va bang tom tat ghi vao thu muc cua file dau tien:
    time_vs_n.png        thoi gian / buoc theo N (log-log)
    omp_speedup.png      speedup & hieu suat OpenMP theo so luong
    gpu_speedup.png      speedup cua cac phien ban so voi tuan tu
    block_size.png       anh huong kich thuoc block (neu co du lieu)
    summary.md           bang tom tat dang Markdown de dan vao bao cao
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

paths = sys.argv[1:] or ["results/results.csv"]
outdir = os.path.dirname(paths[0]) or "."

best = {}  # (impl, mode, n, par) -> row nhanh nhat
for path in paths:
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["n"] = int(r["n"])
            r["steps"] = int(r["steps"])
            r["par"] = int(r["par"])
            r["time_s"] = float(r["time_s"])
            r["gflops"] = float(r["gflops"])
            r["ms_step"] = 1e3 * r["time_s"] / r["steps"]
            key = (r["impl"], r["mode"], r["n"], r["par"])
            if key not in best or r["ms_step"] < best[key]["ms_step"]:
                best[key] = r
rows = list(best.values())

serial = {r["n"]: r["ms_step"] for r in rows if r["mode"] == "serial"}
omp = defaultdict(dict)  # n -> threads -> ms
for r in rows:
    if r["mode"] == "omp":
        omp[r["n"]][r["par"]] = r["ms_step"]
max_t = max((t for d in omp.values() for t in d), default=None)

# CUDA: lay block mac dinh 256 (neu khong co thi block nhanh nhat) cho moi (kernel, N)
cuda = defaultdict(dict)  # kernel -> n -> ms
for k in ("naive", "tiled"):
    by_n = defaultdict(dict)
    for r in rows:
        if r["impl"] == "cuda" and r["mode"] == k:
            by_n[r["n"]][r["par"]] = r["ms_step"]
    for n, d in by_n.items():
        cuda[k][n] = d.get(256, min(d.values()))

# ---- 1. Thoi gian theo N ----
plt.figure(figsize=(7, 5))
if serial:
    ns = sorted(serial)
    plt.plot(ns, [serial[n] for n in ns], "o-", label="Tuan tu (1 luong)")
if max_t:
    ns = sorted(n for n in omp if max_t in omp[n])
    plt.plot(ns, [omp[n][max_t] for n in ns], "s-", label=f"OpenMP ({max_t} luong)")
for k, mk in (("naive", "^-"), ("tiled", "D-")):
    if cuda.get(k):
        ns = sorted(cuda[k])
        plt.plot(ns, [cuda[k][n] for n in ns], mk, label=f"CUDA {k}")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("So vat the N")
plt.ylabel("Thoi gian / buoc (ms)")
plt.title("Thoi gian mo phong N-body theo N")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "time_vs_n.png"), dpi=150)
plt.close()

# ---- 2. Speedup OpenMP (chi cac N co do nhieu so luong) ----
sweep = [n for n in sorted(omp) if n in serial and len(omp[n]) > 1]
if sweep:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for n in sweep:
        ts = sorted(omp[n])
        sp = [serial[n] / omp[n][t] for t in ts]
        ax[0].plot(ts, sp, "o-", label=f"N={n}")
        ax[1].plot(ts, [s / t for s, t in zip(sp, ts)], "o-", label=f"N={n}")
    ts_all = sorted({t for n in sweep for t in omp[n]})
    ax[0].plot(ts_all, ts_all, "k--", alpha=0.5, label="Ly tuong")
    ax[0].set_xlabel("So luong")
    ax[0].set_ylabel("Speedup = T_tuantu / T_omp")
    ax[0].set_title("Speedup OpenMP")
    ax[1].set_xlabel("So luong")
    ax[1].set_ylabel("Hieu suat = Speedup / p")
    ax[1].set_title("Hieu suat song song")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "omp_speedup.png"), dpi=150)
    plt.close()

# ---- 3. Speedup so voi tuan tu ----
common = sorted(serial)
if common:
    labels, series = [], []
    if max_t:
        labels.append(f"OpenMP {max_t}T")
        series.append([serial[n] / omp[n][max_t] if max_t in omp.get(n, {}) else 0 for n in common])
    for k in ("naive", "tiled"):
        if cuda.get(k):
            labels.append(f"CUDA {k}")
            series.append([serial[n] / cuda[k][n] if n in cuda[k] else 0 for n in common])
    if series:
        plt.figure(figsize=(9, 5))
        w = 0.8 / len(series)
        for i, (lab, s) in enumerate(zip(labels, series)):
            plt.bar([x + i * w for x in range(len(common))], s, w, label=lab)
        plt.xticks([x + w * (len(series) - 1) / 2 for x in range(len(common))], [str(n) for n in common])
        plt.yscale("log")
        plt.xlabel("N")
        plt.ylabel("Speedup so voi tuan tu (log)")
        plt.title("Speedup cac phien ban song song")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "gpu_speedup.png"), dpi=150)
        plt.close()

# ---- 4. Kich thuoc block ----
tiled = [r for r in rows if r["impl"] == "cuda" and r["mode"] == "tiled"]
if tiled:
    nmax = max(r["n"] for r in tiled)
    blk = sorted((r["par"], r["gflops"]) for r in tiled if r["n"] == nmax)
    if len(blk) > 1:
        plt.figure(figsize=(6, 4))
        plt.bar([str(b) for b, _ in blk], [g for _, g in blk])
        plt.xlabel("Kich thuoc block (= tile p)")
        plt.ylabel("GFLOP/s")
        plt.title(f"CUDA tiled, N = {nmax}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "block_size.png"), dpi=150)
        plt.close()


# ---- Bang tom tat ----
def f(x):
    return f"{x:.3f}" if x is not None else "-"


def ratio(a, b):
    return f"{a / b:.1f}x" if a and b else "-"


lines = [
    f"Nguon so lieu: {', '.join(paths)}",
    "",
    f"| N | Tuan tu (ms/buoc) | OpenMP {max_t or '-'}T (ms) | Speedup OMP | CUDA naive (ms) "
    "| CUDA tiled (ms) | Tiled / tuan tu | Tiled / OpenMP |",
    "|---|---|---|---|---|---|---|---|",
]
alln = sorted(set(serial) | set(cuda.get("tiled", {})) | set(cuda.get("naive", {})))
for n in alln:
    s = serial.get(n)
    o = omp.get(n, {}).get(max_t) if max_t else None
    cn = cuda.get("naive", {}).get(n)
    ct = cuda.get("tiled", {}).get(n)
    lines.append(f"| {n} | {f(s)} | {f(o)} | {ratio(s, o)} | {f(cn)} | {f(ct)} | {ratio(s, ct)} | {ratio(o, ct)} |")

text = "\n".join(lines)
print(text)
with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as fo:
    fo.write(text + "\n")
print(f"\nDa luu do thi va summary.md vao {outdir}/")
