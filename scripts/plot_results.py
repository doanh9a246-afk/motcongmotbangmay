"""Ve do thi tu results/results.csv (do scripts/benchmark.sh tao ra).

    python scripts/plot_results.py [results/results.csv]

Tao ra trong thu muc results/:
    time_vs_n.png        thoi gian / buoc theo N (log-log)
    omp_speedup.png      speedup & hieu suat OpenMP theo so luong
    gpu_speedup.png      speedup cua cac phien ban so voi tuan tu
    block_size.png       anh huong kich thuoc block (neu co du lieu)
va in bang tom tat dang Markdown de dan vao bao cao.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
outdir = os.path.dirname(path) or "."

rows = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        r["n"] = int(r["n"])
        r["steps"] = int(r["steps"])
        r["par"] = int(r["par"])
        r["time_s"] = float(r["time_s"])
        r["gflops"] = float(r["gflops"])
        r["ms_step"] = 1e3 * r["time_s"] / r["steps"]
        rows.append(r)

serial = {r["n"]: r["ms_step"] for r in rows if r["mode"] == "serial"}
omp = defaultdict(dict)  # n -> threads -> ms
for r in rows:
    if r["mode"] == "omp":
        omp[r["n"]][r["par"]] = r["ms_step"]
max_t = max((t for d in omp.values() for t in d), default=None)

# CUDA: lay block mac dinh 256 (hoac block tot nhat) cho moi (kernel, N)
cuda = defaultdict(dict)  # kernel -> n -> ms
for r in rows:
    if r["impl"] == "cuda":
        prev = cuda[r["mode"]].get(r["n"])
        if r["par"] == 256 or prev is None:
            cuda[r["mode"]][r["n"]] = r["ms_step"]

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

# ---- 2. Speedup OpenMP ----
if omp and serial:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for n in sorted(omp):
        if n not in serial:
            continue
        ts = sorted(omp[n])
        sp = [serial[n] / omp[n][t] for t in ts]
        ax[0].plot(ts, sp, "o-", label=f"N={n}")
        ax[1].plot(ts, [s / t for s, t in zip(sp, ts)], "o-", label=f"N={n}")
    ts_all = sorted({t for d in omp.values() for t in d})
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
        plt.figure(figsize=(8, 5))
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

# ---- 4. Kich thuoc block ----
tiled = [r for r in rows if r["impl"] == "cuda" and r["mode"] == "tiled"]
if tiled:
    nmax = max(r["n"] for r in tiled)
    blk = sorted({(r["par"], r["gflops"]) for r in tiled if r["n"] == nmax})
    if len(blk) > 1:
        plt.figure(figsize=(6, 4))
        plt.bar([str(b) for b, _ in blk], [g for _, g in blk])
        plt.xlabel("Kich thuoc block (= tile p)")
        plt.ylabel("GFLOP/s")
        plt.title(f"CUDA tiled, N = {nmax}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "block_size.png"), dpi=150)

# ---- Bang tom tat ----
print("| N | Tuan tu (ms/buoc) | OpenMP max (ms) | Speedup OMP | CUDA naive (ms) | CUDA tiled (ms) | Speedup tiled |")
print("|---|---|---|---|---|---|---|")
alln = sorted(set(serial) | set(cuda.get("tiled", {})))
for n in alln:
    s = serial.get(n)
    o = omp.get(n, {}).get(max_t) if max_t else None
    cn = cuda.get("naive", {}).get(n)
    ct = cuda.get("tiled", {}).get(n)

    def f(x):
        return f"{x:.3f}" if x is not None else "-"

    print(f"| {n} | {f(s)} | {f(o)} | {f(s / o) if s and o else '-'} | {f(cn)} | {f(ct)} | "
          f"{f(s / ct) if s and ct else '-'} |")
print(f"\nDa luu do thi vao {outdir}/")
