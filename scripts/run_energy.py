"""
run_energy.py - Thi nghiem bao toan nang luong theo dt va eps (N=2048, 2000 buoc, seed 42).
Ghi results/vvduong_energy.csv va ve results/vvduong_energy_dt.png, results/vvduong_energy_eps.png.

    python scripts/run_energy.py

Luu y: chay voi so luong = tat ca luong logic. Voi libgomp cua MinGW, neu -t khac so luong mac dinh
thi ham tinh nang luong (team mac dinh) va nhan tinh toan (team -t) doi kich thuoc team va co the crash.
"""
import csv
import os
import re
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = "./bin/nbody_cpu.exe" if os.name == "nt" else "./bin/nbody_cpu"
N, STEPS, SEED = 2048, 2000, 42
THREADS = os.cpu_count() or 16
DTS = [0.0005, 0.001, 0.002, 0.005, 0.01]
EPSS = [0.001, 0.00316227766, 0.01, 0.0316227766, 0.1]
OUT = "results/vvduong_energy.csv"


def ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-c", cmd], capture_output=True, text=True)
    return r.stdout.strip()


cpu = ps("(Get-CimInstance Win32_Processor).Name") if os.name == "nt" else "CPU"
cores = ps("(Get-CimInstance Win32_Processor).NumberOfCores") if os.name == "nt" else "?"
compiler = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def run(dt, eps):
    t0 = time.perf_counter()
    p = subprocess.run([BIN, "-n", str(N), "-s", str(STEPS), "-dt", str(dt), "-eps", str(eps),
                        "-seed", str(SEED), "-m", "omp", "-t", str(THREADS)],
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    m = re.search(r"\|dE/E0\| = ([0-9.eE+-]+)", p.stdout)
    if p.returncode != 0 or not m:
        raise RuntimeError(f"Loi dt={dt} eps={eps} rc={p.returncode}\n{p.stdout}\n{p.stderr}")
    return float(m.group(1)), wall


rows = []
for study, values in (("dt", DTS), ("eps", EPSS)):
    for v in values:
        dt, eps = (v, 0.01) if study == "dt" else (0.002, v)
        err, wall = run(dt, eps)
        print(f"{study}={v:<12g} |dE/E0| = {err:.3e}  ({wall:.2f} s)", flush=True)
        rows.append([study, study, v, N, STEPS, dt, eps, THREADS, SEED, f"{err:.4g}", f"{wall:.3f}",
                     cpu, compiler, cores, THREADS])

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["study", "parameter", "value", "n", "steps", "dt", "eps", "threads", "seed",
                "energy_error", "wall_s", "cpu", "compiler", "cpu_cores", "logical_processors"])
    w.writerows(rows)
print("Da ghi", OUT)

# ---- Ve do thi ----
short_cpu = re.sub(r"^\d+th Gen ", "", cpu).replace("(R)", "").replace("(TM)", "")
gcc_ver = re.search(r"(\d+\.\d+)\.\d+\s*$", compiler)
header = f"{short_cpu} | {cores} cores / {THREADS} threads | GCC {gcc_ver.group(1) if gcc_ver else ''}"
for study, label, fname in (("dt", "Time step dt", "vvduong_energy_dt.png"),
                            ("eps", "Softening length eps", "vvduong_energy_eps.png")):
    xs = [r[2] for r in rows if r[0] == study]
    ys = [float(r[9]) for r in rows if r[0] == study]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.loglog(xs, ys, "o-", lw=2, ms=8)
    ax.set_xlabel(label)
    ax.set_ylabel("|Delta E / E0|")
    ax.set_title(f"Energy error vs {label} (N={N}, {STEPS} steps)")
    ax.grid(True, which="both", alpha=0.4)
    fig.suptitle(header, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join("results", fname), dpi=150)
    plt.close(fig)
    print("Da ve", fname)
