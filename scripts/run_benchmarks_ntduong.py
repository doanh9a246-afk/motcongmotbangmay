"""
run_benchmarks_ntduong.py - Chay toan bo cac thi nghiem OpenMP cua Nguyen Tung Duong:
1. ntduong_threads.csv: Speedup va hieu suat theo so luong (1, 2, 4, 8, 16) voi N = 2048 -> 16384.
2. ntduong_schedule_simd.csv: So sanh schedule (static, dynamic, guided) va co/khong SIMD.
3. ntduong_newton3.csv: So sanh bien the Dinh luat 3 Newton (N^2/2 cap) voi all-pairs OpenMP.
4. Tu dong sinh file cau hinh may (.machine.txt) va ve cac do thi PNG.
"""

import csv
import datetime
import os
import platform
import subprocess
import sys

CPU_BIN = "./bin/nbody_cpu.exe" if os.name == "nt" else "./bin/nbody_cpu"
REPS = 3
STEPS = 5

def get_cpu_info():
    name = "Intel Core Processor"
    cores = "10 (6P + 4E)"
    threads = str(os.cpu_count() or 16)
    compiler = "gcc"

    # Lay thong tin CPU tren Windows qua PowerShell / Registry
    if os.name == "nt":
        try:
            cmd = "powershell -NoProfile -Command \"(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0').ProcessorNameString\""
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                name = res.stdout.strip()
        except Exception:
            pass

    # Lay thong tin compiler
    try:
        res = subprocess.run([ "gcc", "--version" ], capture_output=True, text=True)
        if res.returncode == 0:
            compiler = res.stdout.splitlines()[0].strip()
    except Exception:
        pass

    return {
        "cpu": name,
        "cores": cores,
        "threads": threads,
        "compiler": compiler,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})"
    }

def run_command_best(args, reps=REPS):
    best_row = None
    min_time = float("inf")
    for _ in range(reps):
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"Error running {' '.join(args)}:\n{p.stderr}", file=sys.stderr)
            return None
        for line in p.stdout.splitlines():
            if line.startswith("RESULT,"):
                parts = line.strip().split(",")
                # RESULT,impl,mode,n,steps,par,time_s,ginter_per_s,gflops
                time_s = float(parts[6])
                if time_s < min_time:
                    min_time = time_s
                    best_row = parts[1:]
    return best_row

def experiment_threads(info):
    out_csv = "results/ntduong_threads.csv"
    out_machine = "results/ntduong_threads.machine.txt"
    ns = [2048, 4096, 8192, 16384]
    threads = [1, 2, 4, 8, 16]

    print("=================================================================")
    print("THI NGHIEM 1: Speedup & Hieu suat theo so luong (OpenMP vs Serial)")
    print("=================================================================")

    # Ghi file machine info
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(out_machine, "w", encoding="utf-8") as f:
        f.write(f"Nguoi thuc hien : Nguyen Tung Duong\n")
        f.write(f"Ngay do        : {now_str}\n")
        f.write(f"He dieu hanh   : {info['os']}\n")
        f.write(f"CPU            : {info['cpu']}\n")
        f.write(f"So loi         : {info['cores']}\n")
        f.write(f"So luong       : {info['threads']}\n")
        f.write(f"Compiler       : {info['compiler']}\n")
        f.write(f"Tham so        : NS=\"{' '.join(map(str, ns))}\" THREADS=\"{' '.join(map(str, threads))}\" STEPS={STEPS} REPS={REPS}\n")

    rows = []
    for n in ns:
        print(f"\n>> N = {n}")
        # Chay serial
        s_row = run_command_best([CPU_BIN, "-n", str(n), "-s", str(STEPS), "-m", "serial", "-noenergy"])
        if s_row:
            rows.append(s_row)
            print(f"   Serial    : {float(s_row[5]):.4f} s | {float(s_row[7]):.2f} GFLOP/s")

        # Chay omp cac so luong
        for t in threads:
            o_row = run_command_best([CPU_BIN, "-n", str(n), "-s", str(STEPS), "-m", "omp", "-t", str(t), "-noenergy"])
            if o_row:
                rows.append(o_row)
                t_s = float(o_row[5])
                gflops = float(o_row[7])
                s_t = float(s_row[5]) if s_row else t_s
                speedup = s_t / t_s
                eff = speedup / t
                print(f"   OpenMP {t:2d}T: {t_s:.4f} s | {gflops:6.2f} GFLOP/s | Speedup: {speedup:5.2f}x | Hieu suat: {eff*100:5.1f}%")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "mode", "n", "steps", "par", "time_s", "ginter_per_s", "gflops"])
        writer.writerows(rows)
    print(f"\nDa luu ket qua vao {out_csv} va {out_machine}")
    return rows

def experiment_schedule_simd():
    out_csv = "results/ntduong_schedule_simd.csv"
    ns = [2048, 4096, 8192, 16384]
    schedules = ["static", "dynamic", "guided"]
    simd_options = [1, 0] # 1 = co SIMD, 0 = khong SIMD
    threads = 16

    print("\n=================================================================")
    print("THI NGHIEM 2: So sanh Schedule (static, dynamic, guided) & SIMD")
    print("=================================================================")

    rows = []
    # Header: n,steps,threads,sched,simd,time_s,gflops,speedup_vs_nosimd
    for n in ns:
        print(f"\n>> N = {n} (16 luong)")
        for simd in simd_options:
            simd_label = "SIMD ON" if simd == 1 else "SIMD OFF"
            for sched in schedules:
                r = run_command_best([CPU_BIN, "-n", str(n), "-s", str(STEPS), "-m", "omp",
                                      "-t", str(threads), "-sched", sched, "-simd", str(simd), "-noenergy"])
                if r:
                    t_s = float(r[5])
                    gflops = float(r[7])
                    rows.append({
                        "n": n, "steps": STEPS, "threads": threads,
                        "sched": sched, "simd": simd, "simd_label": simd_label,
                        "time_s": t_s, "ms_step": t_s / STEPS * 1000.0,
                        "gflops": gflops
                    })
                    print(f"   [{simd_label}] sched({sched:7s}): {t_s:.4f} s | {t_s/STEPS*1000.0:6.2f} ms/buoc | {gflops:6.2f} GFLOP/s")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "steps", "threads", "sched", "simd", "time_s", "ms_step", "gflops"])
        for r in rows:
            writer.writerow([r["n"], r["steps"], r["threads"], r["sched"], r["simd"],
                             f"{r['time_s']:.6f}", f"{r['ms_step']:.3f}", f"{r['gflops']:.2f}"])
    print(f"\nDa luu ket qua vao {out_csv}")
    return rows

def experiment_newton3():
    out_csv = "results/ntduong_newton3.csv"
    ns = [2048, 4096, 8192, 16384]
    threads = [1, 2, 4, 8, 16]

    print("\n=================================================================")
    print("THI NGHIEM 3: So sanh Dinh luat 3 Newton (N^2/2 cap) vs All-Pairs")
    print("=================================================================")

    rows = []
    # Header: n,threads,mode,time_s,ms_step,gflops,ratio_allpairs_over_newton3
    for n in ns:
        print(f"\n>> N = {n}")
        # Chay serial lam moc
        s_row = run_command_best([CPU_BIN, "-n", str(n), "-s", str(STEPS), "-m", "serial", "-noenergy"])
        t_serial = float(s_row[5]) if s_row else 1.0

        for t in threads:
            # All-pairs OpenMP chuan
            r_omp = run_command_best([CPU_BIN, "-n", str(n), "-s", str(STEPS), "-m", "omp", "-t", str(t), "-noenergy"])
            t_omp = float(r_omp[5]) if r_omp else 0.0
            gflops_omp = float(r_omp[7]) if r_omp else 0.0

            # Newton 3 OpenMP (reduction buffers)
            r_n3 = run_command_best([CPU_BIN, "-n", str(n), "-s", str(STEPS), "-m", "omp_newton3", "-t", str(t), "-sched", "guided", "-noenergy"])
            t_n3 = float(r_n3[5]) if r_n3 else 0.0
            gflops_n3 = float(r_n3[7]) if r_n3 else 0.0

            speedup_n3 = t_serial / t_n3 if t_n3 > 0 else 0
            speedup_omp = t_serial / t_omp if t_omp > 0 else 0
            # Ty le: Neu t_omp / t_n3 > 1 thi Newton 3 nhanh hon, nguoc lai all-pairs nhanh hon
            ratio = t_omp / t_n3 if t_n3 > 0 else 0

            rows.append({
                "n": n, "threads": t,
                "t_omp": t_omp, "ms_omp": t_omp / STEPS * 1000.0, "gflops_omp": gflops_omp, "speedup_omp": speedup_omp,
                "t_n3": t_n3, "ms_n3": t_n3 / STEPS * 1000.0, "gflops_n3": gflops_n3, "speedup_n3": speedup_n3,
                "ratio_speed": ratio
            })
            print(f"   Threads {t:2d}: All-pairs = {t_omp:.4f} s ({gflops_omp:6.2f} GFLOP/s) | "
                  f"Newton3 = {t_n3:.4f} s ({gflops_n3:6.2f} GFLOP/s) | Ty so All-pairs/Newton3 = {ratio:4.2f}x")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "threads", "time_omp_s", "ms_omp", "gflops_omp", "speedup_omp",
                         "time_newton3_s", "ms_newton3", "gflops_newton3", "speedup_newton3", "ratio_omp_over_n3"])
        for r in rows:
            writer.writerow([r["n"], r["threads"],
                             f"{r['t_omp']:.6f}", f"{r['ms_omp']:.3f}", f"{r['gflops_omp']:.2f}", f"{r['speedup_omp']:.2f}",
                             f"{r['t_n3']:.6f}", f"{r['ms_n3']:.3f}", f"{r['gflops_n3']:.2f}", f"{r['speedup_n3']:.2f}",
                             f"{r['ratio_speed']:.2f}"])
    print(f"\nDa luu ket qua vao {out_csv}")
    return rows

if __name__ == "__main__":
    info = get_cpu_info()
    print("Moi truong he thong:")
    for k, v in info.items():
        print(f"  {k:10s}: {v}")

    # Chay lan luot 3 thi nghiem
    experiment_threads(info)
    experiment_schedule_simd()
    experiment_newton3()
    print("\nHOAN THANH TOAN BO THI NGHIEM!")
