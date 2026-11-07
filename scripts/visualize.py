"""Truc quan hoa quy dao tu file CSV (step,id,x,y,z) do chuong trinh ghi ra voi -o.

    ./bin/nbody_cpu -n 1024 -s 2000 -dt 0.002 -o results/traj.csv -every 20
    python scripts/visualize.py results/traj.csv            # anh dau/cuoi -> results/snapshot.png
    python scripts/visualize.py results/traj.csv --gif      # them anh dong -> results/nbody.gif
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)
path = sys.argv[1]
make_gif = "--gif" in sys.argv
outdir = os.path.dirname(path) or "."

frames = defaultdict(lambda: ([], [], []))
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        xs, ys, zs = frames[int(r["step"])]
        xs.append(float(r["x"]))
        ys.append(float(r["y"]))
        zs.append(float(r["z"]))
steps = sorted(frames)
LIM = 2.0


def draw(ax, step):
    ax.cla()
    xs, ys, zs = frames[step]
    ax.scatter(xs, ys, zs, s=1, c="tab:orange", alpha=0.7)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-LIM, LIM)
    ax.set_title(f"buoc {step}")
    ax.set_axis_off()


fig = plt.figure(figsize=(10, 5))
for k, st in enumerate((steps[0], steps[-1])):
    draw(fig.add_subplot(1, 2, k + 1, projection="3d"), st)
plt.tight_layout()
snap = os.path.join(outdir, "snapshot.png")
plt.savefig(snap, dpi=150)
print("Da luu", snap)

if make_gif:
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    anim = FuncAnimation(fig, lambda i: draw(ax, steps[i]), frames=len(steps))
    gif = os.path.join(outdir, "nbody.gif")
    anim.save(gif, writer=PillowWriter(fps=15))
    print("Da luu", gif)
