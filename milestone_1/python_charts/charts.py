import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Results

N = [100, 1000, 10000, 100000, 1000000,
     10000000, 100000000, 1000000000]

pi_cpu = [2.6800000000, 3.0960000000, 3.1344000000,
          3.1431600000, 3.1412800000, 3.1415580000,
          3.1417225200, 3.1416849520]

pi_gpu = [3.3200000000, 3.1240000000, 3.1056000000,
          3.1378000000, 3.1405120000, 3.1417292000,
          3.1418661600, 3.1416741480]

error_cpu = [0.4615926536, 0.0455926536, 0.0071926536,
             0.0015673464, 0.0003126536, 0.0000346536,
             0.0001298664, 0.0000922984]

error_gpu = [0.1784073464, 0.0175926536, 0.0359926536,
             0.0037926536, 0.0010806536, 0.0001365464,
             0.0002735064, 0.0000814944]

cpu_time = [0.000003, 0.000017, 0.000153, 0.001558,
            0.014521, 0.147805, 1.541649, 15.911579]

gpu_time = [0.000012, 0.000012, 0.000012, 0.000027,
            0.000058, 0.000100, 0.000671, 0.010879]

speedup = [0.3, 1.4, 12.5, 57.3, 250.4, 1481.4,
           2297.4, 1462.6]

# Style

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 10
})


# Chart 1: CPU vs GPU execution time

fig1, ax1 = plt.subplots(figsize=(7, 5))

ax1.plot(N, [t * 1000 for t in cpu_time],
         marker="o", linewidth=2, label="CPU")
ax1.plot(N, [t * 1000 for t in gpu_time],
         marker="s", linewidth=2, label="GPU")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of points (N)")
ax1.set_ylabel("Execution time (ms)")
ax1.set_title("CPU vs GPU Execution Time", fontweight="bold")
ax1.grid(True, which="both", alpha=0.25)
ax1.legend()

fig1.tight_layout()
fig1.savefig("panel1_execution_time.png", dpi=300,
             bbox_inches="tight")


# Chart 2: CPU vs GPU estimation accuracy

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.plot(N, error_cpu, marker="o",
         linewidth=2, label="CPU error")
ax2.plot(N, error_gpu, marker="s",
         linewidth=2, label="GPU error")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Number of points (N)")
ax2.set_ylabel("Absolute error |estimate - π|")
ax2.set_title("Estimation Accuracy", fontweight="bold")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend()

fig2.tight_layout()
fig2.savefig("panel2_accuracy.png", dpi=300,
             bbox_inches="tight")


# Chart 3: GPU speedup over CPU

fig3, ax3 = plt.subplots(figsize=(7, 5))

ax3.plot(N, speedup, marker="D", linewidth=2,
         label="Measured speedup")
ax3.axhline(1, linestyle="--", linewidth=1.2,
            label="Equal execution time")

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Number of points (N)")
ax3.set_ylabel("Speedup (CPU time / GPU time)")
ax3.set_title("GPU Speedup over CPU", fontweight="bold")
ax3.grid(True, which="both", alpha=0.25)
ax3.legend()

fig3.tight_layout()
fig3.savefig("panel3_speedup.png", dpi=300,
             bbox_inches="tight")


# Chart 4: Grid-block-thread hierarchy in CUDA using shared memory for block sum and atomic update for global sum

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
thread_colors = [BLUE, ORANGE, AQUA, YELLOW]

fig4, ax4 = plt.subplots(figsize=(6.6, 4.4))
ax4.set_xlim(0, 16)
ax4.set_ylim(0, 7.2)
ax4.axis("off")

# Header
ax4.text(0, 6.85,
            r"samples  $i = 0 \ldots N{-}1$   (here $N{=}16$, stride $=4$, one block of 4 threads)",
            fontsize=12, color=INK2, va="center")

# 16 samples, coloured by the thread that processes them (thread i mod 4)
for i in range(16):
    t = i % 4
    ax4.add_patch(Rectangle((i + 0.04, 5.4), 0.92, 0.9,
                            fc=thread_colors[t], ec="white", lw=1.0, alpha=0.30))
    ax4.add_patch(Rectangle((i + 0.04, 5.4), 0.92, 0.9,
                            fc="none", ec=thread_colors[t], lw=1.6))
    ax4.text(i + 0.5, 5.85, f"{i}", ha="center", va="center",
                fontsize=12, color=INK)

ax4.text(8.0, 4.97, r"colour = thread that processes sample $i$  (thread $i$ mod 4)",
            fontsize=11.6, color=INK2, va="center", ha="center")

# Four threads, each with its own generator state and counter
for t in range(4):
    x0 = t * 4 + 0.15
    ax4.add_patch(FancyBboxPatch((x0, 3.2), 3.7, 1.45,
                                    boxstyle="round,pad=0.0,rounding_size=0.15",
                                    fc="white", ec=thread_colors[t], lw=2.0))
    ax4.text(x0 + 1.85, 4.23, f"thread $t_{t}$", ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
    ax4.text(x0 + 1.85, 3.73, "own XORWOW state\nlocal hit counter",
                ha="center", va="center", fontsize=10.2, color=INK2, linespacing=1.15)
    cx = x0 + 1.85
    ax4.annotate("", xy=(cx, 2.55), xytext=(cx, 3.15),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, mutation_scale=12))

# Block sum in shared memory
ax4.add_patch(FancyBboxPatch((0.15, 1.65), 15.7, 0.85,
                                boxstyle="round,pad=0.0,rounding_size=0.15",
                                fc="white", ec=AXIS, lw=1.6))
ax4.text(8.0, 2.07, "block sum in shared memory  (tree reduction, thread 0 holds the total)",
            ha="center", va="center", fontsize=12, color=INK)
ax4.annotate("", xy=(8.0, 1.05), xytext=(8.0, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, mutation_scale=12))

# Global atomic update
ax4.add_patch(FancyBboxPatch((0.15, 0.2), 15.7, 0.85,
                                boxstyle="round,pad=0.0,rounding_size=0.15",
                                fc="#f0efec", ec=AXIS, lw=1.6))
ax4.text(8.0, 0.62, "global hit counter  (one atomicAdd per block)",
            ha="center", va="center", fontsize=13, color=INK)

fig4.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig4.savefig("panel4_cuda_hierarchy.png", dpi=300)


# Display all figures

# plt.show()