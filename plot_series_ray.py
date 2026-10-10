import csv
import matplotlib.pyplot as plt


def read_results(filename):
    rows = []

    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            rows.append(
                {
                    "N": int(row["N"]),
                    "runtime": float(row["runtime_seconds"]),
                }
            )

    return rows


one_cpu = read_results("series_ray_1cpu.csv")
two_cpu = read_results("series_ray_2cpu.csv")

n_values = [row["N"] for row in one_cpu]
one_times = [row["runtime"] for row in one_cpu]
two_times = [row["runtime"] for row in two_cpu]

x = list(range(len(n_values)))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [value - width / 2 for value in x],
    one_times,
    width,
    label="1 CPU Core",
)

plt.bar(
    [value + width / 2 for value in x],
    two_times,
    width,
    label="2 CPU Cores",
)

plt.xticks(
    x,
    [f"{value:,}" for value in n_values],
)

plt.xlabel("Number of Series Terms (N)")
plt.ylabel("Running Time (seconds)")
plt.title("Ray Series-Based Pi Runtime Comparison")
plt.legend()
plt.tight_layout()

output_file = "series_ray_runtime.png"
plt.savefig(output_file, dpi=200)

print(f"Plot saved to: {output_file}")

print("\nSpeedup results:")

for n, t1, t2 in zip(n_values, one_times, two_times):
    speedup = t1 / t2 if t2 > 0 else float("inf")

    print(
        f"N={n:,}: "
        f"1 CPU={t1:.6f}s, "
        f"2 CPUs={t2:.6f}s, "
        f"Speedup={speedup:.3f}x"
    )
