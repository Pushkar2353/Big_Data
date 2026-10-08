import argparse
import csv
import math
import time

import ray


STEPS = [1000, 10000, 100000, 1000000]


@ray.remote(num_cpus=1)
def calculate_partial_sum(start_index, end_index, n):
    dx = 1.0 / n
    partial_sum = 0.0

    for i in range(start_index, end_index):
        x = i * dx
        partial_sum += 4.0 / (1.0 + x * x)

    return partial_sum


def split_range(n, parts):
    ranges = []

    base_size = n // parts
    remainder = n % parts

    start = 0

    for part in range(parts):
        extra = 1 if part < remainder else 0
        end = start + base_size + extra

        ranges.append((start, end))
        start = end

    return ranges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed calculus-based Pi approximation using Ray."
    )

    parser.add_argument(
        "--cores",
        type=int,
        choices=[1, 2],
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ray.init(address="auto")

    output_file = f"calculus_ray_{args.cores}cpu.csv"

    # Warm-up
    warmup_ranges = split_range(1000, args.cores)

    warmup_refs = [
        calculate_partial_sum.remote(start, end, 1000)
        for start, end in warmup_ranges
    ]

    ray.get(warmup_refs)

    results = []

    print("=" * 80)
    print("CALCULUS-BASED PI APPROXIMATION USING RAY")
    print(f"CPU cores used: {args.cores}")
    print("=" * 80)

    for n in STEPS:
        dx = 1.0 / n

        ranges = split_range(n, args.cores)

        start_time = time.perf_counter()

        refs = [
            calculate_partial_sum.remote(start, end, n)
            for start, end in ranges
        ]

        partial_results = ray.get(refs)

        total = sum(partial_results)

        estimated_pi = total * dx

        elapsed = time.perf_counter() - start_time

        absolute_error = abs(estimated_pi - math.pi)

        relative_error_percent = (
            absolute_error / math.pi
        ) * 100.0

        results.append(
            {
                "steps": n,
                "cores": args.cores,
                "estimated_pi": estimated_pi,
                "absolute_error": absolute_error,
                "relative_error_percent": relative_error_percent,
                "runtime_seconds": elapsed,
            }
        )

        print(f"\nSteps                 : {n}")
        print(f"CPU cores             : {args.cores}")
        print(f"Estimated Pi          : {estimated_pi:.12f}")
        print(f"Actual Pi             : {math.pi:.12f}")
        print(f"Absolute error        : {absolute_error:.12e}")
        print(
            f"Relative error (%)    : "
            f"{relative_error_percent:.12e}"
        )
        print(f"Runtime (seconds)     : {elapsed:.6f}")

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "steps",
            "cores",
            "estimated_pi",
            "absolute_error",
            "relative_error_percent",
            "runtime_seconds",
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")

    ray.shutdown()


if __name__ == "__main__":
    main()
