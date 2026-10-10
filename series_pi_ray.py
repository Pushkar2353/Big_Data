import argparse
import csv
import math
import time

import ray


N_VALUES = [100, 1000, 10000, 100000]


@ray.remote(num_cpus=1)
def calculate_partial_series(start_value, end_value):
    partial_sum = 0.0

    for i in range(start_value, end_value + 1):
        partial_sum += 1.0 / (i * i)

    return partial_sum


def split_range(n, parts):
    ranges = []

    base_size = n // parts
    remainder = n % parts

    start = 1

    for part in range(parts):
        extra = 1 if part < remainder else 0

        end = start + base_size + extra - 1

        ranges.append((start, end))

        start = end + 1

    return ranges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed series-based Pi approximation using Ray."
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

    output_file = f"series_ray_{args.cores}cpu.csv"

    warmup_ranges = split_range(1000, args.cores)

    warmup_refs = [
        calculate_partial_series.remote(start, end)
        for start, end in warmup_ranges
    ]

    ray.get(warmup_refs)

    results = []

    print("=" * 80)
    print("SERIES-BASED PI APPROXIMATION USING RAY")
    print(f"CPU cores used: {args.cores}")
    print("=" * 80)

    for n in N_VALUES:
        ranges = split_range(n, args.cores)

        start_time = time.perf_counter()

        refs = [
            calculate_partial_series.remote(start, end)
            for start, end in ranges
        ]

        partial_results = ray.get(refs)

        series_sum = sum(partial_results)

        estimated_pi = math.sqrt(
            6.0 * series_sum
        )

        elapsed = time.perf_counter() - start_time

        absolute_error = abs(
            estimated_pi - math.pi
        )

        relative_error_percent = (
            absolute_error / math.pi
        ) * 100.0

        tail_error_upper_bound = 1.0 / n

        results.append(
            {
                "N": n,
                "cores": args.cores,
                "estimated_pi": estimated_pi,
                "absolute_error": absolute_error,
                "relative_error_percent": relative_error_percent,
                "tail_error_upper_bound": tail_error_upper_bound,
                "runtime_seconds": elapsed,
            }
        )

        print(f"\nN                         : {n}")
        print(f"CPU cores                 : {args.cores}")
        print(f"Estimated Pi              : {estimated_pi:.12f}")
        print(f"Actual Pi                 : {math.pi:.12f}")
        print(
            f"Absolute Pi error          : "
            f"{absolute_error:.12e}"
        )
        print(
            f"Relative error (%)        : "
            f"{relative_error_percent:.12e}"
        )
        print(
            f"Series tail upper bound   : "
            f"{tail_error_upper_bound:.12e}"
        )
        print(f"Runtime (seconds)         : {elapsed:.6f}")

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "N",
            "cores",
            "estimated_pi",
            "absolute_error",
            "relative_error_percent",
            "tail_error_upper_bound",
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
