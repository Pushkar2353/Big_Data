import argparse
import csv
import math
import time

from pyspark.sql import SparkSession


N_VALUES = [100, 1000, 10000, 100000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed series-based Pi approximation using Spark."
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

    spark = (
        SparkSession.builder
        .appName(f"SeriesPiSpark-{args.cores}CPU")
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    output_file = f"series_spark_{args.cores}cpu.csv"

    sc.parallelize(range(1000), args.cores).sum()

    results = []

    print("=" * 80)
    print("SERIES-BASED PI APPROXIMATION USING SPARK")
    print(f"CPU cores used: {args.cores}")
    print("=" * 80)

    for n in N_VALUES:
        start_time = time.perf_counter()

        series_sum = (
            sc.parallelize(range(1, n + 1), args.cores)
            .map(lambda i: 1.0 / (i * i))
            .sum()
        )

        estimated_pi = math.sqrt(6.0 * series_sum)

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

    spark.stop()


if __name__ == "__main__":
    main()
