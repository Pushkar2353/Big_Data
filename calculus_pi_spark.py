import argparse
import csv
import math
import time

from pyspark.sql import SparkSession


STEPS = [1000, 10000, 100000, 1000000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed calculus-based Pi approximation using Spark."
    )
    parser.add_argument(
        "--cores",
        type=int,
        choices=[1, 2],
        required=True,
        help="Number of Spark CPU cores used for the experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName(f"CalculusPiSpark-{args.cores}CPU")
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    output_file = f"calculus_spark_{args.cores}cpu.csv"

    # Small warm-up job so Spark initialization does not dominate
    # the first measured experiment.
    sc.parallelize(range(1000), args.cores).sum()

    results = []

    print("=" * 80)
    print("CALCULUS-BASED PI APPROXIMATION USING SPARK")
    print(f"CPU cores used: {args.cores}")
    print("=" * 80)

    for n in STEPS:
        dx = 1.0 / n

        start_time = time.perf_counter()

        rdd = sc.parallelize(range(n), args.cores)

        total = (
            rdd
            .map(lambda i: 4.0 / (1.0 + (i * dx) ** 2))
            .sum()
        )

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

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
