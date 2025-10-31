#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys


def split_groundtruth(groundtruth_filepath):
    ann_bench_scripts_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "split_groundtruth.pl"
    )
    pwd = os.getcwd()
    path_to_groundtruth = os.path.normpath(groundtruth_filepath).split(os.sep)
    if len(path_to_groundtruth) > 1:
        os.chdir(os.path.join(*path_to_groundtruth[:-1]))
    groundtruth_filename = path_to_groundtruth[-1]
    subprocess.run(
        [ann_bench_scripts_path, groundtruth_filename, "groundtruth"],
        check=True,
    )
    os.chdir(pwd)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--groundtruth",
        help="Path to billion-scale dataset groundtruth file",
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    split_groundtruth(args.groundtruth)


if __name__ == "__main__":
    main()
