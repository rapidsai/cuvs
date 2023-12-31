# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections import defaultdict
import json

import pandas as pd
import numpy as np


def load_dataframe(filename):
    """Loads up the select_k benchmark times as a pandas dataframe

    This loads up the timings from the MATRIX_BENCH script into a pandas dataframe
    This file is generated by running:

    ./cpp/build/MATRIX_BENCH --benchmark_filter=SelectKDataset \
        --benchmark_out_format=json \
        --benchmark_out=select_k_times.json \
        --select_k_dataset

    Note running these MATRIX_BENCH tests takes over 24 hours right now
    """
    benchmarks = json.load(open(filename))["benchmarks"]
    df = pd.DataFrame(benchmarks, columns=["real_time", "run_name"])
    run_info = [
        run[1:4] + list(map(int, run[4:9]))
        for run in df.run_name.str.split("/").tolist()
    ]
    df[
        [
            "key_type",
            "index_type",
            "algo",
            "row",
            "col",
            "k",
            "use_index_input",
            "use_memory_pool",
        ]
    ] = pd.DataFrame(run_info, index=df.index)
    df["time"] = df["real_time"] / 1000
    df = df.drop(["run_name", "real_time"], axis=1)
    df = df.sort_values(
        by=[
            "k",
            "row",
            "col",
            "key_type",
            "index_type",
            "use_index_input",
            "use_memory_pool",
        ]
    )
    df = df.reset_index(drop=True)
    return df


def get_dataset(df):
    """Returns the training features, labels and sample weights from a dataframe"""
    # group the dataframe by the input features
    feature_algo_time = defaultdict(list)
    for row in df.itertuples():
        feature_algo_time[
            (
                row.k,
                row.row,
                row.col,
                row.use_memory_pool,
                row.key_type,
                row.index_type,
            )
        ].append((row.algo, row.time))

    # get the features (x), labels (y) and sample_weights from the grouped times
    X, y, weights = [], [], []
    for feature, algo_times in feature_algo_time.items():
        # we can't yet handle the dtype values in training, remove
        feature = feature[:-2]

        # figure out the fastest algorithm for this set of features
        algo_times = sorted(algo_times, key=lambda x: x[1])
        best_algo, best_time = algo_times[0]

        # set the sample_weight to the absolute speed increase above the
        # time of the next fastest algorithm. the idea here is that
        # we really want to capture the 2x or 10x speedups - but
        # the 1% speedups might just be noise (and this is especially
        # true for the faster runs)
        if len(algo_times) == 1:
            # no other algorithm handles this K value,
            second_best_time = np.inf
        else:
            second_best_time = algo_times[1][1]

        # sample_weight = min((second_best_time / best_time) - 1, 10)
        sample_weight = min((second_best_time - best_time), 10)

        X.append(feature)
        y.append(best_algo)
        weights.append(sample_weight)

    return np.array(X), np.array(y), np.array(weights)
