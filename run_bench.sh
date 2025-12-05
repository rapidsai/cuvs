#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

export RAPIDS_DATASET_ROOT_DIR=/raid/mide/rapids
DATASET=mnist-784-euclidean

python -m cuvs_bench.get_dataset --dataset $DATASET --normalize
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 1 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 2 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 4 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 5 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 10 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 20 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 50 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
python -m cuvs_bench.run --dataset $DATASET --algorithms cuvs_ivf_flat --batch-size 10 -k 100 --groups test -m throughput --dataset-path $RAPIDS_DATASET_ROOT_DIR
mv $RAPIDS_DATASET_ROOT_DIR/$DATASET/result/search/ $RAPIDS_DATASET_ROOT_DIR/$DATASET/result/search_256
