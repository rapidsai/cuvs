#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  -s -- -y
source "$HOME/.cargo/env"
export PATH=$PATH:$HOME/.cargo/bin
cd rust
cargo test
