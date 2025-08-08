#!/bin/bash

# Build and run Rust kmeans example in Docker container
# This script will:
# 1. Build the Docker container with all dependencies
# 2. Build the cuVS library and Rust bindings
# 3. Run the kmeans example

set -e

echo "🚀 Building cuVS Rust kmeans example..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU acceleration may not work."
    echo "   Make sure you have nvidia-docker2 installed."
fi

echo "📦 Building Docker container..."
docker-compose -f docker-compose.rust-kmeans.yml build

echo "🔧 Starting container and building cuVS..."
docker-compose -f docker-compose.rust-kmeans.yml up -d

echo "⏳ Waiting for container to be ready..."
sleep 5

echo "🏗️  Building cuVS library and Rust bindings..."
docker-compose -f docker-compose.rust-kmeans.yml exec cuvs-rust-kmeans ./build.sh libcuvs rust

echo "🧪 Running kmeans example..."
docker-compose -f docker-compose.rust-kmeans.yml exec cuvs-rust-kmeans cargo run --example kmeans --manifest-path rust/cuvs/Cargo.toml

echo "✅ Done! The kmeans example has been executed."
echo ""
echo "To run the example again, use:"
echo "  docker-compose -f docker-compose.rust-kmeans.yml exec cuvs-rust-kmeans cargo run --example kmeans --manifest-path rust/cuvs/Cargo.toml"
echo ""
echo "To get an interactive shell in the container, use:"
echo "  docker-compose -f docker-compose.rust-kmeans.yml exec cuvs-rust-kmeans bash"
echo ""
echo "To stop the container, use:"
echo "  docker-compose -f docker-compose.rust-kmeans.yml down" 