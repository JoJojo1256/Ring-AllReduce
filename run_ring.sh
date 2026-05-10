#!/usr/bin/env bash
# Usage: ./run_ring.sh [num_nodes] [array_size] [bandwidth_mb]
# Defaults: 4 nodes, 1,000,000 floats, unlimited bandwidth

set -e

N=${1:-4}
SIZE=${2:-1000000}
BW=${3:-0}
BASE_PORT=9100

echo "==> building..."
go build -o bin/ring ./ring

echo "==> starting $N ring nodes (size=$SIZE floats each)..."
for i in $(seq 0 $((N - 1))); do
    ./bin/ring -rank "$i" -n "$N" -size "$SIZE" -port "$BASE_PORT" -bw "$BW" &
done

wait
echo "==> done"
