#!/usr/bin/env bash
# Usage: ./run.sh [num_workers] [array_size] [bandwidth_mb]
# Defaults: 4 workers, 1,000,000 floats, unlimited bandwidth

set -e

N=${1:-4}
SIZE=${2:-1000000}
BW=${3:-0}
ADDR="localhost:9000"

echo "==> building..."
go build -o bin/reducer ./reducer
go build -o bin/worker ./worker

echo "==> starting reducer (n=$N)..."
./bin/reducer -addr "$ADDR" -n "$N" -bw "$BW" &
REDUCER_PID=$!

# Give the reducer a moment to start listening.
sleep 0.2

echo "==> starting $N workers (size=$SIZE floats each)..."
for i in $(seq 0 $((N - 1))); do
    ./bin/worker -addr "$ADDR" -id "$i" -size "$SIZE" -bw "$BW" &
done

wait
echo "==> done"
