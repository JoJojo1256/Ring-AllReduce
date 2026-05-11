# Collective Communication Algorithms for Distributed Training

> If the video doesn't work, here is a link: https://www.loom.com/share/d5b4429eb9e249dc95307492ce6adb7e

CSCI 1680 Final Project — Jo & Jeff

Implementation of two AllReduce algorithms used in large-scale distributed ML training: a naive parameter-server approach and Ring-AllReduce. Simulated using Go processes communicating over TCP on a single machine, with per-link bandwidth limiting to model dedicated network hardware.

## Background

When training a neural network across multiple GPUs, each GPU computes its own copy of the gradients on its local batch of data. Before the next training step, all GPUs need to agree on the same gradient values — typically the sum (or average) across all workers. This operation is called **AllReduce**.

The naive way to do this is to route everything through a single parameter server: workers send their gradients up, the server sums them, and broadcasts the result back down. This works, but the server's bandwidth becomes a bottleneck that grows linearly with the number of workers.

**Ring-AllReduce**, described in [a blog post by Andrew Gibiansky at Baidu Research](http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) and later implemented in NVIDIA's NCCL, fixes this by arranging nodes in a logical ring and having every node send and receive simultaneously. The communication cost per node approaches a constant (roughly `2 × array_size × bytes_per_float`) regardless of how many workers you add.

## Algorithms

### Naive AllReduce

One designated reducer node:
1. Accepts connections from all N workers
2. Receives the full gradient array from each worker concurrently
3. Sums all arrays element-wise
4. Broadcasts the result back to each worker sequentially

**Bottleneck:** the reducer must receive `N × array_size` bytes and re-send `N × array_size` bytes. Both phases grow linearly with N. At large N the OS socket buffers can be exhausted entirely.

### Ring-AllReduce

Nodes are arranged in a ring. Each node only ever communicates with its two immediate neighbors. The algorithm runs in two phases:

**Scatter-Reduce (N-1 rounds):** The array is split into N equal chunks. Each round, every node simultaneously sends a chunk to its right neighbor and receives a chunk from its left neighbor, accumulating (adding) into its local copy. After N-1 rounds, every node holds the fully reduced version of exactly one chunk.

**AllGather (N-1 rounds):** Each node passes its finished chunk around the ring. Receivers copy rather than accumulate. After N-1 rounds, every node has the complete reduced array.

**Why the cost is constant:** Each node sends and receives exactly one chunk per round, for `2(N-1)` rounds total. Chunk size is `array_size / N`, so total bytes sent per node = `2(N-1)/N × array_size ≈ 2 × array_size` as N grows — independent of N.

## Implementation

```
proto/proto.go      wire format for float64 arrays + per-link rate limiter
reducer/main.go     naive parameter server
worker/main.go      naive worker client
ring/main.go        ring-allreduce node (both client and server)
run.sh              launch naive allreduce
run_ring.sh         launch ring allreduce
```

**Wire format:** 4-byte big-endian `uint32` length prefix followed by the array as big-endian `float64` values. `encoding/binary` writes the entire slice in one call to avoid per-element syscall overhead.

**Ring topology:** Each node listens on `basePort + rank` and connects to `basePort + (rank+1) % N`. All nodes start listening before dialing, with a retry loop to handle startup races. Each round's send and receive happen in separate goroutines to prevent deadlock (if every node tried to send before receiving, the ring would stall).

**Rate limiting:** Each outbound connection is wrapped in a token bucket rate limiter. This simulates dedicated per-link bandwidth — the key hardware assumption behind Ring-AllReduce's constant-cost guarantee. Without it, all connections share the loopback interface and the ring's scaling advantage is obscured. Setting `-bw 100` (100 MB/s per link) brings the observed scaling behavior in line with the theoretical model.

## Running

```bash
# build everything
go build ./...

# naive allreduce: N workers, array size, bandwidth in MB/s per link
./run.sh [N] [array_size] [bw_mbps]

# ring allreduce: N nodes, array size, bandwidth in MB/s per link
./run_ring.sh [N] [array_size] [bw_mbps]

# example: 8 workers, 1M floats, 100 MB/s simulated links
./run.sh 8 1000000 100
./run_ring.sh 8 1000000 100
```

Array size must be divisible by N for the ring. Omit `bw_mbps` or pass 0 for unlimited (loopback speed).

## Results

All runs: 1,000,000 float64 values (~8 MB per worker), 100 MB/s simulated link bandwidth. Times are the last worker to finish (worst case).

| N workers | Naive (ms) | Ring (ms) |
|-----------|------------|-----------|
| 2         | 35         | 14        |
| 4         | 52         | 20        |
| 8         | 154        | 43        |
| 16        | crashed    | 69        |

Naive roughly doubles with each doubling of workers. Ring grows slowly — from 2 to 16 nodes, time increases by ~5× for naive vs ~5× for ring in absolute terms, but ring's growth is driven by round-trip latency accumulation (2(N-1) rounds), not bandwidth — in a real cluster with sub-millisecond links the ring line would be nearly flat.

**The n=16 crash** is not a bug. With 16 workers each writing 8MB simultaneously to one reducer, the OS socket buffer pool (mbufs on macOS) is exhausted and writes return `ENOBUFS` instead of blocking. This illustrates why the naive approach has a hard scalability ceiling, not just a performance one. The ring never hits this: each node's maximum in-flight data is one chunk (`array_size / N`), which shrinks as N grows.

## Design Notes

**Why simulate on one machine?** Real GPU clusters are expensive and unavailable for coursework. Processes on the same machine communicate over the loopback interface, which behaves like a network but at memory speed. The rate limiter makes each connection behave as if it has dedicated bandwidth, restoring the hardware assumption the ring algorithm relies on.

**Why throttle sends only?** In our model the sender is responsible for respecting link capacity. The receiver reads as fast as it can, which matches real NIC behavior where the bottleneck is the transmitter's line rate.

**Standard library only:** `net`, `sync`, `encoding/binary`, `math/rand`, `time`. No external dependencies.
