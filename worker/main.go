package main

import (
	"flag"
	"log"
	"math/rand"
	"net"
	"time"

	"allreduce/proto"
)

func main() {
	addr := flag.String("addr", "localhost:9000", "reducer address")
	id := flag.Int("id", 0, "worker ID (used as rand seed)")
	size := flag.Int("size", 1000000, "number of float64 gradient values")
	bw := flag.Float64("bw", 0, "link bandwidth in MB/s (0 = unlimited)")
	flag.Parse()

	var limiter *proto.RateLimiter
	if *bw > 0 {
		limiter = proto.NewRateLimiter(*bw * 1e6)
	}

	// Generate a random gradient array seeded by worker ID so each worker
	// has a different array (but runs are reproducible).
	rng := rand.New(rand.NewSource(int64(*id + 1)))
	gradients := make([]float64, *size)
	for i := range gradients {
		gradients[i] = rng.Float64()
	}

	conn, err := net.Dial("tcp", *addr)
	if err != nil {
		log.Fatalf("connecting to reducer: %v", err)
	}
	defer conn.Close()
	log.Printf("worker %d connected to reducer at %s", *id, *addr)

	start := time.Now()

	tc := proto.NewThrottledConn(conn, limiter)
	if err := proto.SendArray(tc, gradients); err != nil {
		log.Fatalf("sending gradients: %v", err)
	}

	result, err := proto.RecvArray(conn)
	if err != nil {
		log.Fatalf("receiving result: %v", err)
	}

	elapsed := time.Since(start)
	log.Printf("worker %d: allreduce done in %v, result[0]=%.6f (array len=%d)",
		*id, elapsed, result[0], len(result))
}
