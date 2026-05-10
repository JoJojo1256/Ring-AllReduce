package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"allreduce/proto"
)

func main() {
	addr := flag.String("addr", ":9000", "address to listen on")
	n := flag.Int("n", 2, "number of workers to wait for")
	bw := flag.Float64("bw", 0, "link bandwidth in MB/s per connection (0 = unlimited)")
	flag.Parse()

	var limiter *proto.RateLimiter
	if *bw > 0 {
		limiter = proto.NewRateLimiter(*bw * 1e6)
		log.Printf("rate limiting to %.0f MB/s per link", *bw)
	}

	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("reducer listening on %s, expecting %d workers", *addr, *n)

	// Accept all worker connections up front.
	conns := make([]net.Conn, *n)
	for i := range conns {
		c, err := ln.Accept()
		if err != nil {
			log.Fatalf("accept: %v", err)
		}
		conns[i] = c
		log.Printf("worker %d connected from %s", i, c.RemoteAddr())
	}

	// Receive gradient arrays from all workers concurrently.
	arrays := make([][]float64, *n)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var recvErr error

	start := time.Now()

	for i, c := range conns {
		wg.Add(1)
		go func(idx int, conn net.Conn) {
			defer wg.Done()
			arr, err := proto.RecvArray(conn)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				recvErr = fmt.Errorf("worker %d recv: %w", idx, err)
				return
			}
			arrays[idx] = arr
		}(i, c)
	}
	wg.Wait()

	if recvErr != nil {
		log.Fatal(recvErr)
	}

	// Validate all arrays are the same length.
	arrayLen := len(arrays[0])
	for i, arr := range arrays[1:] {
		if len(arr) != arrayLen {
			log.Fatalf("worker %d sent array of length %d, expected %d", i+1, len(arr), arrayLen)
		}
	}

	// Sum element-wise.
	result := make([]float64, arrayLen)
	for _, arr := range arrays {
		for j, v := range arr {
			result[j] += v
		}
	}

	elapsed := time.Since(start)
	totalBytes := arrayLen * 8 * *n // bytes received
	log.Printf("reduced %d workers × %d floats in %v (%.2f MB received)",
		*n, arrayLen, elapsed, float64(totalBytes)/1e6)

	// Broadcast result back to all workers.
	for i, c := range conns {
		tc := proto.NewThrottledConn(c, limiter)
		if err := proto.SendArray(tc, result); err != nil {
			log.Fatalf("sending result to worker %d: %v", i, err)
		}
	}

	log.Println("allreduce complete")
}
