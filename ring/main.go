package main

import (
	"flag"
	"log"
	"math/rand"
	"net"
	"strconv"
	"time"

	"allreduce/proto"
)

func main() {
	rank := flag.Int("rank", 0, "this node's rank in the ring")
	n := flag.Int("n", 4, "total number of nodes")
	size := flag.Int("size", 1000000, "array size (must be divisible by n)")
	basePort := flag.Int("port", 9100, "base port; node i listens on basePort+i")
	bw := flag.Float64("bw", 0, "link bandwidth in MB/s (0 = unlimited)")
	flag.Parse()

	var limiter *proto.RateLimiter
	if *bw > 0 {
		limiter = proto.NewRateLimiter(*bw * 1e6)
	}

	if *size%*n != 0 {
		log.Fatalf("size %d must be divisible by n %d", *size, *n)
	}

	// Generate gradient data seeded by rank so each node has a different array.
	rng := rand.New(rand.NewSource(int64(*rank + 1)))
	data := make([]float64, *size)
	for i := range data {
		data[i] = rng.Float64()
	}

	// Split into N chunks. Each chunk is a subslice of data (no copy).
	chunkSize := *size / *n
	chunks := make([][]float64, *n)
	for i := range chunks {
		chunks[i] = data[i*chunkSize : (i+1)*chunkSize]
	}

	// Step 1: start listening for the left neighbor's connection.
	myAddr := ":" + strconv.Itoa(*basePort+*rank)
	ln, err := net.Listen("tcp", myAddr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	// Step 2: connect to right neighbor, retrying until it's listening.
	rightAddr := "localhost:" + strconv.Itoa(*basePort+(*rank+1)%*n)
	var rightConn net.Conn
	for {
		rightConn, err = net.Dial("tcp", rightAddr)
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Step 3: accept the connection from the left neighbor.
	leftConn, err := ln.Accept()
	if err != nil {
		log.Fatalf("accept: %v", err)
	}
	ln.Close()

	// Wrap the outbound connection with the rate limiter (if set).
	// Each node throttles its own sends, simulating a dedicated link.
	rightConn = proto.NewThrottledConn(rightConn, limiter)

	log.Printf("rank %d: ring established (left=%s, right=%s)",
		*rank, leftConn.RemoteAddr(), rightAddr)

	start := time.Now()

	// --- Scatter-Reduce: N-1 rounds ---
	// Each round, node i sends chunk (i-round) mod N to the right and receives
	// chunk (i-round-1) mod N from the left, adding it into its local copy.
	// After N-1 rounds, node i holds the fully reduced chunk (i+1) mod N.
	for round := 0; round < *n-1; round++ {
		sendIdx := (*rank - round + *n) % *n
		recvIdx := (*rank - round - 1 + *n) % *n

		received, err := sendRecv(rightConn, leftConn, chunks[sendIdx])
		if err != nil {
			log.Fatal(err)
		}
		for j := range received {
			chunks[recvIdx][j] += received[j]
		}
	}

	// --- AllGather: N-1 rounds ---
	// Each round, node i sends the chunk it received last round to the right.
	// Receivers copy (not add) into their local chunk.
	// After N-1 rounds, every node has all N fully reduced chunks.
	for round := 0; round < *n-1; round++ {
		sendIdx := (*rank + 1 - round + *n) % *n
		recvIdx := (*rank - round + *n) % *n

		received, err := sendRecv(rightConn, leftConn, chunks[sendIdx])
		if err != nil {
			log.Fatal(err)
		}
		copy(chunks[recvIdx], received)
	}

	elapsed := time.Since(start)
	log.Printf("rank %d: ring allreduce done in %v, result[0]=%.6f (chunk size=%d)",
		*rank, elapsed, data[0], chunkSize)

	rightConn.Close()
	leftConn.Close()
}

// sendRecv sends a chunk to the right neighbor and receives a chunk from the
// left neighbor concurrently. Concurrency is required — doing send then recv
// sequentially would deadlock since every node would be blocked sending.
func sendRecv(right, left net.Conn, send []float64) ([]float64, error) {
	type recvResult struct {
		data []float64
		err  error
	}
	ch := make(chan recvResult, 1)

	go func() {
		data, err := proto.RecvArray(left)
		ch <- recvResult{data, err}
	}()

	if err := proto.SendArray(right, send); err != nil {
		return nil, err
	}

	r := <-ch
	return r.data, r.err
}
