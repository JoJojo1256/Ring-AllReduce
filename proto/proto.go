// Package proto handles binary serialization of float64 arrays over TCP.
//
// Wire format: [4-byte big-endian uint32 length] [length × 8-byte big-endian float64]
package proto

import (
	"encoding/binary"
	"fmt"
	"net"
	"sync"
	"time"
)

// SendArray writes a float64 slice to conn using the wire format.
func SendArray(conn net.Conn, data []float64) error {
	length := uint32(len(data))
	if err := binary.Write(conn, binary.BigEndian, length); err != nil {
		return err
	}
	return binary.Write(conn, binary.BigEndian, data)
}

// RecvArray reads a float64 slice from conn using the wire format.
func RecvArray(conn net.Conn) ([]float64, error) {
	var length uint32
	if err := binary.Read(conn, binary.BigEndian, &length); err != nil {
		return nil, fmt.Errorf("reading length: %w", err)
	}
	data := make([]float64, length)
	if err := binary.Read(conn, binary.BigEndian, data); err != nil {
		return nil, fmt.Errorf("reading array: %w", err)
	}
	return data, nil
}

// RateLimiter is a token bucket that limits throughput to a fixed bytes/sec.
// Each call to Wait(n) blocks until n bytes worth of tokens are available,
// simulating a dedicated link with fixed bandwidth.
type RateLimiter struct {
	rate     float64 // bytes per second
	tokens   float64 // current token count
	lastFill time.Time
	mu       sync.Mutex
}

func NewRateLimiter(bytesPerSec float64) *RateLimiter {
	return &RateLimiter{
		rate:     bytesPerSec,
		tokens:   bytesPerSec, // start with a full bucket
		lastFill: time.Now(),
	}
}

// Wait blocks until n tokens are available, then consumes them.
func (r *RateLimiter) Wait(n int) {
	r.mu.Lock()

	// Refill tokens based on time elapsed since last call.
	now := time.Now()
	r.tokens += now.Sub(r.lastFill).Seconds() * r.rate
	r.lastFill = now
	if r.tokens > r.rate {
		r.tokens = r.rate // cap bucket at 1 second's worth
	}

	r.tokens -= float64(n)
	var sleep time.Duration
	if r.tokens < 0 {
		// Deficit: sleep until enough tokens accumulate.
		sleep = time.Duration(-r.tokens / r.rate * float64(time.Second))
		r.tokens = 0
	}
	r.mu.Unlock()

	if sleep > 0 {
		time.Sleep(sleep)
	}
}

// ThrottledConn wraps a net.Conn and rate-limits writes.
// Reads are not throttled — in our simulation the sender is responsible
// for respecting link bandwidth.
type ThrottledConn struct {
	net.Conn
	limiter *RateLimiter
}

func NewThrottledConn(conn net.Conn, limiter *RateLimiter) net.Conn {
	if limiter == nil {
		return conn
	}
	return &ThrottledConn{Conn: conn, limiter: limiter}
}

func (c *ThrottledConn) Write(b []byte) (int, error) {
	c.limiter.Wait(len(b))
	return c.Conn.Write(b)
}
