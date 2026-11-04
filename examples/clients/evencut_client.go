// Minimal client for the Evencut HTTP service (Go standard library only).
//
//	go run evencut_client.go            # uses http://127.0.0.1:8000
//
// Not exercised in this repository's test run (no Go toolchain was installed); it uses only
// net/http and encoding/json.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const userAgent = "evencut-client-go/1.0.0"

// Partition mirrors the PartitionResponse schema in openapi.json.
type Partition struct {
	Objective    string   `json:"objective"`
	Parts        int      `json:"parts"`
	Ends         []int    `json:"ends"`
	Loads        []uint64 `json:"loads"`
	MaxLoad      uint64   `json:"max_load"`
	Total        uint64   `json:"total"`
	SumOfSquares string   `json:"sum_of_squares"`
	Imbalance    float64  `json:"imbalance"`
}

type APIError struct {
	Status int
	Detail string
}

func (e *APIError) Error() string { return fmt.Sprintf("HTTP %d: %s", e.Status, e.Detail) }

type Client struct {
	BaseURL string
	HTTP    *http.Client
	Retries int
}

func NewClient(baseURL string) *Client {
	return &Client{BaseURL: baseURL, HTTP: &http.Client{Timeout: 60 * time.Second}, Retries: 3}
}

func (c *Client) post(path string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequest(http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", userAgent)
		resp, err := c.HTTP.Do(req)
		if err == nil {
			if done, result := c.handle(resp, out, attempt); done {
				return result
			}
		} else if attempt >= c.Retries {
			return err
		}
		backoff := time.Duration(250*(1<<attempt)) * time.Millisecond
		if backoff > 4*time.Second {
			backoff = 4 * time.Second
		}
		time.Sleep(backoff)
	}
}

// handle consumes one response and closes its body before returning; done reports whether the
// caller should stop retrying.
func (c *Client) handle(resp *http.Response, out any, attempt int) (done bool, result error) {
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return true, json.NewDecoder(resp.Body).Decode(out)
	}
	retryable := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
	if !retryable || attempt >= c.Retries {
		var e struct {
			Detail any `json:"detail"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&e)
		return true, &APIError{Status: resp.StatusCode, Detail: fmt.Sprint(e.Detail)}
	}
	_, _ = io.Copy(io.Discard, resp.Body) // drain so the connection can be reused
	return false, nil
}

// Partition splits weights into k contiguous parts ("min_max" or "sum_squares").
func (c *Client) Partition(weights []uint64, k int, objective string) (*Partition, error) {
	var p Partition
	err := c.post("/v1/partition", map[string]any{"weights": weights, "k": k, "objective": objective}, &p)
	return &p, err
}

func main() {
	base := os.Getenv("EVENCUT_URL")
	if base == "" {
		base = "http://127.0.0.1:8000"
	}
	c := NewClient(base)
	p, err := c.Partition([]uint64{5, 3, 8, 2, 7, 4, 6, 1, 9, 5}, 3, "min_max")
	if err != nil {
		fmt.Println("error:", err)
		os.Exit(1)
	}
	fmt.Println("partition:", p.Ends, p.Loads, "max", p.MaxLoad)
}
