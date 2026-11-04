# HTTP clients

Small clients for the Evencut service (`uvicorn evencut.service:app`). Each one sends a
`User-Agent`, retries 429 / 5xx / network errors with exponential backoff, and turns error
responses into a typed error.

| File | Runtime | Run |
|---|---|---|
| `evencut_client.py` | Python 3.9+, standard library only | `python evencut_client.py` |
| `evencut_client.mjs` | Node.js 18+ (or a browser) | `node evencut_client.mjs` |
| `evencut_client.go` | Go 1.21+ | `go run evencut_client.go` (not run in this repository's checks) |

Set `EVENCUT_URL` to point at another server (default `http://127.0.0.1:8000`). The full API is
described in `../../openapi.json`; any OpenAPI generator can produce clients for other languages.

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s -X POST http://127.0.0.1:8000/v1/partition -H "Content-Type: application/json" \
     -d '{"weights":[5,3,8,2,7,4,6,1,9,5],"k":3,"objective":"sum_squares"}'
curl -s -X POST http://127.0.0.1:8000/v1/ai/pipeline-stages -H "Content-Type: application/json" \
     -d '{"layer_costs":[1,1,1,1,1,1,1,1,1,1,1,1],"num_stages":3,"embedding_cost":2,"head_cost":3}'
```
