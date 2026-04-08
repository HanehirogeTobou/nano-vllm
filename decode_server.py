"""Standalone Decode Server for nano-vllm PD disaggregation.

Accepts KV cache transfers from the prefill server on a background thread
and runs the decode loop continuously.  Results can be polled over HTTP.

Usage
-----
    # Terminal 1 – start decode server first:
    python decode_server.py --model ~/huggingface/Qwen3-0.6B \\
        --kv-port 29500 --http-port 8001

    # Terminal 2 – start prefill server:
    python prefill_server.py --model ~/huggingface/Qwen3-0.6B \\
        --decode-host localhost --kv-port 29500 --http-port 8000

    # Submit a request to the prefill server:
    SEQ_ID=$(curl -s -X POST http://localhost:8000/generate \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "Hello!", "max_tokens": 64}' | python -c "import sys,json; print(json.load(sys.stdin)['seq_id'])")

    # Poll the decode server for the result:
    curl -s "http://localhost:8001/result?seq_id=$SEQ_ID" | python -m json.tool

HTTP API
--------
GET /result?seq_id=<int>
    Response: {"status": "done",    "seq_id": int, "text": str}
           or {"status": "pending", "seq_id": int}

GET /results
    Response: {"results": [{"seq_id": int, "text": str}, ...]}
    Lists all completed sequences.

GET /health
    Response: {"status": "ok", "running": int, "finished": int}
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from nanovllm import DecodeEngine, SamplingParams


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DecodeHandler(BaseHTTPRequestHandler):

    engine: DecodeEngine       # set at server start
    results_lock: threading.Lock

    def log_message(self, fmt: str, *args) -> None:
        pass

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/health":
            with DecodeHandler.results_lock:
                finished = len(self.engine._results)
                running = len(self.engine.scheduler.running)
            self._send_json(200, {"status": "ok", "running": running, "finished": finished})

        elif parsed.path == "/result":
            if "seq_id" not in params:
                self._send_json(400, {"error": "missing seq_id"})
                return
            seq_id = int(params["seq_id"][0])
            with DecodeHandler.results_lock:
                tids = self.engine.get_result(seq_id)
            if tids is None:
                self._send_json(200, {"status": "pending", "seq_id": seq_id})
            else:
                text = self.engine.tokenizer.decode(tids)
                self._send_json(200, {"status": "done", "seq_id": seq_id, "text": text})

        elif parsed.path == "/results":
            with DecodeHandler.results_lock:
                results = [
                    {"seq_id": sid, "text": self.engine.tokenizer.decode(tids)}
                    for sid, tids in self.engine._results.items()
                ]
            self._send_json(200, {"results": results})

        else:
            self._send_json(404, {"error": "not found"})


# ---------------------------------------------------------------------------
# Background decode loop
# ---------------------------------------------------------------------------

def _decode_loop(engine: DecodeEngine, lock: threading.Lock, poll_interval: float = 0.001) -> None:
    """Run the decode engine continuously in a background thread."""
    while True:
        with lock:
            engine.step()
        if not engine.has_pending():
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="nano-vllm Decode Server")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--kv-host", default="0.0.0.0",
                        help="Address to listen for KV transfers (default: 0.0.0.0)")
    parser.add_argument("--kv-port", type=int, default=29500,
                        help="Port to listen for KV transfers (default: 29500)")
    parser.add_argument("--http-port", type=int, default=8001,
                        help="HTTP port for this decode server (default: 8001)")
    parser.add_argument("--http-host", default="0.0.0.0",
                        help="HTTP bind address (default: 0.0.0.0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    engine = DecodeEngine(
        model_path,
        kv_host=args.kv_host,
        kv_port=args.kv_port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    results_lock = threading.Lock()
    DecodeHandler.engine = engine
    DecodeHandler.results_lock = results_lock

    decode_thread = threading.Thread(
        target=_decode_loop,
        args=(engine, results_lock),
        daemon=True,
    )
    decode_thread.start()

    server = HTTPServer((args.http_host, args.http_port), DecodeHandler)
    print(f"[decode] KV receiver listening on {args.kv_host}:{args.kv_port}")
    print(f"[decode] HTTP server listening on {args.http_host}:{args.http_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[decode] Shutting down")


if __name__ == "__main__":
    main()
