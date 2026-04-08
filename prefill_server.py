"""Standalone Prefill Server for nano-vllm PD disaggregation.

Receives generation requests over HTTP and runs the prefill phase, then
transfers the computed KV cache to the decode server.

Usage
-----
    # Terminal 1 – start the decode server first (it owns the KV receiver):
    python decode_server.py --model ~/huggingface/Qwen3-0.6B \\
        --kv-port 29500 --http-port 8001

    # Terminal 2 – start the prefill server:
    python prefill_server.py --model ~/huggingface/Qwen3-0.6B \\
        --decode-host localhost --kv-port 29500 --http-port 8000

    # Client:
    curl -s -X POST http://localhost:8000/generate \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Hello, world!", "max_tokens": 64}' | python -m json.tool
    # → {"seq_id": 0}

    curl -s "http://localhost:8001/result?seq_id=0" | python -m json.tool
    # → {"status": "done", "text": "..."}

HTTP API
--------
POST /generate
    Body (JSON): {"prompt": str | list[int], "max_tokens": int,
                  "temperature": float, "ignore_eos": bool}
    Response:    {"seq_id": int}

GET /health
    Response:    {"status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from nanovllm import PrefillEngine, SamplingParams


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class PrefillHandler(BaseHTTPRequestHandler):

    engine: PrefillEngine  # set at server start

    def log_message(self, fmt: str, *args) -> None:  # silence default access log
        pass

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        prompt = body.get("prompt")
        if prompt is None:
            self._send_json(400, {"error": "missing 'prompt'"})
            return

        sp = SamplingParams(
            temperature=float(body.get("temperature", 0.7)),
            max_tokens=int(body.get("max_tokens", 64)),
            ignore_eos=bool(body.get("ignore_eos", False)),
        )

        with PrefillHandler._lock:
            seq_id = self.engine.add_request(prompt, sp)
            self.engine.step()

        self._send_json(200, {"seq_id": seq_id})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="nano-vllm Prefill Server")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--decode-host", default="localhost",
                        help="Hostname of the decode server (default: localhost)")
    parser.add_argument("--kv-port", type=int, default=29500,
                        help="Port on which the decode server listens for KV transfers (default: 29500)")
    parser.add_argument("--http-port", type=int, default=8000,
                        help="HTTP port for this prefill server (default: 8000)")
    parser.add_argument("--http-host", default="0.0.0.0",
                        help="HTTP bind address (default: 0.0.0.0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    engine = PrefillEngine(
        model_path,
        kv_host=args.decode_host,
        kv_port=args.kv_port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    PrefillHandler.engine = engine
    PrefillHandler._lock = threading.Lock()

    server = HTTPServer((args.http_host, args.http_port), PrefillHandler)
    print(f"[prefill] Listening on {args.http_host}:{args.http_port}")
    print(f"[prefill] Forwarding KV to {args.decode_host}:{args.kv_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[prefill] Shutting down")


if __name__ == "__main__":
    main()
