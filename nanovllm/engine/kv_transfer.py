"""KV cache transfer utilities for Prefill-Decode (PD) disaggregation.

KVSender  — runs on the prefill instance; connects to the decode instance
            and sends KV cache blocks together with sequence metadata.

KVReceiver — runs on the decode instance; listens for incoming transfers
             from one or more prefill instances and exposes received items
             through a thread-safe queue.

Wire format (per message):
    [4B little-endian header_len] [header_bytes (pickle)] [kv_bytes (raw)]

The header is a pickled dict that contains all sequence metadata plus the
shape, dtype, and byte-length of the KV payload so that the receiver can
allocate the exact read buffer.
"""

from __future__ import annotations

import io
import pickle
import queue as _queue
import socket
import struct
import threading
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Low-level socket helpers
# ---------------------------------------------------------------------------

def _send_all(sock: socket.socket, data: bytes) -> None:
    total = len(data)
    sent = 0
    while sent < total:
        n = sock.send(data[sent:])
        if n == 0:
            raise ConnectionError("Socket connection broken during send")
        sent += n


def _recv_all(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    received = 0
    while received < n:
        count = sock.recv_into(view[received:], n - received)
        if count == 0:
            raise ConnectionError("Socket connection broken during recv")
        received += count
    return bytes(buf)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialise a tensor to raw bytes, handling bfloat16 via torch.save."""
    buf = io.BytesIO()
    torch.save(t.cpu().contiguous(), buf)
    return buf.getvalue()


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Deserialise a tensor produced by _tensor_to_bytes."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Transfer metadata
# ---------------------------------------------------------------------------

@dataclass
class KVTransferMeta:
    seq_id: int
    token_ids: list
    num_prompt_tokens: int
    temperature: float
    max_tokens: int
    ignore_eos: bool
    kv_len: int          # byte-length of the serialised KV tensor


# ---------------------------------------------------------------------------
# Sender  (prefill side)
# ---------------------------------------------------------------------------

class KVSender:
    """Sends KV cache blocks and sequence metadata to a decode instance."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self, timeout: float = 60.0) -> None:
        """Connect to the KVReceiver, retrying until *timeout* seconds."""
        import time
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                self._sock = sock
                return
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)

    def send(
        self,
        seq_id: int,
        token_ids: list[int],
        num_prompt_tokens: int,
        temperature: float,
        max_tokens: int,
        ignore_eos: bool,
        kv_data: torch.Tensor,
    ) -> None:
        """Send sequence metadata and KV cache blocks.

        Args:
            kv_data: shape ``[2, num_layers, num_kv_blocks, block_size,
                               num_kv_heads, head_dim]``
        """
        kv_bytes = _tensor_to_bytes(kv_data)
        meta = KVTransferMeta(
            seq_id=seq_id,
            token_ids=token_ids,
            num_prompt_tokens=num_prompt_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            kv_len=len(kv_bytes),
        )
        meta_bytes = pickle.dumps(meta)

        with self._lock:
            # Frame: [4B meta_len][meta_bytes][kv_bytes]
            _send_all(self._sock, struct.pack("<I", len(meta_bytes)))
            _send_all(self._sock, meta_bytes)
            _send_all(self._sock, kv_bytes)

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ---------------------------------------------------------------------------
# Receiver  (decode side)
# ---------------------------------------------------------------------------

class KVReceiver:
    """Listens for KV transfers from prefill instances.

    Received ``(KVTransferMeta, kv_tensor)`` pairs are placed in
    :attr:`queue` for the decode engine to consume.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.queue: _queue.Queue[tuple[KVTransferMeta, torch.Tensor]] = (
            _queue.Queue()
        )
        self._server_sock: socket.socket | None = None
        self._running = False

    def start(self) -> None:
        """Start the background listener thread."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(16)
        self._running = True
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
                t = threading.Thread(
                    target=self._recv_loop, args=(conn,), daemon=True
                )
                t.start()
            except OSError:
                break

    def _recv_loop(self, conn: socket.socket) -> None:
        try:
            while self._running:
                raw_header_len = _recv_all(conn, 4)
                meta_len = struct.unpack("<I", raw_header_len)[0]
                meta: KVTransferMeta = pickle.loads(_recv_all(conn, meta_len))
                kv_bytes = _recv_all(conn, meta.kv_len)
                kv_data = _bytes_to_tensor(kv_bytes)
                self.queue.put((meta, kv_data))
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
