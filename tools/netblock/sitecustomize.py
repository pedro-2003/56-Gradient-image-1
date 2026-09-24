"""Python-level network cut-off for venv runs (the validator runs the trainer with
`docker run --network none`; Vast instances are containers, so docker cannot be used there).

Put this directory first on PYTHONPATH (tools/local_run_venv.sh does it when NETBLOCK=1, the
default): every socket connect to a non-loopback address raises, in the entrypoint and in the
trainer it spawns. Any hidden download dependency therefore fails loudly instead of passing here
and failing on the validator.
"""
import os
import socket

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection


def _is_local(address):
    try:
        host = address[0] if isinstance(address, tuple) else address
    except Exception:  # noqa: BLE001
        return False
    return host in ("127.0.0.1", "::1", "localhost", "")


def _blocked(*a, **k):
    raise OSError("NETBLOCK: network access is disabled for this run (the validator uses --network none)")


def _connect(self, address):
    if _is_local(address):
        return _orig_connect(self, address)
    _blocked()


def _connect_ex(self, address):
    if _is_local(address):
        return _orig_connect_ex(self, address)
    _blocked()


def _create_connection(address, *a, **k):
    if _is_local(address):
        return _orig_create_connection(address, *a, **k)
    _blocked()


if os.environ.get("NETBLOCK", "1") != "0":
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.create_connection = _create_connection
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
