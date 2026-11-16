"""Each snippet in practica/ runs and prints what the video shows (no GPU, no network)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRACTICA = ROOT / "practica"


def run(name: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PRACTICA / f"{name}.py")], cwd=ROOT,
                          capture_output=True, text=True, timeout=60, check=False)


def test_costo_api_matches_the_calculator():
    out = run("01_costo_api").stdout
    assert "al día: 20,000,000 de entrada y 2,000,000 de salida" in out
    assert "entrada 0.75, salida 3.75 -> $675 al mes" in out
    assert "entrada 1.5, salida 7.5 -> $1,350 al mes" in out


def test_costo_gpu_one_and_eight_users():
    out = run("02_costo_gpu").stdout
    assert "25.3 tok/s: 22.0 h de GPU al día, $936 al mes" in out
    assert "192.0 tok/s: 2.9 h de GPU al día, $341 al mes" in out


def test_cruce_api_wins_only_at_low_volume():
    lines = run("03_cruce").stdout.splitlines()
    assert len(lines) == 6
    assert lines[0].endswith("gana API")
    assert all(line.endswith("gana vLLM") for line in lines[1:])
    assert "$  341" in lines[3] and "$  2,700" in lines[3]
    assert "$  493" in lines[4] and "$  987" in lines[5]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), PRACTICA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_client(port: int, tmp_path: Path, name: str = "05_cliente"):
    """A client snippet as written, pointed at `port` instead of 8000."""
    src = (PRACTICA / f"{name}.py").read_text()
    assert "http://localhost:8000/v1" in src
    client = tmp_path / f"{name}.py"
    client.write_text(src.replace("localhost:8000", f"localhost:{port}"))
    return subprocess.run([sys.executable, str(client)], cwd=ROOT, capture_output=True,
                          text=True, timeout=60, check=False)


def test_fake_server_speaks_openai_chat_json(tmp_path):
    server = HTTPServer(("localhost", 0), load("04_servidor_falso").Falso)
    server.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = run_client(server.server_address[1], tmp_path).stdout
    finally:
        server.shutdown()
        server.server_close()
    assert "(respuesta falsa) recibí 8 palabras" in out
    assert "'prompt_tokens': 8" in out and "'completion_tokens': 5" in out


def test_openai_package_gets_the_same_answer(tmp_path):
    pytest.importorskip("openai")
    server = HTTPServer(("localhost", 0), load("04_servidor_falso").Falso)
    server.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        r = run_client(server.server_address[1], tmp_path, "05_cliente_openai")
    finally:
        server.shutdown()
        server.server_close()
    assert r.returncode == 0, r.stderr
    assert "(respuesta falsa) recibí 8 palabras" in r.stdout
    assert "usage: 8 de entrada, 5 de salida" in r.stdout


def test_client_without_server_is_refused(tmp_path):
    probe = HTTPServer(("localhost", 0), load("04_servidor_falso").Falso)
    port = probe.server_address[1]
    probe.server_close()  # nobody listens on this port any more
    r = run_client(port, tmp_path)
    assert r.returncode != 0
    assert "urllib.error.URLError" in r.stderr and "Connection refused" in r.stderr


def test_cabe_roto_fails_with_the_12b_model():
    r = run("06_cabe_roto")
    assert "gemma-4-E4B-it: pesos 14.89 GiB, quedan 3.43 GiB de KV-cache" in r.stdout
    assert r.returncode != 0
    assert "ValueError: gemma-4-12B-it no cabe: sus pesos ocupan 22.3 GiB" in r.stderr
    assert "vLLM tiene 20.27 GiB" in r.stderr


def test_cabe_with_one_byte_per_parameter():
    r = run("06_cabe")
    assert r.returncode == 0
    assert "gemma-4-12B-it: pesos 11.14 GiB, quedan 7.18 GiB de KV-cache" in r.stdout
