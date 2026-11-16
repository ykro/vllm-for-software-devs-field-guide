"""Un servidor falso que habla como vLLM: recibe JSON por HTTP y devuelve JSON."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Falso(BaseHTTPRequestHandler):
    def do_POST(self):
        pedido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        pregunta = pedido["messages"][-1]["content"]
        texto = f"(respuesta falsa) recibí {len(pregunta.split())} palabras"
        cuerpo = {"model": pedido["model"],
                  "choices": [{"message": {"role": "assistant", "content": texto}}],
                  "usage": {"prompt_tokens": len(pregunta.split()),
                            "completion_tokens": len(texto.split())}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(cuerpo).encode())


if __name__ == "__main__":
    print("escuchando en http://localhost:8000/v1", flush=True)
    HTTPServer(("localhost", 8000), Falso).serve_forever()
