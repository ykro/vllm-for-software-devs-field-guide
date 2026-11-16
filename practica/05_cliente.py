"""El cliente: el JSON que le mandarías a la API de OpenAI, a otro lado."""
import json
import urllib.request

BASE_URL = "http://localhost:8000/v1"
pedido = {
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user",
                  "content": "Resume mi ticket de soporte en una línea"}]}
req = urllib.request.Request(
    f"{BASE_URL}/chat/completions", data=json.dumps(pedido).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    respuesta = json.load(r)
print(respuesta["choices"][0]["message"]["content"])
print("usage:", respuesta["usage"])
