"""El mismo pedido con el paquete de OpenAI: cambian dirección y clave."""
from openai import OpenAI

cliente = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
r = cliente.chat.completions.create(
    model="google/gemma-4-E4B-it",
    messages=[{"role": "user",
               "content": "Resume mi ticket de soporte en una línea"}])
print(r.choices[0].message.content)
print("usage:", r.usage.prompt_tokens, "de entrada,",
      r.usage.completion_tokens, "de salida")
