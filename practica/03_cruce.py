"""En qué volumen deja de ganar la API: multiplico mis usuarios."""
import json
from pathlib import Path

api = json.loads(Path("benchmarks/api_prices.json").read_text())
precio = api["usd_per_million"]["from_2027_01_01"]
gpu = json.loads(Path("benchmarks/l4_instance_pricing.json").read_text())
tokens_por_s = 192.0  # ocho usuarios a la vez, medido en una L4
horas_de_uso = 8

for veces in [0.25, 0.5, 1, 2, 4, 8]:
    entrada, salida = 20_000_000 * veces, 2_000_000 * veces
    api_mes = (entrada * precio["input"]
               + salida * precio["output"]) / 1e6 * 30
    horas_gpu = salida / (tokens_por_s * 3600)
    gpu_mes = max(horas_gpu, horas_de_uso) * gpu["usd_per_hour"] * 30
    gana = "API" if api_mes < gpu_mes else "vLLM"
    print(f"x{veces:<5} API ${api_mes:>7,.0f}   vLLM ${gpu_mes:>5,.0f}"
          f"   gana {gana}")
