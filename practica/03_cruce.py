"""En qué volumen deja de ganar la API: multiplico mis usuarios y comparo."""
import json
from pathlib import Path

api = json.loads(Path("benchmarks/api_prices.json").read_text())["usd_per_million"]
precio = api["from_2027_01_01"]
usd_hora = json.loads(Path("benchmarks/l4_instance_pricing.json").read_text())["usd_per_hour"]
tokens_por_s = 193.0  # ocho usuarios a la vez, medido en una L4
horas_de_uso = 8

for veces in [0.25, 0.5, 1, 2, 4, 8]:
    entrada, salida = 20_000_000 * veces, 2_000_000 * veces
    api_mes = (entrada * precio["input"] + salida * precio["output"]) / 1e6 * 30
    horas_gpu = salida / (tokens_por_s * 3600)
    gpu_mes = max(horas_gpu, horas_de_uso) * usd_hora * 30
    gana = "API" if api_mes < gpu_mes else "vLLM"
    print(f"x{veces:<5} API ${api_mes:>7,.0f}   vLLM ${gpu_mes:>5,.0f}   gana {gana}")
