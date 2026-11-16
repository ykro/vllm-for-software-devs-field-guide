"""Cuánto cuesta servirlo yo: la GPU cobra por hora, escriba o no."""
import json
from pathlib import Path

gpu = json.loads(Path("benchmarks/l4_instance_pricing.json").read_text())
medido = json.loads(Path("benchmarks/first_server_throughput.json").read_text())
salida_dia = 2_000_000  # tokens que el modelo escribe al día
horas_de_uso = 8  # horas al día en que tu app tiene usuarios

for fila in medido["rows"]:
    tokens_por_hora = fila["output_tokens_per_s"] * 3600
    horas_gpu = salida_dia / tokens_por_hora
    encendida = max(horas_gpu, horas_de_uso)  # on-demand: se apaga cuando nadie la usa
    mes = encendida * gpu["usd_per_hour"] * 30
    print(f"{fila['users']} usuario(s) a la vez, {fila['output_tokens_per_s']} tokens/s: "
          f"{horas_gpu:.1f} h de GPU al día, ${mes:,.0f} al mes")
