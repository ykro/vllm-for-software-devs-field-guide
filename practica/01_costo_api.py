"""Cuánto me cobra una API al mes: tokens al día por precio por millón."""
import json
from pathlib import Path

precios = json.loads(Path("benchmarks/api_prices.json").read_text())
conversaciones_por_dia = 8_000
tokens_entrada = 2_500  # instrucciones, historial y la pregunta
tokens_salida = 250  # lo que el modelo escribe de vuelta

entrada_dia = conversaciones_por_dia * tokens_entrada
salida_dia = conversaciones_por_dia * tokens_salida
print(f"al día: {entrada_dia:,} de entrada y {salida_dia:,} de salida")
print(f"{precios['model']}, dólares por millón de tokens:")
for periodo, p in precios["usd_per_million"].items():
    mes = (entrada_dia * p["input"] + salida_dia * p["output"]) / 1e6 * 30
    print(f"  {periodo}: entrada {p['input']}, salida {p['output']}"
          f" -> ${mes:,.0f} al mes")
