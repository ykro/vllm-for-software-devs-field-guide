"""Cabe el modelo en una L4: pesos = parámetros por bytes por parámetro."""
GIB = 1024**3
PRESUPUESTO_GIB = 20.27  # lo que vLLM toma de una L4 de 24 GB (medido)
RESERVA_GIB = 1.95  # trabajo de cada paso y memoria del sistema (medido)
PARAMETROS = {"gemma-4-E4B-it": 7_996_156_490, "gemma-4-12B-it": 11_959_730_224,
              "gemma-4-31B-it": 31_273_088_876}


def revisar(modelo, bytes_por_parametro=2):
    pesos = PARAMETROS[modelo] * bytes_por_parametro / GIB
    kv_cache = PRESUPUESTO_GIB - pesos - RESERVA_GIB
    if kv_cache <= 0:
        raise ValueError(f"{modelo} no cabe: sus pesos ocupan {pesos:.1f} GiB "
                         f"y vLLM tiene {PRESUPUESTO_GIB} GiB en esta GPU")
    print(f"{modelo}: pesos {pesos:.2f} GiB, quedan {kv_cache:.2f} GiB de KV-cache")


revisar("gemma-4-E4B-it")
revisar("gemma-4-12B-it")
