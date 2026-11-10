"""API or your own vLLM server? A back-of-the-envelope calculator with measured numbers.

The API charges per token. Your server charges per hour, whether it writes one token or
a million. So the question is how many tokens you get out of each GPU hour, and for how
many hours a day the GPU has to be on.

    python scripts/self_host_or_api.py --output-tokens-per-day 2000000
    python scripts/self_host_or_api.py --input-tokens-per-day 20000000 \
        --output-tokens-per-day 2000000 --busy-hours 10

Throughput comes from C1 (benchmarks/c1_throughput.json, measured on one NVIDIA L4 with
Gemma 4 E4B through vLLM): one user at a time is the pessimistic case, eight at a time
the optimistic one. Prices come from benchmarks/*.json, each with its source and date.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
DAYS = 30


def load(name: str) -> dict:
    return json.loads((BENCH / name).read_text())


def scenario(out_per_day: float, tokens_per_s: float, usd_per_hour: float,
             busy_hours: float) -> dict:
    """On-demand: the GPU is on while it works (at least `busy_hours`). Always-on: 24 h."""
    gpu_hours = out_per_day / (tokens_per_s * 3600)
    gpus = max(1, math.ceil(gpu_hours / 24))
    on_demand_hours = max(gpu_hours, busy_hours)
    return {
        "gpu_hours_per_day": gpu_hours,
        "gpus_needed": gpus,
        "on_demand_usd_month": on_demand_hours * usd_per_hour * DAYS,
        "always_on_usd_month": gpus * 24 * usd_per_hour * DAYS,
    }


def compare(input_per_day: float, out_per_day: float, busy_hours: float) -> dict:
    api = load("api_prices.json")
    l4 = load("l4_instance_pricing.json")
    tput = load("c1_throughput.json")
    api_costs = {
        label: (input_per_day * p["input"] + out_per_day * p["output"]) / 1e6 * DAYS
        for label, p in api["usd_per_million"].items()
    }
    self_host = {
        f"{row['users']}_users": scenario(out_per_day, row["output_tokens_per_s"],
                                          l4["usd_per_hour"], busy_hours)
        for row in tput["rows"]
    }
    return {"api": api, "l4": l4, "throughput": tput, "api_usd_month": api_costs,
            "self_host": self_host}


def money(x: float) -> str:
    return f"${x:,.0f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-tokens-per-day", type=float, default=0.0)
    p.add_argument("--output-tokens-per-day", type=float, required=True)
    p.add_argument("--busy-hours", type=float, default=8.0,
                   help="hours per day your users are active (the GPU stays on that long)")
    a = p.parse_args()
    r = compare(a.input_tokens_per_day, a.output_tokens_per_day, a.busy_hours)

    print(f"Volume  : {a.input_tokens_per_day:,.0f} input + {a.output_tokens_per_day:,.0f} "
          f"output tokens/day, {a.busy_hours:g} busy hours/day, {DAYS} days")
    print(f"API     : {r['api']['model']} ({r['api']['tier']}), checked {r['api']['checked']}")
    for label, cost in r["api_usd_month"].items():
        print(f"  {label:26} {money(cost):>10} / month")
    print(f"vLLM    : {r['l4']['instance']}, ${r['l4']['usd_per_hour']}/h")
    for label, s in r["self_host"].items():
        users = label.split("_")[0]
        note = "" if s["gpus_needed"] == 1 else f"  ({s['gpus_needed']} GPUs)"
        print(f"  {users} user(s) at a time: {s['gpu_hours_per_day']:.1f} GPU h/day -> on-demand "
              f"{money(s['on_demand_usd_month'])} / month, always-on "
              f"{money(s['always_on_usd_month'])} / month{note}")


if __name__ == "__main__":
    main()
