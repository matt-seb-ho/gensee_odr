#!/usr/bin/env python
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from open_deep_research.run_with_timeline import estimate_cost_usd


def parse_iso(ts: str) -> datetime:
    """Parse ISO8601 timestamp with timezone."""
    return datetime.fromisoformat(ts)


def summarize_records(
    records: Dict[str, Any],
    recompute_cost: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[float], int, int]:
    """
    Given a records dict (prompt_id -> record), compute:

    - avg_llm_latency_s: average span duration for LLM spans
    - avg_tool_latency_s: average span duration for tool spans
    - avg_total_cost_per_prompt_usd: average sum(cost_usd) per prompt
    - n_llm_spans
    - n_tool_spans
    """
    llm_latencies: List[float] = []
    tool_latencies: List[float] = []
    prompt_costs: List[float] = []

    for prompt_id, rec in records.items():
        spans = rec.get("spans", [])
        total_cost_prompt = 0.0

        for span in spans:
            try:
                start = parse_iso(span["start_time"])
                end = parse_iso(span["end_time"])
            except Exception:
                # Skip spans with bad timestamps
                continue

            dt = (end - start).total_seconds()
            kind = span.get("kind")

            if kind == "llm":
                llm_latencies.append(dt)
            elif kind == "tool":
                # print(span["tool_name"])
                tool_latencies.append(dt)

            # Cost: if span has cost_usd, add it
            if kind == "llm":
                if recompute_cost:
                    if span.get("model", None) is None:
                        # Can't recompute cost without model info
                        continue
                    cost = estimate_cost_usd(
                        model=span.get("model", None),
                        prompt_tokens=span.get("prompt_tokens", 0),
                        completion_tokens=span.get("completion_tokens", 0),
                    )
                    if cost is not None:
                        total_cost_prompt += cost
                        continue  # skip existing cost_usd
                else:
                    cost = span.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        total_cost_prompt += float(cost)

        # Even if a prompt has 0 cost, we track it; it contributes 0 to average.
        prompt_costs.append(total_cost_prompt)

    def safe_avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    # avg_llm_latency_s = safe_avg(llm_latencies)
    total_llm_latency_s = sum(llm_latencies)
    total_tool_latency_s = sum(tool_latencies)
    avg_total_cost_per_prompt_usd = safe_avg(prompt_costs)

    return (
        total_llm_latency_s,
        total_tool_latency_s,
        avg_total_cost_per_prompt_usd,
        len(llm_latencies),
        len(tool_latencies),
    )


def summarize_file(path: Path) -> None:
    with path.open() as f:
        records = json.load(f)

    (
        total_llm_latency,
        total_tool_latency,
        avg_cost,
        n_llm,
        n_tool,
    ) = summarize_records(records)

    def fmt(x: Optional[float], ndigits: int = 4) -> str:
        return f"{x:.{ndigits}f}" if x is not None else "n/a"

    print(f"File: {path}")
    print(f"  LLM spans:  n={n_llm},  tot latency={fmt(total_llm_latency, 3)} s")
    print(f"  Tool spans: n={n_tool}, tot latency={fmt(total_tool_latency, 3)} s")
    print(f"  Avg total cost per prompt: {fmt(avg_cost, 6)} USD")
    print()


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument(
        "files",
        nargs="+",
        help="One or more JSON files containing records dicts",
    )
    args = psr.parse_args()

    for f in args.files:
        summarize_file(Path(f))


if __name__ == "__main__":
    main()
