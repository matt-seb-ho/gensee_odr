import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from tqdm import tqdm

from .deep_researcher import deep_researcher

REPO_ROOT = Path(__file__).parent.parent.parent
# print(f"REPO_ROOT: {REPO_ROOT}")
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs"
GEMINI_FLASH_CONFIG_PATH = CONFIG_DIR / "gemini_2_5_flash.yaml"

PRICING: dict[str, dict[str, float]] = {
    # Example OpenAI-style entries (replace with your real ones)
    # "gemini-2.0-flash": {...}
    # "claude-3.5-sonnet": {...}
    "gemini-2.5-pro": {
        "input": 1.25,  # per 1M tokens,
        "output": 10.0,  # per 1M tokens, includes thinking tokens
    },
    "gemini-2.5-flash": {
        "input": 0.30,  # per 1M tokens,
        "output": 2.50,  # per 1M tokens, includes thinking tokens
    },
    "gemini-3-pro-preview": {
        "input": 2.0,  # per 1M tokens,
        "output": 12.0,  # per 1M tokens, includes thinking tokens
    },
}
UPDATED_PRICING: dict[str, dict[str, float]] = {
    # Example OpenAI-style entries (replace with your real ones)
    # "gemini-2.0-flash": {...}
    # "claude-3.5-sonnet": {...}
    "gemini-3-pro-preview": {
        "prompt_threshold": 200_000,
        "prompt_price_below_threshold": 2.0,  # per 1M tokens,
        "prompt_price_above_threshold": 4.0,
        "completion_threshold": 200_000,
        "completion_price_below_threshold": 12.0,  # per 1M tokens, includes thinking tokens
        "completion_price_above_threshold": 18.0,
    },
    "gemini-2.5-pro": {
        "prompt_threshold": 200_000,
        "prompt_price_below_threshold": 1.25,  # per 1M tokens,
        "prompt_price_above_threshold": 2.50,
        "completion_threshold": 200_000,
        "completion_price_below_threshold": 10.0,  # per 1M tokens, includes thinking tokens
        "completion_price_above_threshold": 15.0,
    },
    "gemini-2.5-flash": {
        "prompt_price": 0.30,  # per 1M tokens,
        "completion_price": 2.50,  # per 1M tokens, includes thinking tokens
    },
}


@dataclass
class Span:
    component: str
    kind: Literal["llm", "tool"]
    query_id: str
    run_id: str
    parent_run_id: str | None
    start_time: datetime
    end_time: datetime

    # New fields for cost tracking (mostly for kind=="llm")
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    tool_name: str | None = None  # only for kind=="tool" if you care


class UsageInfo(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: int


def est_cost_helper(
    pricing_info: dict[str, float],
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    if "prompt_threshold" in pricing_info:
        input_threshold = pricing_info["prompt_threshold"]
        if prompt_tokens <= input_threshold:
            input_price = pricing_info["prompt_price_below_threshold"]
        else:
            input_price = pricing_info["prompt_price_above_threshold"]
    else:
        input_price = pricing_info["prompt_price"]
    if "completion_threshold" in pricing_info:
        output_threshold = pricing_info["completion_threshold"]
        if completion_tokens <= output_threshold:
            output_price = pricing_info["completion_price_below_threshold"]
        else:
            output_price = pricing_info["completion_price_above_threshold"]
    else:
        output_price = pricing_info["completion_price"]
    total_cost = (prompt_tokens * input_price + completion_tokens * output_price) * 1e-6
    return total_cost


def estimate_cost_usd(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Return estimated USD cost, or None if we lack info."""
    if model is None or prompt_tokens is None or completion_tokens is None:
        print(
            f"warning: missing info for cost estimation: {model}, {prompt_tokens}, {completion_tokens}"
        )
        return None
    pricing_info = UPDATED_PRICING.get(model)
    if not pricing_info:
        print(f"warning: no pricing info for model {model}")
        return None
    total_cost = est_cost_helper(
        pricing_info,
        prompt_tokens,
        completion_tokens,
    )
    return total_cost

    # price = PRICING.get(model)
    # if not price:
    #     print(f"warning: no pricing info for model {model}")
    #     return None
    # total_cost = (
    #     prompt_tokens * price["input"] + completion_tokens * price["output"]
    # ) * 1e-6
    # return total_cost


def extract_usage_from_event(ev: dict[str, Any]) -> UsageInfo | None:
    """Best-effort extraction of token usage from an on_chat_model_end event."""
    data = ev.get("data") or {}

    # Pattern 1: data["output"] is AIMessage with .usage_metadata
    out = data.get("output")
    usage = getattr(out, "usage_metadata", None)
    if isinstance(usage, dict):
        thinking_tokens = usage.get("output_token_details", {}).get("reasoning", 0)
        return UsageInfo(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            thinking_tokens=thinking_tokens,
        )

    # If nothing matched, give up
    return None


async def run_with_timeline(
    query: str,
    config: RunnableConfig,
    query_id: str | None = None,
    capture_all_events: bool = False,  # <-- NEW FLAG
):
    """
    Execute the LangGraph agent and capture:
       - structured spans (LLM/tool)
       - optionally, ALL raw events for debugging

    Returns:
        (final_output, spans, raw_events)
    """
    spans: list[Span] = []
    raw_events: list[dict[str, Any]] = []  # <-- NEW
    in_flight: dict[str, dict[str, Any]] = {}

    query_id_local = query_id
    root_run_id: str | None = None
    final_output = None

    inputs = {"messages": [HumanMessage(content=query)]}

    async for ev in deep_researcher.astream_events(
        inputs,
        config=config,
        version="v2",
    ):
        now = datetime.now(timezone.utc)
        ev_type: str = ev.get("event")
        name: str = ev.get("name")
        run_id: str = str(ev.get("run_id"))
        parent_ids = ev.get("parent_ids") or []
        metadata = ev.get("metadata") or {}
        data = ev.get("data") or {}

        # --------------------------
        # RAW EVENTS CAPTURE (if enabled)
        # --------------------------
        if capture_all_events:
            raw_events.append(
                {
                    "timestamp": now.isoformat(),
                    "event": ev_type,
                    "name": name,
                    "run_id": run_id,
                    "parent_ids": parent_ids,
                    "metadata": metadata,
                    "data": data,
                }
            )

        # --------------------------
        # Track root run ID
        # --------------------------
        if ev_type == "on_chain_start" and not parent_ids and root_run_id is None:
            root_run_id = run_id
            if query_id_local is None:
                query_id_local = root_run_id

        # --------------------------
        # Detect start events
        # --------------------------
        if ev_type in ("on_chat_model_start", "on_llm_start", "on_tool_start"):
            node_name = metadata.get("langgraph_node")
            component_name = node_name or name

            kind: Literal["llm", "tool"]
            tool_name: str | None = None
            model_name: str | None = None

            if ev_type in ("on_chat_model_start", "on_llm_start"):
                kind = "llm"
                # Different LangChain versions put this in different places
                model_name = metadata.get("ls_model_name") or metadata.get("model")
            else:
                # on_tool_start
                tool_name = name  # e.g. "tavily_search", "web_search", "think_tool"

                if tool_name == "think_tool":
                    # treat as "internal LLM thinking"
                    kind = "llm"
                else:
                    kind = "tool"

            in_flight[run_id] = {
                "component": component_name,
                "kind": kind,
                "tool_name": tool_name,
                "model": model_name,
                "parent_run_id": parent_ids[0] if parent_ids else None,
                "start_time": now,
            }

        # --------------------------
        # Detect end events
        # --------------------------
        if ev_type in ("on_chat_model_end", "on_llm_end", "on_tool_end"):
            info = in_flight.pop(run_id, None)
            if info:
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                cost_usd = None

                # Only try usage/cost for LLM spans
                if info["kind"] == "llm" and ev_type in (
                    "on_chat_model_end",
                    "on_llm_end",
                ):
                    usage = extract_usage_from_event(ev)
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                        cost_usd = estimate_cost_usd(
                            info.get("model"),
                            prompt_tokens,
                            completion_tokens,
                        )

                spans.append(
                    Span(
                        component=info["component"],
                        kind=info["kind"],
                        query_id=query_id_local or (root_run_id or "unknown"),
                        run_id=run_id,
                        parent_run_id=info["parent_run_id"],
                        start_time=info["start_time"],
                        end_time=now,
                        model=info.get("model"),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost_usd,
                        tool_name=info.get("tool_name"),
                    )
                )

        # --------------------------
        # Capture root output
        # --------------------------
        if ev_type == "on_chain_end" and run_id == root_run_id:
            final_output = data.get("output", data)

    return final_output or {}, spans, raw_events


async def a_main(odr_config):
    final_state, spans, raw_events = await run_with_timeline(
        query="Explain diffusion policies.",
        config=odr_config,
        capture_all_events=True,  # <--- enable raw event capture
    )

    print("Number of raw events:", len(raw_events))
    print("LLM/tool spans:", len(spans))


def prepend_if_not_absolute(tgt: Path | str, prefix: Path) -> Path:
    """Prepend prefix to tgt if tgt is not absolute."""
    if isinstance(tgt, str):
        tgt = Path(tgt)
    if tgt.is_absolute():
        return tgt
    return prefix / tgt


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / "src/.env")
    psr = argparse.ArgumentParser()
    psr.add_argument("--config", type=str, help="Path to config file")
    args = psr.parse_args()

    with open(prepend_if_not_absolute(args.config, CONFIG_DIR)) as f:
        cfg = yaml.safe_load(f)

    with open(prepend_if_not_absolute(cfg["odr_cfg"], CONFIG_DIR)) as f:
        odr_cfg_dict = yaml.safe_load(f)
    odr_cfg: RunnableConfig = {
        "configurable": odr_cfg_dict,
    }

    records: dict[str, Any] = {}
    for prompt_id, prompt in tqdm(cfg["prompts"].items()):
        print(f"Running prompt ID {prompt_id}...")

        final_state, spans, raw_events = asyncio.run(
            run_with_timeline(
                query=prompt,
                config=odr_cfg,
                capture_all_events=True,
            )
        )

        # Extract just what you care about from final_state
        # deep_researcher returns a dict; "final_report" should be present.
        final_report = final_state.get("final_report")
        # If you want *everything* but with non-serializable values stringified:
        # safe_final_state = json.loads(json.dumps(final_state, default=str))

        # Serialize spans so datetimes become strings
        spans_serialized = []
        for span in spans:
            d = asdict(span)
            d["start_time"] = span.start_time.isoformat()
            d["end_time"] = span.end_time.isoformat()
            spans_serialized.append(d)

        # Optional: strip "data" off raw_events to avoid weird objects
        safe_raw_events: list[dict[str, Any]] = []
        for ev in raw_events:
            ev_copy = dict(ev)
            # Either drop data:
            # ev_copy.pop("data", None)
            # Or stringify it:
            ev_copy["data"] = json.loads(json.dumps(ev_copy["data"], default=str))
            safe_raw_events.append(ev_copy)

        total_cost = sum(s.cost_usd or 0.0 for s in spans)
        total_prompt_tokens = sum(
            (s.prompt_tokens or 0) for s in spans if s.kind == "llm"
        )
        total_completion_tokens = sum(
            (s.completion_tokens or 0) for s in spans if s.kind == "llm"
        )

        records[prompt_id] = {
            "prompt": prompt,
            "final_report": final_report,
            "total_cost_usd": total_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "spans": spans_serialized,
            "raw_events": safe_raw_events,
        }

    out_path = prepend_if_not_absolute(cfg["output_file"], OUTPUT_DIR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved results to {out_path}")
