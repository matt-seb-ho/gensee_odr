# viz_timelines.py

from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt

# Simple type alias
SpanDict = dict[str, Any]


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


def _parse_ts(x: Any) -> float:
    """Convert x to a float seconds timestamp.

    Handles:
      - datetime objects
      - ISO8601 timestamp strings
      - numeric epoch seconds (int/float)
    Returns UNIX epoch seconds as float.
    """
    if isinstance(x, datetime):
        return x.timestamp()

    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        try:
            dt = datetime.fromisoformat(x)
            return dt.timestamp()
        except ValueError:
            raise ValueError(f"Unrecognized timestamp string: {x}")

    raise TypeError(f"Cannot convert timestamp of type {type(x)}: {x}")


def _duration_seconds(start: Any, stop: Any) -> float:
    return _parse_ts(stop) - _parse_ts(start)


def _to_seconds_since_start(start_value: Any, t0_value: Any) -> float:
    return _parse_ts(start_value) - _parse_ts(t0_value)


def _get_kind(span: SpanDict) -> str:
    """Normalize 'kind' / 'type' field."""
    if "kind" in span:
        return span["kind"]
    if "type" in span:
        return span["type"]
    return "unknown"


# -------------------------------------------------------------------
# VIS 1
# -------------------------------------------------------------------
def plot_span_timeline(
    spans: list[SpanDict],
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    VIS 1 (updated):
      - Thin horizontal bars for each event
      - No y-axis labels
      - Sorted by start time
      - X-axis in seconds since first event
    """
    if not spans:
        raise ValueError("No spans provided")

    # Sort by start time
    sorted_spans = sorted(spans, key=lambda s: s["start"])

    # Reference t0
    t0 = sorted_spans[0]["start"]

    starts_s = []
    durations_s = []
    kinds = []

    for span in sorted_spans:
        start = span["start"]
        stop = span["stop"]
        starts_s.append(_to_seconds_since_start(start, t0))
        durations_s.append(_duration_seconds(start, stop))
        kinds.append(_get_kind(span))

    n = len(sorted_spans)

    # AUTO FIG SIZE — compact even for 300 spans
    created_fig = False
    if ax is None:
        height = min(12, 0.3 * n + 1)  # adaptive, capped
        fig, ax = plt.subplots(figsize=(14, height))
        created_fig = True

    # Colors
    color_map = {
        "llm_call": "#b4637a",  # rose
        "web_search": "#286983",  # sapphire
        "unknown": "#ea9d34",
    }

    ys = list(range(n))

    # THIN BAR HEIGHT
    bar_height = 0.25
    row_spacing = 0.5

    for idx, (start_s, dur_s, kind) in enumerate(zip(starts_s, durations_s, kinds)):
        color = color_map.get(kind, color_map["unknown"])
        # ax.barh(
        #     y=idx,
        #     width=dur_s,
        #     left=start_s,
        #     height=bar_height,
        #     color=color,
        #     # edgecolor="black",
        #     linewidth=0.3,
        # )
        ax.barh(
            y=idx * row_spacing,
            left=start_s,
            width=dur_s,
            height=bar_height,
            align="edge",
            color=color,
            linewidth=0,
        )

    # Remove y-axis labels for readability
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Also remove y-axis spine
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("Time since start (s)")
    if title:
        ax.set_title(title)

    # Simple legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map["llm_call"], ec="black", lw=0.3),
        plt.Rectangle((0, 0), 1, 1, color=color_map["web_search"], ec="black", lw=0.3),
    ]
    ax.legend(handles, ["LLM Call", "Web Search"], title="Span Type")

    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    if created_fig:
        plt.tight_layout()

    return ax


# -------------------------------------------------------------------
# VIS 2
# -------------------------------------------------------------------
def plot_run_stack_latency(
    runs: dict[str, list[SpanDict]],
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    VIS 2:
    Input:
      - runs: dict[run_id -> list[span_dict]]
        Each span_dict has:
          - 'kind' or 'type' in {'llm_call', 'web_search'}
          - 'start', 'stop' timestamps (datetime or numeric)
    Output:
      - Stacked bar chart: one bar per run.
        * bottom = total LLM inference latency (s)
        * top    = total Web search latency (s)
        * bars are labeled with run ids.
    """
    if not runs:
        raise ValueError("No runs provided")

    run_ids = list(runs.keys())

    total_llm = []
    total_web = []

    for run_id in run_ids:
        spans = runs[run_id]
        llm_sum = 0.0
        web_sum = 0.0
        for span in spans:
            start = span["start"]
            stop = span["stop"]
            dur = _duration_seconds(start, stop)
            kind = _get_kind(span)
            if kind == "llm_call":
                llm_sum += dur
            elif kind == "web_search":
                web_sum += dur
            # ignore other kinds
        total_llm.append(llm_sum)
        total_web.append(web_sum)

    # Create axes if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(run_ids)), 4))
        created_fig = True

    x = range(len(run_ids))

    # Colors aligned with VIS 1
    llm_color = "#b4637a"  # rose
    web_color = "#286983"  # sapphire

    ax.bar(x, total_llm, color=llm_color, label="LLM inference")
    ax.bar(x, total_web, bottom=total_llm, color=web_color, label="Web search")

    ax.set_xticks(list(x))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")

    ax.set_ylabel("Total latency (s)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("LLM + Web search latency per run")

    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if created_fig:
        plt.tight_layout()

    return ax
