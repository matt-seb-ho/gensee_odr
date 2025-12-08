import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patches
from matplotlib import cycler


ROSE_PINE_DAWN = {
    "rose": "#b4637a",
    "gold": "#ea9d34",
    "peach": "#d7827e",
    "sapphire": "#286983",
    "foam": "#56949f",
    "lavender": "#907aa9",
}

LLM_COLOR = ROSE_PINE_DAWN["rose"]
TOOL_COLOR = ROSE_PINE_DAWN["sapphire"]


def apply_style():
    plt.style.use("default")
    plt.rcParams["figure.dpi"] = 256
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=[
            ROSE_PINE_DAWN["rose"],
            ROSE_PINE_DAWN["gold"],
            ROSE_PINE_DAWN["peach"],
            ROSE_PINE_DAWN["sapphire"],
            ROSE_PINE_DAWN["foam"],
            ROSE_PINE_DAWN["lavender"],
        ]
    )
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cmr10"


def load_records(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def df_from_spans(spans: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(spans)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df["duration_days"] = df["duration_s"] / 86400.0
    df["start_num"] = df["start_time"].map(mdates.date2num)
    df["end_num"] = df["end_time"].map(mdates.date2num)
    return df.sort_values("start_time")


def draw_gantt(
    df: pd.DataFrame, prompt_id: str, output_path: Path, prompt_text: str = ""
):
    apply_style()

    components = sorted(df["component"].unique())
    comp_to_y = {comp: i for i, comp in enumerate(components)}

    fig_height = max(2, 0.5 * len(components) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Draw rectangles
    for _, row in df.iterrows():
        y = comp_to_y[row["component"]]
        start_num = row["start_num"]
        width_days = row["duration_days"]

        color = LLM_COLOR if row["kind"] == "llm" else TOOL_COLOR

        rect = patches.Rectangle(
            (start_num, y - 0.4),
            width_days,
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    # ---- Set axis limits explicitly so bars are in view ----
    if len(df) > 0:
        min_start = df["start_num"].min()
        max_end = df["end_num"].max()
        pad = (max_end - min_start) * 0.05 or 0.001  # tiny pad if all times equal

        ax.set_xlim(min_start - pad, max_end + pad)
        ax.set_ylim(-1, len(components))  # one row per component

    # Y axis labels
    ax.set_yticks(list(comp_to_y.values()))
    ax.set_yticklabels(list(comp_to_y.keys()))

    # X axis as dates
    ax.xaxis_date()
    fig.autofmt_xdate()

    title = f"Gantt Chart — Prompt {prompt_id}"
    if prompt_text:
        preview = (prompt_text[:100] + "...") if len(prompt_text) > 100 else prompt_text
        title += f"\n{preview}"
    ax.set_title(title, fontsize=12)

    ax.set_xlabel("Time")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"[OK] Saved Gantt chart → {output_path}")


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("--input-file", required=True)
    psr.add_argument("--prompt-id", required=True)
    psr.add_argument("--output", required=True)
    args = psr.parse_args()

    records = load_records(args.input_file)
    if args.prompt_id not in records:
        raise KeyError(f"Prompt id '{args.prompt_id}' not found in records.")

    entry = records[args.prompt_id]
    spans = entry["spans"]
    prompt_text = entry.get("prompt", "")

    df = df_from_spans(spans)
    draw_gantt(df, args.prompt_id, Path(args.output), prompt_text)


if __name__ == "__main__":
    main()
