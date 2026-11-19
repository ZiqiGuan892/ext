"""Plot raw values and moving averages for verl metrics in plot/ beside console.log."""

import argparse
import csv
from collections import deque
from hashlib import sha256
import math
from pathlib import Path
import re


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP = re.compile(r"(?<![\w/])(?:step|training/global_step)\s*:\s*(\d+)(?=\s|$)")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
VALUE = rf"(?:{NUMBER}|[+-]?(?:nan|inf(?:inity)?))"
FIELD = re.compile(
    rf"(?<!\S)([\w][^\s:]*):\s*(?:np\.(?:u?int|float)\d+\(({VALUE})\)|({VALUE}))(?=\s|$)",
    re.IGNORECASE,
)


def read_steps(log, metric=None):
    rows = {}
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "step" not in line:
                continue
            line = ANSI.sub("", line)
            step = STEP.search(line)
            if not step:
                continue
            fields = {
                key: float(wrapped or plain)
                for key, wrapped, plain in FIELD.findall(line)
                if key != "step" and (metric is None or key == metric)
            }
            if fields:
                rows.setdefault(int(step[1]), {}).update(fields)
    return dict(sorted(rows.items()))


def metric_stem(key):
    # Keep paths stable across runs and distinct for keys such as a/b and a_b.
    name = re.sub(r"[^\w.-]+", "_", key.replace("/", "__"))[:110]
    return f"{name}_{sha256(key.encode()).hexdigest()[:10]}"


def moving_average(values, window):
    recent = deque()
    total = 0.0
    for value in values:
        if not math.isfinite(value):
            recent.clear()
            total = 0.0
            yield math.nan
            continue
        if len(recent) == window:
            total -= recent.popleft()
        recent.append(value)
        total += value
        yield total / len(recent)


def plot_metric(rows, metric, png, window=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(9, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    # Validation and other periodic metrics have their own observation cadence.
    steps = [step for step, row in rows.items() if metric in row]
    values = [rows[step][metric] for step in steps]
    finite_count = sum(math.isfinite(v) for v in values)
    span = min(window, finite_count) if window is not None else math.isqrt(finite_count)
    raw = [v if math.isfinite(v) else math.nan for v in values]
    # Dense lines need markers only where a gap would hide a lone observation.
    markers = None if finite_count <= 12 else [
        i for i, v in enumerate(raw) if math.isfinite(v)
        and (i == 0 or math.isnan(raw[i - 1]))
        and (i == len(raw) - 1 or math.isnan(raw[i + 1]))
    ]
    ax.plot(steps, raw, color="#afbdc5" if span > 1 else "#165b80",
            linewidth=0.8 if span > 1 else 1.5, marker=".", markersize=4, markevery=markers,
            label="Observed")
    if span > 1:
        ax.plot(steps, list(moving_average(raw, span)), color="#165b80", linewidth=2.2,
                label=f"Moving average ({span} observations)")
        ax.legend(frameon=False)
    ax.set(title=metric, xlabel="Training step", ylabel="Value")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    if len(steps) == 1:
        ax.set_xticks(steps)
    if not finite_count:
        ax.text(0.5, 0.5, "No finite values", ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)
    fig.savefig(png, dpi=150)
    fig.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="path to console.log")
    parser.add_argument("--metric", help="only plot this exact metric (default: all)")
    parser.add_argument("--window", type=int, metavar="N",
                        help="trailing average over N observations per metric (default: square root of the "
                             "finite observation count, rounded down; 1 shows only the raw line)")
    parser.add_argument("-o", "--output", type=Path, help="single PNG path (defaults to the reward metric in this mode)")
    args = parser.parse_args()

    if args.window is not None and args.window < 1:
        parser.error("--window must be at least 1")
    if not args.log.is_file():
        parser.error(f"log file not found: {args.log}")
    if args.output and args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    metric = args.metric or ("critic/rewards/mean" if args.output else None)
    rows = read_steps(args.log, metric)
    if not rows:
        parser.exit(1, f"No step data found for {metric or 'numeric metrics'} in {args.log}\n")

    directory = args.output.parent if args.output else args.log.parent / "plot"
    metrics = sorted({key for row in rows.values() for key in row})
    plots = [args.output or directory / f"{metric_stem(key)}.png" for key in metrics]
    table = plots[0].with_suffix(".csv") if metric else directory / "metrics.csv"
    if any(path.resolve() == args.log.resolve() for path in [table, *plots]):
        parser.error("output path would overwrite the input log")
    directory.mkdir(parents=True, exist_ok=True)
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", *metrics])
        writer.writeheader()
        for step, row in rows.items():
            writer.writerow({"step": step, **row})
    for key, png in zip(metrics, plots):
        plot_metric(rows, key, png, args.window)
    print(f"{len(rows)} steps | {len(metrics)} plots in {directory} | CSV: {table}")


if __name__ == "__main__":
    main()
