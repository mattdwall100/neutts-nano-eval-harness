"""Plot a sweep: stacked per-stage RTF bars and a config x metric heatmap.

Usage:
    uv run plot.py results/b1_*                 # -> results/b1_stages.png, results/b1_heatmap.png
    uv run plot.py results/b1_* --prefix mine   # -> results/mine_*.png
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# validated categorical slots 1-4 + neutral for the residual (dataviz reference palette)
STAGE_COLORS = {"prefill": "#2a78d6", "generate": "#eb6834", "decode": "#1baf7a", "watermark": "#eda100", "other": "#c3c2b7"}
STAGE_LABEL = {"prefill": "prefill (prompt)", "generate": "generate (decode loop)", "decode": "codec decode", "watermark": "watermark", "other": "phonemize + prompt + glue"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# (summary key, label, lower_is_better, format)
METRICS = [
    ("rtf_median", "RTF", True, "{:.2f}"),
    ("stage_rtf_median.prefill", "prefill RTF", True, "{:.2f}"),
    ("stage_rtf_median.generate", "generate RTF", True, "{:.2f}"),
    ("stage_rtf_median.decode", "codec RTF", True, "{:.3f}"),
    ("tok_per_s_median", "decode tok/s", False, "{:.0f}"),
    ("prefill_tok_per_s_median", "prefill tok/s", False, "{:.0f}"),
    ("peak_rss_mb", "peak RSS MB", True, "{:.0f}"),
    ("load_s", "load s", True, "{:.1f}"),
    ("wer_corpus", "WER", True, "{:.3f}"),
    ("spk_sim_mean", "speaker sim", False, "{:.3f}"),
    ("ttfa_median", "TTFA s", True, "{:.2f}"),
    ("stall_max", "worst stall s", True, "{:.2f}"),
    ("seam_logmel_db_median", "seam mel dB", True, "{:.2f}"),
]


def get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def load(dirs):
    out = []
    for d in [p for d in dirs for p in (sorted(glob.glob(d)) or [d])]:  # expand globs here: PowerShell does not
        p = Path(d) / "summary.json"
        if p.exists():
            out.append((Path(d).name, json.loads(p.read_text())))
    return out


def stages_chart(runs, path):
    names = [n for n, _ in runs]
    st = {k: np.array([float(get(s, f"stage_rtf_median.{k}") or 0) for _, s in runs]) for k in ("prefill", "generate", "decode", "watermark")}
    st = {k: np.nan_to_num(v) for k, v in st.items()}
    total = np.array([s["rtf_median"] for _, s in runs])
    st["other"] = np.clip(total - sum(st.values()), 0, None)

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(names) + 2.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for k in ("prefill", "generate", "decode", "watermark", "other"):
        v = st[k]
        ax.barh(y, v, left=left, height=0.62, color=STAGE_COLORS[k], edgecolor="#fcfcfb", linewidth=2, label=STAGE_LABEL[k])
        for yi, l, w in zip(y, left, v):
            if w > 0.06 * total.max():  # direct label on segments big enough to hold it
                ax.text(l + w / 2, yi, f"{w:.2f}", ha="center", va="center", fontsize=8.5, color="#0b0b0b" if k in ("decode", "watermark", "other") else "#ffffff")
        left += v
    for yi, t, end, n in zip(y, total, left, [s["n"] for _, s in runs]):
        ax.text(max(t, end) + 0.1, yi, f"RTF {t:.2f}  (n={n})", va="center", fontsize=9, color=INK2)
    ax.axvline(1.0, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(1.0, len(names) - 0.35, "real time", ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names, fontsize=9.5, color=INK)
    ax.set_xlabel("median RTF per stage (seconds of compute per second of audio)", color=INK2, fontsize=9.5)
    ax.set_xlim(0, total.max() * 1.22)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=5, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Where the time goes, per configuration", loc="left", fontsize=12, color=INK, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def heatmap(runs, path):
    names = [n for n, _ in runs]
    metrics = [m for m in METRICS if any(get(s, m[0]) is not None for _, s in runs)]
    vals = np.array([[np.nan if get(s, k) is None else float(get(s, k)) for k, *_ in metrics] for _, s in runs], dtype=float)
    # per-column index: 0 = best config, 1 = worst (lower_is_better honoured). Raw value printed in every cell.
    norm = np.zeros_like(vals)
    for j, (_, _, low, _) in enumerate(metrics):
        col = vals[:, j]
        ok = ~np.isnan(col)
        lo, hi = np.nanmin(col), np.nanmax(col)
        rng = (hi - lo) or 1.0
        norm[ok, j] = (col[ok] - lo) / rng if low else (hi - col[ok]) / rng
        norm[~ok, j] = np.nan

    fig, ax = plt.subplots(figsize=(1.05 * len(metrics) + 2.5, 0.5 * len(names) + 1.8), facecolor="#fcfcfb")
    ax.imshow(np.nan_to_num(norm, nan=0.0), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j, (_, _, _, fmt) in enumerate(metrics):
            v = vals[i, j]
            txt = "n/a" if np.isnan(v) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="#ffffff" if (not np.isnan(norm[i, j]) and norm[i, j] > 0.55) else INK)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=9, color=INK, rotation=30, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(names)), names, fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(-0.5, len(metrics)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Per column: light = best configuration, dark = worst. Cell shows the raw value.", loc="left", fontsize=9.5, color=INK2, pad=48)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def streaming_chart(runs, path):
    srun = [(n, s) for n, s in runs if s.get("ttfa_median") is not None]
    if not srun:
        return False
    names = [n for n, _ in srun]
    ttfa = np.array([s["ttfa_median"] for _, s in srun])
    stall = np.array([s["stall_max"] for _, s in srun])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(names) + 2), facecolor="#fcfcfb", sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, v, title, color in ((axes[0], ttfa, "time to first audio (s), median", STAGE_COLORS["prefill"]), (axes[1], stall, "worst playback stall (s), max over utterances", STAGE_COLORS["generate"])):
        ax.set_facecolor("#fcfcfb")
        ax.barh(y, v, height=0.6, color=color)
        for yi, x in zip(y, v):
            ax.text(x + v.max() * 0.015, yi, f"{x:.2f}", va="center", fontsize=9, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(0, v.max() * 1.18)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    axes[0].set_yticks(y, names, fontsize=9.5, color=INK)
    batch = [s for n, s in runs if s.get("ttfa_median") is None]
    if batch:
        b = batch[0]
        axes[0].axvline(b["rtf_median"] * 2.7, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
        axes[0].text(b["rtf_median"] * 2.7, len(names) - 0.4, "batch e2e, 2.7 s sentence", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--prefix", default=None, help="output name prefix (default: common prefix of the run names)")
    args = ap.parse_args()
    runs = sorted(load(args.dirs), key=lambda r: r[1]["rtf_median"])  # best first
    if not runs:
        raise SystemExit("no summary.json found")
    prefix = args.prefix or (runs[0][0].split("_")[0] if len(runs) > 1 else runs[0][0])
    out = Path("results")
    stages_chart(runs, out / f"{prefix}_stages.png")
    heatmap(runs, out / f"{prefix}_heatmap.png")
    extra = " and %s_streaming.png" % prefix if streaming_chart(runs, out / f"{prefix}_streaming.png") else ""
    print(f"wrote results/{prefix}_stages.png and results/{prefix}_heatmap.png{extra}")


if __name__ == "__main__":
    main()
