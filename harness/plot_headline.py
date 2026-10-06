"""The headline figure: the nine-step chain (sweep h1) as stacked per-stage RTF, full scale and zoomed.

    uv run harness/plot_headline.py            # -> results/headline.png
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [
    ("h1_1_shipped", "1  As shipped: torch bf16 + torch codec,\n    13 s reference, watermark"),
    ("h1_2_q4_onnx8", "2  + Q4 backbone, ONNX int8 codec"),
    ("h1_3_ref5", "3  + 5 s reference"),
    ("h1_4_nowm", "4  + watermark off"),
    ("h1_5_native_build", "5  + llama.cpp built for this CPU"),
    ("h1_6_sliced", "6  + output rows sliced (our patch)"),
    ("h1_7_pinned", "7  + decode threads pinned to P-cores"),
    ("h1_8_stream_shipped", "8  streaming, shipped streaming settings"),
    ("h1_9_stream_final", "9  streaming + codec spin off, 50-frame chunks"),
]
N_BATCH = 7
C = {"prefill": "#2a78d6", "generate": "#eb6834", "decode": "#1baf7a", "watermark": "#eda100"}
L = {"prefill": "prefill (read prompt)", "generate": "generate tokens", "decode": "codec decode", "watermark": "watermark"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def main():
    S = [json.load(open(f"results/{n}/summary.json")) for n, _ in ROWS]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8), facecolor="#fcfcfb", gridspec_kw={"width_ratios": [1, 1.25]}, sharey=True)
    y = np.arange(len(ROWS))[::-1]
    split = y[N_BATCH - 1] - 0.5  # between the last batch row and the first streaming row
    for a, xmax, title in ((ax[0], 12.5, "Full scale"), (ax[1], 2.75, "Zoomed: rows 2-9 (row 1 runs off the right edge)")):
        a.set_facecolor("#fcfcfb")
        a.axhspan(-0.6, split, color="#efeeea", zorder=0)
        a.axhline(split, color=INK2, linestyle=(0, (1, 3)), linewidth=1.2, zorder=1)
        left = np.zeros(len(ROWS))
        for k in C:
            v = np.nan_to_num(np.array([float(s["stage_rtf_median"].get(k) or 0) for s in S]))
            a.barh(y, v, left=left, height=0.62, color=C[k], edgecolor="#fcfcfb", linewidth=2, label=L[k], zorder=2)
            for yi, l, w in zip(y, left, v):
                if w > 0.075 * xmax and l + w <= xmax:
                    a.text(l + w / 2, yi, f"{w:.2f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if k in ("prefill", "generate") else INK, zorder=3)
            left += v
        for yi, s, end in zip(y, S, left):
            t = s["rtf_median"]
            extra = ""
            if s.get("ttfa_median") is not None and a is ax[1]:
                extra = f"   first audio {s['ttfa_median']:.1f} s, " + ("no stalls" if s["stall_max"] == 0 else f"stalls up to {s['stall_max']:.1f} s")
            if max(t, end) < xmax:
                a.text(max(t, end) + xmax * 0.012, yi, f"RTF {t:.2f}{extra}", va="center", fontsize=9, color=INK2, zorder=3)
            else:
                a.text(xmax * 0.985, yi, f"RTF {t:.1f} →", va="center", ha="right", fontsize=9, color="#ffffff", zorder=3)
        a.axvline(1.0, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
        a.text(1.0, len(ROWS) - 0.42, "real time", ha="center", fontsize=8.5, color=INK2)
        if a is ax[0]:  # rows share the y axis; the zoomed panel's long bars leave no room for the labels
            a.text(xmax * 0.99, split + 0.12, "batched", ha="right", va="bottom", fontsize=9.5, color=INK2, style="italic")
            a.text(xmax * 0.99, split - 0.12, "streamed", ha="right", va="top", fontsize=9.5, color=INK2, style="italic")
        a.set_xlim(0, xmax)
        a.set_ylim(-0.6, len(ROWS) - 0.3)
        a.grid(axis="x", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors="#898781", labelsize=9)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.set_title(title, loc="left", fontsize=10.5, pad=22)
        a.set_xlabel("RTF: seconds of compute per second of audio (median)", fontsize=9.5, color=INK2)
    ax[0].set_yticks(y, [l for _, l in ROWS], fontsize=9.5, color=INK, ha="left")
    ax[0].tick_params(axis="y", pad=235, length=0)
    ax[1].legend(loc="lower right", frameon=False, fontsize=9, ncol=4, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("NeuTTS-Nano on an Intel i7-1265U laptop CPU: one change per row, one interleaved session, 20 utterances per row (row 1: 2)", x=0.008, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig("results/headline.png", dpi=150)
    print("wrote results/headline.png")


if __name__ == "__main__":
    main()
