"""Evidence for slicing the backbone's output projection to the speech rows.

At every decode step over a corpus sample, read the full 194k-row logit vector and measure what the rows we
would drop (everything except the 65,536 speech codes + the end token) contribute.

    uv run slice_stats.py            # -> results/lm_head_slice.json, results/lm_head_slice.png
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

Q4, ONNX8 = "neuphonic/neutts-nano-q4-gguf", "neuphonic/neucodec-onnx-decoder-int8"
REF = "samples/jo6"          # 6.4 s reference: shortest clip with no voice loss (R1)
N_SHORT, N_LONG, TOP_K = 40, 10, 50
SEEDS = (1000, 1001)          # two sampling paths per text


def main():
    from neutts import NeuTTS
    import neutts.neutts as _nn

    with contextlib.redirect_stdout(io.StringIO()):
        tts = NeuTTS(backbone_repo=Q4, codec_repo=ONNX8)
    _nn.print = lambda *a, **k: None
    b = tts.backbone
    V = b.n_vocab()
    s0 = b.tokenize(b"<|speech_0|>", add_bos=False, special=True)[0]
    end_id = b.tokenize(b"<|SPEECH_GENERATION_END|>", add_bos=False, special=True)[0]
    keep = np.zeros(V, bool)
    keep[s0 : s0 + 65536] = True
    keep[end_id] = True
    drop = ~keep

    ref_codes = torch.load(REF + ".pt")
    ref_text = Path(REF + ".txt").read_text(encoding="utf-8").strip()
    lines = [l.strip() for l in Path("corpus.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = lines[:N_SHORT] + lines[60 : 60 + N_LONG]

    from collections import Counter

    mass, kl, best_rank, margin, hits = [], [], [], [], 0
    per_text, best_names = [], Counter()
    drop_idx = np.flatnonzero(drop)
    runs = [(ti, text, seed) for ti, text in enumerate(texts) for seed in SEEDS]
    for ri, (ti, text, seed) in enumerate(runs):
        toks = [b.token_bos()] + b.tokenize(tts._ggml_prompt(ref_codes, ref_text, text).encode(), add_bos=False, special=True)
        b.reset()
        b.eval(toks)
        rng = np.random.default_rng(seed * 1000 + ti)
        n = 0
        for _ in range(tts.max_context - len(toks)):
            lg = np.ctypeslib.as_array(b._ctx.get_logits(), shape=(V,)).astype(np.float64)
            p = np.exp(lg - lg.max())
            p /= p.sum()
            m = float(p[drop].sum())
            mass.append(m)
            kl.append(-np.log1p(-m))                       # KL(sliced || full) in nats = -log(kept mass)
            top = np.argpartition(-lg, TOP_K)[:TOP_K]
            hits += int(drop[top].sum())
            kth = lg[top].min()                             # the 50th-best logit = sampler's cut-off
            dl = lg[drop]
            bi = int(dl.argmax())
            best_drop = dl[bi]
            best_names[int(drop_idx[bi])] += 1
            margin.append(float(kth - best_drop))           # >0: best dropped row is below the cut-off
            best_rank.append(int((lg > best_drop).sum()) + 1)
            nxt = int(rng.choice(top, p=p[top] / p[top].sum()))  # temperature 1, top-k 50, as the reference samples
            n += 1
            if nxt == end_id:
                break
            b.eval([nxt])
        per_text.append({"text_id": ti, "seed": seed, "steps": n})
        print(f"[{ri + 1}/{len(runs)}] {n:4d} steps  running: mean dropped mass {np.mean(mass):.2e}, max {np.max(mass):.2e}, top-{TOP_K} hits {hits}", flush=True)

    mass, kl, margin, best_rank = map(np.array, (mass, kl, margin, best_rank))
    h = 576
    out = {
        "model": Q4, "reference": REF, "n_texts": len(texts), "seeds": list(SEEDS), "n_utterances": len(runs), "n_steps": int(len(mass)), "top_k": TOP_K,
        "steps_with_a_dropped_row_inside_top": {str(k): int((best_rank <= k).sum()) for k in (50, 100, 200, 500, 1000, 5000)},
        "most_frequent_best_dropped_rows": [{"token": b.detokenize([t], special=True).decode("utf-8", "replace"), "id": t, "steps": c} for t, c in best_names.most_common(15)],
        "vocab": int(V), "rows_kept": int(keep.sum()), "rows_dropped": int(drop.sum()), "dropped_fraction": round(float(drop.mean()), 4),
        "dropped_mass": {"mean": float(mass.mean()), "median": float(np.median(mass)), "p99": float(np.percentile(mass, 99)), "p99_9": float(np.percentile(mass, 99.9)), "max": float(mass.max())},
        "kl_sliced_vs_full_nats": {"mean": float(kl.mean()), "max": float(kl.max())},
        "dropped_rows_in_top_k": {"slots": int(hits), "of": int(len(mass) * TOP_K)},
        "steps_where_a_dropped_row_was_in_top_k": int((margin <= 0).sum()),
        "best_dropped_row_rank": {"min": int(best_rank.min()), "median": int(np.median(best_rank)), "p1": int(np.percentile(best_rank, 1))},
        "logit_margin_cutoff_minus_best_dropped": {"min": float(margin.min()), "median": float(np.median(margin))},
        "macs_per_step": {"layers": 24 * (2 * h * h + 2 * h * 192 + 3 * h * 2304), "lm_head_full": int(V) * h, "lm_head_sliced": int(keep.sum()) * h},
        "per_text": per_text,
    }
    Path("results/lm_head_slice.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_text"}, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor="#fcfcfb")
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors="#898781", labelsize=9)
    ax[0].hist(np.log10(np.maximum(mass, 1e-12)), bins=60, color="#2a78d6")
    ax[0].set_title(f"Probability mass on the {int(drop.sum()):,} dropped rows, per decode step", loc="left", fontsize=10.5)
    ax[0].set_xlabel("log10(mass)   (-3 = one in a thousand)", fontsize=9.5, color="#52514e")
    ax[0].set_ylabel("decode steps", fontsize=9.5, color="#52514e")
    ax[1].hist(np.log10(best_rank), bins=60, color="#2a78d6")
    ax[1].axvline(np.log10(TOP_K), color="#52514e", linestyle=(0, (4, 3)), linewidth=1.2)
    ax[1].text(np.log10(TOP_K), ax[1].get_ylim()[1] * 0.95, f" top-{TOP_K} sampling cut-off", fontsize=8.5, color="#52514e", va="top")
    ax[1].set_title("Rank of the best dropped row among all 194k logits", loc="left", fontsize=10.5)
    ax[1].set_xlabel("log10(rank)   (must stay right of the cut-off)", fontsize=9.5, color="#52514e")
    fig.suptitle(f"{len(mass):,} decode steps, {len(texts)} corpus texts x {len(SEEDS)} seeds, Q4 backbone", x=0.01, ha="left", fontsize=9.5, color="#52514e")
    fig.tight_layout()
    fig.savefig("results/lm_head_slice.png", dpi=160)
    print("wrote results/lm_head_slice.json and results/lm_head_slice.png")


if __name__ == "__main__":
    main()
