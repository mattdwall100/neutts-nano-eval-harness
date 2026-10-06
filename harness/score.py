"""Quality floor for bench.py results: WER (faster-whisper) and speaker similarity (ECAPA).

Usage:
    uv run score.py results/baseline_torch [results/q4 ...]

Writes results/<name>/scores.jsonl and merges wer/spk_sim aggregates into summary.json.
Absolute values depend on the ASR and embedding models; compare configurations relative to each other
on the same seeds. The reference clip's own WER is reported as the ASR floor.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ASR_SR = 16_000


# ---------------------------------------------------------------- text / WER
def normalize(s: str) -> str:
    s = s.lower().replace("-", " ").replace("’", "'")  # typographic apostrophe: "I’m" must equal ASR's "I'm"
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) at word level, compound-tolerant.

    ASR spells compounds inconsistently ("drug store" vs "drugstore"); either side may match the
    concatenation of two adjacent words on the other side at zero cost, so spelling convention is not
    counted as a mispronunciation.
    """
    r, h = normalize(ref).split(), normalize(hyp).split()
    R, H = len(r), len(h)
    D = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        D[i][0] = i
    for j in range(H + 1):
        D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            best = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
            if i >= 2 and r[i - 2] + r[i - 1] == h[j - 1]:
                best = min(best, D[i - 2][j - 1])
            if j >= 2 and h[j - 2] + h[j - 1] == r[i - 1]:
                best = min(best, D[i - 1][j - 2])
            D[i][j] = best
    return D[R][H], R


# ---------------------------------------------------------------- audio
def load16k(path) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != ASR_SR:
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, ASR_SR).numpy()
    return w


class Scorer:
    def __init__(self, whisper_model="small.en", beam_size=5):
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.asr = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        self.beam_size = beam_size
        self.spk = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=".cache/ecapa",
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,  # Windows: no symlink privilege
        )

    def transcribe(self, wav16k: np.ndarray) -> str:
        # pass arrays, not paths: faster-whisper's PyAV path is broken against current av
        segs, _ = self.asr.transcribe(wav16k, beam_size=self.beam_size, language="en")
        return " ".join(s.text.strip() for s in segs)

    def embed(self, wav16k: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.spk.encode_batch(torch.from_numpy(wav16k)[None]).squeeze()


def score_dir(d: Path, scorer: Scorer):
    meta = json.loads((d / "meta.json").read_text())
    rows = [r for r in (json.loads(l) for l in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if not r.get("failed")]

    ref_wav = load16k(meta["ref_audio"])
    ref_emb = scorer.embed(load16k(meta.get("sim_ref_audio") or meta["ref_audio"]))
    ref_text = Path(meta["ref_text"]).read_text(encoding="utf-8").strip()
    floor_hyp = scorer.transcribe(ref_wav)
    floor_err, floor_n = word_errors(ref_text, floor_hyp)

    cos = torch.nn.functional.cosine_similarity
    scored, tot_err, tot_n = [], 0, 0
    with open(d / "scores.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            w = load16k(d / r["wav"])
            hyp = scorer.transcribe(w)
            err, n = word_errors(r["text"], hyp)
            sim = float(cos(ref_emb, scorer.embed(w), dim=0))
            tot_err += err
            tot_n += n
            s = {k: r[k] for k in ("text_id", "repeat", "seed", "wav", "audio_s")}
            s.update({"hyp": hyp, "word_errors": err, "ref_words": n, "wer": round(err / n, 4), "spk_sim": round(sim, 4)})
            scored.append(s)
            f.write(json.dumps(s) + "\n")
            print(f"  {r['wav']} wer={s['wer']:.3f} sim={sim:.3f}  | {hyp[:70]}")

    agg = {
        "wer_corpus": round(tot_err / tot_n, 4),  # pooled: total errors / total words
        "wer_median": round(float(np.median([s["wer"] for s in scored])), 4),
        "wer_worst": round(max(s["wer"] for s in scored), 4),
        "spk_sim_mean": round(float(np.mean([s["spk_sim"] for s in scored])), 4),
        "spk_sim_min": round(min(s["spk_sim"] for s in scored), 4),
        "asr_floor_wer_on_reference": round(floor_err / floor_n, 4),
        "asr_floor_hyp": floor_hyp,  # kept so --rescore can recompute the floor without ASR
        "n_scored": len(scored),
    }
    summ_path = d / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    summary.update(agg)
    summ_path.write_text(json.dumps(summary, indent=2))
    print(f"{d.name}: {json.dumps(agg)}")
    return agg


def logmel_dist_db(a: np.ndarray, b: np.ndarray, sr: int = 24_000) -> float:
    """Mean |dB| difference of log-mel spectrograms after cross-correlation alignment. Phase-insensitive, so a
    non-causal vocoder decoded in chunks is judged on what is audible, not on sample phase. Identical audio -> 0."""
    import librosa
    from numpy.fft import irfft, rfft

    n = 1 << (max(len(a), len(b)) * 2 - 1).bit_length()
    xc = irfft(rfft(a, n) * np.conj(rfft(b, n)), n)
    lag = int(np.argmax(xc))
    lag = lag - n if lag > n // 2 else lag
    a = a[lag:] if lag > 0 else a
    b = b[-lag:] if lag < 0 else b
    m = min(len(a), len(b))
    mel = lambda x: librosa.power_to_db(librosa.feature.melspectrogram(y=x[:m], sr=sr, n_fft=1024, hop_length=256, n_mels=80), ref=1.0, top_db=80)
    return float(np.abs(mel(a) - mel(b)).mean())


def seam_dir(d: Path, ref_dir: Path):
    """Seam distortion vs the batch wav of the same text+seed (identical tokens): log-mel distance in dB."""
    dists = []
    for r in map(json.loads, filter(str.strip, (d / "rows.jsonl").read_text(encoding="utf-8").splitlines())):
        if r.get("failed") or not (ref_dir / r["wav"]).exists():
            continue
        a, _ = sf.read(d / r["wav"], dtype="float32")
        b, _ = sf.read(ref_dir / r["wav"], dtype="float32")
        if abs(len(a) - len(b)) > 0.05 * min(len(a), len(b)):  # different token sequence: not comparable
            continue
        dists.append(logmel_dist_db(a, b))
    summ_path = d / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    for k in ("seam_snr_db_median", "seam_snr_db_min"):
        summary.pop(k, None)
    summary.update({"seam_logmel_db_median": round(float(np.median(dists)), 2) if dists else None, "seam_logmel_db_max": round(max(dists), 2) if dists else None, "seam_n": len(dists)})
    summ_path.write_text(json.dumps(summary, indent=2))
    print(f"{d.name}: seam log-mel distance median {summary['seam_logmel_db_median']} dB, max {summary['seam_logmel_db_max']} dB, n={len(dists)}")


def rescore_dir(d: Path):
    """Recompute WER from the stored hypotheses (after a scoring change); no ASR, no embeddings."""
    scored = [json.loads(l) for l in (d / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = {(r["text_id"], r["repeat"]): r for r in map(json.loads, (d / "rows.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    tot_err = tot_n = 0
    for s in scored:
        err, n = word_errors(rows[(s["text_id"], s["repeat"])]["text"], s["hyp"])
        s.update({"word_errors": err, "ref_words": n, "wer": round(err / n, 4)})
        tot_err += err
        tot_n += n
    (d / "scores.jsonl").write_text("".join(json.dumps(s) + chr(10) for s in scored), encoding="utf-8")
    summ_path = d / "summary.json"
    summary = json.loads(summ_path.read_text())
    summary.update({"wer_corpus": round(tot_err / tot_n, 4), "wer_median": round(float(np.median([s["wer"] for s in scored])), 4), "wer_worst": round(max(s["wer"] for s in scored), 4)})
    if "asr_floor_hyp" in summary:  # runs scored before the hypothesis was stored keep their old floor
        err, n = word_errors(Path(summary["ref_text"]).read_text(encoding="utf-8").strip(), summary["asr_floor_hyp"])
        summary["asr_floor_wer_on_reference"] = round(err / n, 4)
    summ_path.write_text(json.dumps(summary, indent=2))
    print(f"{d.name}: wer_corpus={summary['wer_corpus']} asr_floor={summary.get('asr_floor_wer_on_reference')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--rescore", action="store_true", help="recompute WER from stored hypotheses only")
    ap.add_argument("--seam-ref", default=None, help="batch run dir with the same seeds: adds seam log-mel distance (dB) of each wav vs its batch twin")
    args = ap.parse_args()
    # expand globs here (PowerShell does not); tolerate patterns that also match files
    dirs = [Path(p) for d in args.dirs for p in (sorted(glob.glob(d)) or [d]) if (Path(p) / "rows.jsonl").exists()]
    if args.seam_ref:
        for d in dirs:
            seam_dir(d, Path(args.seam_ref))
        return
    if args.rescore:
        for d in dirs:
            rescore_dir(d)
        return
    scorer = Scorer(args.whisper, args.beam)
    for d in dirs:
        score_dir(d, scorer)


if __name__ == "__main__":
    # self-check of the WER kernel
    assert word_errors("the cat sat", "the cat sat") == (0, 3)
    assert word_errors("the cat sat", "the sat") == (1, 3)
    assert word_errors("The cat, sat.", "the cat sat on") == (1, 3)
    assert word_errors("a b c", "") == (3, 3)
    assert word_errors("left the drug store", "left the drugstore") == (0, 4)
    assert word_errors("the sideshow", "the side show") == (0, 2)
    assert word_errors("the drug store", "the drugstores") == (2, 3)
    assert word_errors("I’m here", "I'm here") == (0, 2)
    main()
