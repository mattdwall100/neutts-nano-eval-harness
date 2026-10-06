"""Cut a reference voice to a target length at a word boundary, with matching transcript and cached codes.

    uv run make_refs.py dave 5 emily 5        # -> samples/dave5.wav/.txt/.pt, samples/emily5.wav/.txt/.pt

Word timings come from Whisper; the cut is only accepted if Whisper's words up to the cut match the official
transcript's words (normalised), so the shortened transcript is exactly what is spoken in the shortened clip.
"""
import contextlib
import io
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def main(pairs):
    from faster_whisper import WhisperModel
    from neutts import NeuTTS

    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    with contextlib.redirect_stdout(io.StringIO()):
        tts = NeuTTS(backbone_repo="neuphonic/neutts-nano-q4-gguf", codec_repo="neuphonic/neucodec")
    for voice, target in pairs:
        w, sr = sf.read(f"samples/{voice}.wav", dtype="float32")
        mono = w.mean(1) if w.ndim > 1 else w
        w16 = torchaudio.functional.resample(torch.from_numpy(mono), sr, 16000).numpy()
        segs, _ = asr.transcribe(w16, beam_size=5, language="en", word_timestamps=True)
        words = [x for s in segs for x in s.words]
        ref = Path(f"samples/{voice}.txt").read_text(encoding="utf-8").split()
        ok = [i for i in range(min(len(words), len(ref))) if all(norm(words[j].word) == norm(ref[j]) for j in range(i + 1))]
        cands = [i for i in ok if words[i].end <= target + 0.4]
        if not cands:
            print(f"{voice}: no aligned cut at or before {target}s (aligned words: {len(ok)})")
            continue
        k = cands[-1]
        end = min(words[k].end + 0.15, len(mono) / sr)
        name = f"{voice}{int(target)}"
        sf.write(f"samples/{name}.wav", mono[: int(end * sr)], sr)
        Path(f"samples/{name}.txt").write_text(" ".join(ref[: k + 1]), encoding="utf-8")
        codes = tts.encode_reference(f"samples/{name}.wav")
        torch.save(codes, f"samples/{name}.pt")
        print(f"{name}: {end:.2f}s, {k + 1} words, {len(codes)} codes | ...{' '.join(ref[max(k - 3, 0): k + 1])}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main([(a[i], float(a[i + 1])) for i in range(0, len(a), 2)])
