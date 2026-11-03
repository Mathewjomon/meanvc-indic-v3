"""
Stage 5d - Listening demo: ONE Malayalam sentence, converted into a Malayalam voice and into a
Tamil voice, then played one after the other. Runs on the laptop (CPU, VS Code) or Colab.

WHAT YOU HEAR (in this order)
  1. source            the original Malayalam speaker
  2. Malayalam voice   the reference clip of the target Malayalam speaker
  3. ml -> ml          the source sentence in that Malayalam voice   (same-language conversion)
  4. Tamil voice       the reference clip of the target Tamil speaker
  5. ml -> ta          the source sentence in that Tamil voice       (cross-language conversion)
  With --compare, the original MeanVC (Mandarin) result is played after each of ours.

  The clips come from the TEST speakers (never used in training). Pick your own with
  --source / --ref-ml / --ref-ta (any .wav file, 3-15 s works best for references).

HOW TO RUN (laptop, VS Code terminal with the venv active)
  python "G:\\My Drive\\meanvc_indic_v3\\code\\infer\\demo.py"
  python "G:\\My Drive\\meanvc_indic_v3\\code\\infer\\demo.py" --compare     (also the original MeanVC)
  First run on the laptop: copies the model files (about 2.7 GB) from Drive to D:\\meanvc_cache, once.

OUTPUT  outputs/demo/<source clip>/  source.wav, voice_ml.wav, voice_ta.wav, ml_to_ml_ours.wav, ...
        (the files stay there - click them in VS Code to play them again)
"""

import os
from pathlib import Path

import argparse  # noqa: E402
import sys  # noqa: E402

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert as cv  # noqa: E402

TEST = cv.PROJECT / "data" / "manifests" / "test.tsv"
TEST_AUDIO = cv.PROJECT / "data" / "audio_test"


def pick_clips():
    """Default clips: a Malayalam source of about 6 s; target voices = the longest clip of another
    Malayalam speaker and of a Tamil speaker (longer reference = better voice copy)."""
    t = pd.read_csv(TEST, sep="\t", keep_default_na=False, dtype={"speaker_id": str}).sort_values("utt_id")
    ml, ta = t[t.code == "ml"], t[t.code == "ta"]
    src = ml.iloc[(ml.duration - 6).abs().argsort(kind="stable")].iloc[0]
    others = ml[ml.speaker_id != src.speaker_id]
    ref_ml = others.sort_values("duration", ascending=False).iloc[0]
    ref_ta = ta.sort_values("duration", ascending=False).iloc[0]
    path = lambda row: TEST_AUDIO / row.path
    return path(src), path(ref_ml), path(ref_ta), src.text_verbatim


def play(path, label):
    """Play a .wav and wait until it ends (Windows); elsewhere just print where it is."""
    print(f"  > {label:<34} {path.name}", flush=True)
    if os.name == "nt":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)


def main():
    ap = argparse.ArgumentParser(description="Hear Malayalam -> Malayalam and Malayalam -> Tamil conversion")
    ap.add_argument("--source", default=None, help="Malayalam speech to convert (default: a test clip)")
    ap.add_argument("--ref-ml", default=None, help="a clip of the target Malayalam voice")
    ap.add_argument("--ref-ta", default=None, help="a clip of the target Tamil voice")
    ap.add_argument("--compare", action="store_true", help="also convert with the original MeanVC")
    ap.add_argument("--no-play", action="store_true", help="only save the files")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    d_src, d_ml, d_ta, text = pick_clips()
    src_path = Path(args.source or d_src)
    refs = {"ml": Path(args.ref_ml or d_ml), "ta": Path(args.ref_ta or d_ta)}
    out = cv.PROJECT / "outputs" / "demo" / src_path.stem
    print(f"Source: {src_path.name}" + ("" if args.source else f"  (text: {text})"))
    print(f"Target voices: Malayalam {refs['ml'].name} | Tamil {refs['ta'].name}")

    print("Loading the models (first time on the laptop: a few minutes)...")
    systems = ["ours"] + (["mandarin"] if args.compare else [])
    vc = cv.VoiceConverter(systems[0], args.device)
    src = cv.load_wav(src_path)
    cv.save_wav(out / "source.wav", src)
    e_src = vc.embed(src)
    plan = []                                            # (file, label) in listening order
    plan.append((out / "source.wav", "1. source (Malayalam speaker)"))
    results = {}
    for system in systems:
        if system != vc.system:
            vc.set_system(system)
        for code, name in (("ml", "Malayalam"), ("ta", "Tamil")):
            ref = cv.load_wav(refs[code])
            cv.save_wav(out / f"voice_{code}.wav", ref)
            wav, info = vc.convert(src, ref)
            f = out / f"ml_to_{code}_{system}.wav"
            cv.save_wav(f, wav)
            e = vc.embed(wav)
            results[(system, code)] = (f, cv.similarity(e, vc.embed(ref)), cv.similarity(e, e_src), info)
            print(f"  {system:8s} ml -> {code}: similarity to {name} voice "
                  f"{results[(system, code)][1]:.3f}, to the source {results[(system, code)][2]:.3f} | "
                  f"{info['duration']:.1f} s of speech in {info['seconds']:.1f} s on {vc.device} "
                  f"(RTF {info['rtf']:.2f})")
    for i, (code, name) in enumerate((("ml", "Malayalam"), ("ta", "Tamil"))):
        plan.append((out / f"voice_{code}.wav", f"{2 + 2 * i}. {name} target voice"))
        plan.append((results[("ours", code)][0], f"{3 + 2 * i}. ml -> {code} (ours)"))
        if args.compare:
            plan.append((results[("mandarin", code)][0], f"   ml -> {code} (original MeanVC)"))

    print(f"\nSaved in {out}")
    if not args.no_play:
        print("Playing:" if os.name == "nt" else "Files to listen to:")
        for f, label in plan:
            play(f, label)


if __name__ == "__main__":
    main()