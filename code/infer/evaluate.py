"""
Stage 5b - Evaluate voice conversion on the TEST speakers (never seen in training). Run on Colab.

WHAT IT DOES
  1. Plans the conversions (same plan for every system, saved as pairs.csv):
       9 language pairs (ml/ta/hi -> ml/ta/hi, same-language AND cross-language),
       12 conversions each: a source clip + a different target speaker
       (reference = that speaker's longest test clip).
  2. Converts with each system: mandarin (original MeanVC), enc_only (our encoder + Mandarin
     decoder) and ours (our encoder + our decoder). Same noise for every system.
  3. Measures every converted clip:
       sim_target  voice similarity to the target speaker's reference clip    higher = better
       sim_other   ... to a DIFFERENT clip of the target speaker              higher = better
       sim_source  ... to the source speaker (their voice leaking through)    lower = better
       cer_mms     % of characters wrong when an independent recogniser       lower = better
                   (Meta MMS-1b-all, not trained by us) reads the result
       cer_ours    the same with our own encoder's recogniser (v2's "read-back CER")
       dnsmos      speech quality score 1-5 (Microsoft DNSMOS, overall)       higher = better
       rtf         processing time / audio length on this GPU                 lower = faster
     and the same numbers for the real recordings, as the reference point.
  4. Prints and saves the tables, plus audio examples to listen to.

HOW TO RUN (Colab, after Stage 4)
  !pip install -q speechmos onnxruntime
  quick test:  !python /content/drive/MyDrive/meanvc_indic_v3/code/inference/evaluate.py --n 1
  full:        !python /content/drive/MyDrive/meanvc_indic_v3/code/inference/evaluate.py

OUTPUT (Drive: outputs/eval_v3/)
  summary.md / summary.csv     the tables (per system, same vs cross language, per language pair)
  results.csv                  every conversion, every number
  examples/<pair>/             source.wav, reference.wav, mandarin.wav, enc_only.wav, ours.wav
  wavs/<system>/<pair>.wav     all converted clips
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent                  # code/inference
sys.path.insert(0, str(HERE))
import convert as cv  # noqa: E402  (VoiceConverter, load_wav, save_wav; sets up all other paths)
import train_encoder as te  # noqa: E402
from vocab import load_vocab, normalize  # noqa: E402

PROJECT = cv.PROJECT
DATA = PROJECT / "data"
TEST_AUDIO = DATA / "audio_test"
ENC_DIR = PROJECT / "checkpoints" / "encoder_v3"
OUT = PROJECT / "outputs" / "eval_v3"
LANGS = ("ml", "ta", "hi")
N_PER_PAIR = 12
EXAMPLES_PER_PAIR = 2
MMS_LANG = {"ml": "mal", "ta": "tam", "hi": "hin"}


# ======================================================================================
# The plan
# ======================================================================================
def plan_pairs(test, n_per_pair, seed=0):
    """For each source language x target language: n source clips (spread over speakers)
    and a different target speaker for each. Reference = target speaker's longest clip,
    'other' = their second-longest (used to check similarity on an unseen clip)."""
    rng = np.random.default_rng(seed)
    test = test.assign(spk=test.code + "_" + test.speaker_id)
    clips = {s: g.sort_values(["duration", "utt_id"], ascending=[False, True]).utt_id.tolist()
             for s, g in test.groupby("spk")}
    rows = []
    for s_lang in LANGS:
        groups = [rng.permutation(g.utt_id.tolist()).tolist()
                  for _, g in test[test.code == s_lang].groupby("spk")]
        order = [g[k] for k in range(max(map(len, groups))) for g in groups if k < len(g)]
        for t_i, t_lang in enumerate(LANGS):
            voices = [s for s in rng.permutation(test[test.code == t_lang].spk.unique())
                      if len(clips[s]) >= 2]
            sources = order[t_i * n_per_pair:(t_i + 1) * n_per_pair]   # new sources per pair
            for k, utt in enumerate(sources):
                src = test.loc[test.utt_id == utt].iloc[0]
                options = [v for v in voices if v != src.spk]
                if not options:
                    continue
                tgt = options[k % len(options)]
                rows.append(dict(pair_id=f"{s_lang}2{t_lang}_{k:02d}", src_lang=s_lang, tgt_lang=t_lang,
                                 source=utt, source_spk=src.spk, target_spk=tgt,
                                 reference=clips[tgt][0], other=clips[tgt][1],
                                 text=src.text_verbatim, duration=src.duration))
    return pd.DataFrame(rows)


# ======================================================================================
# Scorers
# ======================================================================================
def char_errors(ref, hyp):
    """Character edit distance after the same text cleaning as training -> (errors, length)."""
    ref, hyp = normalize(ref), normalize(hyp)
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(ref, hyp), len(ref)
    except ImportError:
        prev = list(range(len(hyp) + 1))
        for i, rc in enumerate(ref, 1):
            cur = [i] + [0] * len(hyp)
            for j, hc in enumerate(hyp, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
            prev = cur
        return prev[-1], len(ref)


class OurReader:
    """Our trained encoder + its CTC head as a speech recogniser (greedy decoding)."""

    def __init__(self, device):
        vocab = load_vocab()
        self.id2tok, self.device = {i: t for t, i in vocab.items()}, device
        model = te.build_model(len(vocab), ENC_DIR / "cmvn_from_pretrained.json")
        model.load_state_dict(torch.load(ENC_DIR / "best.pt", map_location="cpu"))   # strict
        te.set_streaming(model)
        self.model = model.to(device).eval()

    @torch.inference_mode()
    def read(self, wav, code):
        feats = te.compute_fbank(wav).unsqueeze(0).to(self.device)
        lens = torch.tensor([feats.shape[1]], device=self.device)
        enc, _, elens = te.run_encoder(self.model, feats, lens, te.CHUNK, te.LEFT_CHUNKS)
        return te.greedy_ctc_text(self.model, enc, elens, self.id2tok)[0]


class MmsReader:
    """Meta MMS-1b-all (1,100+ languages), an independent recogniser we did not train."""

    def __init__(self, device):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device).eval()
        self.device, self.lang = device, None

    @torch.inference_mode()
    def read(self, wav, code):
        if code != self.lang:                                  # switch the language adapter
            self.proc.tokenizer.set_target_lang(MMS_LANG[code])
            self.model.load_adapter(MMS_LANG[code])
            self.model.to(self.device)
            self.lang = code
        x = self.proc(wav, sampling_rate=cv.SR, return_tensors="pt").to(self.device)
        return self.proc.decode(self.model(**x).logits.argmax(-1)[0])


def read_all(reader, items):
    """items: list of (key, wav, language) -> {key: text}. Sorted by language (fewer switches)."""
    return {key: reader.read(wav, code) for key, wav, code in sorted(items, key=lambda it: it[2])}


def dnsmos_all(items):
    from speechmos import dnsmos
    return {key: float(dnsmos.run(np.clip(wav, -1.0, 1.0), sr=cv.SR)["ovrl_mos"])   # must be in [-1, 1]
            for key, wav, _ in items}


# ======================================================================================
# Tables
# ======================================================================================
COLS = ["sim_target", "sim_other", "sim_source", "cer_mms", "cer_ours", "dnsmos", "rtf"]


def summarise(df):
    """Mean of each number; CER as total errors / total characters (in %)."""
    out = {c: df[c].mean() for c in ("sim_target", "sim_other", "sim_source", "dnsmos", "rtf") if c in df}
    for asr in ("mms", "ours"):
        if f"err_{asr}" in df and df[f"err_{asr}"].notna().any():
            out[f"cer_{asr}"] = 100 * df[f"err_{asr}"].sum() / df[f"len_{asr}"].sum()
    out["n"] = len(df)
    return out


def table(rows, title):
    """-> (markdown table text, DataFrame)."""
    df = pd.DataFrame(rows).set_index("system")
    df = df[[c for c in COLS + ["n"] if c in df]]
    lines = ["| system | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    for name, row in df.iterrows():
        cells = []
        for c, v in row.items():
            digits = 0 if c == "n" else (1 if c.startswith("cer") else (2 if c in ("dnsmos", "rtf") else 3))
            cells.append("-" if pd.isna(v) else f"{v:.{digits}f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return f"### {title}\n\n" + "\n".join(lines) + "\n", df


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate voice conversion on the test speakers")
    ap.add_argument("--n", type=int, default=N_PER_PAIR, help="conversions per language pair")
    ap.add_argument("--systems", default="mandarin,enc_only,ours")
    ap.add_argument("--dit", default=None, help="DiT for 'ours' (default checkpoints/dit_v3/dit_best.pt)")
    ap.add_argument("--no-mms", action="store_true", help="skip the independent recogniser")
    ap.add_argument("--no-dnsmos", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    systems = args.systems.split(",")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT if args.n == N_PER_PAIR else OUT.parent / f"eval_v3_n{args.n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | systems: {', '.join(systems)} | output: {out_dir}")

    # ---- 1. plan + audio ----
    print("1. Test set and plan")
    test = pd.read_csv(DATA / "manifests" / "test.tsv", sep="\t", keep_default_na=False,
                       dtype={"speaker_id": str})
    pairs = plan_pairs(test, args.n)
    pairs.to_csv(out_dir / "pairs.csv", index=False)
    path_of = dict(zip(test.utt_id, test.path))
    real = sorted(set(pairs.source) | set(pairs.reference) | set(pairs.other))
    audio = {u: cv.load_wav(TEST_AUDIO / path_of[u]) for u in real}
    print(f"  {len(test)} test clips, {test.code.add(test.speaker_id).nunique()} speakers | "
          f"{len(pairs)} conversions per system ({args.n} per language pair x 9 pairs)")

    # ---- 2. convert ----
    print("2. Converting")
    vc = cv.VoiceConverter(systems[0], device, dit_path=args.dit if systems[0] == "ours" else None)
    emb = {u: vc.embed(audio[u]) for u in real}                       # real clips: once
    voices = {u: vc.voice(audio[u]) for u in sorted(set(pairs.reference))}
    converted, results = {}, []
    for system in systems:
        if system != vc.system:
            vc.set_system(system, args.dit if system == "ours" else None)
        print(f"  {system}: {vc.describe()}")
        t0 = time.time()
        for i, p in enumerate(pairs.itertuples()):
            wav, info = vc.convert(audio[p.source], voice=voices[p.reference], seed=i)
            converted[(system, p.pair_id)] = wav
            e = vc.embed(wav)
            results.append(dict(system=system, pair_id=p.pair_id, src_lang=p.src_lang, tgt_lang=p.tgt_lang,
                                sim_target=cv.similarity(e, emb[p.reference]),
                                sim_other=cv.similarity(e, emb[p.other]),
                                sim_source=cv.similarity(e, emb[p.source]), rtf=info["rtf"]))
            cv.save_wav(out_dir / "wavs" / system / f"{p.pair_id}.wav", wav)
            if (i + 1) % 27 == 0 or i + 1 == len(pairs):
                print(f"    {i + 1}/{len(pairs)} | {(time.time() - t0) / (i + 1):.1f} s per clip", flush=True)
    # the real recordings as the reference point
    for p in pairs.itertuples():
        results.append(dict(system="real speech", pair_id=p.pair_id, src_lang=p.src_lang, tgt_lang=p.tgt_lang,
                            sim_target=cv.similarity(emb[p.other], emb[p.reference]),   # same speaker
                            sim_source=cv.similarity(emb[p.source], emb[p.reference])))  # before conversion
    res = pd.DataFrame(results)

    # examples to listen to: source, reference and every system side by side
    for p in pairs.itertuples():
        if int(p.pair_id[-2:]) < EXAMPLES_PER_PAIR:
            folder = out_dir / "examples" / p.pair_id
            cv.save_wav(folder / "source.wav", audio[p.source])
            cv.save_wav(folder / "reference.wav", audio[p.reference])
            for system in systems:
                cv.save_wav(folder / f"{system}.wav", converted[(system, p.pair_id)])
    lang_of = dict(zip(pairs.pair_id, pairs.src_lang))
    text_of = dict(zip(pairs.pair_id, pairs.text))
    items = [((s, pid), w, lang_of[pid]) for (s, pid), w in converted.items()]
    items += [(("real speech", p.pair_id), audio[p.source], p.src_lang) for p in pairs.itertuples()]
    del vc
    torch.cuda.empty_cache() if device == "cuda" else None

    # ---- 3. score ----
    print("3. Scoring")
    readers = [("ours", OurReader)] + ([] if args.no_mms else [("mms", MmsReader)])
    for name, cls in readers:
        t0 = time.time()
        try:
            reader = cls(device)
            texts = read_all(reader, items)
            del reader
        except Exception as e:                                          # noqa: BLE001
            print(f"  recogniser '{name}' failed ({type(e).__name__}: {e}) - skipped")
            continue
        errs = {k: char_errors(text_of[k[1]], t) for k, t in texts.items()}
        key = list(zip(res.system, res.pair_id))
        res[f"err_{name}"] = [errs[k][0] for k in key]
        res[f"len_{name}"] = [errs[k][1] for k in key]
        res[f"text_{name}"] = [texts[k] for k in key]
        print(f"  recogniser '{name}': {len(texts)} clips in {time.time() - t0:.0f} s")
    if not args.no_dnsmos:
        try:
            t0 = time.time()
            mos = dnsmos_all(items)
            res["dnsmos"] = [mos[k] for k in zip(res.system, res.pair_id)]
            print(f"  DNSMOS: {len(mos)} clips in {time.time() - t0:.0f} s")
        except Exception as e:                                          # noqa: BLE001
            print(f"  DNSMOS failed ({type(e).__name__}: {e}) - skipped (tell Claude)")
    res.to_csv(out_dir / "results.csv", index=False)

    # ---- 4. tables ----
    order = ["real speech"] + systems
    same = res.src_lang == res.tgt_lang
    parts, sums = [], []
    for title, mask in (("All conversions", slice(None)), ("Same language (e.g. ml -> ml)", same),
                        ("Cross-language (e.g. ml -> ta)", ~same)):
        sub = res[mask] if not isinstance(mask, slice) else res
        rows = [dict(system=s, **summarise(sub[sub.system == s])) for s in order]
        text, df = table(rows, title)
        parts.append(text)
        sums.append(df.assign(group=title))
    rows = []
    for s_lang in LANGS:
        for t_lang in LANGS:
            sub = res[(res.system == "ours") & (res.src_lang == s_lang) & (res.tgt_lang == t_lang)]
            if len(sub):
                rows.append(dict(system=f"{s_lang} -> {t_lang}", **summarise(sub)))
    if rows:
        text, df = table(rows, "Ours, per language pair (source -> target voice)")
        parts.append(text)
        sums.append(df.assign(group="ours per pair"))
    legend = ("sim_* = voice similarity (cosine, WavLM-ECAPA): sim_target/sim_other higher = better, "
              "sim_source lower = better. 'real speech' row: sim_target = two real clips of the target "
              "speaker, sim_source = source vs target before conversion, CER/DNSMOS = the source recording. "
              "cer_* in %, lower = better. dnsmos 1-5, higher = better. rtf < 1 = faster than real time.")
    report = "\n".join(parts) + "\n" + legend + "\n"
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    pd.concat(sums).to_csv(out_dir / "summary.csv")
    print("\n" + report)
    print(f"Saved: {out_dir} (summary.md, results.csv, examples/, wavs/)")


if __name__ == "__main__":
    main()
