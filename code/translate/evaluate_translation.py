"""
Stage 6b - Test speech translation + voice conversion end to end on the TEST speakers (never seen in
training). Runs on the laptop (RTX 2050 + CPU, ~35 min) or Colab.

PURPOSE
  Measure every step of speech_translator.py on real spontaneous speech, so the thesis can report
  how well each part works, which speech recogniser is best, and what our voice conversion adds.
  Nothing is trained or changed.

WHAT IT DOES
  A. Speech recognisers compared on ALL 300 test clips (100 per language). Language ID by our
     encoder (script voting), then each recogniser transcribes in the detected language:
       ours            our fine-tuned Fast-U2++ encoder, greedy CTC, 200 ms streaming chunks
       mms             Meta MMS-1b-all (wav2vec 2.0 + language adapter)
       indicconformer  AI4Bharat IndicConformer 600M, RNNT decoding
     -> CER and WER against the real transcripts (same text cleaning as training), time per clip,
        and on the clips of part B: how much each recogniser's errors change the translation
        (chrF of its translation vs the translation of the correct transcript).
  B. The whole pipeline with IndicConformer, on N clips per direction (6 directions: ml->ta, ml->hi,
     ta->ml, ta->hi, hi->ml, hi->ta; default N = 8 -> 48 translations):
       speaker A = the source speaker (language detected, as a real user would run it)
       speaker B = ANOTHER test speaker of the target language: the voice to copy
                   (reference = B's longest clip; B's second-longest clip is kept aside)
     The same TTS speech is given three voices, to see what the voice conversion adds:
       tts        MMS-TTS output, no voice conversion (generic voice)       - lower bound
       mandarin   + original MeanVC (Mandarin encoder + decoder)            - baseline
       ours       + our MeanVC v3 (fine-tuned on ml/ta/hi)                  - the result
     Measured for every output:
       sim_ref     voice similarity to B's reference clip                          higher = better
       sim_other   ... to B's OTHER clip (never given to the model; stricter)      higher = better
       sim_src     ... to speaker A (how much of A's voice is left)                lower = better
       cer_mms     % characters wrong when Meta MMS-1b-all reads the output, vs the translated
                   text it should say                                              lower = better
       dnsmos      speech quality 1-5 (Microsoft DNSMOS overall)                   higher = better
     and for the pipeline: chrf_mt (as in A) and seconds per step; rtf = total time / input length.
  C. Writes the results folder for readers (tables, every number, all audio with clear names).

ROLE IN THE PIPELINE
  Uses SpeechTranslator (speech_translator.py) exactly as a user would, plus the scorers of
  code/infer/evaluate.py. Same test set and speaker-similarity model as the Stage 5 voice-conversion
  evaluation (outputs/eval_v3), so the numbers compare.

HOW TO RUN (laptop, VS Code terminal with the venv active)
  python "G:\\My Drive\\meanvc_indic_v3\\code\\translate\\evaluate_translation.py"            full (N = 8)
  python "G:\\My Drive\\meanvc_indic_v3\\code\\translate\\evaluate_translation.py" --n 1      quick test

OUTPUT  outputs/stage6_speech_translation_voice_conversion_results/   (..._n<N>/ for other N)
  results_tables.md          all tables (open in VS Code, Ctrl+Shift+V)
  summary.csv                the table numbers
  asr_comparison.csv         one row per test clip: language ID + every recogniser's transcript and errors
  results.csv                one row per translation: texts, similarities, CER, DNSMOS, times
  examples/<run>/            1_source_speech_speakerA.wav, 2_reference_voice_speakerB.wav,
                             3_tts_translation_generic_voice.wav, 4_converted_ours.wav,
                             5_converted_original_meanvc.wav, texts.txt
  (README.md in that folder explains everything for readers.)
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_translator as st  # noqa: E402  (pipeline; also sets up the project paths and HF cache)
import evaluate as ev  # noqa: E402  (code/infer/evaluate.py: char_errors, dnsmos_all)
from sacrebleu.metrics import CHRF  # noqa: E402
from vocab import normalize  # noqa: E402

cv = st.cv
PROJECT = st.PROJECT
TEST_AUDIO = PROJECT / "data" / "audio_test"
LANGS = list(st.LANGS)
N_PER_DIRECTION = 8
OUT_NAME = "stage6_speech_translation_voice_conversion_results"
ASRS = ("ours", "mms", "indicconformer")
ASR_NAMES = {"ours": "Our encoder (greedy CTC, streaming)", "mms": "Meta MMS-1b-all",
             "indicconformer": "AI4Bharat IndicConformer 600M (RNNT)"}
NAMES = {c: st.LANGS[c]["name"] for c in LANGS}
EXAMPLE_FILES = {"input": "1_source_speech_speakerA.wav", "reference": "2_reference_voice_speakerB.wav",
                 "tts": "3_tts_translation_generic_voice.wav", "ours": "4_converted_ours.wav",
                 "mandarin": "5_converted_original_meanvc.wav"}


# ======================================================================================
# The plan
# ======================================================================================
def plan_runs(test, n, seed=0):
    """N source clips (speaker A) per direction, 3-12 s, spread over speakers (round robin), new
    clips for each direction. Speaker B (the voice) = a test speaker of the target language,
    rotating over all of them; reference = B's longest clip, other = B's second-longest."""
    rng = np.random.default_rng(seed)
    test = test.assign(spk=test.code + "_" + test.speaker_id)
    clips = {s: g.sort_values(["duration", "utt_id"], ascending=[False, True]).utt_id.tolist()
             for s, g in test.groupby("spk")}
    voices = {c: [s for s in rng.permutation(sorted(test[test.code == c].spk.unique())) if len(clips[s]) >= 2]
              for c in LANGS}
    rows, used = [], {c: 0 for c in LANGS}                     # voices used so far, per target language
    for s in LANGS:
        pool = test[(test.code == s) & test.duration.between(3, 12)]
        groups = [rng.permutation(g.utt_id.tolist()).tolist() for _, g in pool.groupby("spk")]
        groups = [groups[i] for i in rng.permutation(len(groups))]
        order = [g[k] for k in range(max(map(len, groups))) for g in groups if k < len(g)]
        for d, t in enumerate(c for c in LANGS if c != s):
            for k, utt in enumerate(order[d * n:(d + 1) * n]):
                row = test.loc[test.utt_id == utt].iloc[0]
                b = voices[t][used[t] % len(voices[t])]
                used[t] += 1
                rows.append(dict(run_id=f"{s}2{t}_{k:02d}", src_lang=s, tgt_lang=t, source=utt,
                                 speaker_a=row.spk, speaker_b=b, reference=clips[b][0], other=clips[b][1],
                                 text=row.text_verbatim, duration=row.duration))
    return pd.DataFrame(rows)


def load_wav_retry(path, tries=3):
    """Google Drive (stream mode) sometimes serves a file late on the first read -> try again."""
    for k in range(tries):
        try:
            return cv.load_wav(path)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2)


def word_errors(ref, hyp):
    """Word edit distance after the training text cleaning -> (errors, words)."""
    from rapidfuzz.distance import Levenshtein
    r, h = normalize(ref).split(), normalize(hyp).split()
    return Levenshtein.distance(r, h), len(r)


def rate(errors, total):
    return 100 * sum(errors) / max(sum(total), 1)


def fmt(v, digits):
    return "-" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|---" + "|:---:" * (len(header) - 1) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def release_memory():
    """Call after dropping the last reference to a model (x = None) to give its GPU memory back."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="End-to-end test of speech translation + voice conversion")
    ap.add_argument("--n", type=int, default=N_PER_DIRECTION, help="translations per direction")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-dnsmos", action="store_true")
    ap.add_argument("--asr-limit", type=int, default=None,
                    help="part A on only N clips per language (+ the clips of part B) - for a quick test")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT / "outputs" / (OUT_NAME if args.n == N_PER_DIRECTION else f"{OUT_NAME}_n{args.n}")
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"Device: {device} | output: {out_dir}")

    test = pd.read_csv(PROJECT / "data" / "manifests" / "test.tsv", sep="\t", keep_default_na=False,
                       dtype={"speaker_id": str})
    plan = plan_runs(test, args.n)
    audio = {u: load_wav_retry(TEST_AUDIO / p) for u, p in zip(test.utt_id, test.path)}
    print(f"Test set: {len(test)} clips, {test.speaker_id.nunique()} speakers | plan: {len(plan)} translations, "
          f"{plan.speaker_a.nunique()} source speakers, {plan.speaker_b.nunique()} target voices")

    # ---- A. language ID + three recognisers on every test clip ----
    part_a = test
    if args.asr_limit:
        keep = test.groupby("code").head(args.asr_limit).index
        part_a = test[test.index.isin(keep) | test.utt_id.isin(plan.source)]
    print(f"A. Language ID + 3 speech recognisers on {len(part_a)} test clips")
    ours = st.Recognizer(device)
    asr_rows = []
    for r in part_a.itertuples():
        t0 = time.time()
        hyp = ours.transcribe(audio[r.utt_id])
        code, conf, counts = st.identify_language(hyp)
        lang = code or r.code
        asr_rows.append(dict(utt_id=r.utt_id, lang=r.code, detected=code, confidence=conf, used_lang=lang,
                             duration=r.duration, text=r.text_verbatim, text_norm=r.text_normalized,
                             hyp_ours=st.keep_script(hyp, lang),
                             sec_ours=time.time() - t0, **{f"n_{c}": counts[c] for c in LANGS}))
    asr = pd.DataFrame(asr_rows)
    asr["lid_correct"] = asr.detected == asr.lang
    print(f"   language ID {100 * asr.lid_correct.mean():.1f}% correct")
    ic = st.IndicConformerRecognizer()
    for name in ("indicconformer", "mms"):
        model = ic if name == "indicconformer" else st.MmsRecognizer(device)
        hyps, secs = [], []
        for r in asr.itertuples():
            t0 = time.time()
            hyps.append(st.keep_script(model.transcribe(audio[r.utt_id], r.used_lang), r.used_lang))
            secs.append(time.time() - t0)
        asr[f"hyp_{name}"], asr[f"sec_{name}"] = hyps, secs
        print(f"   {name}: done ({np.mean(secs):.2f} s per clip)", flush=True)
    model = None                                                 # MMS off the GPU before the pipeline
    release_memory()
    # errors vs the verbatim transcript (as spoken, incl. colloquial forms) and vs the normalised
    # transcript (standard spelling) - IndicVoices-R provides both
    for name in ASRS:
        for ref_col, tag in (("text", ""), ("text_norm", "n")):
            ce = [ev.char_errors(t, h) for t, h in zip(asr[ref_col], asr[f"hyp_{name}"])]
            we = [word_errors(t, h) for t, h in zip(asr[ref_col], asr[f"hyp_{name}"])]
            asr[f"cerr{tag}_{name}"], asr[f"chars{tag}_{name}"] = [e for e, _ in ce], [n for _, n in ce]
            asr[f"werr{tag}_{name}"], asr[f"words{tag}_{name}"] = [e for e, _ in we], [n for _, n in we]
    asr.to_csv(out_dir / "asr_comparison.csv", index=False, encoding="utf-8")

    # ---- B. the whole pipeline ----
    t0 = time.time()
    pipe = st.SpeechTranslator(device, asr="indicconformer", asr_model=ic)
    print(f"B. Pipeline loaded in {time.time() - t0:.0f} s ({pipe.vc.describe()}); "
          f"{len(plan)} translations, voice = another test speaker (B)")
    # how much each recogniser's errors change the translation (chrF vs translation of the real text)
    hyp_of = {name: dict(zip(asr.utt_id, asr[f"hyp_{name}"])) for name in ASRS}
    mt = {"oracle": [], **{name: [] for name in ASRS}}
    for p in plan.itertuples():
        mt["oracle"].append(pipe.translator.translate([p.text], p.src_lang, p.tgt_lang)[0])
        for name in ASRS:
            mt[name].append(pipe.translator.translate([hyp_of[name][p.source]], p.src_lang, p.tgt_lang)[0])

    p0 = plan.iloc[0]
    pipe.run(audio[p0.source], p0.tgt_lang, reference_wav=audio[p0.reference], seed=0)   # warm-up
    rows, wavs = [], {}
    for i, p in enumerate(plan.itertuples()):
        a, b, b_other = audio[p.source], audio[p.reference], audio[p.other]
        try:
            r = pipe.run(a, p.tgt_lang, reference_wav=b, seed=i)
        except ValueError as e:                                  # e.g. language detected = target
            rows.append(dict(run_id=p.run_id, error=str(e)))
            print(f"   {p.run_id}: FAILED - {e}")
            continue
        e_a, e_ref, e_other = pipe.vc.embed(a), pipe.vc.embed(b), pipe.vc.embed(b_other)
        e_tts, e_out = pipe.vc.embed(r["tts_wav"]), pipe.vc.embed(r["output_wav"])
        rows.append(dict(
            run_id=p.run_id, src_lang=p.src_lang, tgt_lang=p.tgt_lang, speaker_a=p.speaker_a,
            speaker_b=p.speaker_b, source=p.source, reference=p.reference, other=p.other,
            detected=r["source"], lid_conf=r["lid"]["confidence"], input_s=r["input_seconds"],
            output_s=r["output_seconds"], text=p.text, transcript=r["transcript"], translation=r["translation"],
            translation_oracle=mt["oracle"][i],
            sim_real_b=cv.similarity(e_other, e_ref),            # two real clips of B (ceiling)
            sim_real_ab=cv.similarity(e_a, e_ref),               # A vs B before conversion
            **{f"{m}_tts": cv.similarity(e_tts, e) for m, e in (("sim_ref", e_ref), ("sim_other", e_other),
                                                                  ("sim_src", e_a))},
            **{f"{m}_ours": cv.similarity(e_out, e) for m, e in (("sim_ref", e_ref), ("sim_other", e_other),
                                                                   ("sim_src", e_a))},
            **{f"t_{k}": v for k, v in r["times"].items()}, rtf=r["rtf"]))
        wavs[p.run_id] = dict(input=a, reference=b, tts=r["tts_wav"], ours=r["output_wav"])
        print(f"   {i + 1}/{len(plan)} {p.run_id}: voice B {rows[-1]['sim_ref_tts']:.2f} -> "
              f"{rows[-1]['sim_ref_ours']:.2f} | {r['times']['total']:.1f} s", flush=True)
    res = pd.DataFrame(rows)
    ok = res[res["error"].isna()].copy() if "error" in res else res.copy()

    print("   baseline: the same TTS speech through the original (Mandarin) MeanVC")
    pipe.vc.set_system("mandarin")
    base = {m: [] for m in ("sim_ref", "sim_other", "sim_src")}
    for i, p in enumerate(ok.itertuples()):
        w = wavs[p.run_id]
        out, _ = pipe.vc.convert(w["tts"], voice=pipe.vc.voice(w["reference"]), seed=i)
        w["mandarin"] = out
        e_out = pipe.vc.embed(out)
        base["sim_ref"].append(cv.similarity(e_out, pipe.vc.embed(w["reference"])))
        base["sim_other"].append(cv.similarity(e_out, pipe.vc.embed(audio[p.other])))
        base["sim_src"].append(cv.similarity(e_out, pipe.vc.embed(w["input"])))
    for m, v in base.items():
        ok[f"{m}_mandarin"] = v
    pipe = ic = None
    release_memory()

    print("   MMS-1b-all reads every output (intelligibility)")
    mms = st.MmsRecognizer(device)
    for kind in ("tts", "mandarin", "ours"):
        hyp = {p.run_id: mms.transcribe(wavs[p.run_id][kind], p.tgt_lang)
               for p in ok.sort_values("tgt_lang").itertuples()}
        ok[f"mms_{kind}"] = ok.run_id.map(hyp)
        errs = [ev.char_errors(t, hyp[r]) for r, t in zip(ok.run_id, ok.translation)]
        ok[f"err_{kind}"], ok[f"len_{kind}"] = [e for e, _ in errs], [n for _, n in errs]
    mms = None
    release_memory()
    if not args.no_dnsmos:
        print("   DNSMOS (speech quality)")
        for kind in ("input", "reference", "tts", "mandarin", "ours"):
            mos = ev.dnsmos_all([(r, wavs[r][kind], None) for r in ok.run_id])
            ok[f"dnsmos_{kind}"] = ok.run_id.map(mos)
    ok.to_csv(out_dir / "results.csv", index=False, encoding="utf-8")

    # ---- C. audio + texts for listening ----
    for p in ok.itertuples():
        d = out_dir / "examples" / p.run_id
        for kind, fname in EXAMPLE_FILES.items():
            cv.save_wav(d / fname, wavs[p.run_id][kind])
        (d / "texts.txt").write_text(
            f"{NAMES[p.src_lang]} speech of speaker A, translated into {NAMES[p.tgt_lang]}, spoken in the "
            f"voice of speaker B\n\n"
            f"speaker A (source)  {p.speaker_a}   clip {p.source} ({p.input_s:.1f} s)\n"
            f"speaker B (voice)   {p.speaker_b}   clip {p.reference} (reference)\n"
            f"language detected   {NAMES.get(p.detected, p.detected)} ({p.lid_conf:.0%} of letters)\n\n"
            f"real transcript            {p.text}\n"
            f"recognised (IndicConformer) {p.transcript}\n\n"
            f"translation                {p.translation}\n"
            f"translation of real text   {p.translation_oracle}\n"
            f"MMS hears our output as    {p.mms_ours}\n\n"
            f"voice similarity to B (reference clip): TTS {p.sim_ref_tts:.3f} | original MeanVC "
            f"{p.sim_ref_mandarin:.3f} | ours {p.sim_ref_ours:.3f}   (two real clips of B: {p.sim_real_b:.3f})\n"
            f"similarity to A (leakage):              ours {p.sim_src_ours:.3f}   (A vs B before: {p.sim_real_ab:.3f})\n",
            encoding="utf-8")

    # ======================================================================================
    # Tables
    # ======================================================================================
    chrf = CHRF()

    def chrf_of(hyps, refs):
        return chrf.corpus_score(list(hyps), [list(refs)]).score

    def mean(df, col):
        return df[col].mean() if col in df else None

    def cer_out(df, kind):
        return rate(df[f"err_{kind}"], df[f"len_{kind}"])

    parts = [f"# Speech translation + voice conversion: results\n\n"
             f"Test speakers never seen in training. Part A: all {len(asr)} test clips. Part B: {len(ok)} "
             f"translations ({args.n} per direction x 6 directions), {ok.speaker_a.nunique()} source speakers "
             f"(A), {ok.speaker_b.nunique()} target voices (B). Device: {device}.\n"]
    csv_rows = []

    # Table 1 - language ID
    t1 = []
    for c in LANGS + ["all"]:
        d = asr if c == "all" else asr[asr.lang == c]
        t1.append([NAMES.get(c, "All"), len(d), fmt(100 * d.lid_correct.mean(), 1), fmt(d.confidence.mean(), 3),
                   fmt(100 * (d.confidence < st.MIN_CONFIDENCE).mean(), 1)])
        csv_rows.append(dict(table="language_id", group=c, n=len(d), lid_acc=100 * d.lid_correct.mean(),
                             confidence=d.confidence.mean()))
    conf_m = pd.crosstab(asr.lang.map(NAMES), asr.detected.map(NAMES).fillna("none"))
    parts.append("## Table 1 - Step 1: language identification (all test clips)\n\n"
                 + md_table(["Language", "Clips", "Correct %", "Mean confidence", "Asked to confirm %"], t1)
                 + "\nConfusion (rows = real language, columns = detected):\n\n"
                 + md_table(["real \\ detected"] + list(conf_m.columns), [[i] + list(r) for i, r in conf_m.iterrows()]))

    # Table 2 - speech recognisers
    t2, t2n = [], []
    for name in ASRS:
        rows_out = {"": [ASR_NAMES[name]], "n": [ASR_NAMES[name]]}
        for c in LANGS + ["all"]:
            d = asr if c == "all" else asr[asr.lang == c]
            for tag in ("", "n"):
                cer_v = rate(d[f"cerr{tag}_{name}"], d[f"chars{tag}_{name}"])
                wer_v = rate(d[f"werr{tag}_{name}"], d[f"words{tag}_{name}"])
                rows_out[tag].append(f"{cer_v:.1f} / {wer_v:.1f}")
                csv_rows.append(dict(table="asr_verbatim" if tag == "" else "asr_normalised",
                                     group=f"{name}:{c}", n=len(d), cer=cer_v, wer=wer_v))
        rows_out[""] += [fmt(chrf_of(mt[name], mt["oracle"]), 1), fmt(asr[f"sec_{name}"].mean(), 2)]
        csv_rows.append(dict(table="asr_mt", group=name, n=len(plan), chrf_mt=chrf_of(mt[name], mt["oracle"]),
                             sec_per_clip=asr[f"sec_{name}"].mean()))
        t2.append(rows_out[""])
        t2n.append(rows_out["n"])
    parts.append(f"## Table 2 - Step 2: speech recognisers compared ({len(asr)} test clips)\n\n"
                 "### 2a - vs the verbatim transcript (exactly as spoken, colloquial forms included)\n\n"
                 + md_table(["Recogniser (CER % / WER %)", "Malayalam", "Tamil", "Hindi", "All",
                             f"Translation agreement chrF ↑ ({len(plan)} clips)", "Seconds per clip"], t2)
                 + "\n### 2b - vs the normalised transcript (standard spelling)\n\n"
                 + md_table(["Recogniser (CER % / WER %)", "Malayalam", "Tamil", "Hindi", "All"], t2n)
                 + "\nCER / WER: character / word error rate, lower is better. A recogniser that writes the standard "
                   "spelling of a colloquial word is counted wrong in 2a but right in 2b. chrF: translation of the "
                   "recogniser's text vs translation of the real transcript (0-100, higher = its errors change the "
                   "translation less).\n")

    # Table 3 - the voice
    t3 = [["Real speech: two clips of speaker B (ceiling)", "-", fmt(mean(ok, "sim_real_b"), 3), "-", "-",
           fmt(mean(ok, "dnsmos_reference"), 2)],
          ["Real speech: speaker A vs B (before conversion)", fmt(mean(ok, "sim_real_ab"), 3), "-", "-", "-",
           fmt(mean(ok, "dnsmos_input"), 2)]]
    for kind, label in (("tts", "TTS only (no voice conversion)"), ("mandarin", "TTS + original MeanVC"),
                        ("ours", "TTS + our MeanVC v3 (the pipeline)")):
        t3.append([label, fmt(mean(ok, f"sim_ref_{kind}"), 3), fmt(mean(ok, f"sim_other_{kind}"), 3),
                   fmt(mean(ok, f"sim_src_{kind}"), 3), fmt(cer_out(ok, kind), 1), fmt(mean(ok, f"dnsmos_{kind}"), 2)])
        csv_rows.append(dict(table="voice", group=kind, n=len(ok), sim_ref=mean(ok, f"sim_ref_{kind}"),
                             sim_other=mean(ok, f"sim_other_{kind}"), sim_src=mean(ok, f"sim_src_{kind}"),
                             cer_mms=cer_out(ok, kind), dnsmos=mean(ok, f"dnsmos_{kind}")))
    parts.append("## Table 3 - Steps 4-5: the translated speech in speaker B's voice\n\n"
                 + md_table(["Output", "Similarity to B (reference clip) ↑", "Similarity to B (other clip) ↑",
                             "Similarity to A (leakage) ↓", "CER % (MMS) vs translation ↓", "Quality (DNSMOS) ↑"], t3))

    # Table 4 - per direction
    t4 = []
    groups = [((s, t), ok[(ok.src_lang == s) & (ok.tgt_lang == t)]) for s in LANGS for t in LANGS if s != t]
    for (s, t), d in groups + [(("All", None), ok)]:
        if d.empty:
            continue
        label = "All" if s == "All" else f"{NAMES[s]} -> {NAMES[t]}"
        asr_err = [ev.char_errors(a, b) for a, b in zip(d.text, d.transcript)]
        t4.append([label, len(d), fmt(rate([e for e, _ in asr_err], [n for _, n in asr_err]), 1),
                   fmt(chrf_of(d.translation, d.translation_oracle), 1), fmt(d.sim_ref_tts.mean(), 3),
                   fmt(d.sim_ref_ours.mean(), 3), fmt(d.sim_other_ours.mean(), 3), fmt(d.sim_src_ours.mean(), 3),
                   fmt(cer_out(d, "ours"), 1), fmt(mean(d, "dnsmos_ours"), 2), fmt(d.rtf.mean(), 2)])
        csv_rows.append(dict(table="direction", group=label, n=len(d),
                             chrf_mt=chrf_of(d.translation, d.translation_oracle), sim_ref=d.sim_ref_ours.mean(),
                             sim_other=d.sim_other_ours.mean(), sim_src=d.sim_src_ours.mean(),
                             cer_mms=cer_out(d, "ours"), dnsmos=mean(d, "dnsmos_ours"), rtf=d.rtf.mean()))
    parts.append("## Table 4 - The whole pipeline per direction (IndicConformer + our MeanVC v3)\n\n"
                 + md_table(["Direction", "n", "ASR CER %", "Translation chrF ↑", "Voice B: TTS", "Voice B: ours ↑",
                             "Voice B, other clip ↑", "Leakage of A ↓", "CER % (MMS) ↓", "DNSMOS ↑", "RTF ↓"], t4))

    # Table 5 - time per step
    steps = [c for c in ok.columns if c.startswith("t_")]
    t5 = [[c[2:], fmt(ok[c].mean(), 2), fmt(100 * ok[c].mean() / ok.t_total.mean(), 0)] for c in steps]
    parts.append(f"## Table 5 - Time per step ({device} + CPU for IndicConformer, mean of {len(ok)} runs, "
                 f"input {ok.input_s.mean():.1f} s on average)\n\n" + md_table(["Step", "Seconds", "% of total"], t5))

    # Examples
    ex = ["## Examples (one per direction; all audio in examples/<run>/)\n"]
    for _, d in ok.groupby(["src_lang", "tgt_lang"], sort=False):
        e = d.iloc[0]
        ex.append(f"**{e.run_id}** ({NAMES[e.src_lang]} -> {NAMES[e.tgt_lang]}; voice of B: TTS "
                  f"{e.sim_ref_tts:.2f} -> ours {e.sim_ref_ours:.2f})\n\n"
                  f"- real transcript: {e.text}\n- recognised (IndicConformer): {e.transcript}\n"
                  f"- translation: {e.translation}\n- translation of the real transcript: {e.translation_oracle}\n"
                  f"- MMS hears our output as: {e.mms_ours}\n")
    parts.append("\n".join(ex))
    parts.append(f"Runs: {len(res)} planned, {len(ok)} completed, {len(res) - len(ok)} failed. "
                 f"Total evaluation time {(time.time() - started) / 60:.0f} min.\n")

    report = "\n".join(parts)
    (out_dir / "results_tables.md").write_text(report, encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out_dir / "summary.csv", index=False)
    print("\n" + report)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
