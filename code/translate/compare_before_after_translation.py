"""
Stage 6c - Results comparison: voice conversion BEFORE translation (Stage 5) vs AFTER translation (Stage 6).
Runs anywhere in seconds (pandas only) - no GPU, no models.

PURPOSE
  The same voice converter (our MeanVC v3, and the original MeanVC as baseline) was tested twice on the
  same 48 IndicVoices-R test speakers:
    BEFORE translation  Stage 5, outputs/eval_v3/
                        speaker A's REAL speech  ->  speaker B's voice   (A's words, A's language)
    AFTER translation   Stage 6, outputs/stage6_speech_translation_voice_conversion_results/
                        speaker A's speech -> text -> translation -> TTS -> speaker B's voice
                        (A's words, TRANSLATED into B's language)
  This script puts the two side by side, so the thesis can show what translation changes for the voice
  conversion: voice similarity, how much of A's voice is left, clarity, quality.

WHAT IT COMPARES (cross-language only: Stage 5 also has same-language pairs, Stage 6 never does)
  sim_target   similarity of the output to B's reference clip                      higher = better
  sim_other    ... to B's other clip (never given to the model)                    higher = better
  sim_source   similarity to speaker A (A's voice left in the output), next to the
               similarity of A and B themselves (two different people)             lower = better
  cer_added    character errors MMS-1b-all makes on the output MINUS those it makes
               on the input of the voice converter (real speech in Stage 5, TTS in
               Stage 6) = the clarity lost by the conversion. The raw CERs cannot be
               compared: Stage 5 is read against spontaneous speech, Stage 6 against
               clean TTS                                                           lower = better
  dnsmos       speech quality 1-5                                                  higher = better
  Speed is not compared: Stage 5 ran on a Colab A100 (voice conversion only), Stage 6 on the laptop
  (the whole pipeline).

ROLE IN THE PROJECT
  Reads only the two results.csv files written by code/infer/evaluate.py (Stage 5) and
  code/translate/evaluate_translation.py (Stage 6). Changes nothing.

HOW TO RUN
  python "G:\\My Drive\\meanvc_indic_v3\\code\\translate\\compare_before_after_translation.py"

OUTPUT  outputs/results_comparison_before_and_after_translation/
          comparison_tables.md   the tables (open in VS Code, Ctrl+Shift+V)
          comparison.csv         the same numbers
          (README.md in that folder explains the comparison for readers)
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
STAGE5 = PROJECT / "outputs" / "eval_v3" / "results.csv"
STAGE6 = PROJECT / "outputs" / "stage6_speech_translation_voice_conversion_results" / "results.csv"
OUT = PROJECT / "outputs" / "results_comparison_before_and_after_translation"
NAMES = {"ml": "Malayalam", "ta": "Tamil", "hi": "Hindi"}
SYSTEMS = {"ours": "Our MeanVC v3", "mandarin": "Original MeanVC"}


def cer(err, length):
    return 100 * err.sum() / max(length.sum(), 1)


def stage5(df, real, system):
    """Stage 5 numbers for one system on a set of pairs (real = the 'real speech' rows of the same pairs)."""
    return dict(n=len(df), sim_target=df.sim_target.mean(), sim_other=df.sim_other.mean(),
                sim_source=df.sim_source.mean(), sim_ab=real.sim_source.mean(),
                cer_input=cer(real.err_mms, real.len_mms), cer_output=cer(df.err_mms, df.len_mms),
                dnsmos=df.dnsmos.mean())


def stage6(df, system):
    return dict(n=len(df), sim_target=df[f"sim_ref_{system}"].mean(), sim_other=df[f"sim_other_{system}"].mean(),
                sim_source=df[f"sim_src_{system}"].mean(), sim_ab=df.sim_real_ab.mean(),
                cer_input=cer(df.err_tts, df.len_tts), cer_output=cer(df[f"err_{system}"], df[f"len_{system}"]),
                dnsmos=df[f"dnsmos_{system}"].mean())


def md_table(header, rows, center=True):
    lines = ["| " + " | ".join(header) + " |", "|---" + ("|:---:" if center else "|---") * (len(header) - 1) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def main():
    s5 = pd.read_csv(STAGE5)
    s6 = pd.read_csv(STAGE6)
    s5 = s5[s5.src_lang != s5.tgt_lang]                       # cross-language pairs only
    real5 = s5[s5.system == "real speech"]
    directions = [(s, t) for s in NAMES for t in NAMES if s != t]

    rows = []
    for system in SYSTEMS:
        sys5 = s5[s5.system == system]
        for d in directions + [("all", "all")]:
            if d == ("all", "all"):
                a, r, b = sys5, real5, s6
            else:
                pick5 = lambda x: x[(x.src_lang == d[0]) & (x.tgt_lang == d[1])]  # noqa: E731
                a, r, b = pick5(sys5), pick5(real5), s6[(s6.src_lang == d[0]) & (s6.tgt_lang == d[1])]
            for stage, vals in (("before", stage5(a, r, system)), ("after", stage6(b, system))):
                vals["cer_added"] = vals["cer_output"] - vals["cer_input"]
                rows.append(dict(system=system, stage=stage, direction=f"{d[0]}2{d[1]}", **vals))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "comparison.csv", index=False)

    def get(system, stage, direction="all2all"):
        return df[(df.system == system) & (df.stage == stage) & (df.direction == direction)].iloc[0]

    def direction_name(d):
        s, t = d.split("2")
        return "**All directions**" if s == "all" else f"{NAMES[s]} speech → {NAMES[t]} voice"

    def verdict(measure, before, after):
        _, _, higher_better, digits, same = MEASURES[measure]
        change = after - before
        if abs(change) < same:
            word = "About the same"
        else:
            word = "**Better**" if (change > 0) == higher_better else "Worse"
        return f"{change:+.{digits}f}", word

    def cell(measure, v):
        return f"{v:.{MEASURES[measure][3]}f}"

    parts = ["# Results comparison: before translation vs after translation\n\n"
             "**Before translation** (Stage 5): speaker A's real speech is converted into speaker B's voice. "
             "The words stay in A's language.\n\n"
             "**After translation** (Stage 6): speaker A's speech is recognised, translated into B's language, "
             "spoken by TTS, then converted into speaker B's voice.\n\n"
             "Same 48 test speakers (never seen in training) in both tests. Only cross-language pairs: "
             f"{int(get('ours', 'before').n)} conversions before translation, "
             f"{int(get('ours', 'after').n)} after.\n"]

    # 1 + 2: one table per system, one row per measure
    for k, (system, label) in enumerate(SYSTEMS.items(), 1):
        b, a = get(system, "before"), get(system, "after")
        rows = []
        for m, (name, _, _, _, _) in MEASURES.items():
            change, word = verdict(m, b[m], a[m])
            rows.append([name, cell(m, b[m]), cell(m, a[m]), change, word])
        parts.append(f"## {k}. {label}: before vs after translation\n\n"
                     + md_table(["Measure", "Before translation", "After translation", "Change", "Result"], rows))

    # reference points
    b, a = get("ours", "before"), get("ours", "after")
    real5 = s5[s5.system == "real speech"]
    parts.append("## 3. Reference points (real recordings, no conversion)\n\n" + md_table(
        ["Reference", "Before translation", "After translation"],
        [["Two real clips of speaker B: the best similarity possible",
          f"{real5.sim_target.mean():.3f}", f"{s6.sim_real_b.mean():.3f}"],
         ["Speaker A vs speaker B: the normal similarity of two different people",
          f"{b.sim_ab:.3f}", f"{a.sim_ab:.3f}"],
         ["Letters misheard in the voice converter's input (%)",
          f"{b.cer_input:.1f} (real speech)", f"{a.cer_input:.1f} (TTS speech)"]])
        + "\nA leakage value at the 'two different people' level means none of speaker A's voice is left.\n")

    # 4: our model per direction, one table per measure
    parts.append("## 4. Our MeanVC v3 per direction\n")
    for m in ("sim_target", "sim_source", "cer_added"):
        rows = []
        for d in [f"{s}2{t}" for s, t in directions] + ["all2all"]:
            bv, av = get("ours", "before", d)[m], get("ours", "after", d)[m]
            change, word = verdict(m, bv, av)
            rows.append([direction_name(d), cell(m, bv), cell(m, av), change, word])
        better = "higher" if MEASURES[m][2] else "lower"
        parts.append(f"### {MEASURES[m][0]}: {better} is better\n\n"
                     + md_table(["Direction", "Before translation", "After translation", "Change", "Result"], rows))

    # 5: our model vs the original
    rows = []
    for stage in ("before", "after"):
        o, g = get("ours", stage), get("mandarin", stage)
        rows.append([f"{stage.capitalize()} translation", f"{g.sim_target:.3f}", f"{o.sim_target:.3f}",
                     f"{o.sim_target - g.sim_target:+.3f}"])
    per_dir = []
    for d in [f"{s}2{t}" for s, t in directions] + ["all2all"]:
        gains = [get("ours", st, d).sim_target - get("mandarin", st, d).sim_target for st in ("before", "after")]
        per_dir.append([direction_name(d)] + [f"{x:+.3f}" for x in gains])
    parts.append("## 5. Our MeanVC v3 vs the original MeanVC (how much it sounds like speaker B)\n\n"
                 + md_table(["Test", "Original MeanVC", "Our MeanVC v3", "Our gain"], rows)
                 + "\nOur gain per direction:\n\n"
                 + md_table(["Direction", "Gain before translation", "Gain after translation"], per_dir))

    parts.append("## How to read the measures\n\n" + md_table(
        ["Measure", "What it means"],
        [[MEASURES[m][0], MEASURES[m][1]] for m in MEASURES], center=False)
        + "\nAll numbers for every direction and both systems are in `comparison.csv`.\n")

    report = "\n".join(parts)
    (OUT / "comparison_tables.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {OUT}")


# measure -> (row name, explanation, higher is better?, decimals, smaller change = "about the same")
MEASURES = {
    "sim_target": ("Sounds like speaker B (0-1)",
                   "Voice similarity between the output and speaker B's reference clip (higher is better)",
                   True, 3, 0.015),
    "sim_other": ("Sounds like speaker B, on a clip the model never heard (0-1)",
                  "The same, against another recording of B that was not given to the model (higher is better)",
                  True, 3, 0.015),
    "sim_source": ("Still sounds like speaker A: leakage (0-1)",
                   "Voice similarity between the output and speaker A (lower is better)",
                   False, 3, 0.015),
    "cer_added": ("Clarity lost by the conversion (% points)",
                  "Extra % of letters an independent recogniser (MMS) gets wrong after conversion, compared "
                  "with the converter's input (lower is better)",
                  False, 1, 0.5),
    "dnsmos": ("Audio quality, DNSMOS (1-5)",
               "Microsoft DNSMOS overall speech quality (higher is better)",
               True, 2, 0.05),
}


if __name__ == "__main__":
    main()
