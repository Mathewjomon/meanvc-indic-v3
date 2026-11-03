"""
Stage 6d - Draw the Stage 6 results (speech translation + voice conversion) as PNG figures.
Runs anywhere in seconds (pandas + matplotlib) - no GPU, no models, changes no result.

PURPOSE
  The result tables are in Markdown/CSV. This script turns them into pictures for the report, the slides
  and the guide: one chart per question, plus the main tables as an image.

WHAT IT DRAWS  (all from the CSV files the evaluations already wrote)
  1_voice_conversion_result.png     the translated speech in speaker B's voice: TTS only / original
                                    MeanVC / our MeanVC v3 - similarity to B, intelligibility, quality
  2_speech_recognisers.png          our encoder / MMS-1b-all / IndicConformer: CER per language and how
                                    much their errors change the translation (chrF)
  3_similarity_per_direction.png    similarity to speaker B for the 6 directions, three outputs each
  4_before_vs_after_translation.png our MeanVC v3 before (Stage 5) vs after (Stage 6) translation
  5_results_tables.png              the main tables as one image

  Colours: the first three slots of the dataviz reference palette (blue, orange, aqua), which are
  documented as colour-blind safe for every pair; each system keeps the same colour in every chart.

ROLE IN THE PROJECT
  Reads outputs/stage6_speech_translation_voice_conversion_results/{results,asr_comparison}.csv and
  outputs/results_comparison_before_and_after_translation/comparison.csv. Writes PNGs only.

HOW TO RUN
  python "G:\\My Drive\\meanvc_indic_v3\\code\\translate\\plot_results.py"

OUTPUT  outputs/stage6_speech_translation_voice_conversion_results/figures/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
RES = PROJECT / "outputs" / "stage6_speech_translation_voice_conversion_results"
CMP = PROJECT / "outputs" / "results_comparison_before_and_after_translation" / "comparison.csv"
OUT = RES / "figures"

# palette (dataviz reference instance, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a3"
BLUE_LIGHT = "#86b6ef"                       # sequential blue step 250 (ordinal 'before')
SYSTEM_COLOR = {"tts": NEUTRAL, "mandarin": ORANGE, "ours": BLUE}
SYSTEM_NAME = {"tts": "TTS only (no conversion)", "mandarin": "Original MeanVC", "ours": "Our MeanVC v3"}
ASR_COLOR = {"ours": BLUE, "mms": ORANGE, "indicconformer": AQUA}
ASR_NAME = {"ours": "Our encoder", "mms": "Meta MMS-1b-all", "indicconformer": "AI4Bharat IndicConformer"}
LANGS = {"ml": "Malayalam", "ta": "Tamil", "hi": "Hindi"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def style(ax, grid_axis="y"):
    """Recessive frame: no top/right spines, light grid behind the bars."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def bar_labels(ax, bars, fmt, inside=False):
    """Value on each bar: above it, or inside its top (white) where a reference line runs just above."""
    for b in bars:
        v = b.get_height()
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, -4 if inside else 3),
                    textcoords="offset points", ha="center", va="top" if inside else "bottom", fontsize=9,
                    color="white" if inside else INK, fontweight="bold" if inside else "normal")


def ceiling_handle(value):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=INK2, linestyle="--", linewidth=1, label=f"Two real clips of B: {value:.3f} (best possible)")


def cer(err, n):
    return 100 * err.sum() / max(n.sum(), 1)


def save(fig, name, note=None):
    if note:                                    # below everything (bbox 'tight' keeps it in the image)
        fig.text(0.01, -0.03, note, fontsize=8.5, color=INK2, ha="left", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", OUT / name)


def main():
    r = pd.read_csv(RES / "results.csv")
    a = pd.read_csv(RES / "asr_comparison.csv")
    c = pd.read_csv(CMP)
    n = len(r)
    ceiling = r.sim_real_b.mean()

    # ---------------- 1. the voice conversion result ----------------
    kinds = ["tts", "mandarin", "ours"]
    short = {"tts": "TTS only\n(no conversion)", "mandarin": "Original\nMeanVC", "ours": "Our\nMeanVC v3"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.3)
    panels = [("Sounds like speaker B\n(0-1, higher is better)", [r[f"sim_ref_{k}"].mean() for k in kinds], "{:.3f}"),
              ("Intelligibility: CER % by MMS\n(lower is better)",
               [cer(r[f"err_{k}"], r[f"len_{k}"]) for k in kinds], "{:.1f}"),
              ("Audio quality: DNSMOS\n(1-5, higher is better)", [r[f"dnsmos_{k}"].mean() for k in kinds], "{:.2f}")]
    for i, (ax, (title, vals, fmt)) in enumerate(zip(axes, panels)):
        bars = ax.bar([short[k] for k in kinds], vals, width=0.6,
                      color=[SYSTEM_COLOR[k] for k in kinds], edgecolor=SURFACE, linewidth=2)
        bar_labels(ax, bars, fmt, inside=(i == 0))
        ax.set_title(title)
        style(ax)
    axes[0].axhline(ceiling, color=INK2, linestyle="--", linewidth=1)
    axes[0].legend(handles=[ceiling_handle(ceiling)], loc="upper left", fontsize=8.5)
    axes[0].set_ylim(0, 0.95)
    axes[1].set_ylim(0, 24)
    axes[2].set_ylim(1, 4)
    fig.suptitle("Translated speech converted into speaker B's voice", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", y=1.04)
    save(fig, "1_voice_conversion_result.png",
         f"{n} translations, 48 test speakers never seen in training. The same TTS speech is given to each voice converter.")

    # ---------------- 2. speech recognisers ----------------
    asrs = ["ours", "mms", "indicconformer"]
    groups = list(LANGS) + ["all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    w = 0.26
    for i, k in enumerate(asrs):
        vals = []
        for g in groups:
            d = a if g == "all" else a[a.lang == g]
            vals.append(cer(d[f"cerrn_{k}"], d[f"charsn_{k}"]))
        bars = ax.bar([x + (i - 1) * w for x in range(len(groups))], vals, width=w, color=ASR_COLOR[k],
                      edgecolor=SURFACE, linewidth=2, label=ASR_NAME[k])
        bar_labels(ax, bars, "{:.1f}")
    ax.set_xticks(range(len(groups)), [LANGS.get(g, "All languages") for g in groups])
    ax.set_title("Character error rate % (lower is better), 300 test clips")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 40)
    style(ax)
    # translation agreement per recogniser comes from the evaluation summary
    s = pd.read_csv(RES / "summary.csv")
    mt = s[s.table == "asr_mt"].set_index("group").chrf_mt
    ax = axes[1]
    bars = ax.bar([ASR_NAME[k].replace(" ", "\n", 1) for k in asrs], [mt[k] for k in asrs], width=0.6,
                  color=[ASR_COLOR[k] for k in asrs], edgecolor=SURFACE, linewidth=2)
    bar_labels(ax, bars, "{:.1f}")
    ax.set_title("Translation agreement, chrF (higher is better)")
    ax.set_ylim(0, 80)
    style(ax)
    save(fig, "2_speech_recognisers.png",
         "CER against the normalised transcript (standard spelling). chrF: translation of the recognised text vs "
         f"translation of the real transcript, {n} clips.")

    # ---------------- 3. similarity per direction ----------------
    dirs = [(s_, t_) for s_ in LANGS for t_ in LANGS if s_ != t_]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, k in enumerate(kinds):
        vals = [r[(r.src_lang == s_) & (r.tgt_lang == t_)][f"sim_ref_{k}"].mean() for s_, t_ in dirs]
        bars = ax.bar([x + (i - 1) * w for x in range(len(dirs))], vals, width=w, color=SYSTEM_COLOR[k],
                      edgecolor=SURFACE, linewidth=2, label=SYSTEM_NAME[k])
        if k == "ours":
            bar_labels(ax, bars, "{:.2f}", inside=True)
    ax.axhline(ceiling, color=INK2, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(dirs)), [f"{LANGS[s_]} speech\n→ {LANGS[t_]} voice" for s_, t_ in dirs])
    ax.set_ylim(0, 0.9)
    ax.set_title("Sounds like speaker B (0-1, higher is better), per direction")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [ceiling_handle(ceiling)], loc="upper left", fontsize=9, ncol=4,
              bbox_to_anchor=(0, -0.15))
    style(ax)
    save(fig, "3_similarity_per_direction.png", None)

    # ---------------- 4. before vs after translation ----------------
    ours = c[(c.system == "ours") & (c.direction == "all2all")].set_index("stage")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.3)
    panels = [("sim_target", "Sounds like speaker B\n(higher is better)", "{:.3f}", (0, 0.9)),
              ("sim_source", "Still sounds like speaker A: leakage\n(lower is better)", "{:.3f}", (0, 0.4)),
              ("cer_added", "Clarity lost by the conversion,\nCER % points (lower is better)", "{:.1f}", (0, 12))]
    for ax, (col, title, fmt, ylim) in zip(axes, panels):
        vals = [ours.loc["before", col], ours.loc["after", col]]
        bars = ax.bar(["Before translation\n(Stage 5)", "After translation\n(Stage 6)"], vals, width=0.55,
                      color=[BLUE_LIGHT, BLUE], edgecolor=SURFACE, linewidth=2)
        bar_labels(ax, bars, fmt, inside=(col == "sim_source"))
        ax.set_title(title)
        ax.set_ylim(*ylim)
        style(ax)
    for x, stage in enumerate(("before", "after")):          # natural level: two different people
        axes[1].plot([x - 0.3, x + 0.3], [ours.loc[stage, "sim_ab"]] * 2, color=INK2, linestyle="--", linewidth=1)
    axes[1].annotate("dashed: two different\npeople (natural level)", (1.3, ours.loc["after", "sim_ab"]),
                     xytext=(0, 14), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    fig.suptitle("Our MeanVC v3: voice conversion before vs after translation", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", y=1.06)
    save(fig, "4_before_vs_after_translation.png",
         f"Cross-language pairs, same 48 test speakers: {int(ours.loc['before', 'n'])} conversions before, "
         f"{int(ours.loc['after', 'n'])} after.")

    # ---------------- 5. the main tables as an image ----------------
    asr_rows = [[ASR_NAME[k]] + [f"{cer(d[f'cerrn_{k}'], d[f'charsn_{k}']):.1f}"
                                 for d in [a[a.lang == g] for g in LANGS] + [a]] + [f"{mt[k]:.1f}"] for k in asrs]
    voice_rows = [["Two real clips of B (ceiling)", "-", f"{r.sim_real_b.mean():.3f}", "-", "-",
                   f"{r.dnsmos_reference.mean():.2f}"]]
    for k in kinds:
        voice_rows.append([SYSTEM_NAME[k], f"{r[f'sim_ref_{k}'].mean():.3f}", f"{r[f'sim_other_{k}'].mean():.3f}",
                           f"{r[f'sim_src_{k}'].mean():.3f}", f"{cer(r[f'err_{k}'], r[f'len_{k}']):.1f}",
                           f"{r[f'dnsmos_{k}'].mean():.2f}"])
    dir_rows = []
    for s_, t_ in dirs:
        d = r[(r.src_lang == s_) & (r.tgt_lang == t_)]
        dir_rows.append([f"{LANGS[s_]} → {LANGS[t_]}", f"{d.sim_ref_tts.mean():.3f}", f"{d.sim_ref_mandarin.mean():.3f}",
                         f"{d.sim_ref_ours.mean():.3f}", f"{cer(d.err_ours, d.len_ours):.1f}", f"{d.dnsmos_ours.mean():.2f}"])
    dir_rows.append(["All directions", f"{r.sim_ref_tts.mean():.3f}", f"{r.sim_ref_mandarin.mean():.3f}",
                     f"{r.sim_ref_ours.mean():.3f}", f"{cer(r.err_ours, r.len_ours):.1f}", f"{r.dnsmos_ours.mean():.2f}"])
    tables = [
        ("A. Speech recognition: CER % (standard spelling) and translation agreement",
         ["Recogniser", "Malayalam", "Tamil", "Hindi", "All", "chrF ↑"], asr_rows, [len(asr_rows) - 1]),
        ("B. The translated speech in speaker B's voice",
         ["Output", "Like B ↑", "Like B, unseen clip ↑", "Like A (leakage) ↓", "CER % (MMS) ↓", "DNSMOS ↑"],
         voice_rows, [len(voice_rows) - 1]),
        ("C. Per direction: similarity to speaker B, and our output's CER and DNSMOS",
         ["Direction", "TTS only", "Original MeanVC", "Our MeanVC v3 ↑", "Our CER % ↓", "Our DNSMOS ↑"],
         dir_rows, [len(dir_rows) - 1]),
    ]
    heights = [len(t[2]) + 2.2 for t in tables]
    fig, axes = plt.subplots(len(tables), 1, figsize=(12, 0.34 * sum(heights) + 1),
                             gridspec_kw={"height_ratios": heights})
    for ax, (title, header, rows, bold_rows) in zip(axes, tables):
        ax.axis("off")
        ax.set_title(title, pad=4)
        tb = ax.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="center", colLoc="center",
                      colWidths=[0.30] + [0.70 / (len(header) - 1)] * (len(header) - 1))
        tb.auto_set_font_size(False)
        tb.set_fontsize(9.5)
        tb.scale(1, 1.45)
        for (row, col), cell_ in tb.get_celld().items():
            cell_.set_edgecolor(GRID)
            cell_.set_facecolor(SURFACE)
            if col == 0:
                cell_.set_text_props(ha="left")
                cell_.PAD = 0.03
            if row == 0:
                cell_.set_text_props(weight="bold", color=INK)
                cell_.set_facecolor("#f1f0ec")
            elif row - 1 in bold_rows:
                cell_.set_text_props(weight="bold")
    fig.suptitle("Speech translation + voice conversion: results (Stage 6)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    save(fig, "5_results_tables.png",
         f"48 test speakers never seen in training; {n} translations, 6 directions. Similarity = WavLM-ECAPA cosine.")

    # ---------------- 6. one table: every metric, the test data it used, and our system's value ----------------
    after = c[(c.system == "ours") & (c.stage == "after") & (c.direction == "all2all")].iloc[0]
    k = "indicconformer"
    per_lang = a.groupby("lang").size()
    clips = f"{len(a)} clips"
    trans = f"{n} translations"
    metric_rows = [
        ["Understanding", "Language ID accuracy", "Language detected correctly", clips, "Higher",
         f"{100 * a.lid_correct.mean():.1f} %"],
        ["", "CER (character error rate)", "Letters the speech recogniser (IndicConformer) gets wrong", clips, "Lower",
         f"{cer(a[f'cerrn_{k}'], a[f'charsn_{k}']):.1f} %"],
        ["", "WER (word error rate)", "Words the speech recogniser gets wrong", clips, "Lower",
         f"{100 * a[f'werrn_{k}'].sum() / a[f'wordsn_{k}'].sum():.1f} %"],
        ["Translation", "chrF (translation agreement)", "Our translation vs translation of the correct transcript (0-100)",
         f"{n} clips", "Higher", f"{mt[k]:.1f}"],
        ["Voice", "Speaker similarity to B", "Output vs speaker B's reference clip (0-1)", trans, "Higher",
         f"{r.sim_ref_ours.mean():.3f}"],
        ["", "Similarity on B's unseen clip", "Output vs a clip of B the model never received (0-1)", trans, "Higher",
         f"{r.sim_other_ours.mean():.3f}"],
        ["", "Leakage of speaker A", "How much the output still sounds like the source speaker (0-1)", trans, "Lower",
         f"{r.sim_src_ours.mean():.3f}"],
        ["Clarity", "Intelligibility CER", "Letters an independent recogniser (MMS) gets wrong in the output", trans,
         "Lower", f"{cer(r.err_ours, r.len_ours):.1f} %"],
        ["", "Clarity lost by conversion", "Intelligibility CER of the output minus that of the TTS input", trans,
         "Lower", f"{after.cer_added:.1f} points"],
        ["Quality, speed", "DNSMOS", "Speech quality predicted by Microsoft DNSMOS (1-5)", trans, "Higher",
         f"{r.dnsmos_ours.mean():.2f}"],
        ["", "RTF (real-time factor)", "Processing time / speech length (laptop RTX 2050)", trans, "Lower",
         f"{r.rtf.mean():.2f}"],
    ]
    description = (
        "Test data: the official test split of IndicVoices-R, from speakers never used in training.\n"
        f"• {len(a)} test clips ({', '.join(f'{per_lang[c_]} {LANGS[c_]}' for c_ in LANGS)}) from "
        f"{r.speaker_a.nunique()} speakers: language ID and speech recognition.\n"
        f"• {n} translations: {n // 6} per direction × 6 directions (Malayalam, Tamil, Hindi ↔ each other). "
        f"Each speaker is used once as the source (A) and once as the target voice (B), never with themselves.\n"
        "Every value is the average (or the total error rate) over all of these clips, not a single recording.")
    fig = plt.figure(figsize=(14, 0.42 * (len(metric_rows) + 1) + 2.0))
    fig.text(0.0, 0.985, "Speech translation + voice conversion: evaluation metrics and our results",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.0, 0.93, description, fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.77])
    ax.axis("off")
    tb = ax.table(cellText=metric_rows,
                  colLabels=["Group", "Metric", "What it measures", "Tested on", "Better", "Our result"],
                  cellLoc="left", colLoc="left", colWidths=[0.10, 0.19, 0.40, 0.12, 0.08, 0.11], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    for (row, col), cell_ in tb.get_celld().items():
        cell_.set_edgecolor(GRID)
        cell_.set_facecolor(SURFACE)
        cell_.PAD = 0.04
        if row == 0:
            cell_.set_text_props(weight="bold")
            cell_.set_facecolor("#f1f0ec")
        elif col == 0:
            cell_.set_text_props(weight="bold", color=INK2)
        elif col == 5:
            cell_.set_text_props(weight="bold", color=INK)
        if col in (3, 4, 5):
            cell_.set_text_props(ha="center")
    save(fig, "6_metrics_summary.png",
         "Our system: language ID (our encoder) + IndicConformer + IndicTrans2 + MMS-TTS + our MeanVC v3.")


if __name__ == "__main__":
    main()
