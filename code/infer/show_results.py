"""
Stage 5c - Turn the evaluation results into clean tables (for VS Code, the thesis and slides).
Runs anywhere in seconds (needs only pandas + matplotlib) - no GPU, no models.

READS   outputs/eval_v3/summary.csv          (written by evaluate.py)
WRITES  outputs/eval_v3/results_table.md     open in VS Code, press Ctrl+Shift+V to see the tables
        outputs/eval_v3/results_table.png    the same tables as a picture (for slides / the report)
        and prints them in the terminal.

HOW TO RUN (laptop, VS Code terminal with the venv active)
  python "G:\\My Drive\\meanvc_indic_v3\\code\\infer\\show_results.py"
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
LANG = {"ml": "Malayalam", "ta": "Tamil", "hi": "Hindi"}
SYSTEM_NAMES = {
    "real speech": "Real speech (reference point)",
    "mandarin": "Original MeanVC (Mandarin)",
    "enc_only": "Our encoder + Mandarin decoder",
    "ours": "Ours (encoder + decoder fine-tuned)",
}
# column -> (header, number format)
COLUMNS = {
    "sim_target": ("Speaker similarity ↑", "{:.3f}"),
    "sim_other": ("Similarity, other clip ↑", "{:.3f}"),
    "sim_source": ("Source leakage ↓", "{:.3f}"),
    "cer_mms": ("CER % (MMS) ↓", "{:.1f}"),
    "cer_ours": ("CER % (our ASR) ↓", "{:.1f}"),
    "dnsmos": ("Quality (DNSMOS) ↑", "{:.2f}"),
    "rtf": ("RTF ↓", "{:.2f}"),
}
NOTES = [
    "Speaker similarity: cosine of WavLM-ECAPA speaker embeddings, converted speech vs the target "
    "speaker's reference clip (real-speech row: two real clips of the same speaker = the ceiling).",
    "Similarity, other clip: vs a different clip of the target speaker (the stricter measure).",
    "Source leakage: similarity to the source speaker (real-speech row: the two speakers before conversion).",
    "CER: character error rate when a recogniser reads the speech; MMS = Meta MMS-1b-all (independent), "
    "our ASR = our fine-tuned encoder's CTC head. Real-speech row = the original source recordings.",
    "DNSMOS: Microsoft DNSMOS overall quality, 1-5. RTF: processing time / audio length (A100 GPU).",
    "Test set: 48 IndicVoices-R speakers never seen in training; 108 conversions per system "
    "(12 per language pair x 9 pairs).",
]


def fmt_table(df, first_col):
    """DataFrame (index = row names, numeric columns) -> list of rows of strings."""
    cols = [c for c in COLUMNS if c in df.columns and df[c].notna().any()]
    header = [first_col] + [COLUMNS[c][0] for c in cols]
    rows = []
    for name, r in df.iterrows():
        rows.append([str(name)] + ["-" if pd.isna(r[c]) else COLUMNS[c][1].format(r[c]) for c in cols])
    return header, rows


def markdown(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def text_table(title, header, rows):
    widths = [max(len(x) for x in col) for col in zip(header, *rows)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([title, line(header), "-" * (sum(widths) + 2 * len(widths)), *map(line, rows)]) + "\n"


def grid(per_pair, column, fmt):
    """3 x 3 grid: rows = source language, columns = target voice language."""
    g = pd.DataFrame(index=[LANG[c] for c in LANG], columns=[LANG[c] for c in LANG], dtype=object)
    for name, r in per_pair.iterrows():
        s, t = [x.strip() for x in str(name).split("->")]
        g.loc[LANG[s], LANG[t]] = fmt.format(r[column])
    header = ["source \\ target voice"] + list(g.columns)
    return header, [[idx] + list(row) for idx, row in g.iterrows()]


def main():
    ap = argparse.ArgumentParser(description="Make clean result tables from evaluate.py output")
    ap.add_argument("--dir", default=str(PROJECT / "outputs" / "eval_v3"))
    args = ap.parse_args()
    out = Path(args.dir)
    s = pd.read_csv(out / "summary.csv", index_col=0)

    tables = []                                    # (title, header, rows)
    main_t = s[s.group == "All conversions"].rename(index=SYSTEM_NAMES)
    tables.append(("Table 1 - All conversions (same- and cross-language)", *fmt_table(main_t, "System")))

    both = []
    for group, label in (("Same language (e.g. ml -> ml)", "same language"),
                         ("Cross-language (e.g. ml -> ta)", "cross-language")):
        part = s[(s.group == group) & s.index.isin(["mandarin", "ours"])].copy()
        part.index = [f"{SYSTEM_NAMES[i].split(' (')[0]}, {label}" for i in part.index]
        both.append(part)
    tables.append(("Table 2 - Same-language vs cross-language conversion",
                   *fmt_table(pd.concat(both), "System, conversion")))

    per_pair = s[s.group == "ours per pair"]
    if len(per_pair):
        tables.append(("Table 3 - Ours per language pair", *fmt_table(
            per_pair.rename(index=lambda n: " -> ".join(LANG[x.strip()] for x in n.split("->"))),
            "Source -> target voice")))
        tables.append(("Table 4 - Ours: speaker similarity by language pair",
                       *grid(per_pair, "sim_target", "{:.3f}")))
        cer_col = "cer_mms" if "cer_mms" in per_pair and per_pair.cer_mms.notna().any() else "cer_ours"
        tables.append((f"Table 5 - Ours: CER % ({'MMS' if cer_col == 'cer_mms' else 'our ASR'}) "
                       "by language pair", *grid(per_pair, cer_col, "{:.1f}")))

    # ---- markdown (VS Code preview) + terminal ----
    md = "# Voice conversion results (v3)\n\n" + "\n".join(markdown(*t) for t in tables)
    md += "\n**Notes**\n\n" + "\n".join(f"- {n}" for n in NOTES) + "\n"
    (out / "results_table.md").write_text(md, encoding="utf-8")
    print("\n".join(text_table(*t) for t in tables))

    # ---- picture ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    heights = [0.36 * (len(t[2]) + 2) for t in tables]          # +2: the header has two lines
    fig, axes = plt.subplots(len(tables), 1, figsize=(16, sum(heights) + 0.6 * len(tables)),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.45})
    for ax, (title, header, rows) in zip(axes, tables):
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        first = 0.25                                       # wide first column for the names
        widths = [first] + [(1 - first) / (len(header) - 1)] * (len(header) - 1)
        labels = [header[0]] + ["\n".join(textwrap.wrap(h, 14)) for h in header[1:]]
        tb = ax.table(cellText=rows, colLabels=labels, cellLoc="center", colWidths=widths,
                      bbox=[0, 0, 1, 1])
        tb.auto_set_font_size(False)
        tb.set_fontsize(10)
        for (r, c), cell in tb.get_celld().items():
            if r == 0:
                cell.set_height(cell.get_height() * 2)       # room for two-line headers
                cell.set_facecolor("#e8eef7")
                cell.set_text_props(fontweight="bold")
            elif "Ours" in rows[r - 1][0] and not title.startswith("Table 3"):
                cell.set_facecolor("#eaf6ea")             # highlight our system
            if c == 0:
                cell.set_text_props(ha="left")
    fig.savefig(out / "results_table.png", dpi=160, bbox_inches="tight")
    print(f"Saved: {out / 'results_table.md'}  (open it, press Ctrl+Shift+V)")
    print(f"       {out / 'results_table.png'}")


if __name__ == "__main__":
    main()