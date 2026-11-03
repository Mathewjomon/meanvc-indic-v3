"""
Draws the system architecture of MeanVC-Indic v3 as implemented (speech translation + voice conversion).

PURPOSE
  docs/system_architecture.png is the hand-made diagram. This script draws the version that matches
  the code exactly: language identification before the decision, speaker separation marked as planned
  (not implemented), the speaker embedding going to both the timbre encoder and the DiT, and the
  same-language path labelled as the separate convert.py tool.

HOW TO RUN
  python "G:\\My Drive\\meanvc_indic_v3\\docs\\draw_architecture.py"

OUTPUT  docs/system_architecture_implemented.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "system_architecture_implemented.png"
INK, INK2, GREY, WHITE = "#111111", "#555555", "#e3e3e3", "#ffffff"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
fig, ax = plt.subplots(figsize=(22, 8.6))
ax.set_xlim(0, 170)
ax.set_ylim(-1, 46)
ax.axis("off")


def box(x, y, title, sub, fill=WHITE, dashed=False, w=16, h=6):
    """Box centred at (x, y): bold title, second line below."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill, edgecolor=INK,
                           linewidth=1.3, linestyle="--" if dashed else "-"))
    ax.text(x, y + 0.9, title, ha="center", va="center", fontweight="bold", fontsize=11.5)
    ax.text(x, y - 1.1, sub, ha="center", va="center", fontsize=10.5)


def group(x0, y0, x1, y1, label):
    """Dashed group frame; title in the top-right corner (no arrows enter there)."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, linewidth=1.1,
                           linestyle=(0, (4, 3))))
    ax.text(x1 - 1, y1 - 1.3, label, fontweight="bold", fontsize=12.5, va="center", ha="right")


def label(x, y, text, ha="center", va="bottom"):
    ax.text(x, y, text, ha=ha, va=va, style="italic", fontsize=10.5, color=INK2, linespacing=1.4)


def arrow(points):
    """Poly-line through points, arrow head at the end."""
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=INK, linewidth=1.2)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=16, shrinkA=0, shrinkB=0))


# columns: front-end and back-end boxes share x positions
C1, C2, C3, C4 = 88, 110, 132, 155
Y1, Y2, Y3 = 33, 17, 5                      # top row, back-end row, bottom row

group(78, 28.3, 143, 39.2, "Cross-lingual front-end (pretrained)")
group(76, 0.0, 165, 25.4, "Voice conversion back-end (MeanVC v3)")

# ---------------- top row ----------------
box(9, Y1, "Source speech", "(speaker A)", w=15)
box(28, Y1, "Speaker separation", "(planned, not implemented)", dashed=True, w=17)
box(48, Y1, "Language identification", "(our encoder, script voting)", fill=GREY, w=18)
ax.add_patch(Polygon([[67, Y1 + 4.2], [72.5, Y1], [67, Y1 - 4.2], [61.5, Y1]], closed=True,
                     facecolor=WHITE, edgecolor=INK, linewidth=1.3))
ax.text(67, Y1 + 0.8, "Same", ha="center", va="center", fontweight="bold", fontsize=11)
ax.text(67, Y1 - 1.0, "language?", ha="center", va="center", fontsize=10.5)
box(C1, Y1, "Speech recognition", "IndicConformer (text A)")
box(C2, Y1, "Machine translation", "IndicTrans2 (A → B)")
box(C3, Y1, "Text-to-speech", "MMS-TTS (speech B)")

arrow([(16.5, Y1), (19.5, Y1)])
arrow([(36.5, Y1), (39, Y1)])
arrow([(57, Y1), (61.5, Y1)])
arrow([(72.5, Y1), (80, Y1)])
label(76, Y1 + 0.4, "no")
arrow([(96, Y1), (102, Y1)])
arrow([(118, Y1), (124, Y1)])
arrow([(48, Y1 + 3), (48, 42), (86, 42), (86, Y1 + 3)])               # language code to the recogniser
label(67, 42.4, "language of A (ml / ta / hi)")
label(C2, 43.6, "target language B, chosen by the user")
arrow([(C2, 43.4), (C2, Y1 + 3)])

# ---------------- back-end row ----------------
box(C1, Y2, "Streaming content", "encoder (Conformer)", fill=GREY)
box(C2, Y2, "Timbre encoder", "(cross-attention)", fill=GREY)
box(C3, Y2, "Diffusion transformer", "decoder (mean flows)", fill=GREY)
box(C4, Y2, "Neural vocoder", "(Vocos)", w=14)
arrow([(96, Y2), (102, Y2)])
label(99, Y2 + 0.4, "BNF")
arrow([(118, Y2), (124, Y2)])
label(121, Y2 + 0.4, "timbre\nBNF")
arrow([(140, Y2), (148, Y2)])
label(144, Y2 + 0.4, "mel")

arrow([(C3, Y1 - 3), (C3, 26.8), (C1, 26.8), (C1, Y2 + 3)])             # TTS speech -> content encoder
label(C3 + 1, 27.6, "TTS speech (language B)", ha="left", va="center")
arrow([(67, Y1 - 4.2), (67, Y2), (80, Y2)])                              # same language: A's own speech
label(66.2, 25.5, "yes: A's own speech\n(same-language tool,\ncode/infer/convert.py)", ha="right", va="top")

# ---------------- bottom row ----------------
box(9, Y3, "Reference speech", "(speaker B, any language)", w=16)
box(121, Y3, "Speaker encoder", "(WavLM + ECAPA-TDNN)", w=18)
box(C4, Y3, "Converted speech", "(B's voice, language B)", w=16)
arrow([(17, Y3), (112, Y3)])
label(64, Y3 + 0.4, "waveform")
arrow([(9, Y3 + 3), (9, 10.5), (106, 10.5), (106, Y2 - 3)])            # reference mel -> timbre encoder
label(57, 10.9, "reference mel (80 × 10 ms)")
arrow([(115, Y3 + 3), (115, Y2 - 3)])                                    # embedding -> timbre encoder
label(114.4, 9.4, "speaker\nembedding", ha="right", va="center")
arrow([(127, Y3 + 3), (127, 9.5), (C3, 9.5), (C3, Y2 - 3)])            # embedding -> DiT
label(C3 + 0.8, 11.5, "speaker embedding", ha="left", va="center")
arrow([(C4, Y2 - 3), (C4, Y3 + 3)])
label(C4 + 0.8, 10.5, "waveform", ha="left", va="center")

# ---------------- legend ----------------
lx, ly = 2, 23.5
ax.text(lx, ly, "Legend", fontweight="bold", fontsize=12)
for i, (fill, dashed, text) in enumerate([(GREY, False, "Fine-tuned by us (Malayalam, Tamil, Hindi)"),
                                          (WHITE, False, "Pretrained, frozen"),
                                          (WHITE, True, "Planned, not implemented yet")]):
    y = ly - 3.2 - i * 3.0
    ax.add_patch(Rectangle((lx, y - 0.9), 3.6, 1.9, facecolor=fill, edgecolor=INK, linewidth=1.1,
                           linestyle="--" if dashed else "-"))
    ax.text(lx + 5, y, text, va="center", fontsize=11)

ax.text(0.5, 45.5, "MeanVC-Indic v3: speech translation + voice conversion (as implemented)",
        fontweight="bold", fontsize=15, va="top")
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
