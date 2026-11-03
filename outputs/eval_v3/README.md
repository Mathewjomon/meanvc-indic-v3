# eval_v3: voice conversion evaluation results (Stage 5)

This folder holds the **main evaluation of the voice conversion model** (MeanVC-Indic v3) on
Malayalam, Tamil and Hindi. Everything here was produced on **25 Oct 2026** by
[`code/infer/evaluate.py`](../../code/infer/evaluate.py). The tables were then reformatted by
[`code/infer/show_results.py`](../../code/infer/show_results.py).

The test is **voice conversion only, no translation**: a sentence spoken by a real person
(speaker A) is converted so that it sounds like a different real person (speaker B), keeping
A's words.

```
speaker A's speech (source)  ──► content: WHAT is said ───┐
                                                          ├──► converted speech:
speaker B's speech (reference) ─► voice: WHO is speaking ─┘    A's words in B's voice
```

---

## 1. The test set-up

| Item | Value |
|---|---|
| Test speakers | **48 IndicVoices-R speakers never used in training** (official test split). All 48 appear in this test. |
| Language pairs | 9: every source language (ml, ta, hi) × every target voice language (ml, ta, hi). 3 pairs are same-language (ml→ml…), 6 are cross-language (ml→ta…). |
| Conversions | **12 per language pair → 108 per system.** Each conversion has a new source clip and a target speaker different from the source speaker. |
| Reference clip | The target speaker's **longest** test clip (11–15 s) |
| "Other" clip | The target speaker's **second-longest** clip. It is **not** given to the model, and is used for the stricter similarity check. |
| Systems compared | `ours`, `mandarin`, `enc_only` (see below). All three convert the same 108 pairs with the same random noise. |
| Hardware | Colab A100 GPU |

### The three systems

| Name in files | Content encoder | Decoder (DiT) | Purpose |
|---|---|---|---|
| `ours` | Ours (`checkpoints/encoder_v3/encoder_best.pt`, epoch 83) | Ours (`checkpoints/dit_v3/dit_best.pt`, epoch 18) | **The result**: both parts fine-tuned on ml/ta/hi |
| `mandarin` | Original MeanVC (Mandarin) | Original MeanVC (Mandarin) | **Baseline**: the published model, before our work |
| `enc_only` | Ours | Original MeanVC (Mandarin) | **Ablation**: shows that fine-tuning only the encoder is not enough |

### The measurements

| Column | Meaning | Better |
|---|---|---|
| `sim_target` | Voice similarity (cosine of WavLM-ECAPA speaker embeddings) between the converted speech and the **reference clip** | Higher |
| `sim_other` | The same, but against the target speaker's **other clip**, which the model never saw (stricter) | Higher |
| `sim_source` | Similarity to the **source speaker**, i.e. how much of A's voice leaks through | Lower |
| `cer_mms` | Character error rate (%) when **Meta MMS-1b-all**, an independent recogniser we did not train, transcribes the converted speech and is compared with the real transcript | Lower |
| `cer_ours` | The same, using our own encoder's recogniser | Lower |
| `dnsmos` | Microsoft DNSMOS overall speech quality, 1–5 | Higher |
| `rtf` | Real-time factor = processing time / audio length (A100). Below 1 = faster than real time. | Lower |

**The `real speech` row** is the reference point, with no conversion:
- `sim_target`: two real clips of the same target speaker (the ceiling)
- `sim_source`: source and target speakers compared before conversion
- `cer`, `dnsmos`: measured on the original source recordings

---

## 2. Main results

| System | Speaker similarity ↑ | Similarity, other clip ↑ | Source leakage ↓ | CER % (MMS) ↓ | DNSMOS ↑ | RTF ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Real speech (reference point) | 0.753 | – | 0.194 | 23.1 | 3.07 | – |
| Original MeanVC (`mandarin`) | 0.596 | 0.480 | 0.253 | 46.8 | 3.02 | 0.15 |
| Our encoder + Mandarin decoder (`enc_only`) | 0.451 | 0.359 | 0.114 | 68.1 | 2.94 | 0.17 |
| **Ours** | **0.707** | **0.573** | 0.290 | **31.7** | 3.03 | 0.17 |

What the numbers show:

- **Our model sounds much closer to the target speaker than the original MeanVC:** 0.707 vs 0.596, near the 0.753 ceiling.
- **The words are much clearer:** a CER of 31.7% vs 46.8%, against 23.1% for the real recordings.
- **Audio quality is the same as the original model's:** DNSMOS 3.03 vs 3.02.
- **The encoder and decoder must be fine-tuned together.** `enc_only` is the worst system.
- **A little more of the source voice leaks through** than with the original model (0.290 vs 0.253).
- **Same-language conversion scores higher than cross-language** (0.746 vs 0.688). Converting into a **Hindi** voice is the hardest case (0.62–0.65).

The full tables (same vs cross-language, per language pair) are in `results_table.md`.

---

## 3. What is in this folder

```
eval_v3/
├── README.md            this file
├── results_table.md     ★ the clean result tables (open in VS Code, Ctrl+Shift+V)
├── results_table.png    the same tables as an image (for slides / report)
├── summary.md           the tables as printed by evaluate.py (raw format)
├── summary.csv          the table numbers as CSV
├── results.csv          every conversion, every number (one row per conversion per system)
├── pairs.csv            the plan: which source clip was converted into which speaker's voice
├── examples/            ★ 18 cases to LISTEN to (source + reference + all outputs together)
│   └── <pair_id>/       source.wav, reference.wav, ours.wav, mandarin.wav, enc_only.wav
└── wavs/                ALL converted clips (the files the numbers were measured on)
    ├── ours/            108 files: <pair_id>.wav
    ├── mandarin/        108 files
    └── enc_only/        108 files
```

### 3.1 The files

| File | Contents |
|---|---|
| `results_table.md` / `.png` | Five tables: (1) all systems; (2) same vs cross-language; (3) ours per language pair; (4) ours, speaker similarity as a 3×3 grid; (5) ours, CER as a 3×3 grid. Notes explain each column. |
| `summary.md` | The same results in evaluate.py's own format: all conversions / same language / cross-language / ours per pair |
| `summary.csv` | Columns: `system, sim_target, sim_other, sim_source, cer_mms, cer_ours, dnsmos, rtf, n, group` (`group` = all / same / cross / a language pair) |
| `results.csv` | One row per conversion and system (`real speech` included). Columns: `system, pair_id, src_lang, tgt_lang, sim_target, sim_other, sim_source, rtf`, character errors and lengths for both recognisers (`err_ours, len_ours, err_mms, len_mms`), what each recogniser heard (`text_ours, text_mms`), `dnsmos` |
| `pairs.csv` | 108 rows, one per conversion. Columns: `pair_id`, `src_lang`, `tgt_lang`, `source` (source clip ID), `source_spk`, `target_spk`, `reference` (the reference clip ID), `other` (the extra clip used for `sim_other`), `text` (real transcript of the source), `duration` (seconds) |

### 3.2 `examples/`: for listening

18 folders, **2 for each of the 9 language pairs**. Each holds one conversion with all its audio
side by side:

| File | What it is |
|---|---|
| `source.wav` | **Speaker A**: the original speech. Supplies the words. |
| `reference.wav` | **Speaker B**: a different person. Supplies the voice. |
| `ours.wav` | A's words in B's voice: **our model** |
| `mandarin.wav` | The same with the original MeanVC (baseline) |
| `enc_only.wav` | The same with our encoder + Mandarin decoder (ablation) |

**How to listen:** `source.wav` → `reference.wav` → `ours.wav`. The result should say what the
source says, in the voice of the reference. Then compare `ours.wav` with `mandarin.wav`.

**Folder names:** `<source language>2<target voice language>_<number>`. For example, `ml2ta_00`
means a Malayalam speaker's sentence converted into a Tamil speaker's voice.

### 3.3 What each example folder points to

Clip IDs are files in `data/audio_test/<ml|ta|hi>/<clip ID>.wav`. Speaker IDs are IndicVoices-R
speaker IDs (F = female, M = male). The scores are for `ours`, except the "Original MeanVC"
column.

| Folder | Source clip (speaker A) | Speaker A | Reference clip (speaker B) | Speaker B | Similarity, ours ↑ | Similarity, original MeanVC ↑ | Source leakage ↓ | CER % (MMS) ↓ |
|---|---|---|---|---|:---:|:---:|:---:|:---:|
| `hi2hi_00` | hi_te00001_0032 (10.3 s) | S4258561600395718 (F) | hi_te00000_0021 (14.7 s) | S4256820200338001 (F) | 0.705 | 0.679 | 0.461 | 8.0 |
| `hi2hi_01` | hi_te00001_0055 (12.9 s) | S4258858600387188 (M) | hi_te00001_0037 (12.0 s) | S4258561600395718 (F) | 0.736 | 0.617 | 0.114 | 25.6 |
| `hi2ml_00` | hi_te00000_0010 (8.7 s) | S4256820200338001 (F) | ml_te00001_0122 (10.3 s) | S4259631200357141 (M) | 0.751 | 0.644 | 0.218 | 11.4 |
| `hi2ml_01` | hi_te00000_0034 (7.6 s) | S4256942400332561 (F) | ml_te00001_0192 (13.5 s) | S4259900100347536 (F) | 0.775 | 0.727 | 0.289 | 35.0 |
| `hi2ta_00` | hi_te00001_0123 (7.6 s) | S4259898100372712 (M) | ta_te00000_0063 (12.4 s) | S4256817200333561 (F) | 0.680 | 0.613 | 0.236 | 4.7 |
| `hi2ta_01` | hi_te00001_0147 (10.2 s) | S4259898600358163 (F) | ta_te00000_0244 (14.4 s) | S4259640900333333 (M) | 0.635 | 0.378 | 0.129 | 14.3 |
| `ml2hi_00` | ml_te00001_0027 (6.9 s) | S4259485400386489 (M) | hi_te00000_0021 (14.7 s) | S4256820200338001 (F) | 0.630 | 0.651 | 0.287 | 24.2 |
| `ml2hi_01` | ml_te00001_0051 (8.7 s) | S4259502200366806 (F) | hi_te00001_0011 (14.0 s) | S4258486700370849 (M) | 0.757 | 0.622 | 0.263 | 19.2 |
| `ml2ml_00` | ml_te00000_0051 (5.3 s) | S4256324400328999 (M) | ml_te00000_0172 (12.7 s) | S4259267700356708 (F) | 0.784 | 0.524 | 0.291 | 18.2 |
| `ml2ml_01` | ml_te00000_0080 (11.2 s) | S4256368500328803 (M) | ml_te00000_0188 (14.1 s) | S4259278700356316 (F) | 0.730 | 0.686 | 0.233 | 8.4 |
| `ml2ta_00` | ml_te00001_0117 (6.7 s) | S4259631200357141 (M) | ta_te00000_0227 (14.0 s) | S4259592300370250 (M) | 0.701 | 0.671 | 0.312 | 16.2 |
| `ml2ta_01` | ml_te00001_0131 (4.3 s) | S4259695500301137 (F) | ta_te00000_0180 (13.1 s) | S4259418900342635 (F) | 0.656 | 0.450 | 0.349 | 20.6 |
| `ta2hi_00` | ta_te00000_0158 (7.4 s) | S4259410500335240 (M) | hi_te00001_0037 (12.0 s) | S4258561600395718 (F) | 0.549 | 0.431 | 0.240 | 63.0 |
| `ta2hi_01` | ta_te00000_0180 (13.1 s) | S4259418900342635 (F) | hi_te00000_0113 (11.3 s) | S4257878300397507 (F) | 0.660 | 0.610 | 0.447 | 37.2 |
| `ta2ml_00` | ta_te00000_0010 (6.5 s) | S4256620800339114 (M) | ml_te00000_0059 (11.4 s) | S4256324400328999 (M) | 0.794 | 0.723 | 0.220 | 42.4 |
| `ta2ml_01` | ta_te00000_0024 (5.7 s) | S4256700400339825 (F) | ml_te00001_0089 (13.7 s) | S4259627600301041 (M) | 0.696 | 0.676 | 0.211 | 39.2 |
| `ta2ta_00` | ta_te00000_0222 (8.3 s) | S4259592300370250 (M) | ta_te00000_0049 (14.3 s) | S4256730700334621 (F) | 0.830 | 0.766 | 0.059 | 40.5 |
| `ta2ta_01` | ta_te00000_0240 (9.5 s) | S4259640900333333 (M) | ta_te00000_0287 (12.4 s) | S4259953800370105 (M) | 0.636 | 0.455 | 0.431 | 16.7 |

Good cases to play first:
- `ta2ta_00`: the highest similarity (0.830), almost no source leakage, and a male-to-female conversion
- `ml2ml_00`: ours 0.784 vs the original MeanVC's 0.524
- `hi2ta_00`: clear words (CER 4.7%)

Weaker cases:
- `ta2hi_00`: a Tamil source in a Hindi voice, with low similarity and a high CER
- `ml2hi_00`: one of the few cases where the original MeanVC scores slightly higher

These are single examples. The averages in Section 2 are the reliable result.

### 3.4 `wavs/`: every converted clip

- `wavs/ours/`, `wavs/mandarin/` and `wavs/enc_only/` each hold **108 files**, one per conversion, named `<pair_id>.wav`. These are the files every number in the tables was measured on.
- They hold **only the converted audio**. The source and reference recordings are not copied here, to avoid storing them 108 times.

**To find the source and reference of any file in `wavs/`:**
1. Take its name, for example `wavs/ours/ml2ta_07.wav` → `ml2ta_07`.
2. Look up that `pair_id` in `pairs.csv`. The `source` and `reference` columns give the clip IDs.
3. Play `data/audio_test/<language>/<clip ID>.wav`.
4. That conversion's scores are in `results.csv` (same `pair_id`, `system = ours`).

The 18 folders in `examples/` are copies of the first two conversions of each language pair.
For example, `examples/ml2ta_00/ours.wav` is the same clip as `wavs/ours/ml2ta_00.wav`.

---

## 4. Related folders

| Folder | What it is |
|---|---|
| `outputs/eval_v3/` (this folder) | **Stage 5: voice conversion.** Real speaker A → real speaker B's voice, 108 conversions per system. **The main voice-conversion result.** |
| `outputs/stage6_speech_translation_voice_conversion_results/` | **Stage 6: speech translation + voice conversion.** A's sentence is translated into another language and spoken in speaker B's voice (see `README.md`). It has its own README. |

**To re-create this folder:** run `python code/infer/evaluate.py` on Colab after Stage 4, and then
`python code/infer/show_results.py` for `results_table.md` and `.png`.
