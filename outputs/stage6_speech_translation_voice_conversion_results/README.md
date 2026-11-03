# Stage 6 results: speech translation + voice conversion

This folder holds the **end-to-end test of the cross-lingual pipeline** of MeanVC-Indic v3. Speaker A says
a sentence in Malayalam, Tamil or Hindi. The system translates it into one of the other two languages and
speaks the translation **in the voice of a different person, speaker B**.

Everything here was produced on **30 Oct 2026** by
[`code/translate/evaluate_translation.py`](../../code/translate/evaluate_translation.py), on the laptop
(RTX 2050 GPU + CPU). No model was trained for this stage.

```
speaker A's speech ─► 1 language ID ─► 2 speech → text ─► 3 translation ─► 4 text → speech ─► 5 voice conversion ─► output
                      our encoder      IndicConformer      IndicTrans2       MMS-TTS            our MeanVC v3
                                                                                                    ▲
                                                                              speaker B's speech (the voice to copy)
```

| Step | Model | Trained by us? |
|---|---|---|
| 1 Language ID | Our fine-tuned Fast-U2++ encoder + CTC head (script voting) | Yes (Stage 2) |
| 2 Speech → text | AI4Bharat IndicConformer 600M, RNNT decoding | No (pretrained) |
| 3 Translation | AI4Bharat IndicTrans2 Indic↔Indic, 320M | No (pretrained) |
| 4 Text → speech | Meta MMS-TTS (VITS), one voice per language | No (pretrained) |
| 5 Voice conversion | Our MeanVC v3 (content encoder + DiT decoder) + WavLM-ECAPA + Vocos | Yes (Stages 2–4) |

---

## 1. The test set-up

| Item | Value |
|---|---|
| Speakers | The **48 IndicVoices-R test speakers** (16 per language), **never used in training** |
| Part A | **All 300 test clips** (100 per language): language ID, and three speech recognisers compared |
| Part B | **48 translations**: 8 per direction × 6 directions (ml→ta, ml→hi, ta→ml, ta→hi, hi→ml, hi→ta) |
| Speaker A (source) | 48 different speakers, one clip each (3–12 s). The language is **detected**, as a real user would run it. |
| Speaker B (voice) | A test speaker **of the target language**, never the same person as A. All 48 speakers are used once as B. **20 of the 48 pairs are cross-gender** (male ↔ female). |
| B's reference clip | B's **longest** test clip: the voice given to the model |
| B's other clip | B's **second-longest** clip, **never given to the model**. Used for the stricter similarity check. |
| Compared outputs | The same translated TTS speech in three voices: **TTS only** (no conversion), **original MeanVC** (the published Mandarin model), **our MeanVC v3** |

### Measurements

| Name | Meaning | Better |
|---|---|---|
| Language ID correct | The detected language equals the real one | Higher |
| CER / WER | Character / word error rate of a speech recogniser against the real transcript, after the same text cleaning used in training. Given against both the **verbatim** transcript (as spoken, colloquial forms) and the **normalised** one (standard spelling). | Lower |
| Translation agreement (chrF, 0–100) | The translation of the recognised text compared with the translation of the **real** transcript. It shows how much recognition errors change the translation. IndicVoices-R has no human reference translations, so this is not an absolute translation-quality score. | Higher |
| Similarity to B | Cosine similarity of WavLM-ECAPA speaker embeddings between the output and B's reference clip. **Other clip:** the same, against B's unseen clip. | Higher |
| Similarity to A (leakage) | How much the output still sounds like speaker A | Lower |
| CER % (MMS) vs translation | An independent recogniser (Meta MMS-1b-all) transcribes the output, compared with the translated text it should say: **intelligibility** | Lower |
| DNSMOS | Microsoft DNSMOS overall speech quality, 1–5 | Higher |
| RTF | Processing time ÷ input length. Below 1 = faster than real time. | Lower |

---

## 2. Main results

### 2.1 Step 1, language identification (300 clips): 99.3 % correct

Malayalam 100 %, Tamil 100 %, Hindi 98 % (2 Hindi clips were detected as Tamil). The mean confidence is
0.97. Only 1 of 300 clips fell below the 60 % threshold at which the user is asked to confirm.

### 2.2 Step 2, speech recognition (300 clips): IndicConformer is clearly the best

| Recogniser | CER % (verbatim) | WER % (verbatim) | CER % (normalised) | WER % (normalised) | Translation agreement chrF ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| Our encoder (greedy CTC, streaming) | 24.5 | 65.3 | 26.2 | 68.9 | 37.2 |
| Meta MMS-1b-all | 23.3 | 58.8 | 24.1 | 58.4 | 46.9 |
| **AI4Bharat IndicConformer 600M** | **10.3** | **35.3** | **7.3** | **22.8** | **64.7** |

- IndicConformer makes **less than half the character errors** of the other two, in every language
  (normalised CER: Malayalam 8.3 %, Tamil 8.4 %, Hindi 4.6 %).
- Better recognition gives **much better translations**. Agreement rises from 37.2 to 64.7: errors at
  step 2 were the main cause of wrong translations.
- IndicConformer scores better on the normalised transcript, because it writes the **standard spelling**
  of colloquial words. That is also the better input for the translator.
- Our encoder remains the language identifier: it is fast (0.03 s per clip) and 99.3 % correct.

### 2.3 Steps 4–5, the translated speech in speaker B's voice (48 translations)

| Output | Similarity to B ↑ | Similarity to B, other clip ↑ | Similarity to A ↓ | CER % (MMS) ↓ | DNSMOS ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| Real speech: two clips of B (the ceiling) | – | 0.754 | – | – | 3.18 |
| Real speech: A vs B (two different people) | 0.185 | – | – | – | 3.09 |
| TTS only (no voice conversion) | 0.171 | 0.139 | 0.078 | 6.4 | 3.18 |
| TTS + original MeanVC | 0.620 | 0.501 | 0.154 | 20.5 | 2.99 |
| **TTS + our MeanVC v3** | **0.733** | **0.592** | 0.173 | **9.5** | **3.05** |

- **The output sounds like speaker B.** Similarity rises from 0.171 (the TTS voice is a stranger) to
  **0.733**, close to the **0.754** between two real recordings of B. On B's unseen clip it is 0.592,
  against 0.501 for the original MeanVC.
- **Our model beats the original MeanVC** on voice (0.733 vs 0.620) **and** clarity (9.5 % vs 20.5 % CER).
  It is lower than the original in only 3 of the 48 translations.
- **The words stay clear.** Conversion raises the MMS error from 6.4 % (TTS) to only 9.5 %.
- **No leakage of A's voice.** The output is no closer to A (0.173) than B naturally is (0.185). This is
  expected: A's audio never enters the voice converter, only A's words (via text) do.
- **Quality is close to real speech** (DNSMOS 3.05 vs 3.09–3.18).

### 2.4 Per direction (our pipeline)

| Direction | ASR CER % | Translation chrF ↑ | Similarity to B: TTS → ours ↑ | Other clip ↑ | CER % (MMS) ↓ | DNSMOS ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Malayalam → Tamil | 10.1 | 68.8 | 0.219 → 0.769 | 0.639 | 10.1 | 2.97 |
| Malayalam → Hindi | 8.7 | 64.6 | 0.145 → 0.711 | 0.558 | 8.7 | 3.09 |
| Tamil → Malayalam | 14.9 | 54.1 | 0.174 → 0.736 | 0.556 | 13.2 | 3.04 |
| Tamil → Hindi | 11.7 | 55.3 | 0.121 → 0.697 | 0.582 | 11.2 | 3.18 |
| Hindi → Malayalam | 19.7 | 60.7 | 0.136 → 0.759 | 0.571 | 8.6 | 3.03 |
| Hindi → Tamil | 4.8 | 78.2 | 0.232 → 0.729 | 0.648 | 5.5 | 2.97 |
| **All** | **11.6** | **64.7** | **0.171 → 0.733** | **0.592** | **9.5** | **3.05** |

Every direction works. Voices are copied best into Malayalam and Tamil (0.73–0.77). Hindi voices are
a little lower (about 0.70), as in the Stage 5 voice-conversion test. Translations from Hindi into Tamil
agree best (78.2). Tamil sources agree least (54–55), because Tamil spontaneous speech is the hardest to
recognise.

### 2.5 Speed (laptop, RTX 2050 + CPU)

| Step | Seconds | Share |
|---|:---:|:---:|
| 1 Language ID | 0.05 | 1 % |
| 2 Speech → text (IndicConformer, CPU) | 0.73 | 9 % |
| Voice of B (speaker embedding + mel prompt of B's clip) | 2.57 | 33 % |
| 3 Translation | 0.92 | 12 % |
| 4 Text → speech | 1.57 | 20 % |
| 5 Voice conversion | 2.03 | 26 % |
| **Total** | **7.87 s for 7.7 s of speech** | RTF 1.08 |

The system takes about as long as the speech lasts. Reading B's voice is the largest part, but it only
needs to be done **once per voice**, so a second sentence in the same voice takes about 5.3 s. The
pipeline works on a finished recording, not a live stream.

---

## 3. Limitations

- **Translation is the weakest remaining step.** IndicTrans2 was trained on written, punctuated text.
  Spontaneous speech, with its repetitions and half-finished sentences, still produces some wrong or
  awkward translations (see `hi2ml_03`, `ta2hi_07`).
- **Translation agreement is not translation accuracy.** It compares against the translation of the
  correct transcript, because the test set has no human reference translations.
- **Rhythm and intonation come from the TTS.** The output speaks with MMS-TTS's even, read-aloud prosody,
  not B's natural speaking style.
- **8 translations per direction** is enough to compare systems, but per-direction numbers carry noise.
- **Only Malayalam, Tamil and Hindi**, the languages the voice converter was fine-tuned on.

---

## 4. What is in this folder

```
stage6_speech_translation_voice_conversion_results/
├── README.md              this file
├── results_tables.md      ★ all tables, as written by the evaluation (open in VS Code, Ctrl+Shift+V)
├── summary.csv            the table numbers as CSV
├── asr_comparison.csv     Part A: one row per test clip (300): language ID + the three recognisers' text and errors
├── results.csv            Part B: one row per translation (48): all texts, similarities, CER, DNSMOS, times
├── figures/               ★ the results as PNG images (for slides and the report):
│   ├── 1_voice_conversion_result.png       similarity to B, intelligibility, quality: TTS / original / ours
│   ├── 2_speech_recognisers.png            CER per language and translation agreement, 3 recognisers
│   ├── 3_similarity_per_direction.png      similarity to B in each of the 6 directions
│   ├── 4_before_vs_after_translation.png   our model before (Stage 5) vs after (Stage 6) translation
│   ├── 5_results_tables.png                the main tables as one image
│   └── 6_metrics_summary.png               every evaluation metric with our system's result, one table
└── examples/              ★ all 48 translations to LISTEN to, one folder each
    └── <run>/             5 audio files + texts.txt (below)
```

### 4.1 `results_tables.md`

Table 1: language ID (with the confusion table). Table 2: the three recognisers (2a verbatim, 2b
normalised). Table 3: the three voices. Table 4: per direction. Table 5: time per step. Then one
worked example per direction, with all its texts.

### 4.2 The CSV files

| File | Main columns |
|---|---|
| `asr_comparison.csv` | `utt_id, lang, detected, confidence, lid_correct, text` (verbatim), `text_norm` (normalised), `hyp_ours / hyp_mms / hyp_indicconformer` (what each recogniser wrote), `cerr_* chars_* werr_* words_*` (errors vs verbatim), the same with `n` (vs normalised), `sec_*` (time) |
| `results.csv` | `run_id, src_lang, tgt_lang, speaker_a, speaker_b, source, reference, other`, `text` (real transcript), `transcript` (IndicConformer), `translation`, `translation_oracle` (translation of the real transcript), `sim_ref_* / sim_other_* / sim_src_*` for `tts`, `mandarin`, `ours`, `sim_real_b, sim_real_ab`, `mms_*` (what MMS heard), `err_* len_*`, `dnsmos_*`, `t_*` (seconds per step), `rtf` |
| `summary.csv` | One row per table line: `table, group, n` + the numbers |

### 4.3 `examples/`: for listening

**48 folders**, one per translation, named `<source language>2<target language>_<number>`. For example,
`ml2ta_03` is a Malayalam sentence translated into Tamil and spoken in a Tamil speaker's voice.

Each folder holds the files in **listening order**:

| File | What it is |
|---|---|
| `1_source_speech_speakerA.wav` | Speaker A's original sentence (the source language) |
| `2_reference_voice_speakerB.wav` | Speaker B: the voice to copy (the target language) |
| `3_tts_translation_generic_voice.wav` | The translation spoken by MMS-TTS: right words, generic voice |
| `4_converted_ours.wav` | **The result**: the translation in B's voice, our MeanVC v3 |
| `5_converted_original_meanvc.wav` | The same with the original MeanVC (for comparison) |
| `texts.txt` | Speakers, clip IDs, detected language, real transcript, recognised text, translation, translation of the real transcript, what MMS hears in the output, and the similarity numbers |

**How to listen:** play 1 → 2 → 4. File 4 should say A's sentence, translated, in B's voice. Then
compare 3 (before conversion) and 5 (the original model) with 4.

### 4.4 What each example folder points to

Clip IDs are files in `data/audio_test/<ml|ta|hi>/<clip ID>.wav`. F = female, M = male.
Similarity = similarity to speaker B's reference clip. chrF = this translation vs the translation of the
real transcript (0–100). CER = MMS reading our output vs the translation.

| Folder | Speaker A clip | Speaker A | Speaker B reference | Speaker B | Similarity: TTS | Similarity: original | **Similarity: ours** | chrF | CER % |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `ml2ta_00` | ml_te00000_0102 (8.3 s) | S4257213200305882 (M) | ta_te00000_0049 | S4256730700334621 (F) | 0.25 | 0.80 | 0.79 | 63 | 16.9 |
| `ml2ta_01` | ml_te00001_0050 (7.3 s) | S4259502200366806 (F) | ta_te00000_0214 | S4259578000319738 (F) | 0.18 | 0.57 | 0.81 | 48 | 13.3 |
| `ml2ta_02` | ml_te00001_0072 (5.3 s) | S4259627600301041 (M) | ta_te00000_0258 | S4259935800305390 (M) | 0.31 | 0.55 | 0.71 | 87 | 4.3 |
| `ml2ta_03` | ml_te00000_0072 (6.3 s) | S4256368500328803 (M) | ta_te00000_0099 | S4257394700369395 (F) | 0.10 | 0.70 | 0.80 | 69 | 5.8 |
| `ml2ta_04` | ml_te00001_0165 (11.1 s) | S4259786000304613 (F) | ta_te00000_0227 | S4259592300370250 (M) | 0.40 | 0.69 | 0.74 | 71 | 10.9 |
| `ml2ta_05` | ml_te00000_0182 (5.8 s) | S4259278700356316 (F) | ta_te00000_0079 | S4257156700348229 (F) | 0.23 | 0.57 | 0.77 | 100 | 13.9 |
| `ml2ta_06` | ml_te00000_0044 (11.4 s) | S4259564300366216 (M) | ta_te00000_0033 | S4256700400339825 (F) | 0.07 | 0.77 | 0.82 | 66 | 5.6 |
| `ml2ta_07` | ml_te00000_0051 (5.3 s) | S4256324400328999 (M) | ta_te00000_0018 | S4256620800339114 (M) | 0.22 | 0.68 | 0.70 | 75 | 16.9 |
| `ml2hi_00` | ml_te00000_0166 (8.3 s) | S4259267700356708 (F) | hi_te00001_0151 | S4259898600358163 (F) | 0.18 | 0.65 | 0.73 | 92 | 7.7 |
| `ml2hi_01` | ml_te00000_0139 (3.7 s) | S4258813000306083 (M) | hi_te00000_0068 | S4257120600367211 (F) | 0.26 | 0.54 | 0.70 | 92 | 13.8 |
| `ml2hi_02` | ml_te00000_0126 (7.5 s) | S4258010700366114 (F) | hi_te00001_0186 | S4259978700366892 (F) | 0.09 | 0.68 | 0.77 | 41 | 6.0 |
| `ml2hi_03` | ml_te00001_0187 (8.5 s) | S4259900100347536 (F) | hi_te00001_0037 | S4258561600395718 (F) | 0.11 | 0.57 | 0.63 | 60 | 5.3 |
| `ml2hi_04` | ml_te00001_0131 (4.3 s) | S4259695500301137 (F) | hi_te00001_0135 | S4259898100372712 (M) | 0.03 | 0.49 | 0.68 | 100 | 15.6 |
| `ml2hi_05` | ml_te00000_0196 (5.9 s) | S4259451300377433 (F) | hi_te00000_0113 | S4257878300397507 (F) | 0.19 | 0.59 | 0.72 | 36 | 15.7 |
| `ml2hi_06` | ml_te00001_0101 (3.3 s) | S4259631200357141 (M) | hi_te00001_0011 | S4258486700370849 (M) | 0.07 | 0.70 | 0.76 | 29 | 1.7 |
| `ml2hi_07` | ml_te00001_0035 (11.0 s) | S4259485400386489 (M) | hi_te00001_0055 | S4258858600387188 (M) | 0.23 | 0.53 | 0.70 | 64 | 9.6 |
| `ta2ml_00` | ta_te00000_0069 (3.1 s) | S4257156700348229 (F) | ml_te00000_0113 | S4257213200305882 (M) | 0.08 | 0.56 | 0.70 | 100 | 21.1 |
| `ta2ml_01` | ta_te00000_0014 (9.1 s) | S4256620800339114 (M) | ml_te00001_0089 | S4259627600301041 (M) | 0.10 | 0.67 | 0.67 | 36 | 14.7 |
| `ta2ml_02` | ta_te00000_0095 (9.5 s) | S4257394700369395 (F) | ml_te00000_0133 | S4258010700366114 (F) | 0.24 | 0.64 | 0.78 | 96 | 26.5 |
| `ta2ml_03` | ta_te00000_0111 (8.0 s) | S4258842100383591 (M) | ml_te00000_0044 | S4259564300366216 (M) | 0.16 | 0.59 | 0.77 | 30 | 3.2 |
| `ta2ml_04` | ta_te00000_0158 (7.4 s) | S4259410500335240 (M) | ml_te00000_0059 | S4256324400328999 (M) | 0.24 | 0.67 | 0.73 | 34 | 15.4 |
| `ta2ml_05` | ta_te00000_0282 (9.3 s) | S4259953800370105 (M) | ml_te00000_0150 | S4258813000306083 (M) | 0.16 | 0.55 | 0.67 | 76 | 6.8 |
| `ta2ml_06` | ta_te00000_0173 (6.0 s) | S4259418900342635 (F) | ml_te00001_0004 | S4259451300377433 (F) | 0.18 | 0.72 | 0.77 | 54 | 16.3 |
| `ta2ml_07` | ta_te00000_0133 (6.6 s) | S4259255100305814 (F) | ml_te00000_0172 | S4259267700356708 (F) | 0.23 | 0.56 | 0.78 | 41 | 10.2 |
| `ta2hi_00` | ta_te00000_0220 (5.8 s) | S4259592300370250 (M) | hi_te00001_0165 | S4259920000308452 (M) | 0.19 | 0.53 | 0.59 | 48 | 10.0 |
| `ta2hi_01` | ta_te00000_0031 (11.9 s) | S4256700400339825 (F) | hi_te00000_0182 | S4258388600389373 (M) | 0.15 | 0.70 | 0.72 | 46 | 0.0 |
| `ta2hi_02` | ta_te00000_0046 (10.0 s) | S4256730700334621 (F) | hi_te00000_0094 | S4257196900365559 (M) | 0.20 | 0.75 | 0.80 | 32 | 7.6 |
| `ta2hi_03` | ta_te00000_0189 (5.7 s) | S4259519400365771 (M) | hi_te00000_0153 | S4258354000329548 (M) | 0.17 | 0.64 | 0.72 | 100 | 21.0 |
| `ta2hi_04` | ta_te00000_0252 (6.3 s) | S4259935800305390 (M) | hi_te00000_0021 | S4256820200338001 (F) | 0.13 | 0.47 | 0.46 | 100 | 53.3 |
| `ta2hi_05` | ta_te00000_0212 (10.6 s) | S4259578000319738 (F) | hi_te00000_0041 | S4256942400332561 (F) | -0.10 | 0.56 | 0.77 | 63 | 7.0 |
| `ta2hi_06` | ta_te00000_0232 (4.0 s) | S4259640900333333 (M) | hi_te00001_0072 | S4259576300396472 (F) | -0.07 | 0.56 | 0.72 | 100 | 10.9 |
| `ta2hi_07` | ta_te00000_0062 (11.6 s) | S4256817200333561 (F) | hi_te00001_0107 | S4259703700319935 (M) | 0.30 | 0.54 | 0.79 | 24 | 6.9 |
| `hi2ml_00` | hi_te00001_0053 (10.8 s) | S4258858600387188 (M) | ml_te00001_0167 | S4259786000304613 (F) | 0.10 | 0.66 | 0.80 | 58 | 1.5 |
| `hi2ml_01` | hi_te00001_0024 (7.0 s) | S4258561600395718 (F) | ml_te00001_0122 | S4259631200357141 (M) | 0.15 | 0.50 | 0.72 | 52 | 7.8 |
| `hi2ml_02` | hi_te00000_0145 (5.4 s) | S4258354000329548 (M) | ml_te00000_0188 | S4259278700356316 (F) | 0.19 | 0.72 | 0.78 | 64 | 12.5 |
| `hi2ml_03` | hi_te00000_0065 (10.5 s) | S4257120600367211 (F) | ml_te00001_0054 | S4259502200366806 (F) | 0.06 | 0.51 | 0.72 | 16 | 27.3 |
| `hi2ml_04` | hi_te00000_0170 (6.1 s) | S4258388600389373 (M) | ml_te00001_0140 | S4259695500301137 (F) | 0.20 | 0.72 | 0.80 | 90 | 3.3 |
| `hi2ml_05` | hi_te00001_0008 (10.9 s) | S4258486700370849 (M) | ml_te00001_0038 | S4259485400386489 (M) | 0.12 | 0.57 | 0.74 | 77 | 6.5 |
| `hi2ml_06` | hi_te00001_0161 (8.5 s) | S4259920000308452 (M) | ml_te00000_0083 | S4256368500328803 (M) | 0.19 | 0.61 | 0.73 | 100 | 10.0 |
| `hi2ml_07` | hi_te00001_0100 (9.2 s) | S4259703700319935 (M) | ml_te00001_0192 | S4259900100347536 (F) | 0.09 | 0.76 | 0.79 | 44 | 10.4 |
| `hi2ta_00` | hi_te00000_0108 (6.6 s) | S4257878300397507 (F) | ta_te00000_0165 | S4259410500335240 (M) | 0.29 | 0.70 | 0.63 | 98 | 13.9 |
| `hi2ta_01` | hi_te00000_0088 (8.9 s) | S4257196900365559 (M) | ta_te00000_0195 | S4259519400365771 (M) | 0.19 | 0.64 | 0.76 | 79 | 1.5 |
| `hi2ta_02` | hi_te00001_0069 (10.6 s) | S4259576300396472 (F) | ta_te00000_0244 | S4259640900333333 (M) | 0.25 | 0.48 | 0.60 | 81 | 4.4 |
| `hi2ta_03` | hi_te00001_0130 (10.8 s) | S4259898100372712 (M) | ta_te00000_0120 | S4258842100383591 (M) | 0.32 | 0.65 | 0.79 | 69 | 4.3 |
| `hi2ta_04` | hi_te00001_0178 (7.7 s) | S4259978700366892 (F) | ta_te00000_0141 | S4259255100305814 (F) | 0.12 | 0.50 | 0.78 | 62 | 7.1 |
| `hi2ta_05` | hi_te00001_0143 (7.3 s) | S4259898600358163 (F) | ta_te00000_0180 | S4259418900342635 (F) | 0.25 | 0.63 | 0.82 | 100 | 2.7 |
| `hi2ta_06` | hi_te00000_0031 (5.7 s) | S4256942400332561 (F) | ta_te00000_0063 | S4256817200333561 (F) | 0.17 | 0.77 | 0.80 | 56 | 2.6 |
| `hi2ta_07` | hi_te00000_0002 (5.6 s) | S4256820200338001 (F) | ta_te00000_0287 | S4259953800370105 (M) | 0.27 | 0.55 | 0.65 | 100 | 9.5 |

### 4.5 Good examples to play first

| Folder | Why |
|---|---|
| `hi2ta_05` | Best overall: a perfect translation (chrF 100), voice similarity 0.82, clear words (CER 2.7 %) |
| `ml2ta_06` | A **male** Malayalam speaker → a **female** Tamil voice, similarity 0.82 |
| `ml2ta_03` | Male → female, 0.80, with the original MeanVC at 0.70 for comparison |
| `ta2hi_05` | The TTS voice was nothing like B (−0.10); ours reaches 0.77 |
| `hi2ml_04` | Male → female, a near-perfect translation (chrF 90), 0.80 |

Weaker cases, which show the limits:

| Folder | Why |
|---|---|
| `ta2hi_04` | Male → female into Hindi: the lowest similarity (0.46) and a high CER (53 %) |
| `hi2ml_03` | The translation went wrong (chrF 16) even though the voice was copied well (0.72) |

These are single examples. The averages in Section 2 are the reliable result.

---

## 5. Related results

| Folder | What it is |
|---|---|
| `outputs/eval_v3/` | **Stage 5:** voice conversion alone. Real speaker A → real speaker B, no translation, 108 conversions per system. |
| This folder | **Stage 6:** speech translation + voice conversion, with B's voice |

**To re-create this folder:** `python code/translate/evaluate_translation.py` (about 35 minutes on the
laptop). See `code/translate/README.md` for the set-up and the command-line tool.
