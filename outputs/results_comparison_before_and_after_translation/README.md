# Results comparison: before translation vs after translation

This folder compares our voice converter in two situations:

- **Before translation** (Stage 5): speaker A's **real speech** is converted into speaker B's voice. The
  words stay in A's language.
- **After translation** (Stage 6): speaker A's speech is recognised, **translated into B's language**,
  spoken by TTS, and then converted into speaker B's voice.

The voice converters, the test speakers and the measurements are the same in both tests, so the
comparison shows what translation changes.

---

## Summary

| | Before translation | After translation | Result |
|---|:---:|:---:|:---:|
| How much the output sounds like speaker B | 0.688 | **0.733** | **Better** |
| How much speaker A's voice is left (leakage) | 0.279 | **0.173** | **Better**: none left |
| Clarity lost by the conversion | 9.2 % points | **3.1 % points** | **Better** |
| Audio quality (DNSMOS, 1–5) | 3.04 | 3.05 | About the same |
| Our model's gain over the original MeanVC | +0.104 | +0.114 | Holds in both |

**In one sentence:** after translation, our model's output sounds more like the target speaker, keeps
none of the source speaker's voice, and loses less clarity, with the same audio quality.

---

## 1. The two tests

| | Before translation | After translation |
|---|---|---|
| Test | Stage 5, [`outputs/eval_v3`](../eval_v3/README.md) | Stage 6, [`outputs/stage6_speech_translation_voice_conversion_results`](../stage6_speech_translation_voice_conversion_results/README.md) |
| What goes into the voice converter | **Speaker A's real recording** | **TTS speech** of the translated sentence |
| Language of the output | A's language | B's language (translated) |
| Voice | Speaker B (reference = B's longest test clip) | Speaker B (reference = B's longest test clip) |
| Speakers | 48 test speakers, never seen in training | The same 48 |
| Conversions compared | 72 (12 per direction) | 48 (8 per direction) |
| Systems | Our MeanVC v3, original MeanVC | Our MeanVC v3, original MeanVC |

Only **cross-language** pairs are compared (for example, Malayalam speech → a Tamil speaker's voice),
because the translation test has no same-language pairs.

---

## 2. Our MeanVC v3: before vs after translation

| Measure | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Sounds like speaker B (0–1) | 0.688 | 0.733 | +0.045 | **Better** |
| Sounds like speaker B, on a clip the model never heard (0–1) | 0.553 | 0.592 | +0.039 | **Better** |
| Still sounds like speaker A: leakage (0–1) | 0.279 | 0.173 | −0.106 | **Better** |
| Clarity lost by the conversion (% points) | 9.2 | 3.1 | −6.1 | **Better** |
| Audio quality, DNSMOS (1–5) | 3.04 | 3.05 | +0.01 | About the same |

### Reference points (real recordings, no conversion)

| Reference | Before translation | After translation |
|---|:---:|:---:|
| Two real clips of speaker B: the **best** similarity possible | 0.748 | 0.754 |
| Speaker A vs speaker B: the **normal** similarity of two different people | 0.166 | 0.185 |
| Letters misheard in the converter's input (%) | 22.5 (real speech) | 6.4 (TTS speech) |

**What this shows:**

1. **The output sounds more like speaker B.** It goes from 0.688 to **0.733**, close to the best possible
   (about 0.75).
2. **None of speaker A's voice is left.** Before translation, leakage (0.279) was well above the normal
   level for two different people (0.166). After translation, it is at that level (0.173 vs 0.185). The
   reason is that A's recording never reaches the voice converter, only A's **words** (as TTS speech) do.
3. **Less clarity is lost.** Clean TTS speech is easier for the converter to read than spontaneous
   recordings.
4. **Audio quality does not change.**

---

## 3. Original MeanVC: before vs after translation

| Measure | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Sounds like speaker B (0–1) | 0.584 | 0.620 | +0.036 | **Better** |
| Sounds like speaker B, on a clip the model never heard (0–1) | 0.467 | 0.501 | +0.034 | **Better** |
| Still sounds like speaker A: leakage (0–1) | 0.244 | 0.154 | −0.090 | **Better** |
| Clarity lost by the conversion (% points) | 24.6 | 14.2 | −10.4 | **Better** |
| Audio quality, DNSMOS (1–5) | 3.03 | 2.99 | −0.03 | About the same |

The original model improves after translation in the same way. It still loses much more clarity than
ours (14.2 vs 3.1 points).

---

## 4. Our MeanVC v3 vs the original MeanVC

How much the output sounds like speaker B:

| Test | Original MeanVC | Our MeanVC v3 | Our gain |
|---|:---:|:---:|:---:|
| Before translation | 0.584 | 0.688 | +0.104 |
| After translation | 0.620 | 0.733 | +0.114 |

Our model is clearly better in both tests, by about the same margin.

---

## 5. Our MeanVC v3 per direction

### Sounds like speaker B (higher is better)

| Direction | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Malayalam speech → Tamil voice | 0.715 | 0.769 | +0.054 | **Better** |
| Malayalam speech → Hindi voice | 0.649 | 0.711 | +0.063 | **Better** |
| Tamil speech → Malayalam voice | 0.733 | 0.736 | +0.002 | About the same |
| Tamil speech → Hindi voice | 0.623 | 0.697 | +0.074 | **Better** |
| Hindi speech → Malayalam voice | 0.705 | 0.759 | +0.054 | **Better** |
| Hindi speech → Tamil voice | 0.703 | 0.729 | +0.025 | **Better** |
| **All directions** | **0.688** | **0.733** | **+0.045** | **Better** |

### Still sounds like speaker A: leakage (lower is better)

| Direction | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Malayalam speech → Tamil voice | 0.314 | 0.245 | −0.069 | **Better** |
| Malayalam speech → Hindi voice | 0.271 | 0.188 | −0.082 | **Better** |
| Tamil speech → Malayalam voice | 0.304 | 0.292 | −0.012 | About the same |
| Tamil speech → Hindi voice | 0.301 | 0.128 | −0.174 | **Better** |
| Hindi speech → Malayalam voice | 0.238 | 0.108 | −0.130 | **Better** |
| Hindi speech → Tamil voice | 0.243 | 0.077 | −0.166 | **Better** |
| **All directions** | **0.279** | **0.173** | **−0.106** | **Better** |

### Clarity lost by the conversion, in % points (lower is better)

| Direction | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Malayalam speech → Tamil voice | 9.5 | 3.7 | −5.8 | **Better** |
| Malayalam speech → Hindi voice | 12.6 | 3.1 | −9.5 | **Better** |
| Tamil speech → Malayalam voice | 11.1 | 5.1 | −6.0 | **Better** |
| Tamil speech → Hindi voice | 7.3 | 4.9 | −2.4 | **Better** |
| Hindi speech → Malayalam voice | 7.1 | 1.8 | −5.3 | **Better** |
| Hindi speech → Tamil voice | 7.9 | 0.5 | −7.4 | **Better** |
| **All directions** | **9.2** | **3.1** | **−6.1** | **Better** |

The largest gains in similarity are for **Hindi voices** (+0.06 to +0.07), which were the weakest before
translation. Tamil → Malayalam stays about the same.

---

## 6. What each measure means

| Measure | What it means |
|---|---|
| Sounds like speaker B | Voice similarity (cosine of WavLM-ECAPA speaker embeddings) between the output and B's reference clip. 1 = identical voice. |
| … on a clip the model never heard | The same, against B's second-longest clip, which the model was not given. It is a stricter check. |
| Still sounds like speaker A (leakage) | Voice similarity between the output and speaker A. At the "two different people" level, none of A's voice is left. |
| Clarity lost by the conversion | Extra % of letters an independent recogniser (Meta MMS-1b-all) gets wrong after the conversion, compared with the converter's input |
| Audio quality | Microsoft DNSMOS overall speech quality, 1–5 |

**Why "clarity lost" instead of the raw error rate:** the raw error rates come from different kinds of
input. Before translation, the recogniser reads converted **spontaneous real speech**; even the
untouched recordings give 22.5 % errors. After translation, it reads converted **clean TTS speech**,
which gives 6.4 % before conversion. Subtracting each test's own starting point leaves only the part
caused by the voice conversion, and that part can be compared.

---

## 7. Points to keep in mind

- **Different sentences.** Both tests use the same 48 speakers and pick B's reference clip the same way,
  but not the same sentences or the same speaker pairs. The comparison is between averages.
- **Different sizes:** 72 conversions before, 48 after. Per-direction numbers (12 vs 8) are less certain
  than the "All directions" rows.
- **Part of the gain comes from the TTS input.** TTS speech is clean and evenly paced, which makes it
  easier to convert. In return, the output has the TTS's even, read-aloud rhythm, not A's natural way of
  speaking.
- **Speed is not compared.** Stage 5 timed the voice conversion alone on a Colab A100; Stage 6 timed the
  whole pipeline on the laptop.

---

## 8. Listening to both

| | Before translation | After translation |
|---|---|---|
| Folder | `outputs/eval_v3/examples/<pair>/` | `outputs/stage6_speech_translation_voice_conversion_results/examples/<run>/` |
| Play in this order | `source.wav` → `reference.wav` → `ours.wav` | `1_source_speech_speakerA.wav` → `2_reference_voice_speakerB.wav` → `4_converted_ours.wav` |
| You hear | A's sentence **in A's language**, in B's voice | A's sentence **translated into B's language**, in B's voice |

For the same direction, compare for example `eval_v3/examples/ml2ta_00/ours.wav` with
`stage6_…/examples/ml2ta_00/4_converted_ours.wav`. They are different sentences and speakers, but you can
hear the difference between the two situations.

---

## 9. Files

| File | Contents |
|---|---|
| `README.md` | This file |
| `comparison_tables.md` | The same tables, generated automatically by the script |
| `comparison.csv` | Every number: one row per system × test (before/after) × direction |

**To re-create:** `python code/translate/compare_before_after_translation.py`. It takes a few seconds and
only reads the two existing `results.csv` files.
