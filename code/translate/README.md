# code/translate: speech translation + voice conversion (Stage 6)

This folder turns a sentence spoken in **Malayalam, Tamil or Hindi** into the same sentence in another
of these languages, spoken in a **chosen voice**:

- **speaker B's voice**, when a recording of B is given (`--reference`), or
- **the original speaker A's own voice**, when no reference is given.

No model is trained here. The steps use pretrained models plus our MeanVC v3, already fine-tuned in
Stages 2–4.

```
speaker A's speech ─► 1 language ID ─► 2 speech → text ─► 3 translation ─► 4 text → speech ─► 5 voice conversion ─► output
                      (our encoder)    (IndicConformer)   (IndicTrans2)    (MMS-TTS)          (our MeanVC v3)
                                                              ▲                                     ▲
                                                 user picks the target language          speaker B's speech (voice)
```

Only **ml, ta and hi** are supported, as both source and target. Our voice converter is fine-tuned
and tested on these three languages only.

---

## Files

| File | What it is | Run it? |
|---|---|---|
| `speech_translator.py` | **The pipeline.** One class per step (`Recognizer`, `IndicConformerRecognizer`, `MmsRecognizer`, `Translator`, `Synthesizer`), with `SpeechTranslator` joining them. Includes the command-line tool. | Yes: translate your own recordings |
| `evaluate_translation.py` | **The test.** Compares three speech recognisers on all 300 test clips, then runs the full pipeline on 48 translations (test speakers never seen in training) and writes the results folder. | Yes: to reproduce the results |
| `compare_before_after_translation.py` | **The comparison.** Puts the Stage 5 (voice conversion only) and Stage 6 (after translation) results side by side and writes `outputs/results_comparison_before_and_after_translation/`. Seconds, no GPU. | Yes: after either evaluation is re-run |
| `plot_results.py` | **The figures.** Draws the Stage 6 results as PNG charts and a tables image in `outputs/stage6_…/figures/`. Seconds, no GPU. | Yes: after the evaluation is re-run |
| `indic_processor.py` | Text preparation for IndicTrans2: a pure-Python copy of AI4Bharat's IndicProcessor. It is used only when the official `IndicTransToolkit` cannot be installed (Windows without a C compiler). | No: imported automatically |
| `README.md` | This file | |

Every `.py` file starts with a comment block: purpose, what it does, its role, and how to run it.

### Code it reuses (unchanged)

| From | What |
|---|---|
| `code/infer/convert.py` | `VoiceConverter` (MeanVC v3), `load_wav`, `save_wav`, local model copies on the laptop |
| `code/infer/evaluate.py` | Character-error counting and DNSMOS (evaluation only) |
| `code/encoder/train_encoder.py`, `vocab.py` | Building our encoder, fbank features, streaming chunks, CTC decoding, text cleaning |

---

## The five steps

| # | Step | Model | Notes |
|---|---|---|---|
| 1 | Language ID | Our Fast-U2++ encoder + CTC head (`checkpoints/encoder_v3/best.pt`) | Reads the speech as letters. Each letter votes for its script (Malayalam / Tamil / Devanagari); the majority wins. Below 60 % confidence, the user is asked. |
| 2 | Speech → text | **AI4Bharat IndicConformer 600M** (default), RNNT decoding | Trained on IndicVoices, the same kind of spontaneous speech as our data. Runs on the CPU (ONNX Runtime). Options: `--asr ours` (the letters from step 1), `--asr mms` (Meta MMS-1b-all). Stray letters of another script are transliterated into the source script. |
| 3 | Translation | AI4Bharat IndicTrans2 Indic↔Indic (distilled, 320M) | Direct ml↔ta↔hi, no English in between. 5-beam search, with the output limited to about twice the input and no repeated 4-token sequences (stops loops on noisy text). |
| 4 | Text → speech | Meta MMS-TTS (`mms-tts-mal/tam/hin`, VITS) | The right words in a generic voice, 16 kHz |
| 5 | Voice conversion | Our MeanVC v3 (`encoder_v3` + `dit_v3/dit_best.pt`), via `convert.py` | Content from the TTS speech, voice from the reference (speaker B, or A) |

---

## Running it

The VS Code terminal, with the venv active:

```powershell
# translate A's sentence; the language is detected, then you choose the target from a menu
python "G:\My Drive\meanvc_indic_v3\code\translate\speech_translator.py" --input A.wav --reference B.wav

# choose the target in the command (no menu)
python ...\speech_translator.py --input A.wav --reference B.wav --target hi
python ...\speech_translator.py --input A.wav --reference B.wav --target all     # both other languages

# keep A's own voice (no --reference)
python ...\speech_translator.py --input A.wav --target ta
```

| Option | Meaning | Default |
|---|---|---|
| `--input` | Speaker A's speech (3–15 s works best) | required |
| `--reference` | Speaker B's speech: the voice to use | A's own voice |
| `--target` | `ml`, `ta`, `hi` or `all` (both other languages) | ask with a menu |
| `--source` | Skip language detection | detected |
| `--asr` | `indicconformer`, `ours` or `mms` | `indicconformer` |
| `--out` | Results folder | `outputs/translate/` |
| `--device` | `cuda` or `cpu` | `cuda` if available |

What you see without `--target` (Malayalam detected):

```
Step 1  Language     : Malayalam (detected, 97% of letters)
Step 2  Recognised   : ...                                   [indicconformer]

  Translate into which language?
    1) Tamil
    2) Hindi
    3) Both (Tamil + Hindi)
  Choose 1-3:
```

The menu only offers the two other languages. The same language as the source, or any language outside
ml/ta/hi, is refused with a clear message. When no terminal is attached (e.g. called from a script),
`--target` must be given.

**Output**, per target language, in `outputs/translate/<input>_to_<target>/`:

| File | Contents |
|---|---|
| `input.wav` | Speaker A's original speech |
| `reference.wav` | The voice that was copied (B, or A) |
| `tts.wav` | The translation in the generic TTS voice (before voice conversion) |
| `output.wav` | **The result**: the translation in the chosen voice |
| `result.json` | Language and confidence, transcript, translation, time per step, voice similarities |

**From Python:**

```python
import speech_translator as st
p = st.SpeechTranslator()                                  # loads every model once
r = p.run(st.cv.load_wav("A.wav"), target="hi", reference_wav=st.cv.load_wav("B.wav"))
st.cv.save_wav("out.wav", r["output_wav"])                 # r also holds lid, transcript, translation, times
```

---

## Testing it

```powershell
python "G:\My Drive\meanvc_indic_v3\code\translate\evaluate_translation.py"          # full, ~35 min on the laptop
python "G:\My Drive\meanvc_indic_v3\code\translate\evaluate_translation.py" --n 1    # quick check
```

Results go to `outputs/stage6_speech_translation_voice_conversion_results/`. That folder has its own
README explaining every table and file.

---

## Set-up (once)

1. **Python packages.** These are pure Python, and torch / transformers 4.46.3 stay as they are:
   ```powershell
   D:\venvs\meanvc\Scripts\python.exe -m pip install indic-nlp-library-itt regex sacremoses uroman sacrebleu speechmos rapidfuzz "onnxruntime==1.20.1"
   ```
   `onnxruntime` 1.20.1 is the version IndicConformer asks for. DNSMOS works with it too.
2. **Gated models.** IndicTrans2 and IndicConformer need their terms accepted once, logged in to
   Hugging Face:
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M
   - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
3. **Download** (to D:, because C: is nearly full). The saved login is in `C:\Users\<you>\.cache\huggingface\token`:
   ```powershell
   $env:HF_TOKEN_PATH = "$env:USERPROFILE\.cache\huggingface\token"
   $env:HF_HUB_CACHE  = "D:\hf_cache\hub"
   D:\venvs\meanvc\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; [d(r) for r in ['ai4bharat/indictrans2-indic-indic-dist-320M','ai4bharat/indic-conformer-600m-multilingual','facebook/mms-tts-mal','facebook/mms-tts-tam','facebook/mms-tts-hin']]"
   # MMS-1b-all (evaluation / --asr mms): only the main model + the 3 language adapters, not all ~1,100
   D:\venvs\meanvc\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; d('facebook/mms-1b-all', allow_patterns=['*.json','*.txt','model.safetensors','adapter.mal.safetensors','adapter.tam.safetensors','adapter.hin.safetensors'])"
   ```
   `speech_translator.py` reads models from `D:\hf_cache\hub` automatically on the laptop. After the
   download, no login is needed.

**Disk:** about 7 GB in `D:\hf_cache` (IndicConformer 2.6 GB, MMS-1b-all 3.9 GB, IndicTrans2 1.3 GB, MMS-TTS 3 × 0.15 GB).
**GPU:** fits the laptop's 4 GB RTX 2050. IndicConformer runs on the CPU, and MMS-1b-all is only loaded for evaluation.
