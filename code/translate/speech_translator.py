"""
Stage 6 - Speech translation + voice conversion (Malayalam <-> Tamil <-> Hindi).
Works on the laptop (VS Code, RTX 2050 or CPU) and on Colab. No training: pretrained models + our
already-trained MeanVC v3.

PURPOSE
  Speaker A says a sentence in Malayalam, Tamil or Hindi. The program finds the language, the user
  picks a TARGET language (one of the other two of the same three, or both), and the program outputs
  the translated sentence spoken in a chosen voice:
    --reference B.wav given  -> in person B's voice (any recording of B, any of the 3 languages)
    no --reference           -> in speaker A's own voice

THE PIPELINE (one class per step, all in this file)
    1. Recognizer + identify_language   LANGUAGE ID. Our fine-tuned Fast-U2++ encoder + CTC head reads
                                         the speech as letters; the language = the script most letters
                                         belong to (Malayalam / Tamil / Devanagari), share = confidence
                                         (99.3 % correct on the 300 test clips).
    2. IndicConformerRecognizer          SPEECH -> TEXT (default --asr indicconformer). AI4Bharat
                                         IndicConformer 600M (hybrid CTC + RNNT, trained on IndicVoices
                                         spontaneous speech), RNNT decoding, told the language from
                                         step 1. Options: --asr ours (our encoder's letters from step 1),
                                         --asr mms (Meta MMS-1b-all). Stray letters of the other two
                                         scripts are transliterated into the source script (keep_script).
    3. Translator                        TEXT -> TEXT. AI4Bharat IndicTrans2 Indic<->Indic (distilled,
                                         320M): IndicProcessor pre-processing, 5-beam search (output at
                                         most ~2x the input, no repeated 4-token sequence), post-processing.
    4. Synthesizer                       TEXT -> SPEECH. Meta MMS-TTS (VITS, one model per language),
                                         16 kHz. Right words, but a generic voice.
    5. VoiceConverter (code/infer/convert.py, unchanged)
                                         VOICE. Our MeanVC v3: content of the TTS speech + the voice of
                                         the reference (WavLM-ECAPA embedding + mel prompt) -> DiT ->
                                         Vocos -> the translated sentence in the chosen voice.

  Why only ml / ta / hi: our content encoder and DiT are fine-tuned (and evaluated) on these three
  languages only; any other language would be untested input for the voice converter.

ROLE IN THE PROJECT
  Reuses, without changing them: convert.py (VoiceConverter, local model copies on the laptop),
  train_encoder.py (model building, fbank, streaming chunks, CTC decoding) and vocab.py.
  evaluate_translation.py (same folder) imports SpeechTranslator to test the pipeline end to end.

HOW TO RUN (laptop, VS Code terminal with the venv active)
  python "G:\\My Drive\\meanvc_indic_v3\\code\\translate\\speech_translator.py" --input A.wav --reference B.wav
    (no --target: the language is detected, then a menu offers only the valid target languages)
    --target ta | hi | ml   translate into this language          --target all   both other languages
    --reference B.wav       the voice to use (default: speaker A's own voice)
    --source ml             skip language identification
    --asr ours | mms        another speech recogniser (default: indicconformer)
  Models are downloaded once to D:\\hf_cache (see code/translate/README.md for set-up).

  from Python:  from speech_translator import SpeechTranslator
                st = SpeechTranslator()
                r = st.run(load_wav("A.wav"), target="hi", reference_wav=load_wav("B.wav"))

OUTPUT  outputs/translate/<input>_to_<target>/
          input.wav       speaker A's original speech
          reference.wav   the voice that was copied (B, or A itself)
          tts.wav         the translated sentence in the TTS voice (before voice conversion)
          output.wav      the translated sentence in the chosen voice   <- the result
          result.json     language + confidence, transcript, translation, timings, voice similarities
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if os.name == "nt" and Path("D:/").exists():                 # laptop: C: is almost full
    os.environ.setdefault("HF_HUB_CACHE", "D:/hf_cache/hub")
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent                       # code/translate
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "infer"))
import convert as cv  # noqa: E402  VoiceConverter, load_wav, save_wav, local(); sets up all other paths
import train_encoder as te  # noqa: E402  build_model, compute_fbank, run_encoder, greedy_ctc_text
from vocab import load_vocab  # noqa: E402

PROJECT = cv.PROJECT
ENC_DIR = PROJECT / "checkpoints" / "encoder_v3"
ASR_CKPT = ENC_DIR / "best.pt"            # encoder + CTC head (encoder_best.pt has the encoder only)
MT_MODEL = "ai4bharat/indictrans2-indic-indic-dist-320M"
MMS_ASR_MODEL = "facebook/mms-1b-all"
IC_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
ASR_CHOICES = ("indicconformer", "ours", "mms")
SR = cv.SR

# The three languages: FLORES code (IndicTrans2), ISO 639-3 (MMS), Unicode script name (language ID)
LANGS = {
    "ml": dict(name="Malayalam", flores="mal_Mlym", mms="mal", script="MALAYALAM"),
    "ta": dict(name="Tamil", flores="tam_Taml", mms="tam", script="TAMIL"),
    "hi": dict(name="Hindi", flores="hin_Deva", mms="hin", script="DEVANAGARI"),
}
MIN_CONFIDENCE = 0.6                      # below this share of characters: ask the user to confirm


# ======================================================================================
# 1. Language identification + speech recognition
# ======================================================================================
def identify_language(text):
    """Script voting: each recognised letter votes for the script it belongs to.
    -> (language code or None, confidence = winning share, {code: count})."""
    counts = Counter()
    for ch in text:
        name = unicodedata.name(ch, "")
        for code, info in LANGS.items():
            if name.startswith(info["script"]):
                counts[code] += 1
    total = sum(counts.values())
    if not total:
        return None, 0.0, {c: 0 for c in LANGS}
    code, n = counts.most_common(1)[0]
    return code, n / total, {c: counts.get(c, 0) for c in LANGS}


def keep_script(text, code):
    """Rewrite letters of the OTHER two scripts in the detected language's script. Our recogniser
    knows all three alphabets and sometimes writes a few letters of a word in the wrong one
    (e.g. 'கடவിൽ' = Tamil 'கட' + Malayalam 'വിൽ'). The Indic scripts map letter by letter
    (same Unicode layout), so transliterating gives back the intended word ('കടവിൽ');
    deleting the letters would leave broken fragments. Anything that cannot be mapped is dropped."""
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
    for other in LANGS:
        if other != code:
            text = UnicodeIndicTransliterator.transliterate(text, other, code)
    others = tuple(LANGS[c]["script"] for c in LANGS if c != code)
    text = "".join(ch for ch in text if not unicodedata.name(ch, "").startswith(others))
    if code == "ml":                       # a virama after a chillu letter (ൺ ൻ ർ ൽ ൾ ൿ) is never valid
        text = re.sub("([ൺ-ൿ])്", r"\1", text)
    return " ".join(text.split())


class Recognizer:
    """Our fine-tuned Fast-U2++ encoder + CTC head (greedy decoding), 200 ms streaming chunks -
    the same reader as the 'cer_ours' column of evaluate.py, here also used for language ID."""

    def __init__(self, device):
        vocab = load_vocab()
        self.id2tok, self.device = {i: t for t, i in vocab.items()}, device
        model = te.build_model(len(vocab), ENC_DIR / "cmvn_from_pretrained.json")
        model.load_state_dict(torch.load(cv.local(ASR_CKPT), map_location="cpu"))   # strict: all keys
        te.set_streaming(model)
        self.model = model.to(device).eval()

    @torch.inference_mode()
    def transcribe(self, wav):
        feats = te.compute_fbank(wav).unsqueeze(0).to(self.device)
        lens = torch.tensor([feats.shape[1]], device=self.device)
        enc, _, elens = te.run_encoder(self.model, feats, lens, te.CHUNK, te.LEFT_CHUNKS)
        return te.greedy_ctc_text(self.model, enc, elens, self.id2tok)[0]   # <unk> already dropped


class MmsRecognizer:
    """Meta MMS-1b-all (wav2vec 2.0, CTC, per-language adapters). Half precision on the GPU so it fits
    next to the other models on a 4 GB laptop GPU."""

    def __init__(self, device):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(MMS_ASR_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(MMS_ASR_MODEL, torch_dtype=dtype).to(device).eval()
        self.device, self.dtype, self.lang = device, dtype, None

    @torch.inference_mode()
    def transcribe(self, wav, code):
        if code != self.lang:                                  # switch the language adapter
            self.proc.tokenizer.set_target_lang(LANGS[code]["mms"])
            self.model.load_adapter(LANGS[code]["mms"])
            self.model.to(self.device, self.dtype)
            self.lang = code
        x = self.proc(wav, sampling_rate=SR, return_tensors="pt")
        logits = self.model(x.input_values.to(self.device, self.dtype)).logits
        return self.proc.decode(logits.argmax(-1)[0])


class IndicConformerRecognizer:
    """AI4Bharat IndicConformer 600M multilingual: a Conformer with both a CTC and an RNNT head,
    trained on IndicVoices (the same kind of spontaneous speech as our data). We use RNNT decoding
    (the more accurate head). It runs through ONNX Runtime on the CPU, so the GPU stays free for
    translation, TTS and voice conversion. It must be told the language (from our language ID)."""

    def __init__(self, decoding="rnnt"):
        # The model's own from_pretrained() contacts Hugging Face on every start (and the model is
        # gated), so we load its code and files straight from the downloaded copy instead.
        import importlib.util
        from huggingface_hub import snapshot_download
        try:
            folder = snapshot_download(IC_MODEL, local_files_only=True)
        except Exception:
            raise SystemExit(f"IndicConformer is not downloaded yet - see code/translate/README.md (set-up), "
                             f"or use --asr ours")
        spec = importlib.util.spec_from_file_location("indicconformer_model", Path(folder) / "model_onnx.py")
        code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(code)
        self.model = code.IndicASRModel(code.IndicASRConfig(ts_folder=folder))
        self.decoding = decoding

    @torch.inference_mode()
    def transcribe(self, wav, code):
        x = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)    # (1, samples), 16 kHz
        out = self.model(x, code, self.decoding)
        return out[0] if isinstance(out, (list, tuple)) else out


# ======================================================================================
# 2. Translation
# ======================================================================================
class Translator:
    """IndicTrans2 Indic<->Indic distilled model (320M), direct ml/ta/hi translation (no English pivot)."""

    def __init__(self, device):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:                                                   # official toolkit (Colab, Linux)
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:                                    # Windows laptop: same algorithm, pure Python
            from indic_processor import IndicProcessor
        self.ip = IndicProcessor(inference=True)
        self.tok = AutoTokenizer.from_pretrained(MT_MODEL, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, trust_remote_code=True).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def translate(self, texts, src, tgt, beams=5):
        """texts: list of sentences in language src -> list of sentences in language tgt.
        Two guards against the 'stuck in a loop' failure on noisy speech-recognition text (seen in
        testing: one word repeated up to the 256-token limit): the output may be at most about twice
        as long as the input, and no 4-token sequence may repeat."""
        s, t = LANGS[src]["flores"], LANGS[tgt]["flores"]
        batch = self.ip.preprocess_batch(list(texts), src_lang=s, tgt_lang=t)
        x = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                     return_attention_mask=True).to(self.device)
        max_len = min(256, 2 * x["input_ids"].shape[1] + 10)
        out = self.model.generate(**x, use_cache=True, min_length=0, max_length=max_len, num_beams=beams,
                                  num_return_sequences=1, no_repeat_ngram_size=4)
        with self.tok.as_target_tokenizer():
            decoded = self.tok.batch_decode(out.cpu().tolist(), skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(decoded, lang=t)


# ======================================================================================
# 3. Text-to-speech
# ======================================================================================
class Synthesizer:
    """Meta MMS-TTS (VITS: text encoder + duration predictor + flow + HiFi-GAN decoder), 16 kHz.
    One model per language, loaded the first time it is needed."""

    def __init__(self, device):
        self.device, self.models, self._uroman = device, {}, None

    def load(self, code):
        if code not in self.models:
            from transformers import AutoTokenizer, VitsModel
            name = f"facebook/mms-tts-{LANGS[code]['mms']}"
            self.models[code] = (AutoTokenizer.from_pretrained(name),
                                 VitsModel.from_pretrained(name).to(self.device).eval())
        return self.models[code]

    def _romanize(self, text):
        if self._uroman is None:
            import uroman
            self._uroman = uroman.Uroman()
        return self._uroman.romanize_string(text)

    @torch.inference_mode()
    def synthesize(self, text, code, seed=0):
        tok, model = self.load(code)
        if getattr(tok, "is_uroman", False):                   # some MMS voices read romanised text
            text = self._romanize(text)
        x = tok(text, return_tensors="pt").to(self.device)
        if x.input_ids.shape[1] == 0:
            raise ValueError(f"TTS: none of the characters in {text!r} are known to the {code} voice")
        torch.manual_seed(seed)                                # VITS samples noise: same seed -> same audio
        wav = model(**x).waveform[0].float().cpu().numpy()
        assert model.config.sampling_rate == SR, model.config.sampling_rate
        peak = float(np.abs(wav).max())
        return wav * (0.95 / peak) if peak > 0.95 else wav


# ======================================================================================
# The whole pipeline
# ======================================================================================
def other_languages(source):
    """The valid targets for a source language: the other two of ml / ta / hi."""
    return [c for c in LANGS if c != source]


class SpeechTranslator:
    """Loads every model once; recognise() / speak() / run() can then be called many times.
    asr: 'indicconformer' (default), 'ours' or 'mms'. A recogniser or voice converter that is
    already loaded can be passed in (the evaluation does this to avoid loading a model twice)."""

    def __init__(self, device=None, asr="indicconformer", vc_system="ours", mt_device=None,
                 voice_converter=None, asr_model=None):
        if asr not in ASR_CHOICES:
            raise ValueError(f"asr must be one of {ASR_CHOICES}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        timer = time.time()
        self.recognizer = Recognizer(self.device)                      # always: language ID
        self.asr = asr
        if asr_model is not None:
            self.asr_model = asr_model
        elif asr == "indicconformer":
            self.asr_model = IndicConformerRecognizer()
        elif asr == "mms":
            self.asr_model = MmsRecognizer(self.device)
        else:
            self.asr_model = None                                      # 'ours': letters from language ID
        self.translator = Translator(mt_device or self.device)
        self.synthesizer = Synthesizer(self.device)
        self.vc = voice_converter or cv.VoiceConverter(vc_system, self.device)
        self.load_seconds = time.time() - timer

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _timed(self, times, key, fn, *args, **kwargs):
        self._sync()
        t0 = time.time()
        out = fn(*args, **kwargs)
        self._sync()
        times[key] = time.time() - t0
        return out

    # ---- steps 1 + 2: language ID and transcript (once per input) ----
    def recognise(self, wav, source=None, confirm=None):
        """-> dict(source, lid, asr, transcript, transcript_ours, times).
        confirm(code, confidence, counts) -> code is called when language ID is unsure."""
        times = {}
        our_text = self._timed(times, "language_id", self.recognizer.transcribe, wav)
        code, conf, counts = identify_language(our_text)
        lid = dict(detected=code, confidence=conf, script_counts=counts)
        if source is None:
            source = code
            if (code is None or conf < MIN_CONFIDENCE) and confirm is not None:
                source = confirm(code, conf, counts)
        if source not in LANGS:
            raise ValueError("Could not identify the language (no Malayalam/Tamil/Hindi letters "
                             "recognised). Give the source language with --source.")
        text = our_text
        if self.asr_model is not None:
            text = self._timed(times, f"asr_{self.asr}", self.asr_model.transcribe, wav, source)
        text = keep_script(text, source)
        if not text.strip():
            raise ValueError("Speech recognition returned no text - is the input speech?")
        return dict(source=source, lid=lid, asr=self.asr, transcript=text, transcript_ours=our_text,
                    times=times)

    # ---- step 5 input: the voice to copy (once per reference) ----
    def voice(self, reference_wav):
        """Speaker embedding + mel prompt of the voice to copy (B's clip, or A's own speech)."""
        return self.vc.voice(reference_wav)

    # ---- steps 3-5: one target language ----
    def speak(self, rec, target, voice, seed=0):
        """rec: output of recognise(). -> dict(target, translation, tts_wav, output_wav, times)."""
        if target not in LANGS:
            raise ValueError(f"target must be one of {list(LANGS)} (the languages our voice converter knows)")
        if target == rec["source"]:
            raise ValueError(f"The speech is already in {LANGS[target]['name']}: pick "
                             f"{' or '.join(LANGS[c]['name'] for c in other_languages(target))}.")
        times = {}
        translation = self._timed(times, "translate", self.translator.translate,
                                  [rec["transcript"]], rec["source"], target)[0]
        self.synthesizer.load(target)                          # loading is not part of the timing
        tts_wav = self._timed(times, "tts", self.synthesizer.synthesize, translation, target, seed)
        out_wav, _ = self._timed(times, "voice_conversion", self.vc.convert, tts_wav, voice=voice, seed=seed)
        return dict(target=target, translation=translation, tts_wav=tts_wav, output_wav=out_wav, times=times)

    # ---- everything for one target ----
    def run(self, wav, target, source=None, reference_wav=None, seed=0, confirm=None):
        """wav: speaker A, 16 kHz mono float32 (cv.load_wav). reference_wav: the voice to use
        (person B); None = A's own voice. -> dict with every intermediate result."""
        if target not in LANGS:
            raise ValueError(f"target must be one of {list(LANGS)} (the languages our voice converter knows)")
        rec = self.recognise(wav, source, confirm)
        times = dict(rec["times"])
        voice = self._timed(times, "voice_reference", self.voice, wav if reference_wav is None else reference_wav)
        out = self.speak(rec, target, voice, seed)
        times.update(out["times"])
        times["total"] = sum(times.values())
        return {**rec, **out, "times": times, "voice_from": "input" if reference_wav is None else "reference",
                "input_seconds": len(wav) / SR, "output_seconds": len(out["output_wav"]) / SR,
                "rtf": times["total"] / max(len(wav) / SR, 1e-6)}


# ======================================================================================
# Command line
# ======================================================================================
def interactive():
    return sys.stdin.isatty()


def ask(prompt, no_answer):
    """input() that stops with a clear message when nobody can answer (e.g. run from a script)."""
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit(f"\n{no_answer}")


def ask_source(code, conf, counts):
    """Language ID unsure -> show what was found and let the user pick (Enter keeps the guess)."""
    shown = ", ".join(f"{LANGS[c]['name']} {n}" for c, n in counts.items())
    guess = LANGS[code]["name"] if code else "unknown"
    print(f"  Language detection is unsure: {guess} ({conf:.0%} of letters; letters per script: {shown}).")
    if not interactive():
        return code
    codes = list(LANGS)
    menu = "  ".join(f"{i}) {LANGS[c]['name']}" for i, c in enumerate(codes, 1))
    while True:
        answer = ask(f"  Source language? {menu}  [Enter = {guess}]: ",
                     "No answer possible here: give the source language with --source ml / ta / hi.")
        if not answer and code:
            return code
        if answer in {str(i) for i in range(1, len(codes) + 1)}:
            return codes[int(answer) - 1]
        print("  Please type one of the numbers shown.")


def choose_targets(source, given):
    """--target given -> check it; otherwise show a menu with only the valid choices."""
    valid = other_languages(source)
    if given == "all":
        return valid
    if given:
        if given == source:
            raise SystemExit(f"The speech is already in {LANGS[source]['name']}: use --target "
                             f"{' or '.join(valid)} (or all).")
        return [given]
    if not interactive():
        raise SystemExit(f"Add --target ({' / '.join(valid)} / all): no terminal to ask in.")
    print("\n  Translate into which language?")
    options = [[c] for c in valid] + [valid]
    for i, opt in enumerate(options, 1):
        print(f"    {i}) " + ("Both (" + " + ".join(LANGS[c]["name"] for c in opt) + ")" if len(opt) > 1
                              else LANGS[opt[0]]["name"]))
    while True:
        answer = ask(f"  Choose 1-{len(options)}: ",
                     f"No answer possible here: add --target ({' / '.join(valid)} / all).")
        if answer in {str(i) for i in range(1, len(options) + 1)}:
            return options[int(answer) - 1]
        print("  Please type one of the numbers shown.")


def main():
    ap = argparse.ArgumentParser(description="Translate ml/ta/hi speech and speak it in a chosen voice")
    ap.add_argument("--input", required=True, help="speaker A's speech (wav/flac/...), 3-15 s works best")
    ap.add_argument("--reference", default=None,
                    help="person B's speech: the voice to use (default: speaker A's own voice)")
    ap.add_argument("--target", choices=list(LANGS) + ["all"], default=None,
                    help="ml / ta / hi / all (default: ask after detecting the language)")
    ap.add_argument("--source", choices=list(LANGS), default=None, help="skip language identification")
    ap.add_argument("--asr", choices=ASR_CHOICES, default="indicconformer", help="speech recogniser")
    ap.add_argument("--out", default=None, help="folder for the results (default: outputs/translate/)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading models ...", flush=True)
    st = SpeechTranslator(args.device, args.asr)
    print(f"  done in {st.load_seconds:.0f} s on {st.device} (voice converter: {st.vc.describe()})")
    wav = cv.load_wav(args.input)
    ref = cv.load_wav(args.reference) if args.reference else None

    rec = st.recognise(wav, args.source, confirm=ask_source)
    lid = rec["lid"]
    how = (f"detected, {lid['confidence']:.0%} of letters" if args.source is None and lid["detected"] == rec["source"]
           else "given")
    print(f"\nStep 1  Language     : {LANGS[rec['source']]['name']} ({how})")
    print(f"Step 2  Recognised   : {rec['transcript']}   [{rec['asr']}]")
    targets = choose_targets(rec["source"], args.target)

    times_voice = {}
    voice = st._timed(times_voice, "voice_reference", st.voice, wav if ref is None else ref)
    who = f"the voice of {Path(args.reference).name}" if ref is not None else "the speaker's own voice (no --reference)"
    e_in = st.vc.embed(wav)
    e_ref = st.vc.embed(ref) if ref is not None else e_in

    out_root = Path(args.out) if args.out else PROJECT / "outputs" / "translate"
    for k, target in enumerate(targets):
        r = st.speak(rec, target, voice, args.seed)
        out = out_root / f"{Path(args.input).stem}_to_{target}"
        cv.save_wav(out / "input.wav", wav)
        cv.save_wav(out / "reference.wav", wav if ref is None else ref)
        cv.save_wav(out / "tts.wav", r["tts_wav"])
        cv.save_wav(out / "output.wav", r["output_wav"])
        e_tts, e_out = st.vc.embed(r["tts_wav"]), st.vc.embed(r["output_wav"])
        sims = dict(tts_vs_voice=cv.similarity(e_tts, e_ref), output_vs_voice=cv.similarity(e_out, e_ref))
        if ref is not None:
            sims.update(input_vs_voice=cv.similarity(e_in, e_ref), output_vs_input_speaker=cv.similarity(e_out, e_in))
        times = {**rec["times"], **(times_voice if k == 0 else {}), **r["times"]}
        times["total"] = sum(times.values())
        info = {key: v for key, v in {**rec, **r}.items() if not key.endswith("_wav") and key != "times"}
        info.update(input=str(args.input), reference=str(args.reference) if ref is not None else None,
                    similarity=sims, times=times, input_seconds=len(wav) / SR,
                    output_seconds=len(r["output_wav"]) / SR)
        (out / "result.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n--- {LANGS[rec['source']]['name']} -> {LANGS[target]['name']} ---")
        print(f"Step 3  Translation  : {r['translation']}")
        print(f"Step 4  TTS          : {len(r['tts_wav']) / SR:.1f} s of {LANGS[target]['name']} speech")
        print(f"Step 5  Voice        : {who}")
        print(f"        similarity to that voice: TTS {sims['tts_vs_voice']:.3f} -> after conversion "
              f"{sims['output_vs_voice']:.3f} (higher = closer)")
        if ref is not None:
            print(f"        similarity to speaker A: {sims['output_vs_input_speaker']:.3f} "
                  f"(A vs B before: {sims['input_vs_voice']:.3f}; lower = less of A's voice left)")
        print("        time: " + ", ".join(f"{key} {v:.2f} s" for key, v in times.items()))
        print(f"Saved   : {out}  (input.wav, reference.wav, tts.wav, output.wav, result.json)")


if __name__ == "__main__":
    main()
