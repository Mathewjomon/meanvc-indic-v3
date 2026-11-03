"""
Stage 0 - Check that the environment is ready (run on the laptop AND on Colab).

WHAT IT CHECKS
  1. Python, PyTorch, GPU
  2. Library versions that must be pinned (transformers, jiwer, typeguard)
  3. Our fixes are applied to third_party/ (fix_upstream.py)
  4. Every MeanVC / Fast-U2++ module we will use can be imported
  5. Audio tools work: read/write WAV, Kaldi fbank (used for content features)

HOW TO RUN
  Laptop : python code/env/check_env.py
  Colab  : !python /content/drive/MyDrive/meanvc_indic_v3/code/env/check_env.py

Every line prints OK or FAIL. Send the FAIL lines if there are any.
"""

import importlib
import io
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
THIRD = PROJECT / "third_party"
# Make MeanVC ('src', 'vocos') and Fast-U2++ ('wenet') importable, like our scripts will.
for p in (THIRD / "MeanVC", THIRD / "fast-u2pp", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

results = {"ok": 0, "fail": 0}


def check(label, fn):
    """Run one check and print OK / FAIL with the reason."""
    try:
        detail = fn()
        print(f"  OK    {label}" + (f"  ({detail})" if detail else ""))
        results["ok"] += 1
    except Exception as e:
        print(f"  FAIL  {label}  ->  {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        results["fail"] += 1


def version(name):
    """Installed version of a package (works even if the package has no __version__)."""
    from importlib.metadata import version as dist_version
    return dist_version(name)


# ---------------------------------------------------------------------------------------
print("1. Python / PyTorch / GPU")
import torch  # noqa: E402

check("Python", lambda: sys.version.split()[0])
check("PyTorch", lambda: torch.__version__)
check("GPU available", lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() is False")))
check("GPU computes", lambda: f"{(torch.ones(256, 256, device='cuda') @ torch.ones(256, 256, device='cuda')).sum().item():.0f}"
      if torch.cuda.is_available() else "skipped (no GPU)")
check("NumPy works with PyTorch", lambda: f"numpy {version('numpy')}, {torch.zeros(3).numpy().tolist()}")

# ---------------------------------------------------------------------------------------
print("2. Pinned library versions")


def expect(name, want_prefix):
    v = version(name)
    if not v.startswith(want_prefix):
        raise RuntimeError(f"found {v}, need {want_prefix}*")
    return v


check("transformers == 4.46.3 (IndicTransToolkit needs it)", lambda: expect("transformers", "4.46.3"))
check("jiwer == 3.1.0 (MeanVC imports compute_measures)", lambda: expect("jiwer", "3.1.0"))
check("typeguard 2.x (fast-u2pp uses check_argument_types)", lambda: expect("typeguard", "2."))
check("torchaudio", lambda: version("torchaudio"))

# ---------------------------------------------------------------------------------------
print("3. Our fixes in third_party/")


def fixes_applied():
    import fix_upstream as fu
    pending = [f"{p.relative_to(THIRD)}" for p, old, new in fu.TEXT_FIXES
               if fu.apply_text_fix(p, old, new, check_only=True) != "already"]
    pending += [str(p.relative_to(THIRD)) for p in fu.NEW_FILES if not p.exists()]
    pending += [str(d.relative_to(THIRD)) for d in fu.package_dirs() if not (d / "__init__.py").exists()]
    if pending:
        raise RuntimeError(f"{len(pending)} not applied, e.g. {pending[0]} - run fix_upstream.py")
    return "all applied"


check("fix_upstream.py applied", fixes_applied)

# ---------------------------------------------------------------------------------------
print("4. MeanVC / Fast-U2++ imports")
IMPORTS = [
    ("DiT decoder (training)",            "src.model.backbones.dit", "DiT"),
    ("DiT decoder (inference, KV-cache)", "src.infer.dit_kvcache", "DiT"),
    ("Mean-flow model",                   "src.model.cfm_mean_flow", "MeanFlow"),
    ("DiT trainer",                       "src.model.trainer", "Trainer"),
    ("DiT dataset",                       "src.dataset.dataset", "DiffusionDataset"),
    ("Inference helpers",                 "src.infer.infer_ref", "MelSpectrogramFeatures"),
    ("Speaker encoder (ECAPA + WavLM)",   "src.runtime.speaker_verification.verification", "init_model"),
    ("Vocos vocoder",                     "vocos.pretrained", "Vocos"),
    ("Fast-U2++ ASR model",               "wenet.transformer.asr_model", "init_asr_model"),
    ("Fast-U2++ Conformer encoder",       "wenet.transformer.encoder", "ConformerEncoder"),
    ("WavLM loader (s3prl)",              "s3prl.upstream.wavlm.hubconf", "wavlm_large"),
]
for label, module, name in IMPORTS:
    check(f"{label}: {module}.{name}", lambda m=module, n=name: getattr(importlib.import_module(m), n) and "")

# The 'src' found must be MeanVC's, not some other installed package called 'src'.
check("'src' comes from third_party/MeanVC",
      lambda: "" if "third_party" in str(Path(importlib.import_module("src").__file__))
      else (_ for _ in ()).throw(RuntimeError(f"wrong src: {importlib.import_module('src').__file__}")))

# ---------------------------------------------------------------------------------------
print("5. Audio tools")


def wav_roundtrip():
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, (np.sin(np.arange(16000) / 10) * 0.3).astype("float32"), 16000, format="WAV")
    buf.seek(0)
    x, sr = sf.read(buf)
    return f"{len(x)} samples at {sr} Hz"


def kaldi_fbank():
    import torchaudio.compliance.kaldi as kaldi
    wav = torch.randn(1, 16000) * 0.1 * 32768          # MeanVC scales audio by 32768 first
    fb = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=12.5,
                     dither=0.0, energy_floor=0.0, sample_frequency=16000)
    return f"shape {tuple(fb.shape)}"


def librosa_ok():
    import librosa
    librosa.load                                         # forces librosa's lazy import now
    return librosa.__version__


check("soundfile WAV write/read", wav_roundtrip)
check("Kaldi fbank (content features)", kaldi_fbank)
check("librosa", librosa_ok)

# ---------------------------------------------------------------------------------------
print(f"\nResult: {results['ok']} OK, {results['fail']} FAIL")
if results["fail"]:
    print("Run again with --verbose for full error details.")
    sys.exit(1)
print("Environment ready.")