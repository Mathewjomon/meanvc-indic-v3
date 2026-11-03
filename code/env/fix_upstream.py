"""
Stage 0 - Apply all our fixes to the original MeanVC and Fast-U2++ code in third_party/.

WHY
  The original code was written for the authors' own servers and older library versions.
  In v1/v2 we fixed the same problems by hand in every Colab session and again on Windows.
  This script applies every fix automatically, so the laptop and Colab use identical code.

WHAT IT FIXES
  MeanVC (third_party/MeanVC)
    1. 'src' import errors   - adds __init__.py files so Python always finds MeanVC's 'src'
                               (the unexplained "No module named 'src'" on Windows in v2)
    2. dit_kvcache.py        - 'from modules import' only worked from one folder; now a full path
    3. run_wer.py            - Mandarin-only tools (jiwer/zhon/zhconv/funasr) made optional
    4. trainer.py (+ _dis)   - 3 real bugs: stray print(self.model.device) crash, wrong '*' in
                               DiffusionDataset(...), validate() using the authors' private paths;
                               plus the Mandarin-only 'zhconv' import made optional
    5. ecapa_tdnn.py (x3)    - WavLM loaded directly instead of through torch.hub, which imports
                               every s3prl model and crashes on newer torchaudio
  Fast-U2++ (third_party/fast-u2pp)
    6. wenet/cnn             - the code imports a 'wenet.cnn' module that is missing from the repo;
                               we add small placeholder classes (our model never uses them)
    7. __init__.py files     - same reason as 1
    8. wav_distortion.py     - old torchaudio call made safe

HOW TO RUN (once, on the laptop; Colab sees the fixed files through Drive)
  python code/env/fix_upstream.py           apply the fixes (safe to run again)
  python code/env/fix_upstream.py --check   only report what is applied
"""

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]       # .../meanvc_indic_v3
THIRD = PROJECT / "third_party"
MEANVC = THIRD / "MeanVC"
FASTU2 = THIRD / "fast-u2pp"
TAG = "[meanvc-indic fix]"                           # marks every line we changed


# --------------------------------------------------------------------------------------
# Text fixes: (file, original text, fixed text). Each original must appear exactly once.
# --------------------------------------------------------------------------------------
ECAPA_OLD = "                self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)\n"
ECAPA_NEW = (
    f"                # {TAG} torch.hub imports every s3prl model (some crash on new torchaudio);\n"
    "                # load WavLM-Large directly instead.\n"
    "                if feat_type == 'wavlm_large':\n"
    "                    from s3prl.upstream.wavlm.hubconf import wavlm_large\n"
    "                    self.feature_extract = wavlm_large()\n"
    "                else:\n"
    "                    self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)\n"
)

TRAINER_FIXES = [
    ("import zhconv\n",
     f"try:  # {TAG} Mandarin-only, used only in validate()\n"
     "    import zhconv\n"
     "except ImportError:\n"
     "    zhconv = None\n"),
    ("DiffusionDataset(*DiffusionDataset.init_data(",
     "DiffusionDataset(DiffusionDataset.init_data("),
    ("    def validate(self, step):\n        self.model.eval()\n",
     "    def validate(self, step):\n"
     f"        # {TAG} upstream validation uses the authors' private Mandarin paths and models.\n"
     "        # Disabled here; our own validation is done in our training code.\n"
     "        return\n"
     "        self.model.eval()\n"),
    ("        print(self.model.device)\n",
     f"        # print(self.model.device)  # {TAG} crashes: nn.Module has no .device\n"),
]

TEXT_FIXES = [
    (MEANVC / "src/infer/dit_kvcache.py",
     "from modules import (\n",
     f"from src.infer.modules import (  # {TAG} was a bare 'modules' import\n"),
    (MEANVC / "src/eval/run_wer.py",
     "from jiwer import compute_measures\nfrom zhon.hanzi import punctuation\n",
     f"# {TAG} Mandarin-only evaluation tools made optional (only used by upstream validate())\n"
     "try:\n    from jiwer import compute_measures\nexcept ImportError:\n    compute_measures = None\n"
     "try:\n    from zhon.hanzi import punctuation\nexcept ImportError:\n    punctuation = ''\n"),
    (MEANVC / "src/eval/run_wer.py",
     "import zhconv\nfrom funasr import AutoModel\n",
     "try:\n    import zhconv\nexcept ImportError:\n    zhconv = None\n"
     "try:\n    from funasr import AutoModel\nexcept ImportError:\n    AutoModel = None\n"),
    *[(MEANVC / "src/model/trainer.py", old, new) for old, new in TRAINER_FIXES],
    *[(MEANVC / "src/model/trainer_dis.py", old, new) for old, new in TRAINER_FIXES],
    (MEANVC / "src/eval/ecapa_tdnn.py", ECAPA_OLD, ECAPA_NEW),
    (MEANVC / "src/preprocess/models/ecapa_tdnn.py", ECAPA_OLD, ECAPA_NEW),
    (MEANVC / "src/runtime/speaker_verification/ecapa_tdnn.py", ECAPA_OLD, ECAPA_NEW),
    (FASTU2 / "wenet/dataset/wav_distortion.py",
     'torchaudio.set_audio_backend("sox_io")\n',
     f'try:  # {TAG} removed in newer torchaudio\n'
     '    torchaudio.set_audio_backend("sox_io")\n'
     'except Exception:\n'
     '    pass\n'),
]

# --------------------------------------------------------------------------------------
# New files: placeholder for the missing wenet.cnn module.
# --------------------------------------------------------------------------------------
_STUB = (
    f'"""{TAG} Placeholder: fast-u2pp imports this module, but it is missing from the repo.\n'
    'Our 7-layer Conformer never uses it; it only has to exist so the import works."""\n\n\n'
    "class {name}:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise NotImplementedError('{name} is a placeholder (not used by MeanVC-Indic)')\n"
)
NEW_FILES = {
    FASTU2 / "wenet/cnn/__init__.py": f"# {TAG} placeholder package\n",
    FASTU2 / "wenet/cnn/encoder.py": _STUB.format(name="ConformerCnnEncoder"),
    FASTU2 / "wenet/cnn/decoder.py": _STUB.format(name="BiTransformerCnnDecoder"),
}


def package_dirs():
    """Every folder with Python files under MeanVC/src and fast-u2pp/wenet (they lack __init__.py)."""
    dirs = set()
    for root in (MEANVC / "src", FASTU2 / "wenet"):
        dirs.add(root)
        for py in root.rglob("*.py"):
            p = py.parent
            while p != root.parent:          # include every parent folder up to the root
                dirs.add(p)
                p = p.parent
    return sorted(dirs)


# --------------------------------------------------------------------------------------
def apply_text_fix(path, old, new, check_only):
    """Returns 'applied', 'already', 'pending' (check mode) or an error message."""
    if not path.exists():
        return f"FAILED - file not found: {path}"
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"          # keep the file's own line endings
    old_n, new_n = old.replace("\n", nl), new.replace("\n", nl)
    if new_n in raw:
        return "already"
    count = raw.count(old_n)
    if count != 1:
        return f"FAILED - expected the original text once, found it {count} times"
    if check_only:
        return "pending"
    path.write_bytes(raw.replace(old_n, new_n).encode("utf-8"))
    return "applied"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="only report, change nothing")
    args = ap.parse_args()

    for d in (MEANVC, FASTU2):
        if not d.exists():
            raise SystemExit(f"Missing {d} - download the upstream code into third_party/ first.")

    counts = {"applied": 0, "already": 0, "pending": 0, "failed": 0}

    def report(status, label):
        key = status if status in counts else "failed"
        counts[key] += 1
        mark = {"applied": "FIXED  ", "already": "ok     ", "pending": "TO FIX "}.get(status, "ERROR  ")
        print(f"  {mark} {label}" + ("" if key != "failed" else f"  -> {status}"))

    print("Text fixes:")
    for path, old, new in TEXT_FIXES:
        label = f"{path.relative_to(THIRD)}: {old.strip().splitlines()[0][:60]}"
        report(apply_text_fix(path, old, new, args.check), label)

    print("Placeholder files:")
    for path, content in NEW_FILES.items():
        if path.exists():
            status = "already"
        elif args.check:
            status = "pending"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            status = "applied"
        report(status, str(path.relative_to(THIRD)))

    print("__init__.py files:")
    missing = [d for d in package_dirs() if not (d / "__init__.py").exists()]
    if missing and not args.check:
        for d in missing:
            (d / "__init__.py").write_text(f"# {TAG} makes this a regular Python package\n",
                                            encoding="utf-8")
    status = "already" if not missing else ("pending" if args.check else "applied")
    report(status, f"{len(missing)} folder(s) needed one" if missing else "all folders have one")

    print(f"\nSummary: {counts['applied']} fixed now, {counts['already']} already fixed, "
          f"{counts['pending']} still to fix, {counts['failed']} errors")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()