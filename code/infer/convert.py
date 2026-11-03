"""
Stage 5a - Voice conversion: say the SOURCE speech in the REFERENCE speaker's voice.
Works on the laptop (CPU, VS Code) and on Colab (GPU).

THE PIPELINE (same code and settings as training - nothing re-implemented differently)
  source speech    -> content encoder, 200 ms slices          -> bn  (WHAT is said)
  reference speech -> WavLM + ECAPA speaker model             -> xvec (WHO: voice fingerprint)
                   -> mel spectrogram                         -> prompt (WHAT THE VOICE SOUNDS LIKE)
  bn + xvec + prompt -> DiT, 200 ms at a time, 2 steps, kv cache -> mel of the converted speech
  mel -> Vocos vocoder -> 16 kHz waveform

SYSTEMS (which encoder + decoder to use)
  ours       our encoder (encoder_v3) + our decoder (dit_v3/dit_best.pt)       <- the result
  enc_only   our encoder + the original Mandarin decoder                       (comparison)
  mandarin   original MeanVC: Mandarin encoder + Mandarin decoder              (baseline)

HOW TO RUN
  laptop (VS Code terminal, venv active):
    python "G:\\My Drive\\meanvc_indic_v3\\code\\infer\\convert.py" --source a.wav --reference b.wav
  Colab:
    !python /content/drive/MyDrive/meanvc_indic_v3/code/infer/convert.py --source a.wav --reference b.wav

  The speaker model's base file (WavLM-Large, 1.2 GB) is read from
  checkpoints/pretrained/s3prl/ on Drive - downloaded there once, then shared by Colab and laptop.

  from Python:  from convert import VoiceConverter, load_wav
                vc = VoiceConverter("ours");  wav, info = vc.convert(load_wav(src), load_wav(ref))
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
import io
import sys
import time
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent                  # code/infer
PROJECT = HERE.parents[1]
CKPT = PROJECT / "checkpoints"
sys.path.insert(0, str(PROJECT / "code" / "decoder"))   # extract_features.py, train_dit.py

# Where the speaker model's WavLM-Large file (1.2 GB) is kept: on Drive, so it is downloaded only
# once and the laptop gets it through Drive instead of a slow direct download.
import s3prl.util.download  # noqa: E402
S3PRL_DIR = Path(os.environ.get("S3PRL_DIR") or CKPT / "pretrained" / "s3prl")
s3prl.util.download.set_dir(str(S3PRL_DIR))

# Windows laptop: model files are read from a local copy on D: (see local()).
LOCAL_CACHE = os.environ.get("MEANVC_CACHE") or (
    "D:/meanvc_cache" if os.name == "nt" and Path("D:/").exists() else "")


def local(path):
    """Laptop: the G: drive (Google Drive) cannot serve the random reads torch.load needs
    ('OSError: [Errno 22] Invalid argument'), so each model file is copied once to
    D:\\meanvc_cache and read from there. On Colab the Drive path is used directly."""
    path = Path(path)
    if not LOCAL_CACHE:
        return path
    try:
        rel = path.resolve().relative_to(PROJECT)
    except ValueError:
        return path
    dst, size = Path(LOCAL_CACHE) / rel, path.stat().st_size
    if dst.exists() and dst.stat().st_size == size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    print(f"  copying {rel} ({size / 1e9:.2f} GB) to {LOCAL_CACHE} - only the first time", flush=True)
    done, step = 0, max(size // 5, 1)
    try:
        with open(path, "rb") as fi, open(tmp, "wb") as fo:
            while chunk := fi.read(1 << 20):               # small front-to-back reads
                fo.write(chunk)
                if (done + len(chunk)) // step > done // step and size > 200_000_000:
                    print(f"    {100 * (done + len(chunk)) // size}%", flush=True)
                done += len(chunk)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Google Drive could not give Python this file yet ({e}).\n"
                         f"Copy it yourself with File Explorer or PowerShell Copy-Item to\n  {dst}\n"
                         "(same folder), or make the folder 'Available offline' in Drive, then run again.")
    os.replace(tmp, dst)
    return dst


WAVLM_CFG = CKPT / "pretrained" / "wavlm_large_cfg.pt"   # WavLM-Large settings only (a few KB)


def use_wavlm_settings_only():
    """The speaker model file (wavlm_large_finetune.pth) already holds ALL of WavLM's weights, so
    the 1.2 GB WavLM base file is only needed for its settings. When those settings are saved in
    the tiny wavlm_large_cfg.pt, build an empty WavLM from them instead; the real weights are
    loaded from the speaker model file right after (checked: nothing may be missing)."""
    import s3prl.upstream.wavlm.expert as wexp
    import s3prl.upstream.wavlm.hubconf as hub
    cfg = torch.load(local(WAVLM_CFG), map_location="cpu", weights_only=False)["cfg"]

    class EmptyWavLM(wexp.WavLM):
        def load_state_dict(self, state_dict, *args, **kwargs):
            if state_dict is None:                          # weights come later, from the
                return None                                 # speaker model file
            return super().load_state_dict(state_dict, *args, **kwargs)

    class TorchStandIn:                                     # what s3prl's loader calls 'torch'
        def load(self, *args, **kwargs):
            return {"cfg": cfg, "model": None}

        def __getattr__(self, name):
            return getattr(torch, name)

    def wavlm_large(*args, **kwargs):
        saved = wexp.torch, wexp.WavLM
        wexp.torch, wexp.WavLM = TorchStandIn(), EmptyWavLM
        try:
            return wexp.UpstreamExpert("settings-only")
        finally:
            wexp.torch, wexp.WavLM = saved

    hub.wavlm_large = wavlm_large


def prepare_wavlm():
    """Check the WavLM file, and on the laptop use its local copy.
    (An interrupted download leaves a short file that later fails with a confusing
    'failed finding central directory' error - catch it with a clear message instead.)"""
    files = list(S3PRL_DIR.glob("*.wavlm_large.pt"))
    if not files and os.name == "nt":                        # laptop: don't download it slowly here
        raise SystemExit(f"The WavLM file is not on this computer yet ({S3PRL_DIR}).\n"
                         "It comes from Drive: save it there once from Colab, then wait until Drive "
                         "shows it in that folder and run again.")
    for f in files:
        if f.stat().st_size < 1_100_000_000:                 # the complete file is ~1.27 GB
            raise SystemExit(f"{f} is only {f.stat().st_size / 1e9:.2f} GB: an interrupted download.\n"
                             f"Delete it and run again (or copy the complete file there from Colab).")
    if files and LOCAL_CACHE:
        s3prl.util.download.set_dir(str(local(files[0]).parent))

import extract_features as ef  # noqa: E402  encoder / speaker model / mel, same as Stage 3
import train_dit as td         # noqa: E402  DiT building, verified loading, streaming generation

ENCODER_V3 = CKPT / "encoder_v3" / "encoder_best.pt"
ENCODER_MANDARIN = CKPT / "pretrained" / "fastu2++.pt"
DIT_V3 = CKPT / "dit_v3" / "dit_best.pt"
DIT_MANDARIN = CKPT / "pretrained" / "model_200ms.safetensors"
VOCOS = CKPT / "pretrained" / "vocos.pt"
SYSTEMS = {
    "ours": (ENCODER_V3, DIT_V3),
    "enc_only": (ENCODER_V3, DIT_MANDARIN),
    "mandarin": (ENCODER_MANDARIN, DIT_MANDARIN),
}
SR = 16000


def load_wav(path):
    """Any audio file -> 16 kHz mono float32 in [-1, 1]."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return np.clip(wav, -1.0, 1.0)


def save_wav(path, wav):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), SR, subtype="PCM_16")


def load_content_encoder(path, device):
    """The original Mandarin encoder is a TorchScript file; ours is a verified state dict."""
    path = Path(path)
    if path.name == "fastu2++.pt":
        return torch.jit.load(str(path), map_location=device).eval()
    return ef.load_encoder(device, path)


class VoiceConverter:
    """Loads everything once; convert() can then be called many times."""

    def __init__(self, system="ours", device=None, speaker_model=None, dit_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if speaker_model is None:
            settings_only = WAVLM_CFG.exists()                 # no 1.2 GB WavLM base file needed
            if settings_only:
                use_wavlm_settings_only()
            else:
                prepare_wavlm()
            ef.SV_CKPT = local(ef.SV_CKPT)                  # speaker model weights (1.3 GB)
            speaker_model, _, missing, _ = ef.load_speaker_model(self.device)
            if settings_only and missing:
                raise SystemExit(f"Speaker model file lacks WavLM weights: {missing[:3]} - tell Claude")
        self.speaker = speaker_model
        self.mel_fn = ef.load_mel_extractor(self.device)
        self.vocos_bytes = local(VOCOS).read_bytes()
        self.set_system(system, dit_path)

    def set_system(self, system, dit_path=None):
        enc_path, default_dit = SYSTEMS[system]
        self.system, self.dit_path = system, Path(dit_path or default_dit)
        self.encoder = load_content_encoder(local(enc_path), self.device)
        dit = td.build_dit(kv_cache=True)
        td.load_weights(dit, td.read_weights(local(self.dit_path)))   # stops if anything is missing
        self.dit = dit.to(self.device).eval()
        self.dit_epoch = None                                          # our checkpoints record it
        if self.dit_path.suffix == ".pt":
            info = torch.load(local(self.dit_path), map_location="cpu", weights_only=False).get("info", {})
            self.dit_epoch = info.get("epoch")

    def describe(self):
        enc = SYSTEMS[self.system][0]
        dit = f"{self.dit_path.parent.name}/{self.dit_path.name}"
        return (f"encoder {enc.parent.name}/{enc.name}, decoder {dit}"
                + (f" (epoch {self.dit_epoch})" if self.dit_epoch is not None else ""))

    def _tensor(self, wav):
        return torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def embed(self, wav):
        """Speaker embedding (256 numbers) - for the DiT and for measuring similarity."""
        return self.speaker(self._tensor(wav))[0].float()

    @torch.inference_mode()
    def voice(self, reference_wav):
        """Everything the DiT needs about the target voice: (xvec (1,256), prompt mel (1,T,80))."""
        x = self._tensor(reference_wav)
        return self.speaker(x).float(), self.mel_fn(x)[0].T.unsqueeze(0).float()

    @torch.inference_mode()
    def content(self, source_wav):
        """bn from the content encoder, 200 ms slices, stretched to 10 ms frames: (1, T, 256)."""
        bn = ef.extract_bn(self.encoder, source_wav, self.device)
        return td.stretch_bn(bn).unsqueeze(0).to(self.device)

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def convert(self, source_wav, reference_wav=None, voice=None, seed=0):
        """-> (converted waveform, info). Pass voice=self.voice(ref) to reuse a reference."""
        voice = voice or self.voice(reference_wav)
        torch.manual_seed(seed)                                         # same noise -> repeatable
        vocos = torch.jit.load(io.BytesIO(self.vocos_bytes), map_location=self.device)  # fresh (v2)
        self._sync()
        t0 = time.time()
        bn = self.content(source_wav)
        mel = td.generate(self.dit, bn, voice[0], voice[1], self.device)   # (T, 80) in [-1, 1]
        wav = vocos.decode((mel.T.unsqueeze(0) + 1) / 2)[0]
        self._sync()
        seconds = time.time() - t0
        wav = wav.float().cpu().numpy()
        peak = float(np.abs(wav).max())
        if peak > 0.99:                          # the vocoder can overshoot [-1, 1]: turn the
            wav = wav * (0.99 / peak)            # whole clip down a little instead of clipping
        return wav, {"seconds": seconds, "duration": len(wav) / SR,
                     "rtf": seconds / max(len(wav) / SR, 1e-6), "system": self.system}


def similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser(description="Convert SOURCE speech into the REFERENCE voice")
    ap.add_argument("--source", required=True, help="speech to convert (what is said)")
    ap.add_argument("--reference", required=True, help="a clip of the target voice (3-15 s)")
    ap.add_argument("--out", default=None, help="default: outputs/convert/<source>__to__<reference>_<system>.wav")
    ap.add_argument("--system", default="ours", choices=list(SYSTEMS))
    ap.add_argument("--dit", default=None, help="another DiT checkpoint, e.g. checkpoints/dit_v3/dit_epoch_10.pt")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    vc = VoiceConverter(args.system, args.device, dit_path=args.dit)
    print(f"Loaded '{vc.system}' on {vc.device} in {time.time() - t0:.0f} s ({vc.describe()})")
    src, ref = load_wav(args.source), load_wav(args.reference)
    out_wav, info = vc.convert(src, ref, seed=args.seed)
    out = Path(args.out) if args.out else (PROJECT / "outputs" / "convert" /
                                           f"{Path(args.source).stem}__to__{Path(args.reference).stem}_{vc.system}.wav")
    save_wav(out, out_wav)
    e_out, e_ref, e_src = vc.embed(out_wav), vc.embed(ref), vc.embed(src)
    print(f"Converted {len(src) / SR:.1f} s of speech in {info['seconds']:.2f} s "
          f"(real-time factor {info['rtf']:.2f}: below 1 = faster than real time)")
    print(f"Speaker similarity of the result: to the reference voice {similarity(e_out, e_ref):.3f} "
          f"(higher = better) | to the source voice {similarity(e_out, e_src):.3f} (lower = better)")
    print(f"  (before conversion, source vs reference: {similarity(e_src, e_ref):.3f})")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()