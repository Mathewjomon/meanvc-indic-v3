"""
Stage 3 - Extract the DiT training features for every train/val clip (run on Colab).

WHAT THE DiT LEARNS FROM (one set per clip)
  bn    content features from OUR trained encoder (checkpoints/encoder_v3/encoder_best.pt):
        256 numbers every 40 ms, extracted slice by slice (200 ms slices, 2 slices of memory,
        streaming convolution) - exactly as in encoder training and as at inference.
  mel   the clip's mel spectrogram (80 x every 10 ms), with MeanVC's own mel settings
        (the same class MeanVC uses at inference). The DiT learns to generate this.
  xvec  speaker embedding (256 numbers) from WavLM-Large + ECAPA-TDNN (frozen, not trained).

CHECKS BUILT IN (v2 lessons)
  - encoder weights: missing = unexpected = 0 (nothing silently skipped)
  - speaker model: reports which weights loaded (MeanVC itself loads it with strict=False,
    which fails silently) and stops if the ECAPA part did not load
  - slice-by-slice content features == the chunked encoder output used in training
  - after extraction: every file loads, bn length x 4 ~= mel length,
    and same-speaker embeddings are more similar than different-speaker ones

OUTPUT (on Drive)
  data/features/feats_<set>_<k>.tar   bn/, mel/, xvec/ .npy files, 16-bit floats (~0.7 GB per 2000 clips)
                                      -> convert to float32 when loading (the DiT training script does)
  data/features/<set>.tsv             one row per clip: utt_id, speaker, language, frames, shard
  Local copies on Colab's disk are deleted after each set is packed, to keep disk use low.

HOW TO RUN (Colab, after encoder training has finished)
  !python /content/drive/MyDrive/meanvc_indic_v3/code/decoder/extract_features.py
"""

import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent                    # code/decoder
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
ENCODER_DIR = PROJECT / "checkpoints" / "encoder_v3"
SV_CKPT = PROJECT / "checkpoints" / "pretrained" / "wavlm_large_finetune.pth"
OUT = DATA / "features"
sys.path.insert(0, str(PROJECT / "third_party" / "MeanVC"))       # 'src' (MeanVC code)
sys.path.insert(0, str(PROJECT / "code" / "encoder"))             # train_encoder.py, vocab.py

import train_encoder as te  # noqa: E402  (compute_fbank, build_model, set_streaming, ...)

CLIPS_PER_TAR = 2000
KINDS = ("bn", "mel", "xvec")


# ======================================================================================
# Models
# ======================================================================================
def load_encoder(device, encoder_ckpt=None):
    """Our trained content encoder, streaming convolution, eval mode. Load is verified."""
    from vocab import load_vocab
    model = te.build_model(len(load_vocab()), ENCODER_DIR / "cmvn_from_pretrained.json")
    sd = torch.load(encoder_ckpt or ENCODER_DIR / "encoder_best.pt", map_location="cpu")
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    assert not missing and not unexpected, f"encoder: missing={missing} unexpected={unexpected}"
    te.set_streaming(model)
    return model.to(device).eval()


def load_speaker_model(device):
    """WavLM-Large + ECAPA-TDNN speaker encoder, with an honest report of what loaded."""
    from src.runtime.speaker_verification.verification import init_model
    model = init_model("wavlm_large", None)                  # build only, we load ourselves
    sd = torch.load(SV_CKPT, map_location="cpu")
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    head_missing = [k for k in missing if not k.startswith("feature_extract.")]
    report = (f"{len(sd)} tensors in file | missing: {len(missing)} "
              f"({len(head_missing)} in ECAPA head) | unexpected: {len(unexpected)}")
    assert not head_missing, f"speaker model ECAPA head did not load: {head_missing[:5]} | {report}"
    return model.to(device).eval(), report, missing, unexpected


def load_mel_extractor(device):
    """MeanVC's own mel (the class used at inference), so training mels == inference mels."""
    from src.infer.infer_ref import MelSpectrogramFeatures
    return MelSpectrogramFeatures().to(device)               # 16 kHz, n_fft 1024, win 640, hop 160


# ======================================================================================
# One clip
# ======================================================================================
@torch.no_grad()
def extract_bn(encoder, wav, device):
    """Content features slice by slice: 200 ms slices (20 fbank frames step, 23-frame window),
    2 slices of memory (cache 10 frames). Same loop as MeanVC's extract_bn_200ms.py."""
    feats = te.compute_fbank(wav).unsqueeze(0).to(device)
    step, window = te.CHUNK * 4, (te.CHUNK - 1) * 4 + 7
    att = torch.zeros((0, 0, 0, 0), device=device)
    cnn = torch.zeros((0, 0, 0, 0), device=device)
    outs, offset = [], 0
    for i in range(0, feats.shape[1], step):
        piece = feats[:, i:i + window, :]
        if piece.shape[1] < 10:
            break
        y, att, cnn = encoder.forward_encoder_chunk(piece, offset, te.CHUNK * te.LEFT_CHUNKS, att, cnn)
        outs.append(y)
        offset += y.size(1)
    return torch.cat(outs, 1)[0].float().cpu().numpy()      # (frames_40ms, 256)


@torch.no_grad()
def extract_all(models, wav, device):
    encoder, speaker, mel_fn = models
    x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    mel = mel_fn(x)[0].T.float().cpu().numpy()               # (frames_10ms, 80), values in [-1, 1]
    xvec = speaker(x)[0].float().cpu().numpy()               # (256,)
    return extract_bn(encoder, wav, device), mel, xvec


# ======================================================================================
# Worker: one share of the clips (several run in parallel on the same GPU)
# ======================================================================================
def run_shard(args):
    rows, audio_root, work, encoder_ckpt, worker_id = args
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    models = (load_encoder(device, encoder_ckpt), load_speaker_model(device)[0], load_mel_extractor(device))
    failed, t0 = [], time.time()
    for n, (utt, path) in enumerate(rows, 1):
        outs = [Path(work) / k / f"{utt}.npy" for k in KINDS]
        if not all(o.exists() for o in outs):                # skip clips already done (re-run)
            try:
                wav, sr = sf.read(Path(audio_root) / path, dtype="float32")
                assert sr == 16000, f"{sr} Hz"
                for o, arr in zip(outs, extract_all(models, wav, device)):
                    np.save(o, arr.astype(np.float16))       # 16-bit: half the disk, plenty precise
            except Exception as e:                            # noqa: BLE001 - report, don't stop
                failed.append((utt, f"{type(e).__name__}: {e}"))
        if worker_id == 0 and (n % 500 == 0 or n == len(rows)):
            print(f"  worker 0: {n:,}/{len(rows):,} clips | {(time.time() - t0) / 60:.1f} min "
                  f"(other workers progress at the same pace)", flush=True)
    return failed


# ======================================================================================
def verify(df, work):
    """Every file loads, lengths line up, and the speaker embeddings make sense."""
    bn_frames, mel_frames, bad = [], [], []
    for utt in df.utt_id:
        try:
            bn = np.load(work / "bn" / f"{utt}.npy")
            mel = np.load(work / "mel" / f"{utt}.npy")
            xv = np.load(work / "xvec" / f"{utt}.npy")
            assert bn.shape[1] == 256 and mel.shape[1] == 80 and xv.shape == (256,)
            assert np.isfinite(bn).all() and np.isfinite(mel).all() and np.isfinite(xv).all()
            bn_frames.append(bn.shape[0])
            mel_frames.append(mel.shape[0])
        except Exception as e:                                # noqa: BLE001
            bad.append((utt, str(e)))
            bn_frames.append(-1)
            mel_frames.append(-1)
    df = df.assign(bn_frames=bn_frames, mel_frames=mel_frames)
    ok = df[df.bn_frames > 0]
    gap = ok.mel_frames - ok.bn_frames * 4
    print(f"  files OK: {len(ok):,} | bad: {len(bad)}" + (f" e.g. {bad[:3]}" if bad else ""))
    print(f"  mel frames - 4 x bn frames: median {gap.median():.0f}, range {gap.min()} to {gap.max()} "
          f"(small = content and mel line up; the DiT trims to the shorter one)")

    # Speaker check: same-speaker pairs should be clearly more similar than different-speaker pairs.
    sample = ok.groupby("speaker_id").head(4)
    sample = sample[sample.speaker_id.map(sample.speaker_id.value_counts()) >= 2].head(400)
    if len(sample) >= 4:
        X = np.stack([np.load(work / "xvec" / f"{u}.npy") for u in sample.utt_id]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ X.T
        spk = sample.speaker_id.to_numpy()
        same = (spk[:, None] == spk[None, :]) & ~np.eye(len(spk), dtype=bool)
        diff = spk[:, None] != spk[None, :]
        print(f"  speaker embeddings: same-speaker similarity {S[same].mean():.3f} vs "
              f"different-speaker {S[diff].mean():.3f} -> "
              f"{'OK' if S[same].mean() - S[diff].mean() > 0.2 else 'SUSPICIOUS - tell Claude'}")
    return df, bad


def pack(df, work, name):
    """Pack bn/mel/xvec files into a few large .tar files on Drive."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("utt_id").reset_index(drop=True)
    df["shard"] = ""
    for k, start in enumerate(range(0, len(df), CLIPS_PER_TAR)):
        chunk = df.iloc[start:start + CLIPS_PER_TAR]
        shard = f"feats_{name}_{k:03d}.tar"
        local = work / shard
        with tarfile.open(local, "w") as tar:
            for utt in chunk.utt_id:
                for kind in KINDS:
                    tar.add(work / kind / f"{utt}.npy", arcname=f"{kind}/{utt}.npy")
        dst = OUT / shard
        shutil.copy(local, dst)                                 # build locally, then copy to Drive
        local.unlink()
        df.loc[chunk.index, "shard"] = shard
        print(f"  packed {shard} ({len(chunk)} clips, {dst.stat().st_size / 1e9:.2f} GB)")
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract bn / mel / xvec features for DiT training")
    ap.add_argument("--sets", default="train,val")
    ap.add_argument("--workers", type=int, default=4, help="parallel extractors on the GPU")
    ap.add_argument("--audio-root", default="/content/work/audio16k")
    ap.add_argument("--work", default="/content/work/features")
    ap.add_argument("--encoder", default=None, help="default: checkpoints/encoder_v3/encoder_best.pt")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips per set (testing)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_root, work = Path(args.audio_root), Path(args.work)
    for k in KINDS:
        (work / k).mkdir(parents=True, exist_ok=True)

    print("1. Audio")
    te.unpack_audio(audio_root)

    print("2. Models (checks)")
    encoder = load_encoder(device, args.encoder)
    print("  content encoder: loaded (missing 0, unexpected 0), streaming convolution")
    _, report, missing, unexpected = load_speaker_model(device)
    print(f"  speaker model: {report}")
    if missing or unexpected:
        print(f"    e.g. missing {missing[:3]} unexpected {unexpected[:3]}")
    val = pd.read_csv(DATA / "manifests" / "val.tsv", sep="\t", keep_default_na=False)
    wav, _ = sf.read(audio_root / val.path[0], dtype="float32")
    diff, cos, _, _ = te.streaming_check(encoder, wav, device)
    print(f"  slice-by-slice == training check: max difference {diff:.2e}, cosine {cos:.6f}")
    del encoder
    torch.cuda.empty_cache() if device == "cuda" else None

    for name in args.sets.split(","):
        done_table = OUT / f"{name}.tsv"
        if done_table.exists() and not args.limit:
            print(f"\n{name}: already done ({done_table.name}) - skipping")
            continue
        df = pd.read_csv(DATA / "manifests" / f"{name}.tsv", sep="\t", keep_default_na=False,
                         dtype={"speaker_id": str})
        if args.limit:
            df = df.head(args.limit)
        need_gb = len(df) * 0.35e-3 * 1.2                    # ~0.35 MB per clip in 16-bit, +20%
        free_gb = shutil.disk_usage(work).free / 1e9
        print(f"\n3. Extracting {name}: {len(df):,} clips with {args.workers} workers "
              f"(needs ~{need_gb:.0f} GB local disk, {free_gb:.0f} GB free)")
        if free_gb < need_gb:
            print("  WARNING: probably not enough disk space - free some space first (see Claude)")
        t0 = time.time()
        rows = list(zip(df.utt_id, df.path))
        shares = [(rows[i::max(args.workers, 1)], str(audio_root), str(work), args.encoder, i)
                  for i in range(max(args.workers, 1))]
        if args.workers <= 1:
            failed = run_shard(shares[0])
        else:
            import torch.multiprocessing as mp
            with mp.get_context("spawn").Pool(args.workers) as pool:
                failed = [f for part in pool.map(run_shard, shares) for f in part]
        print(f"  done in {(time.time() - t0) / 60:.1f} min | failed: {len(failed)}"
              + (f" e.g. {failed[:3]}" if failed else ""))

        print(f"4. Checking {name}")
        df, bad = verify(df, work)
        df = df[df.bn_frames > 0]
        tag = name if not args.limit else f"{name}_limit{args.limit}"   # test runs never overwrite
        print(f"5. Packing {name} to Drive")
        df = pack(df, work, tag)
        cols = ["utt_id", "set", "code", "speaker_id", "gender", "duration",
                "bn_frames", "mel_frames", "shard"]
        df[cols].to_csv(OUT / f"{tag}.tsv", sep="\t", index=False)
        # This set is now safely on Drive (tar files + table): free the local disk for the next set.
        for utt in df.utt_id:
            for kind in KINDS:
                (work / kind / f"{utt}.npy").unlink(missing_ok=True)
        print(f"  local files removed, {shutil.disk_usage(work).free / 1e9:.0f} GB free")
    print("\nDone. Features in data/features/ (tar files + <set>.tsv)")


if __name__ == "__main__":
    main()