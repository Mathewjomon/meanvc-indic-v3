"""
Stage 4 - Fine-tune the MeanVC DiT decoder on our Malayalam/Tamil/Hindi features (run on Colab).

WHAT IT DOES
  1. Unpacks the feature files from Stage 3 (data/features/feats_*.tar) onto Colab's fast disk.
  2. Builds MeanVC's DiT exactly as released (config_200ms.json: 4 blocks, dim 512, chunk 20 =
     200 ms) and loads the ORIGINAL Mandarin weights (model_200ms.safetensors). The load is
     verified: every weight must be found, nothing silently skipped (v2 lesson).
  3. One training example = one clip:
       content  bn from OUR encoder (one frame per 40 ms), stretched x4 to 10 ms (as MeanVC does)
       target   the clip's mel spectrogram - this is what the DiT learns to generate
       speaker  the clip's speaker embedding (xvec)
       prompt   20 s of mel from OTHER clips of the same speaker (the voice to copy).
                The clip itself is never in its own prompt, so the model cannot cheat.
     A random window of at most 5 s per clip each epoch, cut to whole 200 ms chunks.
  4. Loss: MeanVC's own mean-flow loss with the paper's settings (flow ratio 0.5,
     CFG ratio 0.1, CFG scale 2.0, p 0.5). Nothing new invented.
  5. A smoothed copy of the model (EMA) is what we evaluate and save - it is more stable.
  6. Once before training (= the Mandarin model on our data, our baseline) and after every epoch:
       val loss       mean-flow loss on held-out speakers (same noise every time -> comparable)
       val mel error  real streaming generation, exactly like inference (2 steps, 200 ms chunks,
                      kv cache), compared with the real clip. Per language. Lower = better.
                      The best epoch by this number is saved as dit_best.pt.
  7. Saves last.pt every 1000 steps: after a Colab reset, run the same command to continue.

HOW TO RUN (Colab)
  quick test (~5 min):  !python /content/drive/MyDrive/meanvc_indic_v3/code/decoder/train_dit.py --smoke
  full training:        !python /content/drive/MyDrive/meanvc_indic_v3/code/decoder/train_dit.py

OUTPUT (Drive: checkpoints/dit_v3/)
  dit_best.pt (+ .json)  best epoch, EMA weights - the decoder we use at inference
  dit_epoch_XX.pt        EMA weights after every epoch (for comparing checkpoints by listening)
  last.pt                everything needed to continue training
  log.csv                one line per epoch (epoch -1 = the Mandarin model before training)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # hide TensorFlow start-up messages
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory waste
import argparse
import contextlib
import copy
import csv
import json
import random
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")                       # library deprecation notices

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                  # code/decoder
PROJECT = HERE.parents[1]                               # meanvc_indic_v3
DATA = PROJECT / "data"
FEATURES = DATA / "features"                            # Stage 3 output (tar files + tables)
MEANVC = PROJECT / "third_party" / "MeanVC"
CONFIG = MEANVC / "src" / "config" / "config_200ms.json"
PRETRAINED = PROJECT / "checkpoints" / "pretrained" / "model_200ms.safetensors"
sys.path.insert(0, str(MEANVC))                         # 'src' (MeanVC code)

from src.infer.dit_kvcache import DiT as DiTKV   # noqa: E402  inference DiT (kv cache)
from src.model.backbones.dit import DiT          # noqa: E402  training DiT
from src.model.cfm_mean_flow import MeanFlow     # noqa: E402  mean-flow loss

# --------------------------------------------------------------------------------------
# Settings (MeanVC's own values unless noted)
# --------------------------------------------------------------------------------------
CHUNK = 20            # DiT chunk = 20 mel frames = 200 ms (config_200ms.json)
MAX_LEN = 500         # longest training window: 500 frames = 5 s (MeanVC defaults.ini). The DiT
                      # only looks 1 s back, so longer windows add cost, not information.
PROMPT_LEN = 2000     # voice prompt: 2000 frames = 20 s
BATCH = 32            # ours: 32 clips x 5 s per batch (~25 GB of GPU memory)
EPOCHS = 12           # ours: ~27,000 steps; every clip seen ~12 times (a new random 5 s each time)
LR = 5e-5             # ours: MeanVC trained from scratch at 1e-4, v2 fine-tuned at 2e-5
WARMUP = 1000         # steps of slowly rising learning rate, then constant
WEIGHT_DECAY = 0.01   # AdamW default
GRAD_CLIP = 1.0
EMA_DECAY = 0.999     # ours: averages roughly the last 1000 steps (library default would
                      # average ~100,000 steps and stay close to the Mandarin model)
SAVE_EVERY = 1000     # steps between last.pt saves
MEANFLOW = dict(flow_ratio=0.5, time_dist=["lognorm", -0.4, 1.0], cfg_ratio=0.1,
                cfg_scale=2.0, cfg_uncond="u", p=0.5)
GEN_STEPS = 2         # inference steps for the 200 ms model (MeanVC default)
GEN_PER_LANG = 20     # validation clips per language for the generation check
LANGS = ("ml", "ta", "hi")
KINDS = ("bn", "mel", "xvec")


# ======================================================================================
# Data
# ======================================================================================
def read_tables(names):
    """Feature tables from Stage 3 -> one DataFrame. 'spk' = language + speaker id."""
    frames = []
    for name in names:
        path = FEATURES / f"{name}.tsv"
        if not path.exists():
            raise SystemExit(f"Missing {path} - did Stage 3 finish for '{name}'?")
        frames.append(pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"speaker_id": str}))
    df = pd.concat(frames, ignore_index=True)
    df["spk"] = df.code + "_" + df.speaker_id
    df["frames"] = np.minimum(df.bn_frames * 4, df.mel_frames)          # usable 10 ms frames
    df["n"] = np.minimum(df.frames, MAX_LEN) // CHUNK * CHUNK           # training window
    return df


def keep_multi_clip_speakers(df):
    """A clip needs at least one OTHER clip of its speaker to build its prompt."""
    counts = df.spk.map(df.spk.value_counts())
    return df[(counts >= 2) & (df.n >= CHUNK)].reset_index(drop=True), int((counts < 2).sum())


def unpack_features(df, root):
    """Extract the tar files used by df to fast local disk (skipped if already done)."""
    for kind in KINDS:
        (root / kind).mkdir(parents=True, exist_ok=True)
    for shard in sorted(df.shard.unique()):
        done = root / f".{Path(shard).stem}.done"
        if done.exists():
            continue
        t0 = time.time()
        with tarfile.open(FEATURES / shard) as tar:
            try:
                tar.extractall(root, filter="data")
            except TypeError:                                  # older Python: no filter arg
                tar.extractall(root)
        done.touch()
        print(f"  unpacked {shard} ({time.time() - t0:.0f} s)")


def load_npy(root, kind, utt):
    return np.load(root / kind / f"{utt}.npy").astype(np.float32)       # stored as 16-bit


def stretch_bn(bn):
    """40 ms content frames -> 10 ms frames, linear, exactly like MeanVC's dataset/inference."""
    x = torch.from_numpy(bn).T.unsqueeze(0)                             # (1, 256, T)
    x = F.interpolate(x, size=x.shape[2] * 4, mode="linear", align_corners=True)
    return x[0].T.contiguous()                                          # (4T, 256)


class DiTData(torch.utils.data.Dataset):
    """One item = content, target mel, speaker embedding and voice prompt for one clip."""

    def __init__(self, df, root, train):
        self.df, self.root, self.train = df, root, train
        self.utts = df.utt_id.tolist()
        self.by_spk = df.groupby("spk").indices                         # speaker -> row numbers
        self.spk = df.spk.tolist()
        self.n = df.n.tolist()

    def __len__(self):
        return len(self.df)

    def prompt(self, i, rng):
        """PROMPT_LEN frames of mel from other clips of the same speaker (never clip i)."""
        others = [j for j in self.by_spk[self.spk[i]] if j != i]
        parts, total = [], 0
        while total < PROMPT_LEN:                                       # repeats clips if needed
            for j in rng.permutation(others):
                mel = load_npy(self.root, "mel", self.utts[j])
                parts.append(mel)
                total += len(mel)
                if total >= PROMPT_LEN:
                    break
        mel = np.concatenate(parts)
        start = rng.integers(0, len(mel) - PROMPT_LEN + 1)
        return torch.from_numpy(mel[start:start + PROMPT_LEN])

    def __getitem__(self, i):
        # training: fresh randomness each time; validation: the same "random" choices every time
        rng = np.random.default_rng() if self.train else np.random.default_rng(1000 + i)
        utt = self.utts[i]
        bn = stretch_bn(load_npy(self.root, "bn", utt))
        mel = torch.from_numpy(load_npy(self.root, "mel", utt))
        frames = min(len(bn), len(mel))
        n = self.n[i]
        start = int(rng.integers(0, frames - n + 1)) if self.train else 0
        return dict(bn=bn[start:start + n], mel=mel[start:start + n],
                    xvec=torch.from_numpy(load_npy(self.root, "xvec", utt)),
                    prompt=self.prompt(i, rng), n=n)


def collate(items):
    """Pad to the longest clip in the batch: content with 0, mel with -1 (MeanVC's values)."""
    n_max = max(it["n"] for it in items)
    bn = torch.zeros(len(items), n_max, items[0]["bn"].shape[1])
    mel = torch.full((len(items), n_max, 80), -1.0)
    for k, it in enumerate(items):
        bn[k, :it["n"]] = it["bn"]
        mel[k, :it["n"]] = it["mel"]
    return dict(bn=bn, mel=mel, xvec=torch.stack([it["xvec"] for it in items]),
                prompt=torch.stack([it["prompt"] for it in items]),
                lens=torch.tensor([it["n"] for it in items]))


class BatchPlan(torch.utils.data.Sampler):
    """Batches of BATCH clips with similar length (little padding), in random order.
    The plan of an epoch is fixed by (seed, epoch), so a resumed run can skip what is done."""

    def __init__(self, lengths, batch, shuffle, seed=0):
        self.lengths, self.batch, self.shuffle, self.seed = np.asarray(lengths), batch, shuffle, seed
        self.epoch, self.start = 0, 0

    def batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        jitter = rng.uniform(0, 3 * CHUNK, len(self.lengths)) if self.shuffle else 0
        order = np.argsort(self.lengths + jitter, kind="stable")
        out = [order[i:i + self.batch].tolist() for i in range(0, len(order), self.batch)]
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.batches()[self.start:])

    def __len__(self):
        return len(self.batches()) - self.start


def worker_init(worker_id):
    torch.set_num_threads(1)


def make_loader(ds, plan, workers, device):
    kw = dict(batch_sampler=plan, collate_fn=collate, num_workers=workers,
              pin_memory=(device == "cuda"), worker_init_fn=worker_init)
    if workers > 0:
        # MeanVC's dataset module switches Python to 'spawn' (slow start); 'fork' is fine here
        kw.update(persistent_workers=True, multiprocessing_context="fork", prefetch_factor=4)
    return torch.utils.data.DataLoader(ds, **kw)


def speaker_check(df, root, n_speakers=200):
    """Same-speaker embeddings should be clearly more similar than different-speaker ones.
    (Speakers are picked first, then 2 clips each - so every speaker gives a same-speaker pair.)"""
    rng = np.random.default_rng(0)
    spks = rng.permutation(df.spk.unique())[:n_speakers]
    pairs = [df[df.spk == s].utt_id.sample(2, random_state=0).tolist() for s in spks]
    X = np.stack([load_npy(root, "xvec", u) for p in pairs for u in p])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    lab = np.repeat(np.arange(len(pairs)), 2)
    same = (lab[:, None] == lab[None, :]) & ~np.eye(len(lab), dtype=bool)
    return S[same].mean(), S[lab[:, None] != lab[None, :]].mean()


# ======================================================================================
# Model
# ======================================================================================
def build_dit(kv_cache=False):
    cfg = json.loads(CONFIG.read_text())["model"]
    return (DiTKV if kv_cache else DiT)(**cfg)


def read_weights(path):
    """A DiT state dict from any of the formats we meet:
    MeanVC release (.safetensors), MeanVC trainer checkpoints (EMA keys 'ema_model.*'),
    our own files ({'model_state_dict': ...})."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if "ema_model_state_dict" in sd:
        sd = sd["ema_model_state_dict"]
    elif "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.removeprefix("ema_model."): v for k, v in sd.items() if k not in ("initted", "step")}
    return sd


def load_weights(model, sd):
    """Load and VERIFY: every weight found with the right shape (MeanVC itself uses strict=False,
    which fails silently - the v2 EMA-prefix bug). Only the fixed rotary table may be absent."""
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("rotary_embed.inv_freq")]
    assert not missing and not unexpected, \
        f"DiT weights did not load: missing {missing[:5]} unexpected {unexpected[:5]}"
    # proj_out starts at zero in a new model: if it is not zero now, real weights arrived
    assert model.proj_out.weight.abs().sum() > 0, "DiT weights look untrained (output layer is zero)"
    return len(sd)


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1.0 - decay)


@contextlib.contextmanager
def fixed_seed(seed):
    """Same noise every validation (comparable numbers), without disturbing training randomness."""
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def meanflow_loss(mf, model, batch, device):
    return mf.loss(model, x=batch["mel"].to(device, non_blocking=True),
                   bn=batch["bn"].to(device, non_blocking=True),
                   spks=batch["xvec"].to(device, non_blocking=True),
                   prompts=batch["prompt"].to(device, non_blocking=True),
                   inputs_length=batch["lens"].to(device, non_blocking=True))


def val_loss(ema, mf, loader, device):
    """Mean-flow loss of the EMA model on the validation speakers, fixed noise."""
    total, mse, n = 0.0, 0.0, 0
    batches = iter(loader)                                    # (starting the loader uses randomness)
    with fixed_seed(123), torch.enable_grad():                # the loss needs derivatives (jvp)
        for batch in batches:
            loss, m = meanflow_loss(mf, ema, batch, device)
            total, mse, n = total + loss.item(), mse + m.item(), n + 1
    return total / max(n, 1), mse / max(n, 1)


@torch.inference_mode()
def generate(kv_model, bn, spk, prompt, device, steps=GEN_STEPS, chunk=CHUNK):
    """Streaming generation, 200 ms at a time - a copy of MeanVC's inference() (infer_ref.py)
    without the vocoder. bn (1,T,256) at 10 ms, spk (1,256), prompt (1,P,80) -> mel (T,80)."""
    times = torch.tensor([1.0, 0.8, 0.0] if steps == 2 else
                         torch.linspace(1.0, 0.0, steps + 1).tolist(), device=device)
    cache, kv_cache, offset, out = None, None, 0, []
    for start in range(0, bn.shape[1], chunk):
        bn_chunk = bn[:, start:start + chunk]
        x = torch.randn(1, bn_chunk.shape[1], 80, device=device)
        for i in range(steps):
            t, r = times[i], times[i + 1]
            u, new_kv = kv_model(x, t.expand(1), r.expand(1), cache=cache, cond=bn_chunk,
                                 spks=spk, prompts=prompt, offset=offset, is_inference=True,
                                 kv_cache=kv_cache)
            x = x - (t - r) * u
        kv_cache, offset, cache = new_kv, offset + x.shape[1], x
        out.append(x)
        if offset > 40 and kv_cache is not None and kv_cache[0][0].shape[2] > 100:
            kv_cache = [(k[:, :, -100:, :], v[:, :, -100:, :]) for k, v in kv_cache]
    return torch.cat(out, 1)[0]


def gen_set(val_df, root, per_lang):
    """Validation clips for the generation check: per language, spread over speakers. The voice
    reference is ANOTHER clip of the same speaker (its mel = prompt, its xvec = speaker), as at
    inference. Returns (language, bn, real mel, ref mel, ref xvec) tuples."""
    items = []
    for code in LANGS:
        d = val_df[val_df.code == code].sort_values("utt_id")
        groups = [g.utt_id.tolist() for _, g in d.groupby("spk") if len(g) >= 2]
        k = 0
        while groups and k < per_lang * 50:
            g = groups[k % len(groups)]
            i = k // len(groups)
            if i + 1 < len(g):
                utt, ref = g[i], g[i + 1]
                items.append((code, stretch_bn(load_npy(root, "bn", utt)),
                              torch.from_numpy(load_npy(root, "mel", utt)),
                              torch.from_numpy(load_npy(root, "mel", ref)),
                              torch.from_numpy(load_npy(root, "xvec", ref))))
                if sum(it[0] == code for it in items) >= per_lang:
                    break
            k += 1
    return items


def val_generation(kv_model, items, device):
    """Mel error (mean absolute difference, mel values in [-1, 1]) per language."""
    errs = {code: [] for code in LANGS}
    with fixed_seed(321):
        for code, bn, mel, ref_mel, ref_xvec in items:
            n = min(len(bn), len(mel))
            gen = generate(kv_model, bn[:n].unsqueeze(0).to(device), ref_xvec.unsqueeze(0).to(device),
                           ref_mel.unsqueeze(0).to(device), device)
            errs[code].append((gen[:n].float().cpu() - mel[:n]).abs().mean().item())
    return {c: float(np.mean(v)) for c, v in errs.items() if v}


def validate(ema, kv_model, mf, val_loader, gen_items, device):
    kv_model.load_state_dict(ema.state_dict())
    kv_model.eval()
    loss, mse = val_loss(ema, mf, val_loader, device)
    mel_err = val_generation(kv_model, gen_items, device)
    return loss, mse, mel_err


def save_atomic(obj, path):
    """Write to a temporary file first, so a disconnect can't leave a half-written checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def log_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(list(row))
        w.writerow(list(row.values()))


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Fine-tune the MeanVC DiT decoder")
    ap.add_argument("--smoke", action="store_true", help="quick test: few steps, separate folder")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt and start again")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-sets", default="train,train_extra")
    ap.add_argument("--feat-root", default="/content/work/dit_feats")
    ap.add_argument("--pretrained", default=str(PRETRAINED))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True              # faster float32 maths on A100
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    out_dir = PROJECT / "checkpoints" / ("dit_v3_smoke" if args.smoke else "dit_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | output: {out_dir}")

    # ---- 1. data ----
    print("1. Data")
    train_all = read_tables(args.train_sets.split(","))
    val_all = read_tables(["val"])
    full_steps = -(-len(train_all) // args.batch)              # for the smoke-test time estimate
    if args.smoke:                                             # one tar file of each
        train_all = train_all[train_all.shard == sorted(train_all.shard)[0]]
        val_all = val_all[val_all.shard == sorted(val_all.shard)[0]]
    train_df, dropped = keep_multi_clip_speakers(train_all)
    val_df, _ = keep_multi_clip_speakers(val_all)
    root = Path(args.feat_root)
    unpack_features(pd.concat([train_df, val_df]), root)
    hours = train_df.duration.sum() / 3600
    print(f"  train: {len(train_df):,} clips, {train_df.spk.nunique():,} speakers, {hours:.0f} h "
          f"({dropped} clips dropped: only clip of their speaker)")
    print("  per language: " + ", ".join(f"{c} {n:,}" for c, n in train_df.code.value_counts().items()))
    print(f"  val: {len(val_df):,} clips, {val_df.spk.nunique()} speakers (not in training)")
    same, diff = speaker_check(train_df, root)
    print(f"  speaker embeddings: same-speaker {same:.3f} vs different {diff:.3f} -> "
          f"{'OK' if same - diff > 0.2 else 'SUSPICIOUS - tell Claude'}")

    train_ds = DiTData(train_df, root, train=True)
    val_ds = DiTData(val_df, root, train=False)
    sample = train_ds[0]
    print(f"  one example: content {tuple(sample['bn'].shape)}, mel {tuple(sample['mel'].shape)} "
          f"(values {sample['mel'].min():.2f} to {sample['mel'].max():.2f}), "
          f"prompt {tuple(sample['prompt'].shape)}, speaker {tuple(sample['xvec'].shape)}")
    plan = BatchPlan(train_df.n, args.batch, shuffle=True)
    train_loader = make_loader(train_ds, plan, args.workers, device)
    val_loader = make_loader(val_ds, BatchPlan(val_df.n, args.batch, shuffle=False),
                             min(args.workers, 4), device)
    gen_items = gen_set(val_df, root, 3 if args.smoke else GEN_PER_LANG)
    steps_per_epoch = len(plan.batches())
    print(f"  {steps_per_epoch:,} batches of {args.batch} per epoch | "
          f"generation check on {len(gen_items)} val clips")

    # ---- 2. model ----
    print("2. Model")
    model = build_dit().to(device)
    n = load_weights(model, read_weights(args.pretrained))
    print(f"  DiT {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters | "
          f"loaded {n} tensors from {Path(args.pretrained).name} (all found, verified)")
    kv_model = build_dit(kv_cache=True).to(device)
    kv_model.load_state_dict(model.state_dict())               # inference DiT: same weights
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    mf = MeanFlow(**MEANFLOW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    step, epoch0, batch0, best = 0, 0, 0, float("inf")
    last, log_path = out_dir / "last.pt", out_dir / "log.csv"
    if last.exists() and not args.fresh:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        step, epoch0, batch0, best = ck["step"], ck["epoch"], ck["batch"], ck["best"]
        print(f"  RESUMED from last.pt: epoch {epoch0}, batch {batch0}, step {step}")
    else:
        if log_path.exists():
            log_path.unlink()
        print("  baseline: the Mandarin model on our validation data (before any training)")
        t0 = time.time()
        loss, mse, mel_err = validate(ema, kv_model, mf, val_loader, gen_items, device)
        print(f"=== before training: val loss {loss:.3f} (mse {mse:.3f}) | mel error "
              + " ".join(f"{c} {v:.3f}" for c, v in mel_err.items())
              + f" | avg {np.mean(list(mel_err.values())):.3f} | {time.time() - t0:.0f} s")
        log_row(log_path, dict(epoch=-1, step=0, lr=0, train_loss="", val_loss=f"{loss:.4f}",
                               val_mse=f"{mse:.4f}", **{f"mel_{c}": f"{v:.4f}" for c, v in mel_err.items()},
                               mel_avg=f"{np.mean(list(mel_err.values())):.4f}", minutes=0))

    # ---- 3. training ----
    print("3. Training")
    epochs = 1 if args.smoke else args.epochs
    smoke_steps = 30
    lr = args.lr * min(1.0, (step + 1) / WARMUP)
    for epoch in range(epoch0, epochs):
        plan.epoch, plan.start = epoch, (batch0 if epoch == epoch0 else 0)
        model.train()
        t0, t_log, sums, nb = time.time(), time.time(), np.zeros(2), 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        b = plan.start
        for batch in train_loader:
            if args.smoke and nb >= smoke_steps:
                break
            lr = args.lr * min(1.0, (step + 1) / WARMUP)
            for g in optimizer.param_groups:
                g["lr"] = lr
            try:
                loss, mse = meanflow_loss(mf, model, batch, device)
            except torch.cuda.OutOfMemoryError:
                raise SystemExit(f"Out of GPU memory at batch {b}. Run the same command with "
                                 f"--batch {args.batch * 3 // 4} (it continues from last.pt).")
            if not torch.isfinite(loss):
                print(f"  skipped batch {b}: loss is {loss.item()}")
                optimizer.zero_grad()
                b += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema_update(ema, model, EMA_DECAY)
            step, b, nb = step + 1, b + 1, nb + 1
            sums += [loss.item(), mse.item()]

            if nb % 100 == 0 or (args.smoke and nb % 10 == 0):
                k = 10 if args.smoke else 100
                sec = (time.time() - t_log) / k
                t_log = time.time()
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                print(f"  epoch {epoch} batch {b:,}/{steps_per_epoch:,} step {step:,} | "
                      f"loss {sums[0] / nb:.3f} (mse {sums[1] / nb:.3f}) | lr {lr:.1e} | "
                      f"{sec:.2f} s/batch | GPU {mem:.1f} GB | "
                      f"epoch done in ~{(steps_per_epoch - b) * sec / 60:.0f} min", flush=True)
                if args.smoke and nb == smoke_steps:
                    print(f"  SMOKE estimate for the full run: {full_steps:,} batches/epoch x "
                          f"{sec:.2f} s = ~{full_steps * sec / 60:.0f} min per epoch")
            if step % SAVE_EVERY == 0:
                save_atomic({"model": model.state_dict(), "ema": ema.state_dict(),
                             "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                             "batch": b, "best": best}, last)

        # end of epoch: training part done -> save, then validate
        save_atomic({"model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                     "batch": steps_per_epoch, "best": best}, last)
        train_mins = (time.time() - t0) / 60
        loss_v, mse_v, mel_err = validate(ema, kv_model, mf, val_loader, gen_items, device)
        avg = float(np.mean(list(mel_err.values())))
        tr = sums / max(nb, 1)
        tr_txt = f"{tr[0]:.3f}" if nb else "- (resumed)"
        print(f"=== epoch {epoch}: train loss {tr_txt} | val loss {loss_v:.3f} (mse {mse_v:.3f}) | "
              f"mel error " + " ".join(f"{c} {v:.3f}" for c, v in mel_err.items())
              + f" | avg {avg:.3f} | {train_mins:.1f} min")

        info = {"epoch": epoch, "step": step, "val_loss": loss_v, "val_mse": mse_v,
                "mel_error": mel_err, "mel_error_avg": avg, "config": "config_200ms.json",
                "encoder": "checkpoints/encoder_v3/encoder_best.pt"}
        weights = {"model_state_dict": ema.state_dict(), "info": info}
        torch.save(weights, out_dir / f"dit_epoch_{epoch:02d}.pt")
        if avg < best:
            best = avg
            torch.save(weights, out_dir / "dit_best.pt")
            (out_dir / "dit_best.json").write_text(json.dumps(info, indent=2))
            print(f"    new best (mel error {avg:.3f}) -> dit_best.pt")
        save_atomic({"model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch + 1,
                     "batch": 0, "best": best}, last)
        log_row(log_path, dict(epoch=epoch, step=step, lr=f"{lr:.1e}", train_loss=f"{tr[0]:.4f}" if nb else "",
                               val_loss=f"{loss_v:.4f}", val_mse=f"{mse_v:.4f}",
                               **{f"mel_{c}": f"{v:.4f}" for c, v in mel_err.items()},
                               mel_avg=f"{avg:.4f}", minutes=f"{train_mins:.1f}"))

    print(f"Done. Best mel error {best:.3f} -> {out_dir / 'dit_best.pt'}")


if __name__ == "__main__":
    main()