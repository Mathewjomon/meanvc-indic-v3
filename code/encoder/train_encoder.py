"""
Stage 2b - Fine-tune the Fast-U2++ content encoder on Malayalam + Tamil + Hindi (run on Colab).

WHAT IT DOES
  1. Unpacks the training audio (data/audio_train/*.tar) onto Colab's fast local disk.
  2. Builds the 7-layer Conformer (same shape as the original MeanVC encoder) and loads the
     ORIGINAL Mandarin weights (checkpoints/pretrained/fastu2++.pt). The CTC head and the
     attention decoder are new, sized for our 174-token vocabulary.
  3. Switches every layer to STREAMING convolution (past-only), exactly like the original
     MeanVC encoder - v2 used the non-streaming default by mistake.
  4. Trains with 200 ms chunks and 2 chunks of memory (chunk 5, 2 left chunks): the exact
     setting used later to extract content features, so training == extraction.
     (20% of batches use random chunk sizes for robustness.)
  5. Loss: 0.3 CTC + 0.7 attention. Augmentation: noise/babble, volume, SpecAugment, dither.
  6. After every epoch: validation loss + character error rate (CER) per language on the
     held-out speakers, a log line in log.csv, and checkpoints on Drive.
  7. Can resume after a Colab disconnect: just run the same command again.

LESSONS FROM v2 BUILT IN
  - checks that pretrained weights really loaded (missing = unexpected = 0, values changed)
  - checks that chunked training gives the same output as slice-by-slice extraction
  - vocabulary: '▁' for spaces, real <sos/eos> (see vocab.py)

HOW TO RUN (Colab)
  quick test (~2 min):  !python .../code/encoder/train_encoder.py --smoke
  full training:        !python .../code/encoder/train_encoder.py
OUTPUT
  checkpoints/encoder_v3/encoder_best.pt   encoder weights for feature extraction (Stage 3)
  checkpoints/encoder_v3/log.csv           one line per epoch (loss, CER per language)
"""

import argparse
import contextlib
import csv
import io
import json
import math
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                 # code/encoder
PROJECT = HERE.parents[1]                              # meanvc_indic_v3
DATA = PROJECT / "data"
PRETRAINED = PROJECT / "checkpoints" / "pretrained" / "fastu2++.pt"
sys.path.insert(0, str(PROJECT / "third_party" / "fast-u2pp"))   # 'wenet' (Fast-U2++ code)
sys.path.insert(0, str(HERE))                                    # vocab.py

from vocab import BLANK, SOS_EOS, SPACE, UNK, load_vocab, text_to_ids  # noqa: E402

# --------------------------------------------------------------------------------------
# Model shape: identical to the original MeanVC encoder (checked against fastu2++.pt)
# --------------------------------------------------------------------------------------
MODEL_CONFIG = {
    "encoder": "conformer",
    "encoder_conf": dict(
        output_size=256, attention_heads=4, linear_units=2048, num_blocks=7,
        dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.1,
        input_layer="conv2d", normalize_before=True, cnn_module_kernel=9,
        cnn_module_norm="layer_norm", use_cnn_module=True, activation_type="swish",
        pos_enc_layer_type="rel_pos", selfattention_layer_type="rel_selfattn",
        macaron_style=True, causal=True, use_dynamic_chunk=True,
        use_dynamic_left_chunk=False, dual=True),
    "decoder": "transformer",
    "decoder_conf": dict(
        attention_heads=4, linear_units=2048, num_blocks=3, dropout_rate=0.1,
        positional_dropout_rate=0.1, self_attention_dropout_rate=0.1,
        src_attention_dropout_rate=0.1),
    "model_conf": dict(
        ctc_weight=0.3, lsm_weight=0.1, length_normalized_loss=False, reverse_weight=0.0,
        s_chunk_weight=0.0, distill_weight=0.0, regular=False, is_l_ctc=False,
        is_l_ctc_non_stream=False, output_cache=None, distill_loss_type="l1"),
    "input_dim": 80,
    "is_json_cmvn": True,
}

# --------------------------------------------------------------------------------------
# Training settings - change values here
# --------------------------------------------------------------------------------------
CHUNK, LEFT_CHUNKS = 5, 2      # 200 ms chunks, 2 chunks of memory = same as feature extraction
P_FIXED_CHUNK = 0.8            # share of batches trained with exactly (5, 2); rest random chunks
EPOCHS = 30
MAX_FRAMES = 60000             # fbank frames per batch (10 ms each) -> ~600 s of audio per batch
LR, WARMUP = 5e-4, 1000        # peak learning rate and warm-up steps (Noam schedule, as in v2)
GRAD_CLIP = 5.0
CTC_WEIGHT = 0.3               # loss = 0.3 * CTC + 0.7 * attention
DITHER = 0.1                   # tiny random noise in fbank (training only)
NOISE_PROB = 0.5               # share of clips that get background noise or babble
SPEC_AUG = dict(num_t_mask=2, max_t=50, num_f_mask=2, max_f=10)
KEEP_LAST = 3                  # epoch checkpoints to keep (plus best and last)


# ======================================================================================
# Features: shared with feature extraction (Stage 3) so both compute EXACTLY the same thing
# ======================================================================================
def compute_fbank(wav, dither=0.0):
    """16 kHz float waveform in [-1, 1] -> (frames, 80) log-mel fbank, as MeanVC expects:
    scaled to 16-bit range (x32768), 25 ms window, 10 ms shift, no energy floor."""
    x = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0) * (1 << 15)
    return kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=dither,
                       energy_floor=0.0, sample_frequency=16000)


def build_model(vocab_size, cmvn_json):
    """Fast-U2++ model (encoder + CTC head + attention decoder) with our vocabulary size."""
    from wenet.transformer.asr_model import init_asr_model
    cfg = dict(MODEL_CONFIG, output_dim=vocab_size, cmvn_file=str(cmvn_json))
    with contextlib.redirect_stdout(io.StringIO()):          # init_asr_model prints the whole model
        return init_asr_model(cfg)


def set_streaming(model):
    """Use the past-only (streaming) convolution in every layer, like the original MeanVC.
    Attention chunking is controlled per call (chunk size / left chunks), not here."""
    for layer in model.encoder.encoders:
        layer.conv_module.set_dual_mode(True)
    assert all(layer.conv_module.streaming for layer in model.encoder.encoders)


def encoder_weights_from(path):
    """Encoder weights from the original TorchScript file, or from a saved state dict."""
    try:
        sd = torch.jit.load(str(path), map_location="cpu").state_dict()
    except Exception:
        sd = torch.load(str(path), map_location="cpu")
        sd = sd.get("model", sd)
    return {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}


def write_cmvn_json(enc_sd, path):
    """Fast-U2++ needs a CMVN file to build the model. We write one that reproduces the
    pretrained encoder's own normalisation (it is overwritten by the weights anyway)."""
    mean = enc_sd["global_cmvn.mean"].double()
    istd = enc_sd["global_cmvn.istd"].double()
    var_stat = mean ** 2 + 1.0 / istd ** 2
    path.write_text(json.dumps({"mean_stat": mean.tolist(), "var_stat": var_stat.tolist(),
                                "frame_num": 1}))


def load_pretrained_encoder(model, enc_sd):
    """Load encoder weights and PROVE they loaded (v2 lesson: strict=False fails silently)."""
    key = "encoders.0.self_attn.linear_q.weight"
    before = model.encoder.state_dict()[key].clone()
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    after = model.encoder.state_dict()[key]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    assert not torch.equal(before, after), "weights did not change - load failed"
    assert torch.equal(after, enc_sd[key].to(after.dtype)), "loaded values differ from file"
    return len(enc_sd)


# ======================================================================================
# Data
# ======================================================================================
def unpack_audio(audio_root):
    """Extract data/audio_train/*.tar to fast local disk (skipped if already done)."""
    audio_root.mkdir(parents=True, exist_ok=True)
    for tar_path in sorted((DATA / "audio_train").glob("*.tar")):
        done = audio_root / f".{tar_path.stem}.done"
        if done.exists():
            continue
        t0 = time.time()
        with tarfile.open(tar_path) as tar:
            try:
                tar.extractall(audio_root, filter="data")
            except TypeError:                                  # older Python: no filter arg
                tar.extractall(audio_root)
        done.touch()
        print(f"  unpacked {tar_path.name} ({time.time() - t0:.0f} s)")


class SpeechDataset(torch.utils.data.Dataset):
    """One item = (fbank features, token ids, language) for one clip."""

    def __init__(self, manifest, audio_root, vocab, train):
        df = pd.read_csv(manifest, sep="\t", keep_default_na=False, dtype={"speaker_id": str})
        df["ids"] = [text_to_ids(t, vocab) for t in df.text_verbatim]
        # CTC needs at least one encoder frame per token: 25 encoder frames per second.
        feasible = df.ids.str.len().between(1, (df.duration * 25 - 2).astype(int))
        if (~feasible).sum():
            print(f"  skipped {(~feasible).sum()} clips (empty transcript or too many characters per second)")
        self.df = df[feasible].reset_index(drop=True)
        self.audio_root, self.train = Path(audio_root), train

    def __len__(self):
        return len(self.df)

    def read(self, i):
        wav, sr = sf.read(self.audio_root / self.df.path[i], dtype="float32")
        assert sr == 16000, f"{self.df.path[i]}: {sr} Hz"
        return wav

    def augment(self, wav):
        """Real-world robustness: background noise or another speaker's babble, and volume."""
        if random.random() < NOISE_PROB:
            power = float(np.mean(wav ** 2)) + 1e-8
            if random.random() < 0.5:                            # coloured background noise
                noise = np.random.randn(len(wav)).astype(np.float32)
                k = random.choice([1, 3, 9])                     # 1 = white, larger = duller
                noise = np.convolve(noise, np.ones(k) / k, mode="same").astype(np.float32)
                snr = random.uniform(10, 30)
            else:                                                # babble: another clip, quieter
                other = self.read(random.randrange(len(self.df)))
                noise = np.resize(other, len(wav)).astype(np.float32)
                snr = random.uniform(12, 25)
            noise_power = float(np.mean(noise ** 2)) + 1e-8
            wav = wav + noise * math.sqrt(power / (noise_power * 10 ** (snr / 10)))
        wav = wav * 10 ** (random.uniform(-6, 6) / 20)           # volume +-6 dB
        return np.clip(wav, -1.0, 1.0)

    def __getitem__(self, i):
        wav = self.read(i)
        if self.train:
            wav = self.augment(wav)
        feats = compute_fbank(wav, dither=DITHER if self.train else 0.0)
        if self.train:                                           # SpecAugment
            T, F = feats.shape
            for _ in range(SPEC_AUG["num_t_mask"]):
                w = random.randint(0, min(SPEC_AUG["max_t"], max(T // 5, 1)))
                s = random.randint(0, max(T - w, 0))
                feats[s:s + w, :] = 0
            for _ in range(SPEC_AUG["num_f_mask"]):
                w = random.randint(0, SPEC_AUG["max_f"])
                s = random.randint(0, F - w)
                feats[:, s:s + w] = 0
        return feats, torch.tensor(self.df.ids[i], dtype=torch.long), self.df.code[i]


class BucketBatches(torch.utils.data.Sampler):
    """Batches of clips with similar length (little padding), at most MAX_FRAMES padded frames."""

    def __init__(self, durations, max_frames, shuffle, seed=0):
        self.frames = (np.asarray(durations) * 100).astype(int)
        self.max_frames, self.shuffle, self.epoch, self.seed = max_frames, shuffle, 0, seed

    def batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        jitter = rng.uniform(0, 50, len(self.frames)) if self.shuffle else 0
        order = np.argsort(self.frames + jitter)
        out, cur, longest = [], [], 0
        for i in order:
            longest_new = max(longest, self.frames[i])
            if cur and longest_new * (len(cur) + 1) > self.max_frames:
                out.append(cur)
                cur, longest_new = [], self.frames[i]
            cur.append(int(i))
            longest = longest_new
        if cur:
            out.append(cur)
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())


def collate(items):
    feats, ids, langs = zip(*items)
    return (pad_sequence(feats, batch_first=True), torch.tensor([len(f) for f in feats]),
            pad_sequence(ids, batch_first=True, padding_value=-1),          # -1 = ignore
            torch.tensor([len(t) for t in ids]), list(langs))


def worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


# ======================================================================================
# Training pieces
# ======================================================================================
def noam_lr(step):
    """Warm up for WARMUP steps to LR, then decay with 1/sqrt(step) (same formula as v2)."""
    step = max(step, 1)
    return LR * WARMUP ** 0.5 * min(step ** -0.5, step * WARMUP ** -1.5)


def run_encoder(model, feats, lens, chunk, left):
    """Encoder forward with chunked attention. chunk=5,left=2 == slice-by-slice extraction."""
    out = model.encoder(feats, lens, s_decoding_chunk_size=chunk,
                        s_num_decoding_left_chunks=left)
    enc, mask = out[0], out[1]
    return enc, mask, mask.squeeze(1).sum(1)


def losses(model, batch, device, chunk, left):
    feats, flens, ids, ilens, _ = batch
    feats, flens, ids, ilens = feats.to(device), flens.to(device), ids.to(device), ilens.to(device)
    enc, mask, elens = run_encoder(model, feats, flens, chunk, left)
    loss_ctc = model.ctc(enc.float(), elens, ids, ilens)
    loss_att, acc = model._calc_att_loss(enc, mask, ids, ilens)
    loss = CTC_WEIGHT * loss_ctc + (1 - CTC_WEIGHT) * loss_att.float()
    return loss, loss_ctc, loss_att, enc, elens


def greedy_ctc_text(model, enc, elens, id2tok):
    """Best-path CTC decoding -> text (for the CER check during training)."""
    best = model.ctc.log_softmax(enc.float()).argmax(-1).cpu()
    texts = []
    for seq, n in zip(best, elens.cpu()):
        prev, chars = None, []
        for t in seq[:n].tolist():
            if t != prev and t != 0:
                tok = id2tok[t]
                chars.append(" " if tok == SPACE else ("" if tok in (UNK, SOS_EOS) else tok))
            prev = t
        texts.append(" ".join("".join(chars).split()))
    return texts


@torch.no_grad()
def evaluate(model, loader, device, id2tok, amp):
    import jiwer
    model.eval()
    tot, n, refs, hyps = 0.0, 0, {}, {}
    for batch in loader:
        with amp():
            loss, _, _, enc, elens = losses(model, batch, device, CHUNK, LEFT_CHUNKS)
        tot, n = tot + loss.item(), n + 1
        for lang, ref_ids, hyp in zip(batch[4], batch[2], greedy_ctc_text(model, enc, elens, id2tok)):
            ref = "".join(" " if id2tok[i] == SPACE else id2tok[i] for i in ref_ids.tolist() if i >= 0)
            refs.setdefault(lang, []).append(" ".join(ref.split()))
            hyps.setdefault(lang, []).append(hyp)
    model.train()
    cer = {lang: jiwer.cer(refs[lang], hyps[lang]) * 100 for lang in sorted(refs)}
    return tot / max(n, 1), cer, (refs, hyps)


@torch.no_grad()
def streaming_check(model, wav, device):
    """Prove training == extraction: chunked full-utterance encoder output must match the
    slice-by-slice output used for content features (200 ms slices, 2 slices of memory)."""
    model.eval()
    feats = compute_fbank(wav).unsqueeze(0).to(device)
    full, _, _ = run_encoder(model, feats, torch.tensor([feats.shape[1]], device=device),
                             CHUNK, LEFT_CHUNKS)
    step, window = CHUNK * 4, (CHUNK - 1) * 4 + 7          # 20 frames step, 23 frames window
    att = torch.zeros((0, 0, 0, 0), device=device)
    cnn = torch.zeros((0, 0, 0, 0), device=device)
    outs, offset = [], 0
    for i in range(0, feats.shape[1], step):
        piece = feats[:, i:i + window, :]
        if piece.shape[1] < 10:
            break
        y, att, cnn = model.forward_encoder_chunk(piece, offset, CHUNK * LEFT_CHUNKS, att, cnn)
        outs.append(y)
        offset += y.size(1)
    stream = torch.cat(outs, 1)
    n = min(full.shape[1], stream.shape[1])
    diff = (full[0, :n] - stream[0, :n]).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(full[0, :n], stream[0, :n], dim=-1).mean().item()
    model.train()
    return diff, cos, full.shape[1], stream.shape[1]


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Fine-tune the Fast-U2++ content encoder")
    ap.add_argument("--smoke", action="store_true", help="quick test: few batches, separate folder")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt and start again")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--audio-root", default="/content/work/audio16k")
    ap.add_argument("--pretrained", default=str(PRETRAINED))
    ap.add_argument("--train-manifest", default="train.tsv",
                    help="training list in data/manifests, e.g. train_extra.tsv for round 2")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" \
        else contextlib.nullcontext
    out_dir = PROJECT / "checkpoints" / ("encoder_v3_smoke" if args.smoke else "encoder_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | output: {out_dir}")

    # ---- data ----
    print("1. Audio")
    audio_root = Path(args.audio_root)
    unpack_audio(audio_root)
    vocab = load_vocab()
    id2tok = {i: t for t, i in vocab.items()}
    assert vocab[BLANK] == 0 and vocab[SOS_EOS] == len(vocab) - 1, "vocab.txt has the wrong layout"
    train_ds = SpeechDataset(DATA / "manifests" / args.train_manifest, audio_root, vocab, train=True)
    val_ds = SpeechDataset(DATA / "manifests" / "val.tsv", audio_root, vocab, train=False)
    if args.smoke:
        val_ds.df = val_ds.df.groupby("code").head(10).reset_index(drop=True)
    train_sampler = BucketBatches(train_ds.df.duration, args.max_frames, shuffle=True)
    loader_kw = dict(num_workers=args.workers, collate_fn=collate, worker_init_fn=worker_init,
                     pin_memory=(device == "cuda"), persistent_workers=args.workers > 0)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler=train_sampler, **loader_kw)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=BucketBatches(val_ds.df.duration, args.max_frames, shuffle=False),
        **loader_kw)
    print(f"  train {len(train_ds):,} clips in {len(train_sampler)} batches | "
          f"val {len(val_ds):,} clips | vocab {len(vocab)} tokens")

    # ---- model ----
    print("2. Model")
    enc_sd = encoder_weights_from(args.pretrained)
    cmvn_json = out_dir / "cmvn_from_pretrained.json"
    write_cmvn_json(enc_sd, cmvn_json)
    model = build_model(len(vocab), cmvn_json)
    n_loaded = load_pretrained_encoder(model, enc_sd)
    set_streaming(model)
    model.to(device)
    print(f"  loaded {n_loaded} pretrained encoder tensors (missing 0, unexpected 0, values changed)")
    print(f"  streaming convolution in all {len(model.encoder.encoders)} layers")
    diff, cos, n_full, n_stream = streaming_check(model, val_ds.read(0), device)
    print(f"  training-vs-extraction check: max difference {diff:.2e}, cosine {cos:.6f} "
          f"({n_full} vs {n_stream} frames) -> {'OK' if diff < 1e-3 else 'MISMATCH - tell Claude'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    step, start_epoch, best_cer = 0, 0, float("inf")
    last = out_dir / "last.pt"
    if last.exists() and not args.fresh:
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, start_epoch, best_cer = ck["step"], ck["epoch"] + 1, ck["best_cer"]
        print(f"  RESUMED from last.pt: epoch {start_epoch}, step {step}, best CER {best_cer:.2f}%")

    # ---- training ----
    print("3. Training")
    log_path = out_dir / "log.csv"
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(start_epoch, epochs):
        t0, model_sums, nb = time.time(), np.zeros(3), 0
        train_sampler.epoch = epoch
        model.train()
        for b, batch in enumerate(train_loader):
            if args.smoke and b >= 20:
                break
            lr = noam_lr(step + 1)
            for g in optimizer.param_groups:
                g["lr"] = lr
            fixed = random.random() < P_FIXED_CHUNK
            with amp():
                loss, l_ctc, l_att, _, _ = losses(model, batch, device,
                                                  CHUNK if fixed else 0, LEFT_CHUNKS if fixed else -1)
            if not torch.isfinite(loss):
                print(f"  skipped batch {b}: loss is {loss.item()}")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            step, nb = step + 1, nb + 1
            model_sums += [loss.item(), l_ctc.item(), l_att.item()]
            if b % 50 == 0:
                print(f"  epoch {epoch} batch {b}/{len(train_sampler)} step {step} lr {lr:.2e} "
                      f"loss {loss.item():.2f} (ctc {l_ctc.item():.2f}, att {l_att.item():.2f})")

        val_loss, cer, (refs, hyps) = evaluate(model, val_loader, device, id2tok, amp)
        avg_cer = float(np.mean(list(cer.values())))
        mins = (time.time() - t0) / 60
        tr = model_sums / max(nb, 1)
        print(f"=== epoch {epoch}: train loss {tr[0]:.2f} | val loss {val_loss:.2f} | CER "
              + " ".join(f"{k} {v:.1f}%" for k, v in cer.items()) + f" | avg {avg_cer:.1f}% | {mins:.1f} min")
        lang = next(iter(refs))
        print(f"    example ({lang}) ref: {refs[lang][0][:70]}\n                 hyp: {hyps[lang][0][:70]}")

        # ---- checkpoints (every epoch, so a disconnect loses at most one epoch) ----
        state = model.state_dict()
        torch.save(state, out_dir / f"epoch_{epoch:02d}.pt")
        for old in sorted(out_dir.glob("epoch_*.pt"))[:-KEEP_LAST]:
            old.unlink()
        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save(state, out_dir / "best.pt")
            torch.save({f"encoder.{k}": v for k, v in model.encoder.state_dict().items()},
                       out_dir / "encoder_best.pt")
            (out_dir / "encoder_best.json").write_text(json.dumps(
                {"epoch": epoch, "val_cer": cer, "avg_cer": avg_cer, "chunk": CHUNK,
                 "left_chunks": LEFT_CHUNKS, "streaming_conv": True, "vocab": "data/vocab.txt"},
                indent=2))
            print(f"    new best (avg CER {avg_cer:.1f}%) -> encoder_best.pt")
        torch.save({"model": state, "optimizer": optimizer.state_dict(), "step": step,
                    "epoch": epoch, "best_cer": best_cer}, last)

        new_file = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["epoch", "step", "lr", "train_loss", "train_ctc", "train_att", "val_loss"]
                           + [f"cer_{k}" for k in cer] + ["cer_avg", "minutes"])
            w.writerow([epoch, step, f"{lr:.2e}", *[f"{x:.3f}" for x in tr], f"{val_loss:.3f}"]
                       + [f"{v:.2f}" for v in cer.values()] + [f"{avg_cer:.2f}", f"{mins:.1f}"])

    print(f"Done. Best average CER {best_cer:.1f}% -> {out_dir / 'encoder_best.pt'}")


if __name__ == "__main__":
    main()