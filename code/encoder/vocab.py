"""
Stage 2a - Character vocabulary for the content encoder (Malayalam + Tamil + Hindi).

WHY
  The encoder learns to recognise speech as characters (CTC + attention loss).
  v2's vocabulary had 3 problems, fixed here:
    1. spaces were stored as '<space>', but Fast-U2++ turns spaces into '▁',
       so every space was trained as <unk>                        -> we use '▁'
    2. no <sos/eos>: Fast-U2++ uses the LAST entry, which was the quote mark '”'
                                                                   -> <sos/eos> is last
    3. stray Latin letters and symbols from transcript noise       -> only real script characters

WHAT IT DOES
  - Reads the training transcripts (data/manifests/train.tsv, column text_verbatim).
  - Removes punctuation and symbols (not spoken) and bracket tags like [noise] (not words).
  - Keeps Malayalam, Tamil and Devanagari characters (plus the zero-width joiners that
    Malayalam/Hindi spelling needs) that appear at least MIN_COUNT times.
  - Anything else that is spoken but not in the vocabulary (English letters, digits,
    very rare characters) becomes <unk>, so the timing of the transcript stays right.
  - Writes data/vocab.txt in Fast-U2++ format:  <blank> 0, <unk> 1, ▁ 2, ..., <sos/eos> last
  - Prints a report: characters per script, what was removed, and how much became <unk>.

HOW TO RUN (laptop or Colab, takes seconds)
  python code/encoder/vocab.py

USED BY TRAINING
  from vocab import load_vocab, text_to_ids
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
VOCAB_FILE = DATA / "vocab.txt"

SPACE = "▁"                          # what Fast-U2++ uses for a space between words
BLANK, UNK, SOS_EOS = "<blank>", "<unk>", "<sos/eos>"
MIN_COUNT = 20                       # rarer characters become <unk>
SCRIPTS = ("MALAYALAM", "TAMIL", "DEVANAGARI")
JOINERS = {"\u200c", "\u200d"}       # zero-width non-joiner / joiner (needed for spelling)
TAG_PATTERN = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")   # [noise], <laugh>, {breath}


def script_of(ch):
    """Which writing system a character belongs to ('MALAYALAM', 'TAMIL', 'DEVANAGARI' or None)."""
    name = unicodedata.name(ch, "")
    return next((s for s in SCRIPTS if name.startswith(s)), None)


def is_dropped(ch):
    """Punctuation and symbols are not spoken, so they are removed from the text."""
    return unicodedata.category(ch)[0] in "PS"     # P* = punctuation, S* = symbols


def normalize(text):
    """Same cleaning for building the vocab, training and measuring CER later."""
    text = unicodedata.normalize("NFC", str(text))
    text = TAG_PATTERN.sub(" ", text)
    text = "".join(" " if is_dropped(c) else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def to_tokens(text):
    """'ഒരു വാക്ക്' -> ['ഒ', 'ര', 'ു', '▁', 'വ', ...] (one token per character, ▁ for spaces)."""
    return [SPACE if c == " " else c for c in normalize(text)]


def load_vocab(path=VOCAB_FILE):
    """Read vocab.txt -> {token: id}."""
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok, idx = line.rstrip("\n").rsplit(" ", 1)
            table[tok] = int(idx)
    return table


def text_to_ids(text, table):
    """Transcript -> list of token ids for CTC / attention training."""
    unk = table[UNK]
    return [table.get(t, unk) for t in to_tokens(text)]


# ======================================================================================
def build():
    import pandas as pd

    train = pd.read_csv(DATA / "manifests" / "train.tsv", sep="\t", keep_default_na=False,
                        usecols=["code", "text_verbatim"])

    counts, dropped, tags = Counter(), Counter(), 0
    for text in train.text_verbatim:
        raw = unicodedata.normalize("NFC", str(text))
        tags += len(TAG_PATTERN.findall(raw))
        dropped.update(c for c in TAG_PATTERN.sub(" ", raw) if is_dropped(c))
        counts.update(c for c in normalize(raw) if c != " ")

    # Letters/vowel signs of the 3 scripts (not their digits: a digit is spoken as a word).
    keep = sorted(c for c, n in counts.items()
                  if (script_of(c) or c in JOINERS) and n >= MIN_COUNT
                  and unicodedata.category(c)[0] != "N")
    tokens = [BLANK, UNK, SPACE] + keep + [SOS_EOS]
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        for i, tok in enumerate(tokens):
            f.write(f"{tok} {i}\n")

    # ---------------------------- report ----------------------------
    print(f"Wrote {VOCAB_FILE}  ({len(tokens)} tokens: <blank>, <unk>, ▁, "
          f"{len(keep)} characters, <sos/eos>)")
    per_script = Counter(script_of(c) or "joiner" for c in keep)
    print("Characters per script:", dict(per_script))

    rare = {c: n for c, n in counts.items() if (script_of(c) or c in JOINERS) and n < MIN_COUNT}
    other = {c: n for c, n in counts.items() if not (script_of(c) or c in JOINERS)}
    print(f"Removed (not spoken): {tags:,} bracket tags, {sum(dropped.values()):,} punctuation/symbols"
          + (f", most common: {''.join(c for c, _ in dropped.most_common(15))}" if dropped else ""))
    print(f"Become <unk>: {len(rare)} rare script characters, {len(other)} other characters "
          f"(e.g. English letters, digits)"
          + (f": {''.join(sorted(other, key=lambda c: -other[c])[:30])}" if other else ""))

    table = {t: i for i, t in enumerate(tokens)}
    print("\nShare of characters that become <unk> (lower is better, aim for < 1%):")
    for code, g in train.groupby("code"):
        ids = [i for t in g.text_verbatim for i in text_to_ids(t, table) if i != table[SPACE]]
        unk = sum(i == table[UNK] for i in ids)
        print(f"  {code}: {unk / max(len(ids), 1) * 100:5.2f}%  ({unk:,} of {len(ids):,})")

    sample = train.text_verbatim.iloc[0]
    print("\nExample:", sample[:60])
    print("Tokens :", " ".join(to_tokens(sample)[:30]))


if __name__ == "__main__":
    build()