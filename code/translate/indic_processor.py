"""
Stage 6 - Text pre/post-processing for IndicTrans2 (pure-Python copy of IndicTransToolkit's IndicProcessor).

PURPOSE
  IndicTrans2 does not read raw text. Every sentence must be prepared the way the model was trained:
    before translation  punctuation + digit normalisation, Indic normalisation, tokenisation,
                        script unification (Malayalam/Tamil script -> Devanagari, the model's
                        shared script) and the two language tags "<src> <tgt>" in front;
    after translation   Devanagari -> the target language's own script, placeholders restored,
                        detokenisation.
  The official tool for this is IndicTransToolkit.processor.IndicProcessor (AI4Bharat, MIT licence).

WHY A COPY
  IndicTransToolkit ships only as Cython source. On the Windows laptop there is no C compiler, so
  `pip install IndicTransToolkit` fails ("Failed building wheel"). This file is the same algorithm
  (version 1.1.1, processor.pyx) written as plain Python, limited to what Indic <-> Indic translation
  needs (the English path is not used here). It needs only pure-Python packages:
    pip install indic-nlp-library-itt regex sacremoses

ROLE IN THE PIPELINE
  speech_translator.py -> Translator uses the official IndicProcessor when it is installed (Colab),
  otherwise this one. Both give the same text.
    ASR text -> preprocess_batch -> IndicTrans2 (beam search) -> postprocess_batch -> TTS

  Original: https://github.com/VarunGumma/IndicTransToolkit  (MIT licence, Copyright (c) Varun Gumma)
"""

from queue import Queue

import regex as re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_detokenize, indic_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# FLORES language code -> ISO code used by the Indic NLP library (the languages of this project)
FLORES_TO_ISO = {"mal_Mlym": "ml", "tam_Taml": "ta", "hin_Deva": "hi"}

# Indic digits -> ASCII digits, the same table as the toolkit (Telugu: 1-9 only, as there)
_DIGITS = {}
for _zero in (0x09E6, 0x0AE6, 0x0CE6, 0x0966, 0x0660, 0xABF0, 0x0B66, 0x0A66, 0x1C50, 0x06F0):
    for _d in range(10):
        _DIGITS[_zero + _d] = str(_d)
for _d in range(1, 10):
    _DIGITS[0x0C66 + _d] = str(_d)

_MULTISPACE = re.compile(r"[ ]{2,}")
_DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
_DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
_DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
_END_BRACKET_SPACE_PUNC = re.compile(r"\) ([\.!:?;,])")
_URL = re.compile(r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b")
_NUMERAL = re.compile(r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?"
                      r"|\d+[-\/.:'+]\d+(?:\.\d+)?)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
_OTHER = re.compile(r"[A-Za-z0-9]*[#|@]\w+")
_PUNC_REPLACEMENTS = [
    (re.compile(r"\r"), ""),
    (re.compile(r"\(\s*"), "("),
    (re.compile(r"\s*\)"), ")"),
    (re.compile(r"\s:\s?"), ":"),
    (re.compile(r"\s;\s?"), ";"),
    (re.compile(r"[`´‘‚’]"), "'"),
    (re.compile(r"[„“”«»]"), '"'),
    (re.compile(r"[–—]"), "-"),
    (re.compile(r" %"), "%"),
    (re.compile(r" [?!;]"), lambda m: m.group(0).strip()),
]
# Ways the model sometimes rewrites a placeholder "<ID1>" (the toolkit restores all of them)
_PLACEHOLDER_FORMS = ["<ID{n}>", "< ID{n} >", "[ID{n}]", "[ ID{n} ]", "[ID {n}]", "<ID{n}]", "< ID{n}]",
                      "<ID{n} ]"]
_INDIC_ID = ["आईडी", "आई . डी . ", "आई . डी .", "आई. डी. ", "आई. डी.", "आय. डी. ", "आय. डी.",
             "आय . डी . ", "आय . डी .", "आइ . डी . ", "आइ . डी .", "आइ. डी. ", "आइ. डी.", "ऐटि", "आयडी",
             "ऐडि", "आइडि"]
_INDIC_ID_FORMS = ["<{c}{n}>", "< {c}{n} >", "< {c} {n} >", "<{c} {n}]", "< {c} {n} ]", "[{c}{n}]",
                   "[{c} {n}]", "[ {c}{n} ]", "[ {c} {n} ]", "{c} {n}", "{c}{n}"]


class IndicProcessor:
    """Same interface as IndicTransToolkit.processor.IndicProcessor (Indic languages only)."""

    def __init__(self, inference=True):
        self.inference = inference
        self._maps = Queue()                       # placeholder maps, one per preprocessed sentence
        self._xlit = UnicodeIndicTransliterator()

    # ---------------- preprocessing ----------------
    @staticmethod
    def _punc_norm(text):
        for pattern, repl in _PUNC_REPLACEMENTS:
            text = pattern.sub(repl, text)
        text = _MULTISPACE.sub(" ", text)
        text = _END_BRACKET_SPACE_PUNC.sub(r")\1", text)
        text = _DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = _DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = _DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    def _wrap_with_placeholders(self, text):
        """Numbers, e-mails, URLs, @tags -> <ID1>, <ID2>, ... so the model copies them unchanged."""
        serial, entity_map = 1, {}
        for pattern in (_EMAIL, _URL, _NUMERAL, _OTHER):
            for match in set(pattern.findall(text)):
                if pattern is _URL and len(match.replace(".", "")) < 4:
                    continue
                if pattern is _NUMERAL and len(match.replace(" ", "").replace(".", "").replace(":", "")) < 4:
                    continue
                for form in _PLACEHOLDER_FORMS:
                    entity_map[form.format(n=serial)] = match
                    entity_map[form.replace("ID", "id").format(n=serial)] = match
                for c in _INDIC_ID:
                    for form in _INDIC_ID_FORMS:
                        entity_map[form.format(c=c, n=serial)] = match
                text = text.replace(match, f"<ID{serial}>")
                serial += 1
        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        self._maps.put(entity_map)
        return text

    def _preprocess(self, sent, src_lang, tgt_lang, normalizer, is_target):
        iso = FLORES_TO_ISO[src_lang]
        sent = self._punc_norm(sent).translate(_DIGITS)
        if self.inference:
            sent = self._wrap_with_placeholders(sent)
        tokens = indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso)
        sent = self._xlit.transliterate(" ".join(tokens), iso, "hi").replace(" ् ", "्").strip()
        return sent if is_target else f"{src_lang} {tgt_lang} {sent}"

    def preprocess_batch(self, batch, src_lang, tgt_lang=None, is_target=False, visualize=False):
        normalizer = IndicNormalizerFactory().get_normalizer(FLORES_TO_ISO[src_lang])
        return [self._preprocess(s, src_lang, tgt_lang, normalizer, is_target) for s in batch]

    # ---------------- postprocessing ----------------
    def _postprocess(self, sent, lang, entity_map):
        if isinstance(sent, (tuple, list)):
            sent = sent[0]
        for k, v in entity_map.items():
            sent = sent.replace(k, v)
        iso = FLORES_TO_ISO[lang]
        return indic_detokenize.trivial_detokenize(self._xlit.transliterate(sent, "hi", iso), iso)

    def postprocess_batch(self, sents, lang="hin_Deva", visualize=False, num_return_sequences=1):
        maps = [self._maps.get() for _ in range(len(sents) // num_return_sequences)]
        out = [self._postprocess(s, lang, maps[i // num_return_sequences]) for i, s in enumerate(sents)]
        self._maps.queue.clear()
        return out
