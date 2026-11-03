# Voice conversion results (v3)

### Table 1 - All conversions (same- and cross-language)

| System | Speaker similarity ↑ | Similarity, other clip ↑ | Source leakage ↓ | CER % (MMS) ↓ | CER % (our ASR) ↓ | Quality (DNSMOS) ↑ | RTF ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Real speech (reference point) | 0.753 | - | 0.194 | 23.1 | 25.0 | 3.07 | - |
| Original MeanVC (Mandarin) | 0.596 | 0.480 | 0.253 | 46.8 | 47.9 | 3.02 | 0.15 |
| Our encoder + Mandarin decoder | 0.451 | 0.359 | 0.114 | 68.1 | 68.1 | 2.94 | 0.17 |
| Ours (encoder + decoder fine-tuned) | 0.707 | 0.573 | 0.290 | 31.7 | 32.1 | 3.03 | 0.17 |

### Table 2 - Same-language vs cross-language conversion

| System, conversion | Speaker similarity ↑ | Similarity, other clip ↑ | Source leakage ↓ | CER % (MMS) ↓ | CER % (our ASR) ↓ | Quality (DNSMOS) ↑ | RTF ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original MeanVC, same language | 0.619 | 0.505 | 0.272 | 46.2 | 46.2 | 3.02 | 0.16 |
| Ours, same language | 0.746 | 0.612 | 0.313 | 31.7 | 30.5 | 3.01 | 0.17 |
| Original MeanVC, cross-language | 0.584 | 0.467 | 0.244 | 47.1 | 48.8 | 3.03 | 0.15 |
| Ours, cross-language | 0.688 | 0.553 | 0.279 | 31.7 | 32.9 | 3.04 | 0.17 |

### Table 3 - Ours per language pair

| Source -> target voice | Speaker similarity ↑ | Similarity, other clip ↑ | Source leakage ↓ | CER % (MMS) ↓ | CER % (our ASR) ↓ | Quality (DNSMOS) ↑ | RTF ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Malayalam -> Malayalam | 0.752 | 0.583 | 0.347 | 38.8 | 36.5 | 2.91 | 0.17 |
| Malayalam -> Tamil | 0.715 | 0.603 | 0.314 | 28.3 | 29.6 | 3.02 | 0.17 |
| Malayalam -> Hindi | 0.649 | 0.544 | 0.271 | 36.1 | 40.4 | 2.97 | 0.17 |
| Tamil -> Malayalam | 0.733 | 0.564 | 0.304 | 44.2 | 41.0 | 2.93 | 0.17 |
| Tamil -> Tamil | 0.761 | 0.660 | 0.323 | 33.7 | 29.6 | 3.00 | 0.17 |
| Tamil -> Hindi | 0.623 | 0.529 | 0.301 | 39.6 | 37.0 | 3.05 | 0.17 |
| Hindi -> Malayalam | 0.705 | 0.509 | 0.238 | 20.5 | 23.7 | 3.12 | 0.17 |
| Hindi -> Tamil | 0.703 | 0.568 | 0.243 | 19.4 | 23.9 | 3.14 | 0.17 |
| Hindi -> Hindi | 0.724 | 0.593 | 0.270 | 19.7 | 23.4 | 3.12 | 0.17 |

### Table 4 - Ours: speaker similarity by language pair

| source \ target voice | Malayalam | Tamil | Hindi |
|---|:---:|:---:|:---:|
| Malayalam | 0.752 | 0.715 | 0.649 |
| Tamil | 0.733 | 0.761 | 0.623 |
| Hindi | 0.705 | 0.703 | 0.724 |

### Table 5 - Ours: CER % (MMS) by language pair

| source \ target voice | Malayalam | Tamil | Hindi |
|---|:---:|:---:|:---:|
| Malayalam | 38.8 | 28.3 | 36.1 |
| Tamil | 44.2 | 33.7 | 39.6 |
| Hindi | 20.5 | 19.4 | 19.7 |

**Notes**

- Speaker similarity: cosine of WavLM-ECAPA speaker embeddings, converted speech vs the target speaker's reference clip (real-speech row: two real clips of the same speaker = the ceiling).
- Similarity, other clip: vs a different clip of the target speaker (the stricter measure).
- Source leakage: similarity to the source speaker (real-speech row: the two speakers before conversion).
- CER: character error rate when a recogniser reads the speech; MMS = Meta MMS-1b-all (independent), our ASR = our fine-tuned encoder's CTC head. Real-speech row = the original source recordings.
- DNSMOS: Microsoft DNSMOS overall quality, 1-5. RTF: processing time / audio length (A100 GPU).
- Test set: 48 IndicVoices-R speakers never seen in training; 108 conversions per system (12 per language pair x 9 pairs).
