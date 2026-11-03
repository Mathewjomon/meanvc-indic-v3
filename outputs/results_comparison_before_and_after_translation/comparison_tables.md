# Results comparison: before translation vs after translation

**Before translation** (Stage 5): speaker A's real speech is converted into speaker B's voice. The words stay in A's language.

**After translation** (Stage 6): speaker A's speech is recognised, translated into B's language, spoken by TTS, then converted into speaker B's voice.

Same 48 test speakers (never seen in training) in both tests. Only cross-language pairs: 72 conversions before translation, 48 after.

## 1. Our MeanVC v3: before vs after translation

| Measure | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Sounds like speaker B (0-1) | 0.688 | 0.733 | +0.045 | **Better** |
| Sounds like speaker B, on a clip the model never heard (0-1) | 0.553 | 0.592 | +0.039 | **Better** |
| Still sounds like speaker A: leakage (0-1) | 0.279 | 0.173 | -0.106 | **Better** |
| Clarity lost by the conversion (% points) | 9.2 | 3.1 | -6.1 | **Better** |
| Audio quality, DNSMOS (1-5) | 3.04 | 3.05 | +0.01 | About the same |

## 2. Original MeanVC: before vs after translation

| Measure | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Sounds like speaker B (0-1) | 0.584 | 0.620 | +0.036 | **Better** |
| Sounds like speaker B, on a clip the model never heard (0-1) | 0.467 | 0.501 | +0.034 | **Better** |
| Still sounds like speaker A: leakage (0-1) | 0.244 | 0.154 | -0.090 | **Better** |
| Clarity lost by the conversion (% points) | 24.6 | 14.2 | -10.4 | **Better** |
| Audio quality, DNSMOS (1-5) | 3.03 | 2.99 | -0.03 | About the same |

## 3. Reference points (real recordings, no conversion)

| Reference | Before translation | After translation |
|---|:---:|:---:|
| Two real clips of speaker B: the best similarity possible | 0.748 | 0.754 |
| Speaker A vs speaker B: the normal similarity of two different people | 0.166 | 0.185 |
| Letters misheard in the voice converter's input (%) | 22.5 (real speech) | 6.4 (TTS speech) |

A leakage value at the 'two different people' level means none of speaker A's voice is left.

## 4. Our MeanVC v3 per direction

### Sounds like speaker B (0-1): higher is better

| Direction | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Malayalam speech → Tamil voice | 0.715 | 0.769 | +0.054 | **Better** |
| Malayalam speech → Hindi voice | 0.649 | 0.711 | +0.063 | **Better** |
| Tamil speech → Malayalam voice | 0.733 | 0.736 | +0.002 | About the same |
| Tamil speech → Hindi voice | 0.623 | 0.697 | +0.074 | **Better** |
| Hindi speech → Malayalam voice | 0.705 | 0.759 | +0.054 | **Better** |
| Hindi speech → Tamil voice | 0.703 | 0.729 | +0.025 | **Better** |
| **All directions** | 0.688 | 0.733 | +0.045 | **Better** |

### Still sounds like speaker A: leakage (0-1): lower is better

| Direction | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Malayalam speech → Tamil voice | 0.314 | 0.245 | -0.069 | **Better** |
| Malayalam speech → Hindi voice | 0.271 | 0.188 | -0.082 | **Better** |
| Tamil speech → Malayalam voice | 0.304 | 0.292 | -0.012 | About the same |
| Tamil speech → Hindi voice | 0.301 | 0.128 | -0.174 | **Better** |
| Hindi speech → Malayalam voice | 0.238 | 0.108 | -0.130 | **Better** |
| Hindi speech → Tamil voice | 0.243 | 0.077 | -0.166 | **Better** |
| **All directions** | 0.279 | 0.173 | -0.106 | **Better** |

### Clarity lost by the conversion (% points): lower is better

| Direction | Before translation | After translation | Change | Result |
|---|:---:|:---:|:---:|:---:|
| Malayalam speech → Tamil voice | 9.5 | 3.7 | -5.8 | **Better** |
| Malayalam speech → Hindi voice | 12.6 | 3.1 | -9.5 | **Better** |
| Tamil speech → Malayalam voice | 11.1 | 5.1 | -6.0 | **Better** |
| Tamil speech → Hindi voice | 7.3 | 4.9 | -2.4 | **Better** |
| Hindi speech → Malayalam voice | 7.1 | 1.8 | -5.3 | **Better** |
| Hindi speech → Tamil voice | 7.9 | 0.5 | -7.4 | **Better** |
| **All directions** | 9.2 | 3.1 | -6.1 | **Better** |

## 5. Our MeanVC v3 vs the original MeanVC (how much it sounds like speaker B)

| Test | Original MeanVC | Our MeanVC v3 | Our gain |
|---|:---:|:---:|:---:|
| Before translation | 0.584 | 0.688 | +0.104 |
| After translation | 0.620 | 0.733 | +0.114 |

Our gain per direction:

| Direction | Gain before translation | Gain after translation |
|---|:---:|:---:|
| Malayalam speech → Tamil voice | +0.139 | +0.101 |
| Malayalam speech → Hindi voice | +0.079 | +0.118 |
| Tamil speech → Malayalam voice | +0.127 | +0.117 |
| Tamil speech → Hindi voice | +0.094 | +0.103 |
| Hindi speech → Malayalam voice | +0.065 | +0.129 |
| Hindi speech → Tamil voice | +0.120 | +0.114 |
| **All directions** | +0.104 | +0.114 |

## How to read the measures

| Measure | What it means |
|---|---|
| Sounds like speaker B (0-1) | Voice similarity between the output and speaker B's reference clip (higher is better) |
| Sounds like speaker B, on a clip the model never heard (0-1) | The same, against another recording of B that was not given to the model (higher is better) |
| Still sounds like speaker A: leakage (0-1) | Voice similarity between the output and speaker A (lower is better) |
| Clarity lost by the conversion (% points) | Extra % of letters an independent recogniser (MMS) gets wrong after conversion, compared with the converter's input (lower is better) |
| Audio quality, DNSMOS (1-5) | Microsoft DNSMOS overall speech quality (higher is better) |

All numbers for every direction and both systems are in `comparison.csv`.
