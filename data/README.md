# Dataset

## Source

**IndicVoices-R**, by AI4Bharat (IIT Madras)
https://huggingface.co/datasets/ai4bharat/indicvoices_r
Licence: CC BY 4.0 (free to use with credit to AI4Bharat)

IndicVoices-R is a large Indian-language speech dataset: real, **spontaneous** speech (people talking
naturally, not reading sentences) from thousands of speakers across India, recorded in everyday
conditions and cleaned for quality. Every clip comes with a speaker ID, gender, and a verbatim and
a normalised transcript.

## The part we used

| | |
|---|---|
| Languages | **Malayalam, Tamil, Hindi** |
| Clip length | 2–15 seconds |
| Quality filters | Transcript error (CER) ≤ 0.10, signal-to-noise ratio ≥ 15 dB |
| Audio format | Converted from 44.1 kHz stereo to **16 kHz mono, 16-bit** |

## Splits

The splits are **by speaker**: no speaker appears in more than one split, so validation and test
speakers are never heard during training.

| Split | Clips | Hours | Speakers | Malayalam | Tamil | Hindi | Used for |
|---|---:|---:|---:|---:|---:|---:|---|
| Train, round 1 | 30,000 | 69.2 | 1,758 | 10,000 | 10,000 | 10,000 | Training (encoder + decoder) |
| Train, round 2 (extra) | 41,556 | 95.0 | 1,782 | 14,778 | 21,778 | 5,000 | More training (encoder round 2, decoder) |
| **Training total** | **71,556** | **164.2** | | | | | |
| Validation | 557 | 1.3 | 24 (8 per language) | 200 | 157 | 200 | Checking training progress |
| **Test** | **300** | **0.7** | **48 (16 per language)** | **100** | **100** | **100** | **All reported results** |

The test clips come from the dataset's **official test split**. These 48 speakers are used in every
evaluation: Stage 5 voice conversion and Stage 6 speech translation + voice conversion.
