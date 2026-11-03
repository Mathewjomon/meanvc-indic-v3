# Speech translation + voice conversion: results

Test speakers never seen in training. Part A: all 300 test clips. Part B: 48 translations (8 per direction x 6 directions), 48 source speakers (A), 48 target voices (B). Device: cuda.

## Table 1 - Step 1: language identification (all test clips)

| Language | Clips | Correct % | Mean confidence | Asked to confirm % |
|---|:---:|:---:|:---:|:---:|
| Malayalam | 100 | 100.0 | 0.975 | 0.0 |
| Tamil | 100 | 100.0 | 0.983 | 0.0 |
| Hindi | 100 | 98.0 | 0.958 | 1.0 |
| All | 300 | 99.3 | 0.972 | 0.3 |

Confusion (rows = real language, columns = detected):

| real \ detected | Hindi | Malayalam | Tamil |
|---|:---:|:---:|:---:|
| Hindi | 98 | 0 | 2 |
| Malayalam | 0 | 100 | 0 |
| Tamil | 0 | 0 | 100 |

## Table 2 - Step 2: speech recognisers compared (300 test clips)

### 2a - vs the verbatim transcript (exactly as spoken, colloquial forms included)

| Recogniser (CER % / WER %) | Malayalam | Tamil | Hindi | All | Translation agreement chrF ↑ (48 clips) | Seconds per clip |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Our encoder (greedy CTC, streaming) | 29.4 / 82.8 | 25.9 / 75.0 | 16.8 / 46.8 | 24.5 / 65.3 | 37.2 | 0.03 |
| Meta MMS-1b-all | 25.5 / 75.4 | 29.9 / 79.4 | 12.4 / 32.5 | 23.3 / 58.8 | 46.9 | 0.17 |
| AI4Bharat IndicConformer 600M (RNNT) | 10.6 / 43.8 | 13.3 / 52.0 | 6.2 / 17.2 | 10.3 / 35.3 | 64.7 | 0.80 |

### 2b - vs the normalised transcript (standard spelling)

| Recogniser (CER % / WER %) | Malayalam | Tamil | Hindi | All |
|---|:---:|:---:|:---:|:---:|
| Our encoder (greedy CTC, streaming) | 30.8 / 85.5 | 28.9 / 82.3 | 17.2 / 47.9 | 26.2 / 68.9 |
| Meta MMS-1b-all | 26.3 / 75.8 | 31.3 / 81.1 | 11.7 / 29.9 | 24.1 / 58.4 |
| AI4Bharat IndicConformer 600M (RNNT) | 8.3 / 31.2 | 8.4 / 29.5 | 4.6 / 12.2 | 7.3 / 22.8 |

CER / WER: character / word error rate, lower is better. A recogniser that writes the standard spelling of a colloquial word is counted wrong in 2a but right in 2b. chrF: translation of the recogniser's text vs translation of the real transcript (0-100, higher = its errors change the translation less).

## Table 3 - Steps 4-5: the translated speech in speaker B's voice

| Output | Similarity to B (reference clip) ↑ | Similarity to B (other clip) ↑ | Similarity to A (leakage) ↓ | CER % (MMS) vs translation ↓ | Quality (DNSMOS) ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| Real speech: two clips of speaker B (ceiling) | - | 0.754 | - | - | 3.18 |
| Real speech: speaker A vs B (before conversion) | 0.185 | - | - | - | 3.09 |
| TTS only (no voice conversion) | 0.171 | 0.139 | 0.078 | 6.4 | 3.18 |
| TTS + original MeanVC | 0.620 | 0.501 | 0.154 | 20.5 | 2.99 |
| TTS + our MeanVC v3 (the pipeline) | 0.733 | 0.592 | 0.173 | 9.5 | 3.05 |

## Table 4 - The whole pipeline per direction (IndicConformer + our MeanVC v3)

| Direction | n | ASR CER % | Translation chrF ↑ | Voice B: TTS | Voice B: ours ↑ | Voice B, other clip ↑ | Leakage of A ↓ | CER % (MMS) ↓ | DNSMOS ↑ | RTF ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Malayalam -> Tamil | 8 | 10.1 | 68.8 | 0.219 | 0.769 | 0.639 | 0.245 | 10.1 | 2.97 | 1.20 |
| Malayalam -> Hindi | 8 | 8.7 | 64.6 | 0.145 | 0.711 | 0.558 | 0.188 | 8.7 | 3.09 | 1.16 |
| Tamil -> Malayalam | 8 | 14.9 | 54.1 | 0.174 | 0.736 | 0.556 | 0.292 | 13.2 | 3.04 | 1.08 |
| Tamil -> Hindi | 8 | 11.7 | 55.3 | 0.121 | 0.697 | 0.582 | 0.128 | 11.2 | 3.18 | 0.98 |
| Hindi -> Malayalam | 8 | 19.7 | 60.7 | 0.136 | 0.759 | 0.571 | 0.108 | 8.6 | 3.03 | 0.95 |
| Hindi -> Tamil | 8 | 4.8 | 78.2 | 0.232 | 0.729 | 0.648 | 0.077 | 5.5 | 2.97 | 1.11 |
| All | 48 | 11.6 | 64.7 | 0.171 | 0.733 | 0.592 | 0.173 | 9.5 | 3.05 | 1.08 |

## Table 5 - Time per step (cuda + CPU for IndicConformer, mean of 48 runs, input 7.7 s on average)

| Step | Seconds | % of total |
|---|:---:|:---:|
| language_id | 0.05 | 1 |
| asr_indicconformer | 0.73 | 9 |
| voice_reference | 2.57 | 33 |
| translate | 0.92 | 12 |
| tts | 1.57 | 20 |
| voice_conversion | 2.03 | 26 |
| total | 7.87 | 100 |

## Examples (one per direction; all audio in examples/<run>/)

**ml2ta_00** (Malayalam -> Tamil; voice of B: TTS 0.25 -> ours 0.79)

- real transcript: ഒരു ഇത് ഇണ്ട ഇണ്ടാവൊള്ളൂ കാരണം നമ്മള് എടക്കെടക്ക് അതായത് ഡെയിലി ഒരു രണ്ടുമൂന്നെണ്ണം ഒരു രണ്ട് മൂന്ന് പിക്ചറെങ്കിലും നമ്മൾ നോക്കി വരച്ചിരിക്കണം
- recognised (IndicConformer): ഒരു ഇതുണ്ടാവുക ഉണ്ടാവുകയുള്ളൂ കാരണം നമ്മൾ ഇടയ്ക്കിടയ്ക്ക് അതായത് ഡെയിലി ഒരു രണ്ട് മൂന്നെണ്ണ ഒരു രണ്ട് മൂന്ന് പിക്ചറെങ്കിലും നമ്മൾ നോക്കി വരച്ചിരിക്കണം
- translation: ஒன்று இருக்க வேண்டும், ஏனென்றால் நாம் அவ்வப்போது, குறைந்தபட்சம் ஒரு டெய்லி இரண்டு மூன்று அல்லது இரண்டு மூன்று படங்களைப் பார்த்து வரைந்திருக்க வேண்டும்.
- translation of the real transcript: ஒரு இட் இன்டா இன்டாவோலு ஏனென்றால் நாம் குறைந்தபட்சம் ஒரு இரண்டும் மூன்றும் இரண்டும் மூன்று படங்களை வரைந்திருக்க வேண்டும், அதாவது ஒரு டெய்லி.
- MMS hears our output as: ஒந்டு இருக்கவேண்டும் எனந்தால் நாம் அமமப்போது குறைந்த வட்றும் ஒருடை இரண்டு ூன்டு அல்லது இரண்டு மூன்டு படங்களைப் பார்த்து வறைந்திருக்கவேண்டும்

**ml2hi_00** (Malayalam -> Hindi; voice of B: TTS 0.18 -> ours 0.73)

- real transcript: ഒരു തൊഴിൽ മേഖലയാണ് വീട്ടിലിരിക്കുന്നവർക്കൊരു തൊഴിൽ മേഖലയാണ് അതുപോലെ വീടും പരിസരവും വൃത്തിയായിട്ട് എപ്പൊഴും സൂക്ഷിക്കാനും കഴിയും അതുപോലെത്തന്നെ
- recognised (IndicConformer): ഒരു തൊഴിൽ മേഖലയാണ് വീട്ടിലിരിക്കുന്നവർക്കൊരു തൊഴിൽ മേഖലയാണ് അതുപോലെ വീടും പരിസരവും വൃത്തിയായിട്ട് എപ്പോഴും സൂക്ഷിക്കാനും കഴിയും അതുപോലെ തന്നെ
- translation: एक कार्य क्षेत्र उन लोगों के लिए एक कार्य क्षेत्र है जो घर में रहते हैं और हमेशा घर और पड़ोस को साफ-सुथरा रख सकते हैं।
- translation of the real transcript: एक कार्य क्षेत्र उन लोगों के लिए एक कार्य क्षेत्र है जो घर में रहते हैं और घर और पड़ोस को हमेशा साफ-सुथरा रख सकते हैं।
- MMS hears our output as: एक कार्य क्षेत्रपूर्ण लोगों के लिए एक कार्य क्षेत्र है जो घर में रहते हैं और हमेशा घर और पड़ुस को साथ सूथा रख सकते हैं

**ta2ml_00** (Tamil -> Malayalam; voice of B: TTS 0.08 -> ours 0.70)

- real transcript: ஸோ எனக்கு இப்ப சமையல் ரொம்ப நல்லாவே தெரிஞ்சிச்சி
- recognised (IndicConformer): ஸோ எனக்கு இப்போ சமையல் ரொம்ப நல்லாவே தெரிஞ்சிச்சு
- translation: സോ എനിക്ക് ഇപ്പോൾ പാചകം നന്നായി അറിയാം.
- translation of the real transcript: സോ എനിക്ക് ഇപ്പോൾ പാചകം നന്നായി അറിയാം.
- MMS hears our output as: സോ എനിക്കിപ്പോൾ പാചകം നന്നായറിയ

**ta2hi_00** (Tamil -> Hindi; voice of B: TTS 0.19 -> ours 0.59)

- real transcript: கொறைச்சி கொள்வதனாலையும் லாரி தொழில் மிகவும் லாபகராமான தொழிலா இருந்து இப்ப பின்னோக்கி போயிட்ருக்கு
- recognised (IndicConformer): குறைச்சிக்கொள்வதனாலையும் லாரி தொழிலும் மிகவும் லாபகரமாக தொழிலாக இருந்து இப்போ பின்னோக்கி போயிட்டுருக்கு
- translation: कम होने के साथ-साथ लॉरी उद्योग अब एक बहुत ही लाभदायक व्यवसाय बन गया है।
- translation of the real transcript: डकैती उद्योग अब सबसे अधिक लाभदायक व्यवसाय से पीछे हट गया है।
- MMS hears our output as: कम होने के साथ साथ सलारी उद्योग आप एक बहुत ही लावदायक व्यवसाय बन तिया है

**hi2ml_00** (Hindi -> Malayalam; voice of B: TTS 0.10 -> ours 0.80)

- real transcript: गाँव से ही होता है और अपनी भीत प्रतिभा को बिकसित करने के लिए लोग गाँव की ही तरफ जा जा के सीखते हैं क्युकी यहाँ पे जादा
- recognised (IndicConformer): गाँव से ही होता है और अपनी प्रतिभा को विकसित करने के लिए लोग गाँव की ही तरफ ज़्यादा सीखते हैं क्योंकि यहाँ पे ज़्यादा
- translation: ഗ്രാമത്തിൽ നിന്നാണ് വരുന്നത്, ഇവിടെ കൂടുതൽ ഉള്ളതിനാൽ ആളുകൾ അവരുടെ കഴിവുകൾ വികസിപ്പിക്കുന്നതിനായി ഗ്രാമത്തിൽ തന്നെ കൂടുതൽ പഠിക്കുന്നു.
- translation of the real transcript: ഗ്രാമത്തിൽ നിന്നാണ് വരുന്നത്, ആളുകൾ അവരുടെ ഭയപ്പെടുത്തുന്ന കഴിവുകൾ വികസിപ്പിക്കാൻ ഗ്രാമത്തിലേക്ക് പോകുകയും പഠിക്കുകയും ചെയ്യുന്നു.
- MMS hears our output as: രാമത്തിൽ നിന്നാണ് വരുന്നത് ഇവിടെ കൂടുതൽ ഉള്ളതിനാൽ ആളുകൾ അവരുടെ കഴിവുകൾ വികസിപ്പിക്കുന്നതിനായി ഗ്രാമത്തിൽ തന്നെ കൂടുതൽ പഠിക്കുന്നു

**hi2ta_00** (Hindi -> Tamil; voice of B: TTS 0.29 -> ours 0.63)

- real transcript: ग्रेवी बनाई जाती है इसके बाद पनीर डाला जाता है मटर डाला जाता है मैंने खीर में बी बहोत सारे
- recognised (IndicConformer): ग्रेवी बनाई जाती है इसके बाद पनीर डाला जाता है मटर डाला जाता है मैंने खीर में भी बहुत सारे
- translation: கிரேவி தயாரிக்கப்படுகிறது, பின்னர் பாலாடைக்கட்டி சேர்க்கப்படுகிறது, பீட்ரூட் சேர்க்கப்படுகிறது, நான் பீர் நிறைய உள்ளது
- translation of the real transcript: கிரேவி தயாரிக்கப்படுகிறது, பின்னர் பாலாடைக்கட்டி சேர்க்கப்படுகிறது, பீட் சேர்க்கப்படுகிறது, நான் பீர் நிறைய உள்ளது
- MMS hears our output as: கிரேவி தயாரிக்கப்படுகிறது பின்ட பாலாடைக் கட்டிசேர்க்கப்படுகிறது பீர்குன் சேர்க்கப்படுகிறது நான் ிபேர்கை உள்ளது

Runs: 48 planned, 48 completed, 0 failed. Total evaluation time 32 min.
