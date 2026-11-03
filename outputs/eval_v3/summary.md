### All conversions

| system | sim_target | sim_other | sim_source | cer_mms | cer_ours | dnsmos | rtf | n |
|---|---|---|---|---|---|---|---|---|
| real speech | 0.753 | - | 0.194 | 23.1 | 25.0 | 3.07 | - | 108 |
| mandarin | 0.596 | 0.480 | 0.253 | 46.8 | 47.9 | 3.02 | 0.15 | 108 |
| enc_only | 0.451 | 0.359 | 0.114 | 68.1 | 68.1 | 2.94 | 0.17 | 108 |
| ours | 0.707 | 0.573 | 0.290 | 31.7 | 32.1 | 3.03 | 0.17 | 108 |

### Same language (e.g. ml -> ml)

| system | sim_target | sim_other | sim_source | cer_mms | cer_ours | dnsmos | rtf | n |
|---|---|---|---|---|---|---|---|---|
| real speech | 0.762 | - | 0.249 | 24.2 | 25.6 | 3.02 | - | 36 |
| mandarin | 0.619 | 0.505 | 0.272 | 46.2 | 46.2 | 3.02 | 0.16 | 36 |
| enc_only | 0.474 | 0.367 | 0.127 | 68.5 | 66.2 | 2.85 | 0.17 | 36 |
| ours | 0.746 | 0.612 | 0.313 | 31.7 | 30.5 | 3.01 | 0.17 | 36 |

### Cross-language (e.g. ml -> ta)

| system | sim_target | sim_other | sim_source | cer_mms | cer_ours | dnsmos | rtf | n |
|---|---|---|---|---|---|---|---|---|
| real speech | 0.748 | - | 0.166 | 22.5 | 24.7 | 3.09 | - | 72 |
| mandarin | 0.584 | 0.467 | 0.244 | 47.1 | 48.8 | 3.03 | 0.15 | 72 |
| enc_only | 0.439 | 0.354 | 0.108 | 68.0 | 69.0 | 2.99 | 0.17 | 72 |
| ours | 0.688 | 0.553 | 0.279 | 31.7 | 32.9 | 3.04 | 0.17 | 72 |

### Ours, per language pair (source -> target voice)

| system | sim_target | sim_other | sim_source | cer_mms | cer_ours | dnsmos | rtf | n |
|---|---|---|---|---|---|---|---|---|
| ml -> ml | 0.752 | 0.583 | 0.347 | 38.8 | 36.5 | 2.91 | 0.17 | 12 |
| ml -> ta | 0.715 | 0.603 | 0.314 | 28.3 | 29.6 | 3.02 | 0.17 | 12 |
| ml -> hi | 0.649 | 0.544 | 0.271 | 36.1 | 40.4 | 2.97 | 0.17 | 12 |
| ta -> ml | 0.733 | 0.564 | 0.304 | 44.2 | 41.0 | 2.93 | 0.17 | 12 |
| ta -> ta | 0.761 | 0.660 | 0.323 | 33.7 | 29.6 | 3.00 | 0.17 | 12 |
| ta -> hi | 0.623 | 0.529 | 0.301 | 39.6 | 37.0 | 3.05 | 0.17 | 12 |
| hi -> ml | 0.705 | 0.509 | 0.238 | 20.5 | 23.7 | 3.12 | 0.17 | 12 |
| hi -> ta | 0.703 | 0.568 | 0.243 | 19.4 | 23.9 | 3.14 | 0.17 | 12 |
| hi -> hi | 0.724 | 0.593 | 0.270 | 19.7 | 23.4 | 3.12 | 0.17 | 12 |

sim_* = voice similarity (cosine, WavLM-ECAPA): sim_target/sim_other higher = better, sim_source lower = better. 'real speech' row: sim_target = two real clips of the target speaker, sim_source = source vs target before conversion, CER/DNSMOS = the source recording. cer_* in %, lower = better. dnsmos 1-5, higher = better. rtf < 1 = faster than real time.
