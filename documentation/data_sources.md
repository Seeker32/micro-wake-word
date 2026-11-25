# Data Sources for Training Wake Words

This page lists the audio sources behind the training notebook and records the
licence position of each one. The per-file provenance workflow lives in
`microwakeword/audio/licensed_data.py` (FSD50K / LibriSpeech / SLR28) and
`microwakeword/audio/chime6.py` (CHiME-6); the step-by-step run order is in
[data_pipeline.zh.md](data_pipeline.zh.md). The dataset that the current
notebook actually trains on is described in
[dataset_composition.zh.md](dataset_composition.zh.md).

## Current pipeline

| Source | Role | Licence | Split |
| --- | --- | --- | --- |
| Bryce Beattie's Piper **Clean 100** | Wake-word and hard-negative TTS (22.05 kHz, 245 speaker IDs) | Model trained on LibriTTS-R train-clean-100 (CC BY 4.0); model weights served by the author | Positives and `hey hi` use separate directories; splits assigned per clip |
| FSD50K v1.0 eval | Background noise for augmentation, plus noise negatives | Dataset CC BY 4.0, per-clip CC0/CC-BY allowlist (fail-closed) | uploader `sha256 % 10`: 0 → validation, 1 → testing, rest → training |
| LibriSpeech SLR12 | Speech negatives and long ambient clips | CC BY 4.0 | Official `train-clean-100` / `dev-clean` / `test-clean` |
| OpenSLR SLR28 `RIRS_NOISES` | Reverberation for augmentation | Apache 2.0 (simulated RIRs only) | Shared by all splits |
| CHiME-6 (OpenSLR SLR150) | Real far-field ambient evaluation | CC BY-SA 4.0 (ShareAlike if derived data is distributed) | Official dev → `validation_ambient`, eval → `testing_ambient` |

Everything except Clean 100 and CHiME-6 is downloaded and verified by
`prepare_licensed_data()`; a run writes `data_v1/prepared/manifest.json`,
`manifest.csv`, `excluded_fsd.csv`, `ATTRIBUTION.md` and snapshots of every
licence text under `prepared/sources/`.

## Generated samples

The notebook synthesises wake-word and hard-negative audio with
[Bryce Beattie's Piper Clean 100](https://brycebeattie.com/files/tts/), trained from
scratch on [LibriTTS-R train-clean-100](https://www.openslr.org/141/) (CC BY 4.0).
Three speaking rates (`length_scale` 0.65 / 1.0 / 1.25) × 1,000 samples each are
written per phrase, and `generation.json` records the phonemes, speaker IDs,
noise scales and the SHA-256 of both model files. TTS model provenance and the
product dependency audit are tracked separately from the audio dataset audit.

Earlier versions of this notebook used the [Piper sample
generator](https://github.com/rhasspy/piper-sample-generator) plus
[openWakeWord](https://github.com/dscripka/openWakeWord) for adversarial phrases,
and a Kokoro-based generator (`microwakeword/audio/kokoro_v1.py` is still in the
repository). None of these are part of the current notebook pipeline.

## Augmentation sources

Positive samples are augmented with background noise and room impulse responses:

- **FSD50K** (eval subset) supplies the background-noise pool. Only clips whose
  per-clip licence is CC0, CC BY 3.0 or CC BY 4.0 are kept — of the 10,231 eval
  clips, 8,403 pass (6,975 training / 628 validation / 800 testing) and 1,828 are
  excluded (1,425 CC BY-NC 3.0, 403 CC Sampling+ 1.0). Each split has its own pool,
  so held-out audio never leaks into training augmentation. FSD50K asks
  commercial users to contact the dataset maintainers; see the archived source
  page in `prepared/sources/FSD50K.html`.
- **SLR28 simulated RIRs** (Apache 2.0) are shared by all three splits. Real RIRs,
  point-source noises and isotropic noises from the same archive are not used.

FSD50K's dataset-level licence is CC BY 4.0 with per-clip licences underneath, so
the dataset-level statement is not a substitute for the per-clip allowlist. The
same rule applies to any newly added source.

## Negative speech

**LibriSpeech** (`train-clean-100`, `dev-clean`, `test-clean`, CC BY 4.0) provides
the speech negatives, using the official subset split without re-splitting.
Clips are repeated to at least 6 s, then cut into non-overlapping 5 s chunks; no
audio augmentation is applied to negatives.

## Ambient evaluation

False accepts per hour are estimated on long, unaugmented recordings:

- Clips of at least 6 s from LibriSpeech validation/testing and FSD50K
  validation/testing, kept whole (no repetition, no 5 s chunking).
- **CHiME-6** (OpenSLR SLR150, CC BY-SA 4.0), real four-person dinner
  conversations recorded with far-field arrays. One channel (CH1) per selected
  room, cut into 600 s segments: official dev → `validation_ambient` (27 clips,
  4.46 h) and eval → `testing_ambient` (96 clips, 15.62 h). Both ambient sets are
  merged when the notebook reports false accepts per hour. Cite Barker et al.,
  Interspeech 2018, as required by the licence.

## Splits

| Set | Content |
| --- | --- |
| `training` | 2,400 augmented positives, LibriSpeech `train-clean-100`, FSD50K training bucket, 2,400 `hey hi` samples |
| `validation` | 300 positives, `dev-clean`, FSD50K validation bucket, 300 `hey hi`, plus `validation_ambient` (LibriSpeech + FSD50K ≥6 s clips and CHiME-6) |
| `testing` | 300 positives, `test-clean`, FSD50K testing bucket, 300 `hey hi`, plus `testing_ambient` (same composition) |

FSD50K is **not** an official 90-10 benchmark split here: it is re-bucketed by
uploader for this wake-word task and must not be reported as an FSD50K result.

## Datasets from earlier versions (not used)

These sources appear in the project's history and in earlier revisions of this
page, but the current notebook does not download them:

- **FMA** — audio licences vary per track, some restrict commercial use.
- **WHAM!** — CC BY-NC 4.0, not usable commercially.
- **BIRD** — impulse responses; replaced by SLR28 simulated RIRs.
- **VOiCES** — CC BY 4.0 ("free for commercial, academic, and government use");
  still a plausible ambient candidate, not wired in.
- **Common Voice** — near-field read speech; no ambient value.
- **DiPCo** — the original release claimed CDLA-Permissive 1.0, but its
  distribution channel is gone and the claim cannot be verified from a primary
  source. Use the official re-issue instead: CHiME-5/CHiME-6 on OpenSLR SLR150,
  CC BY-SA 4.0 since 2024-01-01.

For verified licence conclusions and the remaining open questions (MUSAN,
DEMAND, AMI, VOiCES, Freesound API terms), see
[ambient_commercial_datasets.zh.md](ambient_commercial_datasets.zh.md).

## Pending

- Clean 100 TTS weights and the product dependency/distribution audit are
  separate reviews; the audio audit covers FSD50K, LibriSpeech, SLR28 and CHiME-6.
- CHiME-6 is CC BY-SA 4.0: derived data that leaves the project must carry the
  same licence.
- Synthetic positives, pooled ambient sets and far-field recordings do not
  replace testing on the target device in the target room.
