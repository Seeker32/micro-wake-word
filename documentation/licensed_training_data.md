# Replacement data for the basic training notebook

The notebook replaces its augmentation and negative-audio inputs and writes a
file-level provenance record. This is **not a clearance of the whole product**:
the legacy Piper generator weights, their base-model history, product software
dependencies, new training, and device evaluation remain pending.

## Run

Run `notebooks/basic_training_notebook.ipynb` from this updated checkout. The
installation cell accepts a working directory at the repository root, inside
`notebooks`, or above a `micro-wake-word` or `microWakeWord` checkout. In Colab,
clone or upload this checkout to `/content/micro-wake-word` before running the
installation cell. Use `%cd /content/micro-wake-word` to keep generated files
under the checkout. The checkout must contain `microwakeword/audio/licensed_data.py`.

The data cell calls `prepare_licensed_data("licensed_data_v1")`. It needs the
system `zip` command (Info-ZIP, including split-archive conversion), approximately
15 GB of downloads, and at least 40 GB of free disk space for cached archives,
the joined FSD50K archive, selected audio, and features. Downloads and feature
generation can take a long time. This change does not download that audio or
train a model until those cells are executed.

## Sources and selection

| Input | Pinned source | Selection and recorded license |
| --- | --- | --- |
| Background augmentation and environmental negatives | [FSD50K v1.0, record 4060432](https://zenodo.org/records/4060432), eval audio and metadata | Exact CC0 1.0, CC BY 3.0 or CC BY 4.0 URL allowlist. Missing or other licenses are excluded, including NC, ND, SA and Sampling+. Original uploader and title are required. The dataset-level CC BY 4.0 notice is also retained. |
| Reverberation | [OpenSLR SLR28 RIRS_NOISES](https://www.openslr.org/28/) | Only `RIRS_NOISES/simulated_rirs/**/*.wav`, under the source's Apache 2.0 listing. Real RIRs, point-source noise and isotropic noise are excluded. Both original README files and the Apache license are archived. |
| Speech negatives | [OpenSLR SLR12 LibriSpeech](https://www.openslr.org/12/) | `train-clean-100`, `dev-clean`, and `test-clean`, respectively, under CC BY 4.0. Only original FLAC audio is selected. |

FSD50K's published page asks commercial users to contact its maintainers. This
request is preserved in the attribution output and the source snapshot; the
code does not contact anyone or represent that contact as completed. Per-file
selection is not a substitute for reviewing the dataset-level terms for your
distribution. Keep the dataset authors' credit and the included license notices.

FSD50K's eval subset is used as a source pool to avoid downloading the much larger
development archive. It is re-split approximately 80/10/10 by a stable hash of
the uploader, so an uploader cannot occur in more than one split. These are
custom wake-word splits, **not an official FSD50K benchmark**. Augmentation uses
only the corresponding split's background pool. Simulated RIRs are shared.
LibriSpeech retains its official train/development/test split.

The new backgrounds are broader environmental negatives, not a guaranteed
speech-free corpus. LibriSpeech is read English; it does not replace spontaneous
household conversations or explicitly generated confusing phrases. Review for
accidental occurrences of the target phrase and add appropriately authorized
real-world recordings before judging product quality.

## Files produced by the data cell

All selected original audio bytes are stored under `licensed_data_v1/prepared`:

- `noise/{training,validation,testing}/`: selected FSD50K WAVs.
- `speech/{training,validation,testing}/`: LibriSpeech FLACs.
- `rirs/`: simulated RIR WAVs, retaining their original relative paths.
- `manifest.json` and `manifest.csv`: one row per selected audio file, with
  upstream path, source URL, dataset version, author, title, exact license URL,
  split, modifications and SHA-256. The JSON also records archive URLs,
  published checksums where available, downloaded archive SHA-256s, and pending
  work outside this audit. License-document records include their download URLs
  and SHA-256s. CC license texts come from a pinned revision of Creative Commons'
  official `cc-legal-tools-data` repository to avoid blocked website deed pages.
- `ATTRIBUTION.md`: file-level credits and dataset-level notices.
- `excluded_fsd.csv`: rejected clip IDs, license values and rejection reasons.
- `sources/`: the original FSD metadata, dataset notices, RIR READMEs, source-page
  snapshots and copies of the selected licenses.

FSD50K and LibriSpeech downloads are checked against their published MD5s. SLR28
has no published checksum in the source listing used here; its downloaded bytes
are recorded with SHA-256, and ZIP CRC checks apply while reading selected files.
Original archives remain in `licensed_data_v1/downloads`, including excluded FSD
clips inside the original archive. Only selected prepared directories are passed
to the audio pipeline; do not distribute the raw FSD archive as a cleared subset.

Failed downloads do not become cache entries. Preparation publishes the final
directory only after every required split and attribution record is written.
Rerunning verifies all prepared audio hashes and refuses modified or extra audio;
use a fresh data directory when changing the selection policy or source version.

## Notebook integration

The old MIT RIR, FMA, AudioSet, and precomputed NC negative features are no longer
downloaded or referenced by the notebook. Existing files are not deleted.

Hey Kamana positive features use `licensed_features_v1/positive_hey_kamana`.
Replacement negative features use `licensed_features_v1/negative/{speech,noise,ambient}`. The feature
cell resamples through the existing `Clips` loader at 16 kHz, computes features
at a 10 ms step, and creates classification and ambient evaluation sets using
the existing `SpectrogramGeneration` and `RaggedMmap` APIs. Short classification
negatives are repeated to at least six seconds and split into five-second
feature chunks. Ambient evaluation retains original clips longer than six
seconds without repetition; these clips are only a proxy for long real-room
recordings and on-device false-accept measurements.

`licensed_features_v1/feature_sources.json` links those features to the exact
audio manifest hash and documents the preprocessing and augmentation pools.
Carry this file, the manifest, notices, and relevant license copies with your
experiment records and prepare appropriate notices for anything you distribute.

The YAML and output model paths include `hey_kamana` to keep checkpoints
separate from other wake words. This does not retrain anything now or resolve
the legacy TTS model's provenance. Do not
describe a model as fully commercially cleared based on these two steps alone.
