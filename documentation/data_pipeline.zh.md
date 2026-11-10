# basic_training_notebook 的数据处理流程

本文只讲 `notebooks/basic_training_notebook.ipynb` 里**数据从原始音频到训练 batch** 的全过程，按 cell 顺序对应到代码位置。许可与合规问题见 [licensed_training_data.md](licensed_training_data.md)，数据源清单见 [data_sources.md](data_sources.md)。

## 0. 总览

```
target_word (IPA)
   │  cell 3/4  Piper TTS
   ▼
generated_samples/hey_kamana/*.wav          1000 条正样本音频
   │
   │  cell 5     download + license filter + hash
   ▼
data_v1/prepared/                     FSD50K 噪声 / LibriSpeech 语音 / SLR28 RIR
   │          ├── noise/{training,validation,testing}/
   │          ├── speech/{training,validation,testing}/
   │          ├── rirs/
   │          └── manifest.{csv,json}, ATTRIBUTION.md, excluded_fsd.csv, sources/
   │
   │  cell 7     每个 split 一个增广器（背景池 = 同 split 的 noise/）
   │  cell 8     单条试听
   ▼
   ├─ cell 9   正样本 → features_v1/positive_hey_kamana/{split}/wakeword_mmap
   └─ cell 10  负样本 → features_v1/negative/{speech,noise}/{split}/negative_mmap
                ambient  → features_v1/negative/ambient/{validation,testing}_ambient/
   │
   │  cell 11    yaml：sampling_weight / penalty_weight / truncation_strategy
   ▼
   cell 12    训练：batch 按权重随机抽，validation 全量枚举
```

## 1. 正样本：TTS 生成

- cell 3：`target_word = 'hˈeɪ kɑmˈɑnɑ'`，Piper `en_US-libritts_r-medium`，**音素输入**，先只生成 1 条人工试听。
- cell 4：同一命令改为 `--max-samples 1000`，落到 `generated_samples/hey_kamana/`。
- 这个目录**只有一个类别**，没有负样本；负样本全部来自阶段 2。

## 2. 负样本与增广音频：下载 + 许可过滤

cell 5 调 `prepare_licensed_data("data_v1")`（`microwakeword/audio/licensed_data.py`），约 15 GB 下载、40 GB 磁盘。

| 输入 | 用途 | 过滤规则 | 落盘 |
|---|---|---|---|
| FSD50K eval | 背景噪声（既是训练增广池，也是噪声负样本） | 许可 URL 必须命中 CC0/CC-BY 允许列表，且必须有 uploader 与 title | `noise/<split>/*.wav` |
| LibriSpeech | 语音负样本 | 只取 `train-clean-100` / `dev-clean` / `test-clean` 的原始 FLAC | `speech/<split>/*.flac` |
| OpenSLR SLR28 | 房间冲激响应（RIR） | 只取 `simulated_rirs/**/*.wav` | `rirs/`（保留原始相对路径） |

**划分方式（关键）**

- FSD50K 没有用官方 split，而是 `licensed_data.py:101-104` 对 **uploader** 做 `sha256 % 10` 分桶：0→validation、1→testing、其余→training。同一 uploader 的所有片段被锁在同一个 split，避免同源泄漏。
- LibriSpeech 直接沿用上游 subset（`licensed_data.py:35-39`），不再二次切分。
- RIR **不划分**，三个 split 共用同一份。

**产物**：`manifest.csv` / `manifest.json`（每文件一行：上游路径、源 URL、数据集版本、作者、标题、精确许可 URL、split、修改说明、SHA-256），`ATTRIBUTION.md`，`excluded_fsd.csv`（被拒片段及原因），`sources/`（原始元数据与许可文本快照）。重跑会校验哈希，拒绝被修改或多余的文件。

`train-result/` 里的实测统计（`manifest.csv`，102,265 行）：

| 目录 | training | validation | testing | 合计 |
|---|---|---|---|---|
| `noise/`（FSD50K） | 6,975 | 628 | 800 | 8,403 |
| `speech/`（LibriSpeech） | 28,539 | 2,703 | 2,620 | 33,862 |
| `rirs/` | 60,000（三集合共用） | — | — | 60,000 |

## 3. 增广器：每个 split 独立的背景池

cell 7。`clips` 用 `Clips(...)` 加载 1000 条正样本：

```python
Clips(input_directory='generated_samples/hey_kamana',
      file_pattern='*.wav',
      max_clip_duration_s=None,
      remove_silence=False,
      random_split_seed=10,
      split_count=0.1)
```

`augmenters` 是 `{split: Augmentation}` 的字典，三个 split 只是**背景噪声池不同**（`data_root/noise/<split>`），RIR 池共享：

| 参数 | 值 | 作用 |
|---|---|---|
| `augmentation_duration_s` | 3.2 | 输出固定 3.2 s：过长裁掉开头，过短在开头补零 |
| `min_jitter_s` / `max_jitter_s` | 0.195 / 0.205 | 正样本结束位置相对音频尾部的抖动 |
| `AddBackgroundNoise` | 0.75 | 从**同 split** 的背景池抽噪声 |
| `RIR` | 0.5 | 混响 |
| `Gain` | 1.0 | 每条都做 |
| `SevenBandParametricEQ` / `TanhDistortion` / `PitchShift` / `BandStopFilter` / `AddColorNoise` | 各 0.1 | 轻量音色扰动 |
| `background_min_snr_db` / `max_snr_db` | -5 / 10 | 信噪比范围 |

cell 8 抽一条随机片段试听，用来确认增广没有把目标词吃掉。

## 4. 正样本特征：频谱 + 切分 + 重复

cell 9，`SpectrogramGeneration(clips=clips, augmenter=augmenters[split], slide_frames=..., step_ms=10)`，逐 split 写入 `positive_hey_kamana/<split>/wakeword_mmap`。

| split | `repeat`（片段重复次数） | `slide_frames` | 说明 |
|---|---|---|---|
| training | 2 | 10 | 每段频谱产出 10 个错位 1 帧的窗口，模拟流式推理 |
| validation | 1 | 10 | 同上，但不重复 |
| testing | 1 | **1** | 真流式，不做人工重复 |

**切分比例**：`clips.py:146-152`

```python
train_testvalid = audio_dataset.train_test_split(test_size=2 * split_count, seed=10)  # 20% 留出
test_valid     = train_testvalid["test"].train_test_split(test_size=0.5)              # 留出部分对半
```

即 1000 条 → **800 / 100 / 100（8 : 1 : 1）**；训练集再 `repeat=2`，按样本计有效比例 **88.9 : 5.6 : 5.6**。第二次对半没有传 `seed`，所以“哪条进 validation、哪条进 test”每次运行可能不同，但比例恒为 10/10。

## 5. 负样本特征：重复到 6 s，切 5 s 块

cell 10，对 `speech` 和 `noise` 两个语料 × 三个 split 各建一个 mmap：

```python
Clips(input_directory=..., file_pattern='*.flac'|'*.wav',
      repeat_clip_min_duration_s=6)          # 短片段重复到 >= 6 s
SpectrogramGeneration(clips=..., step_ms=10, split_spectrogram_duration_s=5)
```

- 5 s 块 → 约 500 帧（10 ms 步长），块之间不重叠；前 20 帧会被丢弃（`spectrograms.py:87-96`）。
- 负样本**不做音频增广**，只有时长规整。
- 每个 split 目录整体属于该 split，**没有再次随机切分**。

**ambient（ambient 只用于评估）**：同一 cell 内 `if split != "training"` 额外生成，取**原始片段 ≥ 6 s**、`repeat_clip_min_duration_s` 不生效、`split_spectrogram_duration_s=None`（整段保留），落到 `negative/ambient/{validation_ambient,testing_ambient}`。目的是模拟真实房间的长录音，用来估 FAPH，而非训练。

## 6. 训练配置：权重决定 batch 组成

cell 11 写 `training_parameters_hey_kamana.yaml`：

| features_dir | sampling_weight | penalty_weight | truth | truncation_strategy |
|---|---|---|---|---|
| `positive_hey_kamana` | 2.0 | 1.0 | True | `truncate_start` |
| `negative/speech` | 10.0 | 1.0 | False | `random` |
| `negative/noise` | 5.0 | 1.0 | False | `random` |
| `negative/ambient` | 0.0 | 1.0 | False | `split` |

其余关键项：`window_step_ms=10`、`clip_duration_ms=1500`、`batch_size=128`、`training_steps=[10000]`、`learning_rates=[0.001]`、`positive_class_weight=[1]`、`negative_class_weight=[20]`、`eval_step_interval=500`、SpecAugment 全部为 0。

**训练窗口长度是算出来的**（`model_train_eval.py:56-89` + cell 12 的 mixednet 参数）：`clip_duration_ms=1500`、`stride=3`、first conv k=5、mixconv `[5],[7,11],[9,15],[23]` → `spectrogram_length = 50 + 154 = 204` 帧 ≈ **2.04 s**。正样本 3.2 s（320 帧）按 `truncate_start` 保留尾部，负样本 5 s 按 `random` 随机截取 204 帧。

## 7. 训练/验证时的取值路径

`microwakeword/data.py` 里同一个函数按 mode 分叉：

**训练 batch —— 按权重随机抽（`data.py:556-566`）**

```python
random.choices(providers_with_training_dir, weights=sampling_weight, k=batch_size)
```

- 每个 step 抽 128 个来源 → 约 **12% 正样本 / 59% 语音负 / 29% 噪声负**。
- ambient 没有 `training/` 目录，`get_mode_size("training")==0`，被过滤掉；`sampling_weight: 0.0` 是第二重保险。
- 抽到后按需 `spec_augment`（本 notebook 全关）。

**validation / testing —— 全量枚举（`data.py:540-545, 575-596`）**

- 不做加权采样，也没有“随机抽取子集”：遍历**所有** provider 的 `validation/` 目录，把频谱全部拼起来，只 `np.random.shuffle` 打乱顺序。
- 因此验证集 = 正样本 validation + 语音 validation + 噪声 validation + `validation_ambient` 全量。
- ambient 频谱用 `truncation_strategy="split"`，按 200 ms 步进切成 1.5 s 窗口，一条长录音变多个样本。
- `train.py:75-114`：`validation` 指标与 `validation_ambient` 合并累计，只有 FAPH 的分母用 ambient 时长。

目录名的硬约束（`data.py:163-207`，`**/*_mmap/` 递归 glob）：

```
training/  validation/  testing/  validation_ambient/  testing_ambient/
```

缺哪个就只是少一组，ambient 目录只有后两个是正常的。

## 8. 划分比例汇总

| 来源 | train | validation | test | 依据 |
|---|---|---|---|---|
| 正样本（TTS hey_kamana） | 800 条（×2 重复） | 100 | 100 | `split_count=0.1`，seed=10 |
| 语音负样本（LibriSpeech） | 28,539 | 2,703 | 2,620 = **84.3 : 8.0 : 7.7** | 上游 subset；按时长约 90 : 5 : 5 |
| 噪声负样本（FSD50K） | 6,975 | 628 | 800 = **83.0 : 7.5 : 9.5** | uploader 哈希分桶，期望 80/10/10 |
| RIR | — | — | — | 三集合共享，不参与划分 |
| ambient | 不生成 | 有 | 有 | 仅评估，`sampling_weight=0` |

对外表述时注意：正样本是 8:1:1，负样本实测约 84:8:8，两者**不是同一个比例**，也没有一个全局的 train/val/test 比例。

## 9. 已知限制与注意点

1. **同源泄漏只在作者维度被挡住**：FSD50K 按 uploader 分桶能防止同一录音者的片段跨 split，但同一个录音的多个衍生文件（例如 `rain.wav`、`rain_1.mp3` 来自不同作者）无法识别。
2. **验证/测试不重切**：LibriSpeech 与 FSD50K 的 split 完全由上游/哈希决定，notebook 没有“再随机抽 10% 做验证”这一步，所以调参空间受限。
3. **两次 shuffle 无固定种子**（`data.py:205`、`596`，以及 `clips.py:150` 的第二次对半）：成员集合固定，但顺序/边界可变，因此两次训练之间的验证指标不会逐位复现。
4. **正负样本评估方式不对称**：正样本 validation/testing 用 `slide_frames` 模拟流式，负样本不使用；ambient 用长录音切片。三者指标不能直接横向比较。
5. **特征落盘没有版本号**：换增广参数后必须手动删掉 `features_v1/` 再重建，否则 `RaggedMmap` 会读到旧特征；`feature_sources.json` 只记录音频清单哈希，不记录增广参数哈希。
6. **`manifest.csv` 里没有时长字段**：条数比例 ≠ 时长比例 ≠ 频谱帧数比例（负样本 5 s 定长切片后，帧数比例才接近条数比例）。
7. cell 14 用 tar 打包 `trained_models/`、yaml、`feature_sources.json`、manifest/attribution 等，作为实验记录归档；不含音频本体。
