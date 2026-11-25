# basic_training_notebook 最终训练数据集的组成

本文描述 [`notebooks/basic_training_notebook.ipynb`](../notebooks/basic_training_notebook.ipynb) 跑完后**真正喂给训练的那份数据集**由哪些部分组成：来源、数量、划分方式，以及训练/验证时各自的取用方式。数据处理过程（怎么从原始音频走到特征）见 [data_pipeline.zh.md](data_pipeline.zh.md)，数据源与许可见 [data_sources.md](data_sources.md) 和 `microwakeword/audio/licensed_data.py`。

默认配置为 `wake_word_name="hi_circle"`、`wake_word_phonemes="hˈaɪ sˈɜːkəl"`、`positive_dataset_name="clean100_v1_three_speeds"`（cell 1）。hey hi 困难负样本是必需项：生成、特征、训练配置、评估和归档各环节都无条件包含它，配置文件与输出目录固定带 `_clean100_v1_hey_hi` 后缀。

## 1. 总览

训练读取 `training_parameters_hi_circle_clean100_v1_hey_hi.yaml`（cell 18）里 `features` 列表列出的 **6 组特征**。所有特征都是 10 ms 步长的频谱（RaggedMmap）；训练窗口统一为 **204 帧 ≈ 2.04 s**（`clip_duration_ms=1500` 加上 mixednet 结构丢弃的 154 帧，cell 18 末尾有一致性断言）。

| # | features_dir（`features_v1/` 下） | truth | sampling_weight | penalty_weight | truncation_strategy | 作用 |
|---|---|---|---|---|---|---|
| 1 | `positive_hi_circle_clean100_v1_three_speeds` | True | 2.0 | 1.0 | `truncate_start` | 正样本 |
| 2 | `negative/speech` | False | 10.0 | 1.0 | `random` | 语音负样本 |
| 3 | `negative/noise` | False | 5.0 | 1.0 | `random` | 噪声负样本 |
| 4 | `negative/hey_hi_clean100_v1` | False | 2.0 | 1.0 | `truncate_start` | 困难负样本（近似短语） |
| 5 | `negative/ambient` | False | 0.0 | 1.0 | `split` | 仅验证/测试的长录音误触发评估，不参与训练 |
| 6 | `negative/ambient_chime6` | False | 0.0 | 1.0 | `split` | CHiME-6 真实远场误触发评估，不参与训练 |

## 2. 特征落盘结构

```
features_v1/
├── positive_hi_circle_clean100_v1_three_speeds/
│   ├── training/wakeword_mmap
│   ├── validation/wakeword_mmap
│   └── testing/wakeword_mmap
├── negative/
│   ├── speech/{training,validation,testing}/negative_mmap
│   ├── noise/{training,validation,testing}/negative_mmap
│   ├── hey_hi_clean100_v1/
│   │   ├── {training,validation,testing}/hey_hi_mmap
│   │   └── feature_sources.json          # 音频清单、上下文参数、逐条 sha256
│   ├── ambient/
│   │   ├── validation_ambient/{speech,noise}_mmap
│   │   └── testing_ambient/{speech,noise}_mmap
│   └── ambient_chime6/
│       ├── validation_ambient/chime6_mmap
│       ├── testing_ambient/chime6_mmap
│       └── feature_sources.json
└── feature_sources.json                  # 与 data_v1/prepared/manifest.json 的哈希关联
```

特征的上游原始音频在 `data_v1/prepared/`（`noise/`、`speech/`、`rirs/` 加 `manifest.csv`、`manifest.json`、`ATTRIBUTION.md`、`excluded_fsd.csv`、`sources/`），由 cell 11 的 `prepare_licensed_data("data_v1")` 生成；CHiME-6 的音频与审计文件在 `chime6_v1/prepared/`，由 cell 17 的 `ensure_chime6()` 准备。

## 3. 各组数据的组成

### 3.1 正样本：Clean 100 合成的 "Hi Circle"（cell 5、6、14）

- **发音与引擎**：音素 `hˈaɪ sˈɜːkəl`，用 Bryce Beattie 的 Piper **Clean 100**（基于 LibriTTS-R train-clean-100，CC BY 4.0 从零训练）以音素输入合成，输出 22.05 kHz 单声道 PCM16。模型权重与配置的 SHA-256 记录在 `generation.json`。
- **说话人与噪声档**：Clean 100 提供 245 个说话人 ID。生成时按索引轮转 `speaker_id`（245 个）与 `noise_scale`（0.60、0.667、0.72、0.78、0.83 共 5 档，`noise_w_scale=0.8`），因此相邻样本更换说话人，同一说话人还有 5 种发音噪声。三档语速只改 `length_scale`。
- **三种语速**：`fast`（length_scale=0.65）、`normal`（1.0）、`slow`（1.25）各 1000 条，共 **3000 条**原始 WAV，落在 `generated_samples/hi_circle/clean100_v1_three_speeds/{fast,normal,slow}/{0..999}.wav`，生成参数写入 `generation.json`。目录已存在时按 `generation.json` 与逐档文件数校验复用，参数不符会报错而不是混用。
- **划分**：`Clips(random_split_seed=10, split_count=0.1)` 按 **8 : 1 : 1** 切成 training 2400 / validation 300 / testing 300。注意第二次把留出集合对半分时没有显式传 seed（`clips.py`），所以 validation 与 testing 的成员划分依赖全局 RNG 状态，不能假定每次都相同。
- **增广**（每个 split 一个独立增广器，背景噪声池只用**同一 split** 的 `noise/<split>/`，RIR 池三集合共享）：输出固定 3.2 s，词尾抖动 0.195–0.205 s；`AddBackgroundNoise` 概率 0.75（SNR −5…10 dB）、`RIR` 0.5、`Gain` 1.0，EQ/失真/变调/带阻/彩色噪声各 0.1。
- **特征数量**（3.2 s 增广片段产出 317 帧；`slide_frames=10` 丢掉尾部 9 帧，切成 10 个 308 帧的错位窗口，训练时 `truncate_start` 再截成 204 帧）：

  | split | 条数 | repeat | slide_frames | 特征数 |
  |---|---|---|---|---|
  | training | 2400 | 2 | 10 | **48,000** |
  | validation | 300 | 1 | 10 | **3,000** |
  | testing | 300 | 1 | 1（真流式） | **300** |
- **repeat=2 为什么不是简单复制**：增广不在合成 TTS 时做，而在生成特征 mmap 时逐条现场做（`clips.audio_generator(repeat=2)` 只是把同一批原始 PCM 顺序遍历两遍；每条 yield 后立即重新随机增广——jitter 长度、是否加背景噪声及 SNR、是否混响及房间、增益等都重新抽）。因此两遍得到的是同一发音的两种不同“录音条件”；而同一次增广切出的 10 个滑窗只是整体平移 1 帧，几乎相同。若增广是确定性的，repeat=2 就只是复制两份。

### 3.2 语音负样本：LibriSpeech（cell 11、15）

- **来源**：LibriSpeech `train-clean-100` / `dev-clean` / `test-clean` 的原始 FLAC（CC BY 4.0），**直接沿用上游 subset 作为 split，不再二次切分**。
- **实测文件数**（`train-result/hi_circle_v2/prepared/manifest.csv`）：training 28,539 / validation 2,703 / testing 2,620，约 **84 : 8 : 8**（按时长约 90 : 5 : 5）。
- **处理**：不足 6 s 的片段循环重复到 ≥6 s（`repeat_clip_min_duration_s=6`），再切成**不重叠的 5 s 块**（500 帧，前 20 帧丢弃）；**不做音频增广**。特征条数等于 5 s 块数，多于文件数。

### 3.3 噪声负样本：FSD50K eval（cell 11、15）

- **来源**：FSD50K v1.0 eval 子集共 10,231 条。只保留逐条许可命中 CC0 / CC BY 允许列表的片段（fail-closed），并要求 uploader 与 title 齐全；**按 uploader 的 `sha256 % 10` 分桶**（0→validation，1→testing，其余→training，期望 80/10/10），同一 uploader 不会跨集合。
- **实测数量**：10,231 条中选用 **8,403** 条（training 6,975 / validation 628 / testing 800），排除 **1,828** 条（CC BY-NC 3.0 共 1,425 条、CC Sampling+ 1.0 共 403 条），逐条写在 `prepared/excluded_fsd.csv`。
- **处理**：与语音负样本相同（重复到 ≥6 s、切 5 s 块、无音频增广）。
- **双重身份**：`noise/<split>/` 同时是正样本增广的背景噪声池，且池与集合严格对应，评估音频不会在训练增广里出现过。

### 3.4 hey hi 困难负样本（cell 7–9）

- **来源**：同一份 Clean 100，用音素 `hˈeɪ hˈaɪ` 合成三种语速的 "hey hi"，每档 1,000 条共 **3,000 条**原始 WAV，不再依赖 `hey_hi_negative_dataset.ipynb`。生成时说话人与 `noise_scale` 同样轮转，同一批内按原始 PCM 的 SHA-256 去重。
- **时长上限**：每条合成音频必须满足 `((204 - 1) * 10 - 200) / 1000 = 1.83 s`，超长样本丢弃并重新合成（最多 10 次尝试），保证训练窗口截尾后不切断发音。
- **划分**：`Random(42)` 打乱后逐档落到 800/100/100，即 training **2400** / validation **300** / testing **300**；划分结果按目录固定，重跑只校验和复用。
- **特征**：每条音频补零到 3.2 s（尾部留 0.2 s 静音）、重采样 16 kHz 后产出 **317 帧**轨道（不是 204 帧）；训练用 `truncate_start` 取末 204 帧。不重复成 6 秒、不重新划分、不参与 ambient 评估；音频超过上限时 `prepare_hard_negative_features()` 直接报错，而不是截掉部分发音。
- **作用**：采样权重与正样本相同（2.0），专门压制与唤醒词首音节相同的 "hey hi" 误触发；训练后由 cell 21 在 hey hi 测试集上**单独评估**误触发率，不与 ambient 的每小时误唤醒率混算。

### 3.5 ambient 评估集：长环境录音（cell 15）

- **只有 validation / testing 两份**（`validation_ambient`、`testing_ambient`，各含 `speech_mmap` 与 `noise_mmap`），**没有 training 目录**——这是它不进训练 batch 的第一重保证，`sampling_weight=0.0` 是第二重。
- **组成**：speech 与 noise 两个语料中**原始时长 ≥6 s 的片段**，不做 6 秒重复、不切块，整段生成特征；评估时按 `truncation_strategy="split"` 切成窗口，用来估算每小时误唤醒（FAPH）。

### 3.6 CHiME-6 真实远场 ambient（cell 16–17）

- **来源**：CHiME-6（OpenSLR SLR150，CC BY-SA 4.0），家庭多人晚餐对话；取每个房间参考阵列的 **CH1**，按 600 s 连续切片，原始 ≥6 s 才保留。
- **划分**：官方 dev → `validation_ambient`（27 段，**4.46 h**），eval → `testing_ambient`（96 段，**15.62 h**），沿用官方 session，不跨 split。房间选择用 `prepare_chime6()` 默认值 `dev_rooms="busiest"`、`eval_rooms="all"`，所以 validation 侧只有 2 个阵列、testing 侧 6 个阵列。
- **特征**：整段生成，不做增广、重复或随机切块，写入 `features_v1/negative/ambient_chime6/<split>_ambient/chime6_mmap`；`feature_sources.json` 记录逐段 sha256 与来源，可校验复用。
- **口径**：与 3.5 的片段一起进入合并后的 FAPH（`ambient_eval.find_ambient_sets()` 同时读两个目录），训练期的 `validation_ambient` 还会参与 `average_viable_recall`，因此影响 `best_weights.weights.h5` 的挑选。

## 4. 数量汇总

| 组 | 原始音频 | training | validation | testing | 训练用特征形态 |
|---|---|---|---|---|---|
| 正样本（Clean 100 hi_circle） | 3000（3 语速 × 1000） | 2400（→48,000 特征） | 300（→3,000 特征） | 300（→300 特征） | 3.2 s 增广片段，截尾 204 帧 |
| 语音负样本（LibriSpeech） | 33,862 个 FLAC | 28,539 | 2,703 | 2,620 | 重复到 ≥6 s 后切 5 s 块 |
| 噪声负样本（FSD50K eval） | 8,403 个 WAV（10,231 中选） | 6,975 | 628 | 800 | 同上 |
| hey hi 困难负样本 | 3000（3 语速 × 1000） | 2400 | 300 | 300 | 每条 1 条 317 帧轨道，截尾 204 帧 |
| ambient speech/noise（仅评估） | — | 不生成 | ≥6 s 片段整段 | 同左 | 整段特征，评估时切窗 |
| CHiME-6 ambient（仅评估） | dev+eval 共 123 段（600 s 切片） | 不生成 | 27 段 / 4.46 h | 96 段 / 15.62 h | 整段特征，评估时切窗 |

注意三点：

1. speech/noise 的条数是 `train-result/hi_circle_v2/prepared/manifest.csv`（102,265 行，含 60,000 条 RIR）的**实测值**，许可过滤或上游版本变化会小幅改变数字；正样本与 hey hi 的条数由生成参数固定。
2. 负样本的"特征条数"（5 s 块）多于文件数，上表按文件计。
3. 正样本 8:1:1、LibriSpeech 84:8:8、FSD50K 约 83:7.5:9.5、hey hi 80:10:10——**各组划分比例并不一致**，也不存在一个全局的 train/val/test 比例。

## 5. 训练/验证时怎么取用

**训练 batch（cell 18 的权重、cell 19 的调用，`batch_size=128`，共 10,000 步）**：按 sampling_weight 随机抽（权重 2 : 10 : 5 : 2，合计 19），每个 batch 的期望组成为：

| 组 | 权重占比 | 每 batch 期望条数 |
|---|---|---|
| 正样本 | 10.5% | ≈13 |
| 语音负样本 | 52.6% | ≈67 |
| 噪声负样本 | 26.3% | ≈34 |
| hey hi | 10.5% | ≈13 |

配合类别权重 `positive_class_weight=[1]`、`negative_class_weight=[20]`：正样本稀缺但错分代价高，负样本主导 batch 数量。

**重复使用程度**：训练没有 epoch 概念，抽取是两层独立的有放回采样（`data.py`：先 `random.choices` 按 19 份权重抽 provider，再在 provider 内 `random.choice` 抽频谱）。10,000 步 × 128 = 约 128 万次抽全部来自生成阶段冻结好的固定 mmap 池，训练时不再重新增广。由此正样本平均每条特征被抽 ≈2.8 次，hey hi ≈56 次（池小、角色是反复压制特定误触发）。speech/noise 用 `random` 截取，同一条 5 s 频谱每次随机截不同的 204 帧窗口，重复抽到不等于看到同一训练样本；正样本和 hey hi 用 `truncate_start` 固定截尾，重复抽到时样本完全相同。

**验证 / 测试**：不做加权采样，**全量枚举**所有 6 组特征的对应 split 并打乱顺序——即验证集 = 正样本 validation + speech validation + noise validation + hey hi validation，外加单独计的 `validation_ambient`（speech、noise、CHiME-6）；测试集同理换成 testing。ambient 频谱按 `split` 策略切窗；FAPH 的分母只统计 ambient 时长，正样本的漏唤醒不计入 FAPH。

## 6. 这份数据集没有的东西

- **没有任何真人朗读的唤醒词录音**：正样本 100% 是 Clean 100 TTS（三语速 + 增广），真实麦克风场景的表现依赖增广的逼真程度。
- **唯一的对抗短语是 "hey hi"**：没有其他近似唤醒词或干扰语句；FSD50K 背景池也不是保证无语音的语料。
- **正样本与 hey hi 都不保证说话人隔离**：两者都按样本随机划分，同一说话人会同时出现在 training、validation 和 testing。hey hi 只保证原始 PCM 不重复。
- **ambient 是"别人家的房间"**：CHiME-6 是 2018 年的英文晚餐对话，只能近似家庭电视/交谈场景，不能代替目标设备与目标房间的实测。
- **许可遗留项**：Clean 100 的 TTS 权重与产品依赖仍待独立审查；本数据集的许可审计覆盖 FSD50K / LibriSpeech / SLR28 三类输入（CC0/CC-BY/Apache 2.0 白名单）与 CHiME-6（CC BY-SA 4.0，对外分发衍生数据需同许可）。详见 `data_v1/prepared/manifest.json`、`chime6_v1/prepared/manifest.json` 与 [data_sources.md](data_sources.md)。
