# basic_training_notebook 最终训练数据集的组成

本文描述 `notebooks/basic_training_notebook.ipynb` 跑完后**真正喂给训练的那份数据集**由哪些部分组成：来源、数量、划分方式，以及训练/验证时各自的取用方式。数据处理过程（怎么从原始音频走到特征）见 [data_pipeline.zh.md](data_pipeline.zh.md)，许可与合规见 [licensed_training_data.md](licensed_training_data.md)。

默认配置为 `wake_word_name="hey_kamana"`、`positive_dataset_name="three_speeds_v1"`（cell 1）。hey hi 困难负样本是必需项：生成、特征、训练配置、评估和归档各环节都无条件包含它，配置文件与输出目录固定带 `_hey_hi` 后缀。

## 1. 总览

训练读取 `training_parameters_hey_kamana_hey_hi.yaml`（cell 14）里 `features` 列表列出的 **5 组特征**。所有特征都是 10 ms 步长的频谱（RaggedMmap）；训练窗口统一为 **204 帧 ≈ 2.04 s**（`clip_duration_ms=1500` 加上 mixednet 结构丢弃的 154 帧，cell 14 末尾有一致性断言）。

| # | features_dir（`features_v1/` 下） | truth | sampling_weight | penalty_weight | truncation_strategy | 作用 |
|---|---|---|---|---|---|---|
| 1 | `positive_hey_kamana_three_speeds_v1` | True | 2.0 | 1.0 | `truncate_start` | 正样本 |
| 2 | `negative/speech` | False | 10.0 | 1.0 | `random` | 语音负样本 |
| 3 | `negative/noise` | False | 5.0 | 1.0 | `random` | 噪声负样本 |
| 4 | `negative/hey_hi` | False | 2.0 | 1.0 | `truncate_start` | 困难负样本（近似短语） |
| 5 | `negative/ambient` | False | 0.0 | 1.0 | `split` | 仅验证/测试的长录音误触发评估，不参与训练 |

## 2. 特征落盘结构

```
features_v1/
├── positive_hey_kamana_three_speeds_v1/
│   ├── training/wakeword_mmap
│   ├── validation/wakeword_mmap
│   └── testing/wakeword_mmap
├── negative/
│   ├── speech/{training,validation,testing}/negative_mmap
│   ├── noise/{training,validation,testing}/negative_mmap
│   ├── hey_hi/{training,validation,testing}/hey_hi_mmap   # 另有 feature_sources.json
│   └── ambient/
│       ├── validation_ambient/{speech,noise}_mmap
│       └── testing_ambient/{speech,noise}_mmap
└── feature_sources.json        # 记录与 data_v1/prepared/manifest.json 的哈希关联
```

特征的上游原始音频在 `data_v1/prepared/`（`noise/`、`speech/`、`rirs/` 加 `manifest.csv`、`manifest.json`、`ATTRIBUTION.md`、`excluded_fsd.csv`、`sources/`），由 cell 7 的 `prepare_licensed_data("data_v1")` 生成。

## 3. 各组数据的组成

### 3.1 正样本：TTS 合成的 "Hey Kamana"（cell 5、6、9、11）

- **发音**：音素 `hˈeɪ kɑmˈɑnɑ`，用 Piper `en_US-libritts_r-medium` 以音素输入合成。
- **三种语速**：`fast`（length_scale=0.65）、`normal`（1.0）、`slow`（1.25）各 1000 条，共 **3000 条**原始 WAV，落在 `generated_samples/hey_kamana/three_speeds_v1/{fast,normal,slow}/`，生成参数写入 `generation.json`。
- **划分**：`Clips(random_split_seed=10, split_count=0.1)` 按 **8 : 1 : 1** 切成 training 2400 / validation 300 / testing 300。这是唯一一处"同一批音频被切进三个集合"的数据，因此正样本三个集合之间是同源不同增广。
- **增广**（每个 split 一个独立增广器，背景噪声池只用**同一 split** 的 `noise/<split>/`，RIR 池三集合共享）：输出固定 3.2 s，词尾抖动 0.195–0.205 s；`AddBackgroundNoise` 概率 0.75（SNR −5…10 dB）、`RIR` 0.5、`Gain` 1.0，EQ/失真/变调/带阻/彩色噪声各 0.1。
- **特征数量**（`slide_frames=10` 表示每段频谱产出 10 个错位 1 帧的窗口）：

  | split | 条数 | repeat | slide_frames | 特征数 |
  |---|---|---|---|---|
  | training | 2400 | 2 | 10 | **48,000** |
  | validation | 300 | 1 | 10 | **3,000** |
  | testing | 300 | 1 | 1（真流式） | **300** |

  3.2 s 频谱共 320 帧，训练时按 `truncate_start` 截取**末尾** 204 帧——词尾靠近片段结尾，保证完整发音在窗口内。
- **repeat=2 为什么不是简单复制**：增广不在合成 TTS 时做，而在生成特征 mmap 时逐条现场做（`clips.audio_generator(repeat=2)` 只是把同一批原始 PCM 顺序遍历两遍；每条 yield 后立即重新随机增广——jitter 长度、是否加背景噪声及 SNR、是否混响及房间、增益等都重新抽）。因此两遍得到的是同一发音的两种不同“录音条件”；而同一次增广切出的 10 个滑窗只是整体平移 1 帧，几乎相同。若增广是确定性的，repeat=2 就只是复制两份。

### 3.2 语音负样本：LibriSpeech（cell 7、13）

- **来源**：LibriSpeech `train-clean-100` / `dev-clean` / `test-clean` 的原始 FLAC（CC BY 4.0），**直接沿用上游 subset 作为 split，不再二次切分**。
- **实测文件数**（`train-result/` 的 `manifest.csv`）：training 28,539 / validation 2,703 / testing 2,620，约 **84 : 8 : 8**（按时长约 90 : 5 : 5）。
- **处理**：不足 6 s 的片段循环重复到 ≥6 s（`repeat_clip_min_duration_s=6`），再切成**不重叠的 5 s 块**（500 帧，前 20 帧丢弃）；**不做音频增广**。特征条数等于 5 s 块数，多于文件数。

### 3.3 噪声负样本：FSD50K eval（cell 7、13）

- **来源**：FSD50K v1.0 eval 子集，仅保留许可 URL 命中 CC0 / CC BY 允许列表、且有 uploader 与 title 的片段；**按 uploader 的 `sha256 % 10` 分桶**（0→validation，1→testing，其余→training，期望 80/10/10），同一 uploader 不会跨集合。
- **实测文件数**：training 6,975 / validation 628 / testing 800。
- **处理**：与语音负样本相同（重复到 ≥6 s、切 5 s 块、无音频增广）。
- **双重身份**：`noise/<split>/` 同时是正样本增广的背景噪声池，且池与集合严格对应，评估音频不会在训练增广里出现过。

### 3.4 hey hi 困难负样本（cell 3、4；音频来自 `hey_hi_negative_dataset.ipynb`）

- **来源**：Piper 合成的易混淆短语 "hey hi"（音素 `hˈeɪ hˈaɪ`），同样三种语速 × 1000 条共 3000 条原始 WAV；按原始 PCM 去重后逐档 80/10/10 划分（seed 42），**保留生成时的原始划分**。
- **数量**（notebook 实际输出）：training **2400** / validation **300** / testing **300**。
- **特征**：每条音频恰好产出 **1 条 204 帧特征**（`context_frames=204`，重采样 16 kHz、左侧补零、完整发音保留在尾部）——不重复成 6 秒、不重新划分、不参与 ambient 评估；过长音频直接报错而不是截掉部分发音。
- **作用**：采样权重与正样本相同（2.0），专门压制与唤醒词首音节相同的 "hey hi" 误触发；训练后由 cell 16 在 hey hi 测试集上**单独评估**误触发率，不与 ambient 的每小时误唤醒率混算。

### 3.5 ambient 评估集：长环境录音（cell 13）

- **只有 validation / testing 两份**（`validation_ambient`、`testing_ambient`，各含 `speech_mmap` 与 `noise_mmap`），**没有 training 目录**——这是它不进训练 batch 的第一重保证，`sampling_weight=0.0` 是第二重。
- **组成**：speech 与 noise 两个语料中**原始时长 ≥6 s 的片段**，不做 6 秒重复、不切块，整段生成特征；评估时按 `truncation_strategy="split"` 切成窗口，用来估算每小时误唤醒（FAPH）。

## 4. 数量汇总

| 组 | 原始音频 | training | validation | testing | 训练用特征形态 |
|---|---|---|---|---|---|
| 正样本（TTS hey_kamana） | 3000（3 语速 × 1000） | 2400（→48,000 特征） | 300（→3,000 特征） | 300（→300 特征） | 3.2 s 增广片段，截尾 204 帧 |
| 语音负样本（LibriSpeech） | 33,862 个 FLAC | 28,539 | 2,703 | 2,620 | 重复到 ≥6 s 后切 5 s 块 |
| 噪声负样本（FSD50K eval） | 8,403 个 WAV | 6,975 | 628 | 800 | 同上 |
| hey hi 困难负样本 | 3000（3 语速 × 1000） | 2400 | 300 | 300 | 每条 1 个 204 帧特征 |
| ambient（仅评估） | — | 不生成 | speech+noise ≥6 s 片段 | 同左 | 整段特征，评估时切窗 |

注意三点：

1. speech/noise 的条数是 `train-result/hey_kamana_v1/data_v1/prepared/manifest.csv`（102,265 行）的**实测值**，许可过滤或上游版本变化会小幅改变数字；正样本与 hey hi 的条数由生成参数固定。
2. 负样本的"特征条数"（5 s 块）多于文件数，上表按文件计。
3. 正样本 8:1:1、LibriSpeech 84:8:8、FSD50K 约 83:7.5:9.5、hey hi 80:10:10——**各组划分比例并不一致**，也不存在一个全局的 train/val/test 比例。

## 5. 训练/验证时怎么取用

**训练 batch（cell 15，`batch_size=128`，共 10,000 步）**：按 sampling_weight 随机抽（权重 2 : 10 : 5 : 2，合计 19），每个 batch 的期望组成为：

| 组 | 权重占比 | 每 batch 期望条数 |
|---|---|---|
| 正样本 | 10.5% | ≈13 |
| 语音负样本 | 52.6% | ≈67 |
| 噪声负样本 | 26.3% | ≈34 |
| hey hi | 10.5% | ≈13 |

配合类别权重 `positive_class_weight=[1]`、`negative_class_weight=[20]`：正样本稀缺但错分代价高，负样本主导 batch 数量。

**重复使用程度**：训练没有 epoch 概念，抽取是两层独立的有放回采样（`data.py`：先 `random.choices` 按 19 份权重抽 provider，再在 provider 内 `random.choice` 抽频谱）。10,000 步 × 128 = 约 128 万次抽全部来自生成阶段冻结好的固定 mmap 池，训练时不再重新增广。由此正样本平均每条特征被抽 ≈2.8 次，hey hi ≈56 次（池小、角色是反复压制特定误触发）。speech/noise 用 `random` 截取，同一条 5 s 频谱每次随机截不同的 204 帧窗口，重复抽到不等于看到同一训练样本；正样本和 hey hi 用 `truncate_start` 固定截尾，重复抽到时样本完全相同。

**验证 / 测试**：不做加权采样，**全量枚举**所有特征组的对应 split 并打乱顺序——即验证集 = 正样本 validation + speech validation + noise validation + hey hi validation + `validation_ambient`（测试集同理换成 testing）。ambient 频谱按 `split` 策略切窗；FAPH 的分母只统计 ambient 时长。

## 6. 这份数据集没有的东西

- **没有任何真人朗读的唤醒词录音**：正样本 100% 是 Piper TTS（三语速 + 增广），真实麦克风场景的表现依赖增广的逼真程度。
- **唯一的对抗短语是 "hey hi"**：没有其他近似唤醒词或干扰语句；FSD50K 背景池也不是保证无语音的语料。
- **hey hi 不保证说话人隔离**：其 80/10/10 划分只保证原始录音不重复（见生成 notebook 的说明）。
- **许可遗留项**：Piper 生成权重等产物依赖仍待独立审查，本数据集的许可审计只覆盖 FSD50K / LibriSpeech / SLR28 三类输入（见 notebook 开头的 licensing 声明与 [licensed_training_data.md](licensed_training_data.md)）。
