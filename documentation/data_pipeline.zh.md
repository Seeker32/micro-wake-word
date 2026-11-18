# basic_training_notebook 全流程

本文对应 [`notebooks/basic_training_notebook.ipynb`](../notebooks/basic_training_notebook.ipynb) 当前代码，按从上到下的执行顺序说明环境、数据、训练、评估和归档。下面的 cell 编号从 0 开始，插入或删除单元格后可能变化。许可细节见 [licensed_training_data.md](licensed_training_data.md)，数据源见 [data_sources.md](data_sources.md)。

## 流程总览

```text
设置路径与环境 → Clean 100 试听、生成唤醒词与 hey hi
  → 准备许可音频和 CHiME-6 → 生成正样本、负样本和 ambient 特征
  → 写训练配置 → 训练、选择权重、转换并测试量化流式模型
  → hey hi 独立测试 → CHiME-6 独立测试 → 打包模型与来源记录
```

## 1. 路径与依赖（cell 1–4）

顶部集中设置仓库、数据、模型和输出路径。默认唤醒词为 `hi_fondcircle`，音素为 `hˈaɪ fˈɒnd sˈɜːkəl`；困难负样本为 `hey hi`。修改唤醒词、音素或数据版本后，应同步检查样本目录、版本名与配置，并从路径单元格开始运行。

Colab 默认仓库位于 `/content/micro-wake-word`；目录不存在时会克隆仓库。依赖单元以 editable 模式安装项目，再安装 `piper-tts==1.3.0`，移除已有 ONNX Runtime CPU/GPU 包并安装 `onnxruntime-gpu[cuda,cudnn]`。接着预加载 CUDA 库并打印可用 provider。若当前会话已导入旧版 ONNX Runtime，安装后需要重启会话。Clean 100 的推理设备日志与 TensorFlow 训练设备分别判断；Colab 需在运行时设置中选择 GPU。

首次准备原始许可数据约需下载 15 GB；notebook 建议至少预留 50 GB 磁盘，并要求系统有 `zip` 命令。CHiME-6 另有下载和空间需求，见第 6 节。

## 2. 合成正样本与困难负样本（cell 5–9）

`load_clean100()` 下载或校验 Clean 100 模型及配置，先生成一条试听。正样本使用三档语速 `fast`、`normal`、`slow`，每档生成 1,000 条，共 3,000 条，保存于 `generated_samples/hi_fondcircle/clean100_v1_three_speeds/`，并记录 `generation.json`。生成使用不同的说话人和合成参数；重跑会校验并复用匹配的结果。

`hey hi` 使用同一个模型，每档生成 1,000 条，按每档 800/100/100 划为 training、validation、testing，默认保存在 `generated_datasets/hey_hi_negative_clean100_v1/raw/`。随后 `prepare_hard_negative_features()` 按已有划分导出特征到 `features_v1/negative/hey_hi_clean100_v1/`，并验证来源与特征以支持复用。它使用 204 帧上下文；修改音素时，应按 notebook 注释更新数据版本，避免复用旧缓存。

## 3. 许可音频准备（cell 10–11）

可选的 Drive 单元把预先下载的 8 个压缩包复制到 `data_v1/downloads/`，并检查有官方 MD5 的新复制文件；也可跳过该单元，由 `prepare_licensed_data()` 下载。准备过程提取并校验音频及来源记录：

| 来源 | 用途 | 划分 |
|---|---|---|
| FSD50K eval | 噪声负样本和正样本背景增广 | 按 uploader 哈希分桶，避免同一 uploader 跨集合；只保留符合代码许可及元数据要求的片段 |
| LibriSpeech train-clean-100、dev-clean、test-clean | 语音负样本 | 分别对应 training、validation、testing |
| OpenSLR SLR28 simulated RIR | 混响增广 | 三个集合共用 |

结果位于 `data_v1/prepared/`，包括 `noise/`、`speech/`、`rirs/`、`manifest.json`、`manifest.csv`、`ATTRIBUTION.md`、排除记录和来源快照。重跑会检查已准备数据的完整性。来源和许可记录不等于已经完成产品依赖审计或真实设备测试。

## 4. 正样本增广和特征（cell 12–14）

`Clips` 从三档语速目录加载 WAV，以 `split_count=0.1` 划出约 80% training、10% validation、10% testing。代码对第一次划分设置 seed 10，第二次将留出集合分半时没有显式传 seed，因此不能据此保证每次验证与测试成员完全相同。

每个 split 使用对应的 `noise/<split>` 作为背景池，RIR 共用。增广输出为 3.2 秒，背景噪声概率 0.75、RIR 0.5、Gain 1.0，其他音色扰动各为 0.1，背景信噪比范围为 -5 到 10 dB。试听单元生成并播放 `augmented_clip.wav`，可用于检查目标词是否清晰。

特征步长为 10 ms，输出到 `features_v1/positive_hi_fondcircle_clean100_v1_three_speeds/<split>/wakeword_mmap`。training 的 `repeat=2`、`slide_frames=10`；validation 为 1 和 10；testing 为 1 和 1。testing 使用流式模型，不做人工滑窗重复。

## 5. 普通负样本与 ambient 特征（cell 15）

语音和噪声保持准备阶段的划分，短分类负样本重复至至少 6 秒，再按 5 秒切分频谱，写入 `features_v1/negative/{speech,noise}/<split>/negative_mmap`。

validation 和 testing 额外生成 ambient 特征：只选满足最短时长条件的原始长片段，不做重复或 5 秒切块，写入 `features_v1/negative/ambient/{validation_ambient,testing_ambient}/`。它们只用于评估。`features_v1/feature_sources.json` 记录许可音频清单、正样本生成记录、增广来源及困难负样本信息。

## 6. CHiME-6 远场评估集（cell 16–17）

CHiME-6 提供真实家庭远场录音，只进入 ambient 验证和测试，不参与训练采样。代码优先复用已校验的 `chime6_v1/prepared/`，其次从 `data_v1/downloads` 或可选 Drive 目录恢复 `chime6_v1.zip`／已解压数据，最后才从 OpenSLR 下载。Notebook 注释给出的下载量约 11.9 GB、峰值磁盘约 12 GB。

默认将官方 dev 映射到 `validation_ambient`、eval 映射到 `testing_ambient`，notebook 标注时长分别约 4.46 小时和 15.62 小时。准备过程选取参考阵列的 CH1、切成最长约 10 分钟的连续片段；特征生成保持这些片段完整，不做增广、重复或随机切块。特征和来源记录位于 `features_v1/negative/ambient_chime6/`，相同输入的已校验特征可复用。

训练时 validation ambient 会整体载入，notebook 估计 CHiME-6 特征约占 1.2 GB，后续验证切片还会增加内存占用。内存不足时，可把准备调用改为：

```python
chime6_lineage = prepare_ambient_features(
    chime6_prepared_dir, chime6_features_dir, splits=("testing",)
)
```

这会重建该 CHiME-6 特征集合，只保留 testing；训练期 ambient 验证将使用原有语音和噪声数据。

## 7. 配置与训练（cell 18–19）

配置写入 `training_parameters_hi_fondcircle_clean100_v1_hey_hi.yaml`。采样权重决定训练 batch 的来源；ambient 不参与训练采样：

| 来源 | sampling_weight | truth | 截取方式 |
|---|---:|---|---|
| 唤醒词正样本 | 2 | 正 | `truncate_start` |
| LibriSpeech 语音 | 10 | 负 | `random` |
| FSD50K 噪声 | 5 | 负 | `random` |
| hey hi 困难负样本 | 2 | 负 | `truncate_start` |
| 原有 ambient、CHiME-6 ambient | 各 0 | 负 | `split` |

非 ambient 权重总和为 19，训练时按权重抽取，约为 10.5% 正样本、52.6% 语音、26.3% 噪声、10.5% hey hi；实际 batch 会随机波动。`penalty_weight` 均为 1，正负类别权重分别为 1 和 20。主要参数为 10,000 steps、batch size 128、学习率 0.001、每 500 steps 验证一次、音频窗口 1,500 ms，SpecAugment 关闭。

当前 mixednet 的 stride 为 3、卷积上下文丢弃 154 帧；结合 10 ms 步长，配置检查得到输入上下文 204 帧。修改模型结构或窗口长度时，需同步检查 `hey_hi_context_frames` 并重新准备相应特征。

训练期间仍用非流式验证的 `average_viable_recall` 保存 `best_weights.weights.h5`，并按验证间隔保存快照。训练步数和 checkpoint 恢复方式不变，不启用早停。

## 8. 用量化流式验证选型，再验收（cell 20–22）

- **选权重**：转换非流式最佳、首个、最接近最后一步 75% 的阶段和最后一份快照，去掉重复权重。每个候选共用 500 条训练侧校准特征；从 validation 的错位窗恢复每条正样本，只计一次。合并原有语音、噪声和已准备的 CHiME-6 validation 背景数据，在 `STREAMING_FAPH_BUDGET`（默认 0.1 次/小时）内选漏唤醒率最低的候选。背景时长不足以允许 1 次误唤醒时，排名实际比较的是零观测误唤醒工作点。
- **锁定阈值测试**：只让入选模型进入 testing，沿用验证阈值，报告漏唤醒、每小时误唤醒和 `hey hi` 误触发数。排名与测试写入训练目录的 `streaming_checkpoint_selection.json`。入选权重保存为 `best_streaming_weights.weights.h5`；原 `best_weights.weights.h5` 不变。若全部候选都无法在预算内检出正样本，不替换现有模型。
- **部署文件与诊断曲线**：原有 `tflite_stream_state_internal_quant/stream_state_internal_quant.tflite` 路径现在存放实际入选的 TFLite。同目录的 `tflite_streaming_roc.txt` 仍记录 testing 诊断曲线；它不参与选权重或调阈值。每条录音独立初始化流式状态，误唤醒率按合并背景时长计算。
- **hey hi 独立测试**：使用量化流式模型逐条统计困难负样本在 0.95、0.98、0.99 阈值下的误触发数和比例，写入 `hey_hi_test_results.json`；此比例不与 ambient 的 FAPH 混算。
- **CHiME-6 独立测试**：只对 testing 计算 0.90、0.95、0.98、0.99 阈值下的 FAPH、估计误唤醒次数及触发片段数，并列出峰值最高的片段；结果写入 `chime6_ambient_results.json`。这与汇总测试使用相同的每小时误唤醒计算口径。

合成正样本的测试和远场录音评估不能代替真实用户、真实设备和目标部署环境中的测试。

## 9. 打包与复用（cell 23）

最后生成 `hi_fondcircle_clean100_v1_hey_hi_run.tar.gz`，位于仓库父目录。归档包含训练目录中的模型和报告、训练 YAML、特征来源记录、正样本与 hey hi 的 `generation.json`，以及许可音频和 CHiME-6 的清单、归属和来源文件；不包含完整原始音频与特征集。

只复评已有权重时，确保训练目录保留原 `training_config.yaml` 和权重，且当前路径下有配套特征。完成路径、依赖和配置单元（1、2、18）后，将 cell 19 的 `RUN_TRAINING` 设为 `False`，继续执行选型和测试。

跨会话复用 hey hi 音频时，保留完整 `raw/` 目录及 `generation.json` 并更新顶部路径。更改正样本增广、音频或特征参数时，应检查对应输出和缓存是否需要重建；普通正负样本的来源记录不能替代对增广参数变化的检查。
