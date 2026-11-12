# 预下载数据集档案说明

WSL 训练 notebook（`notebooks/basic_training_notebook_wsl.ipynb`）把数据集目录固定在
`/mnt/e/data/wake_word`。下列压缩包提前下载到该目录后，数据准备单元会把它们挪进
`downloads/`，`prepare_licensed_data` 直接复用，免去约 11 GiB 的重新下载。各档案在
下游流程中的作用详见[许可训练数据](licensed_training_data.md)；本页是原始下载文件的
逐一清单。

所有 MD5 均来自官方发布页，由 `microwakeword/audio/licensed_data.py` 在下载时强制校验。
SLR28 未公布校验值；其解压文件在读取时由 ZIP CRC 校验兜底。

## FSD50K v1.0（Zenodo record 4060432）

背景声增强与环境负样本的来源。只挑选 CC0/CC-BY 许可的 eval 片段；eval 音频是两段式
分卷 ZIP，读取前必须先用 `zip -s 0` 合并。

| 文件 | 大小 | MD5 | 内容与用途 |
| --- | --- | --- | --- |
| `FSD50K.eval_audio.zip` | 2.9 GiB | `6fa47636c3a3ad5c7dfeba99f2637982` | 分卷 ZIP 的第一部分，存放 eval 子集的原始 WAV。合并后按许可白名单提取为噪声池，并按上传者哈希约 80/10/10 重分成 `prepared/noise/{training,validation,testing}`。 |
| `FSD50K.eval_audio.z01` | 3.0 GiB | `3090670eaeecc013ca1ff84fe4442aeb` | 同一分卷 ZIP 的第二部分。单独无用；合并后生成 `FSD50K.eval_audio.joined.zip`（约 5.9 GiB，需要系统 `zip` 命令）。 |
| `FSD50K.metadata.zip` | 6.4 MiB | `b9ea0c829a411c1d42adb9da539ed237` | `eval_clips_info_FSD50K.json`：每个片段的许可 URL、上传者、标题。驱动许可白名单过滤、split 分配和全部署名记录；副本存入 `prepared/sources/`。 |
| `FSD50K.doc.zip` | 6.8 KiB | `3516162b82dc2945d3e7feba0904e800` | 数据集级 `LICENSE-DATASET`（CC BY 4.0 声明）和 `README.md`；解压到 `prepared/sources/`。数据集级条款在逐片段许可之外同样适用。 |

## OpenSLR SLR28 — RIRS_NOISES

混响增强（`RIR` 增强器，`impulse_paths`）。

| 文件 | 大小 | MD5 | 内容与用途 |
| --- | --- | --- | --- |
| `rirs_noises.zip` | 1.25 GiB | 未公布 | 房间脉冲响应与噪声集合。只挑选 `RIRS_NOISES/simulated_rirs/**/*.wav`（Apache 2.0）到 `prepared/rirs/`；真实 RIR、点源噪声和各向同性噪声被排除。两份 README 存档到 `prepared/sources/`。 |

## OpenSLR SLR12 — LibriSpeech

负特征集的语音负样本；沿用官方 train/dev/test split，保证留出音频不进入训练。

| 文件 | 大小 | MD5 | 内容与用途 |
| --- | --- | --- | --- |
| `train-clean-100.tar.gz` | 6.0 GiB | `2a93770f6d5c6c964bc36631d331a522` | 100 小时有声书朗读英文 FLAC。提取为 `prepared/speech/training/` 负样本。 |
| `dev-clean.tar.gz` | 322 MiB | `42e2234ba48799c1f50f24a7926300a1` | 验证集朗读者。`prepared/speech/validation/` 负样本，外加 `validation_ambient` 集（原始 ≥6 s 片段，不做重复拼接）。 |
| `test-clean.tar.gz` | 331 MiB | `32fa31d27d2e1cad72775fee3f4849a9` | 测试集朗读者。`prepared/speech/testing/` 负样本，外加 `testing_ambient` 集。 |

## 磁盘与运行提示

- 压缩包合计约 10.9 GiB。数据准备还要写合并后的 FSD50K ZIP、`prepared/` 下挑选出的
  音频以及特征文件，因此按[许可训练数据](licensed_training_data.md)所述预留约 40 GiB
  空闲空间。
- 请保留原始压缩包：`prepared/` 会对照它们做哈希校验，manifest 也记录了其 SHA-256。
  不要把原始 FSD50K 档案当作已清理子集分发——只有 `prepared/noise` 是逐文件核对过
  许可的，凭据是 `ATTRIBUTION.md` 和 `manifest.csv`。

## Colab：默认从 Google Drive 复制

`notebooks/ambient_testing_notebook.ipynb`（以及任何用 `microwakeword.audio.drive_data`
的入口）不再默认走公网下载，而是先挂载 Drive 并复制已有副本。`DRIVE_ROOT` 默认
`/content/drive/MyDrive/wake-word`，按下列条目有哪个复制哪个：

| Drive 里的路径 | 复制到（run 目录） | 说明 |
| --- | --- | --- |
| `trained_models` | `trained_models` | 训练产物，评测用的 `tflite_stream_state_internal_quant/stream_state_internal_quant.tflite` 就在里面 |
| `data_v1/downloads` | `data_v1/downloads` | 本文档列出的 8 个压缩包，复制后核对官方 MD5 |
| `data_v1/prepared` | `data_v1/prepared` | 已挑选好的音频；`prepare_licensed_data` 会逐文件校验 |
| `chime6_v1/prepared` | `chime6_v1/prepared` | 已解压的 CHiME-6 |
| `chime6_v1.zip` | `data_v1/downloads/chime6_v1.zip` | `ensure_chime6` 认这个位置 |
| `features_v1/negative/ambient*` | 同路径 | 已建好的 ambient 特征，可直接复用 |

复制是幂等的：大小一致的文件直接跳过，字节先写 `.part` 再改名，中断不会留下半截文件；
MD5 不符的压缩包会被删除并报错。只有 `data_v1/prepared` 不存在而压缩包齐全时才在本地解压。
