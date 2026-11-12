# 允许商用的 ambient 音频数据集调研

目的：为 microWakeWord 唤醒词项目扩充 **ambient 误唤醒评估集**（原始 ≥6 s 长音频、可流式拼接）与**训练背景噪声池**（次要），筛选**允许商用**的数据集，并逐一给出**一手来源**（官方主页 / OpenSLR / Zenodo / GitHub 官方仓库 / 许可证原文）的许可结论。

术语约定：

- **ambient 评估集**：原始 ≥6 s 连续音频（真实场景混响/多人对话/媒体人声），评估时按流式顺序拼接，不做增广改写。对应 `microwakeword/audio/licensed_data.py` 之外新增的 `negative/ambient` 评测面。
- **训练背景池**：进增广器与噪声负样本池的音频，允许切段/混音/加混响等派生处理。
- 核实日期：2026-11-11。所有「主源链接」均为调研当日实际访问并抓取原文的页面；无法访问的来源如实标注**「未能核实一手来源」**。

> 本文件只做调研与接入规划，未修改任何代码。现行数据管线与许可白名单见 `microwakeword/audio/licensed_data.py`，现状说明见 [data_pipeline.zh.md](data_pipeline.zh.md)、[pre_downloaded_archives.md](pre_downloaded_archives.md)。

## 1. 结论速览表

| 数据集 | 许可（主源结论） | 商用结论 | 规模 | 用途建议 | 核实状态 |
|---|---|---|---|---|---|
| MUSAN (SLR17) | 数据集整体 CC BY 4.0；内含 FMA 音乐/LibriVox 语音/Freesound、SoundBible 噪声，逐源许可待审 | 有条件可用（整体声明已核，逐源 LICENSE 内文未能核实） | ≈109 h（论文口径） | 训练背景池（noise/music 部分） | 整体许可一手已核；语料内 LICENSE 未能核实 |
| WHAM! Noise | CC BY-NC 4.0 | ❌ 不可商用（NC） | noise 包 17 GB→35 GB；48 kHz 全长版 68.1 GB→76 GB | 排除 | 一手已核 |
| DiPCo / CHiME-5/6 | CHiME-6 (SLR150)：CC BY-SA 4.0；DiPCo 原发布声称 CDLA-Permissive 1.0 | 可商用（用 SLR150 版本；注意 ShareAlike 义务）；CDLA 声称未能核实 | 压缩包约 120 GB（97/11/12）；录音约 50 h | **ambient 评估集首选**（真实家庭多人对话） | SLR150 一手已核；CDLA 声称未能核实一手来源 |
| Common Voice | 官方页面 JS 渲染，许可行未能抓取 | 待核（不可商用结论也不成立，只是未核实） | 每版本 ~20 h+ 英文朗读 | 训练语音负样本池（候选） | **未能核实一手来源** |
| The People's Speech | 官方卡片：CC-BY 与 CC-BY-SA **混合子集**（2.0/2.5/3.0/4.0 家族） | 可商用（注意 CC-BY-SA 子集的 ShareAlike 传染） | 30,000+ h | 训练语音/对话负样本池 | 一手已核（MLCommons 官方 HF 卡片） |
| GigaSpeech | 仓库 LICENSE 文件为 Apache 2.0；README 无数据集许可章节 | 音频重分发高风险（见 §5）；不建议商用产品数据源 | 10,000 h 转写 / 33,005 h 总量 | 仅限内部评估实验，谨慎 | 仓库 LICENSE 一手已核；数据集级许可声明缺失 |
| VoxPopuli | 数据 CC0（README License 表）；模型/代码 CC BY-NC 4.0（不用即可） | 可商用（仅数据） | 欧洲议会演讲，量级数十万小时（未标注）+数百小时转写 | 训练背景池（媒体演讲人声） | 一手已核 |
| ESC-50 | 整体 CC BY-NC 3.0；仅 ESC-10 子集 CC BY 3.0 | ❌ 整体不可商用（NC）；ESC-10 可用但片段仅 5 s | 2,000 clips × 5 s | 排除 | 一手已核（仓库 LICENSE 原文） |
| AMI Meeting Corpus | CC BY 4.0 | ✅ 可商用 | 多方会议语料（数百段会议） | **ambient 评估集推荐**（办公室多人会议） | 一手已核（官方 Licence 页） |
| OpenSLR SLR28 (RIRS_NOISES) | **Apache 2.0**；点源噪声子目录自带 LICENSE 声明选取自 Freesound Public Domain | 可商用 | RIR 6 万条；噪声 ≈9.6 h（独立源仅 37 个公共域源 + 多房间双通道录音） | 维持现状（RIR 增广）；噪声可作背景池小补充，不适合扩 ambient | 一手已核（含 zip 内 LICENSE/ANNOTATIONS ranged 读取） |
| FSD50K | 数据集级 CC BY 4.0 + 逐 clip CC 许可（eval：CC0 4,914 / CC-BY 3,489 / CC-BY-NC 1,425 / Sampling+ 403） | 现状已在用（仓库 CC0/CC-BY 白名单） | eval 10,231 clips | 维持现状 | 一手已核（Zenodo record 元数据） |
| VOiCES | CC BY 4.0（"free for commercial, academic, and government use"） | ✅ 可商用 | 真实房间重放语音 + 真实背景噪声 | **ambient 评估集候选** | 一手已核（IQT Labs 官网） |
| Freesound（CC0 筛选） | 声音逐条 CC 许可（CC0 可商用）；**但 API 免费使用仅限非商用** | 声音本身 CC0 可商用；API 商用需联系官方 | 海量（逐 clip） | 训练背景池扩充方案 | 一手已核（API Terms of Use） |
| DEMAND | CC BY 4.0 | ✅ 可商用（标注 attribution） | 16 通道环境噪声库，16 环境 × 3 段 × 5 min | 训练背景池推荐 | 一手已核（经 DataCite 出版方存档元数据；Zenodo 页面本机被拒） |
| AudioSet | 无数据集许可；音频托管于 YouTube | ❌ 排除 | 2,084,320 × 10 s clips | 排除 | 一手已核 |
| TAU/DCASE 城市场景系列 | Zenodo 页面本机被拒（403），DataCite 元数据仅 "Open Access" 无标准许可字段 | 未能核实一手来源 | 每版 ~40 h 场景录音 | 核实前不引入 | **未能核实一手来源** |
| UrbanSound8K | 预期 CC BY-NC；官方页面文字未能定位（weebly 页面动态渲染） | ❌ 排除（待补逐字核实） | 8,732 clips ≤4 s | 排除 | 许可原文待核 |
| VoxCeleb | 官方页"research purposes only"口径（页面文本未能定位） | ❌ 排除 | — | 排除 | 许可原文待核 |
| TED-LIUM 3 | LIUM 官网迁移后不可达；TED Talks 官方 Usage Policy：CC BY-NC-ND 4.0 | ❌ 排除（底流 NC-ND） | — | 排除 | TED 政策一手已核；TED-LIUM 3 页面未能核实一手来源 |
| CHiME-5（旧 EULA 口径） | **2024-01-01 起重发为 CC BY-SA 4.0**（"free for both academic and commercial purposes"） | ✅ 可商用（SA 义务），建议直接用 SLR150 CHiME-6 | 同 DiPCo 条目 | 并入 DiPCo/CHiME-6 条目 | 一手已核（官方 Datasets 页） |
| AISHELL-2 | 官方页为中文动态页，许可条款未能定位；免费版口径限非商用 | ❌ 排除（普通话 + 许可不明） | — | 排除 | 许可原文待核 |

## 2. 逐数据集详情

### 2.1 MUSAN（OpenSLR SLR17）

- 主源：<https://www.openslr.org/17/>。页面 License 字段：**Attribution 4.0 International (CC BY 4.0)**；Summary："A corpus of music, speech, and noise"。
- 构成：音乐来自 FMA（FMA 逐 track 许可不一）、语音来自 LibriVox、噪声来自 Freesound/SoundBible（论文 arXiv:1510.08484 口径）。SLR28 官方页面也说明其 point-source noises 直接"extracted from the MUSAN corpus"（<https://www.openslr.org/28/>），见 §2.9。
- **许可结论：数据集整体 CC BY 4.0（主源已核：https://www.openslr.org/17/）。语料内逐源 LICENSE 文件（含 FMA 逐 track 许可映射）本次未能核实**——`musan.tar.gz` 约 11 GB，本调研未下载解包。MUSAN 属"聚合 CC 声明"型语料，整体声明不能替代逐源审核，与 FSD50K 的教训一致（FSD50K eval 中约 18% 为 NC/Sampling+，见 §2.10）。
- 内容类型：音乐人声 / 朗读与有声书语音 / 短噪声事件。
- 规模：≈109 h（论文口径，未逐字核实）。
- 下载形态：单一 `musan.tar.gz`（OpenSLR 直链，无注册）。
- 用途建议：**仅训练背景池候选**。clip 多为 15–60 s，整体可用，但必须先解包审 `LICENSE` 文件，把 FMA 音乐里 NC/Sampling 之外不明的 track 全部落入 `excluded` 清单后才可接入。
- 接入归档要求（沿用 `licensed_data.py` lineage 模式）：manifest 逐文件记 `dataset="MUSAN (SLR17)"`、`upstream_path`、`source_url`、精确 `license_url`（MUSAN LICENSE 内文中的逐源许可）、`author`、`title`；`sources/` 存 MUSAN `LICENSE` 全文快照；不明许可 → `excluded_musan.csv`。

### 2.2 WHAM! Noise

- 主源：<https://wham.whisper.ai/>（官网 License 区块，`rel="license"` 指向 <http://creativecommons.org/licenses/by-nc/4.0/>）。
- 页面原文："This work is licensed under a **Creative Commons Attribution-NonCommercial 4.0 International License**."
- **许可结论：CC BY-NC 4.0（主源已核：https://wham.whisper.ai/）。仓库 `documentation/data_sources.md` 的 BY-NC 声称正确；不是 CC BY。**
- 内容类型：真实室内环境噪声（咖啡馆/餐厅/办公室等，记录自真实场景），已处理移除可懂语音；48 kHz 版为**未按 WSJ0 切分的全长度原始录音**，"wide distribution of clip durations"。
- 规模：`wham_noise.zip` 17 GB→35 GB；`WHAM!48kHz` 68.1 GB→76 GB（官方下载页数字）。
- 下载形态：S3 直链 zip/tar.gz，无注册。
- 用途建议：❌ **商用排除**（NC 禁止商用）。内容形态本来是 ambient 评估集的理想素材（真实室内 + 全长度），**若未来仅做内部研究评估（无商用分发）可再评估**；商用产品线不得引入。

### 2.3 DiPCo（Amazon Science）/ CHiME-5 / CHiME-6（SLR150）

- DiPCo 主源：<https://www.amazon.science/publications/dipco-dinner-party-corpus>。该页面为论文发布页，**不含许可声明**（已核，页面仅含论文 PDF 下载与 BibTeX）；原始数据分发（`dipco-alignment` S3 桶）**已下线**（本次访问返回 `NoSuchBucket`，已核）。**仓库 `documentation/data_sources.md` 声称的 CDLA-Permissive 1.0：未能核实一手来源（原分发渠道不存在了）**。
- 可商用替代主源：**OpenSLR SLR150 = CHiME-6 数据集**（即 DiPCo 同源录音的官方再发布）。页面：<https://www.openslr.org/150/>，License 字段：**Attribution-ShareAlike 4.0 (CC BY-SA 4.0 International)**；页面提供逐文件 `LICENSE.txt`（<https://openslr.trmal.net/resources/150/LICENSE.txt>，全文为 CC BY-SA 4.0 法律文本，要求引用 Barker et al., Interspeech 2018）。
- CHiME-5 官方 Datasets 页：<https://chimechallenge.org/datasets/chime5>，原文："**As of 1st Jan 2024, the CHiME-5 dataset has been re-issued under a standard CC BY-SA 4.0 license, and is free for both academic and commercial purposes.**"（并提示 CHiME-6 修复了原 CHiME-5 的阵列同步问题，新工作建议用 CHiME-6。）
- **许可结论：商用可用，许可 CC BY-SA 4.0（主源已核：https://www.openslr.org/150/ 与 https://chimechallenge.org/datasets/chime5）。注意 ShareAlike：若对外分发衍生数据需同许可；仅内部评估/训练无再分发义务问题。**
- 内容类型：真实家庭餐厅场景 **4 人自然多人对话**（Amazon 雇员志愿者），近讲单麦 + 5 组 7 麦远场阵列（Amazon Science 页面描述原文）。正好命中"真实室内多人对话"缺口。
- 规模：SLR150 提供 `CHiME6_train.tar.gz` 97 GB、`CHiME6_dev.tar.gz` 11 GB、`CHiME6_eval.tar.gz` 12 GB（官方页 2026-11-11 实时核对，下载合计约 120 GB；另有 transcriptions 2.4 MB、floorplans 1.4 MB）。音频内容：20 场真实家庭晚餐聚会（每场 ≥2 h，约 2–2.5 h），官方口径标注语音时长 train 40:33 h / dev 4:27 h / eval 5:12 h，总录音约 50 h；每场含多通道阵列（16 kHz WAV），取单通道远场即有约 20 × 2.4 h ≈ 48 h 可用素材。
- 下载形态：tar.gz 直链，无注册、无 EULA（旧 CHiME-5 的注册/EULA 流程已被 2024 再发布取代）。
- 用途建议：**ambient 评估集首选**。取远场阵列单通道（更接近智能音箱收音），原始 ≥6 s 片段直接流式拼接；同一 session 不跨 split。也适合训练背景池（可整段混入）。
- 接入归档要求：manifest 记 `dataset="CHiME-6 (OpenSLR SLR150)"`、`license_url="https://creativecommons.org/licenses/by-sa/4.0/"`、`source_url` 指向 SLR150 页、`author="CHiME consortium (Barker et al. 2018)"`；`sources/` 存 SLR150 `LICENSE.txt` 快照；ATTRIBUTION 按 LICENSE.txt 要求引用 Barker et al. 2018 论文。

### 2.4 Common Voice（Mozilla）

- 主源：<https://commonvoice.mozilla.org/en/datasets>。**该页面为 JS 应用壳，许可文字无法从抓取的 HTML 中定位**；官方仓库 README（<https://github.com/common-voice/common-voice>）只声明代码 MPL-2.0 与句子文本 CC0，未覆盖音频数据集；HuggingFace 官方镜像卡（`mozilla-foundation/common_voice_11_0`）为 gated 资源，无法匿名读取卡片全文。
- **许可结论：未能核实一手来源（常见口径为 CC0，但本次未能在主源页面逐字核实，不得引用）。**
- 内容类型：众包朗读短句、近讲单说话人。
- 用途建议：对 **ambient 评估集无价值**（无长环境音、非远场）；作为训练语音负样本池的候选，待许可核实后再评估。优先级低于 §2.5。

### 2.5 The People's Speech（MLCommons）

- 主源：MLCommons 官方 HuggingFace 卡片 <https://huggingface.co/datasets/MLCommons/peoples_speech>（组织官方账号；mlcommons.org 官网对数据中心返回 403 反爬页，未能读取）。卡片 Dataset Summary："**licensed for academic and commercial usage under CC-BY-SA and CC-BY 4.0**"；Licensing Information 章节："**We provide CC-BY and CC-BY-SA subsets of the dataset.**"；卡片 YAML license 列表含 cc-by-2.0/2.5/3.0/4.0 与 cc-by-sa-3.0/4.0。
- GitHub 官方仓库：<https://github.com/mlcommons/peoples-speech>（已核：仓库根无 LICENSE 文件，README 为数据流水线文档）。
- **许可结论：可商用，但许可为 CC-BY 与 CC-BY-SA 混合子集，并非统一 CC BY 4.0（主源已核：https://huggingface.co/datasets/MLCommons/peoples_speech）。仓库文档若按"整体 CC BY 4.0"理解则不准确。**
- 内容类型：英语语音大杂烩——政府/法律程序录音、公开演讲、口述历史、播客等（卡片 Social Impact 描述），多元说话人与声学条件。
- 规模：30,000+ 小时（卡片原文）。
- 下载形态：官方渠道登记 + HuggingFace/官方发布（官网 403，具体下载路径未在本机走通）。
- 用途建议：**训练语音/对话负样本池**（量大、真实多说话人）；不适合 ambient 评估集（来源异质、逐文件许可与归因依赖上游 manifest，拼接评估面的可审计性差）。
- 接入归档要求：接入前先拉取上游 per-file manifest，把许可字段并入 `licensed_data.py` 的 records 结构；**CC-BY-SA 子集若涉及对外分发衍生数据将触发 ShareAlike**，需在 manifest 中单独标记 SA 文件；`sources/` 存卡片与上游 LICENSE 声明快照。

### 2.6 GigaSpeech（SpeechColab）

- 主源：<https://github.com/SpeechColab/GigaSpeech>。仓库根 `LICENSE` 文件全文为 **Apache License 2.0**（已核原文）；但 **README 中没有任何数据集级许可章节**（已核：全文仅在未来工作里出现 "Making new datasets with permissive licenses"），即"数据集以 Apache 2.0 发布"并无 README 级声明。
- 内容构成（README 表格原文，已核）：Audiobook 11,982 h（转写 2,655 h）、Podcast 9,254 h（转写 3,498 h）、YouTube 11,768 h（转写 3,845 h）；合计 33,005 h，其中转写 10,000 h。Dev/Test 评估子集"randomly selected from the crawled Podcast and YouTube data; part of it was manually collected through other channels"。
- 下载形态：需填 Google Form 并按回复邮件指示获取原始发布（README Step 1/2，已核）。
- **许可结论：仓库代码与标注整理可按 Apache 2.0 使用（主源已核：https://github.com/SpeechColab/GigaSpeech 及其 LICENSE 文件）；数据集音频本身的再分发权利没有可靠的许可声明背书，商用风险见 §5。**
- 用途建议：**不纳入商用产品的数据源**。若仅做内部对照实验（含播客/YouTube 噪声背景的转录音频可做评估参考面），必须遵守原始获取渠道，且不得再分发。

### 2.7 VoxPopuli（Meta AI）

- 主源：<https://github.com/facebookresearch/voxpopuli> README License 表（原文）：
  - "**VoxPopuli Data | CC0** (see also European Parliament's legal notice for the raw data)"（EP legal notice：<https://www.europarl.europa.eu/legal-notice/en/>）
  - Pre-trained Models / Code | CC BY-NC 4.0（不使用模型与代码即不受影响）
- **许可结论：数据部分 CC0，可商用（主源已核：https://github.com/facebookresearch/voxpopuli）。模型与代码为 CC BY-NC 4.0，接入时只用其数据获取脚本产物、不引用其权重。**
- 内容类型：欧洲议会全体会议**公开演讲**（多语种，英语为主）；正式演讲腔，非对话，接近"广播电视/演讲媒体人声"。
- 规模：原始未标注音频数十万小时级 + 数百小时转写子集（README/论文口径，小时数未逐字核实）。
- 下载形态：仓库脚本从欧洲议会站点抓取 + 官方 fbaipublicfiles 元数据包；无注册。
- 用途建议：**训练背景池**（媒体人声背景）；ambient 评估集可少量纳入"演讲媒体"评估面，但与咖啡馆/家庭场景的分布差异大，权重不宜高。
- 接入归档要求：manifest 记 `dataset="VoxPopuli"`、`license_url="https://creativecommons.org/publicdomain/zero/1.0/"`、`author="European Parliament (via VoxPopuli, Meta AI)"`；`sources/` 存 README License 表与 EP legal notice 快照。

### 2.8 ESC-50

- 主源：<https://github.com/karolpiczak/ESC-50> 的 `LICENSE` 文件（原文）："The dataset as a whole is available under the terms of the Creative Commons **Attribution-NonCommercial license** (<http://creativecommons.org/licenses/by-nc/3.0/>). **The ESC-10 subset is licensed as a Creative Commons Attribution 3.0 Unported** dataset."；文件其余部分为逐 clip 的 Freesound 归属与许可标注。
- README 徽章同样标注 CC BY-NC（已核）。
- **许可结论：ESC-50 整体 CC BY-NC 3.0，❌ 不可商用；仅 ESC-10 子集为 CC BY 3.0 可商用（主源已核：https://github.com/karolpiczak/ESC-50/blob/master/LICENSE）。若仓库旧文档曾声称 ESC-50 为 CC BY 4.0，是错误口径。**
- 内容类型：环境声分类 clips，2000 条 × 5 s（ESC-10 为 400 条 × 5 s）。
- 用途建议：❌ 排除。整体 NC；ESC-10 虽可商用但单 clip 5 s 短于 ambient 评估 ≥6 s 要求，仅能进训练背景池且体量（≈0.5 h）过小，性价比低。

### 2.9 OpenSLR SLR28（RIRS_NOISES，仓库已在用）

- 主源：<https://www.openslr.org/28/>。页面 License 字段：**Apache 2.0**（已核原文）。
- 页面 About 原文：包含 RWCP sound scene database 与 REVERB challenge 的 real RIRs 和 isotropic noises、AIR 数据库、自研 simulated RIRs，以及"the point-source noises that **extracted from the MUSAN corpus**"。
- **许可结论：SLR28 为 Apache 2.0，可商用（主源已核：https://www.openslr.org/28/）。注意：这不是 CC BY 4.0——任务假设与其不符；仓库 `licensed_data.py` 中 RIR 记录的 `license_url=Apache-2.0` 与官方页面一致，无需修改。**
- 噪声部分内容量（本次对官方镜像 `rirs_noises.zip` 中央目录与关键成员做 ranged 读取，一手核实）：
  - `pointsource_noises/`：**843 个 wav，16 kHz 单声道 16-bit，合计 ≈5.9 h**；其 `LICENSE` 文件原文："All selected recordings were marked as in the **Public Domain** on Free Sound (url: https://www.freesound.org)."；`ANNOTATIONS` 列出 37 个 `noise-free-sound-*` 源——843 个文件是少量源在不同 RIR/位置下的**大量卷积变体**，独立噪声内容只有 37 源。
  - `real_rirs_isotropic_noises/`：417 个 wav（16 kHz，含双通道 binaural），按通道折算 **≈3.7 h** 真实房间噪声（RWCP/REVERB challenge 来源）。
  - `simulated_rirs/`：60,000 个 wav ≈ 2.2 GB（RIR 脉冲，非噪声）。
- 用途建议：**维持现状**（simulated RIR 增广，Apache 2.0）。噪声部分 ≈9.6 h 总量看似可用，但**独立内容少**（点源部分仅 37 个公共领域源、其余为变体；isotropic 部分为多房间双通道录音），更适合作为**训练背景池的小体量补充**（`real_rirs_isotropic_noises` 的真实房间噪声）而非 ambient 评估集主力；扩背景池优先用 §2.12 DEMAND 与 §2.13 Freesound CC0。

### 2.10 FSD50K（现状标注）

- 主源：Zenodo record 4060432 API 元数据 <https://zenodo.org/api/records/4060432>（本机可访问；网页版被 Zenodo 反爬拦截）。元数据 description 原文：全部 clip 为 Freesound 逐 clip CC 许可，"some of them requiring attribution ... and some forbidding further commercial reuse"；eval 集构成：**CC0 4,914 / CC-BY 3,489 / CC-BY-NC 1,425 / CC Sampling+ 403**；数据集整体另有 CC-BY（`LICENSE-DATASET` 文件）；逐 clip 许可映射在 `eval_clips_info_FSD50K.json`。
- **许可结论：现状不变——数据集级 CC BY 4.0 + 逐 clip 许可（主源已核：https://zenodo.org/records/4060432）。仓库 `allowed_fsd_license` 白名单（CC0 / CC-BY 3.0 / CC-BY 4.0，fail-closed）与上游元数据驱动的归因记录是正确做法，维持。**
- 备注：仓库 `ATTRIBUTION.md` 已记录"commercial users 请联系 FSD50K 维护者"的注意事项（源自官方页存档）。

### 2.11 VOiCES（IQT Labs / Lab41 + SRI）

- 主源：官方站点 <https://iqtlabs.github.io/voices/>（原 `voices18.github.io` 重定向至此）。页面原文："The corpus will be released as open source, **Creative Commons BY 4.0**, free for commercial, academic, and government use."；引用章节："VOiCES is publicly available released under **Creative Commons BY 4.0**, free for commercial, academic, and government use."（要求引用 Richey et al. 2018, arXiv:1804.05053）。
- **许可结论：CC BY 4.0，可商用（主源已核：https://iqtlabs.github.io/voices/）。仓库 `documentation/data_sources.md` 的 CC BY 4.0 声称正确。**
- 内容类型：真实房间（远场麦阵列 + binaural）中**重放 LibriSpeech 语音 + MUSAN 干扰源**，同时记录 HVAC、走廊脚步等真实背景与混响；前景语音许可继承 LibriSpeech（CC BY 4.0）。
- 规模：多房间/多距离/多信噪配置的长录音（官网未给总小时数，未核实）。
- 下载形态：官网提供官方下载渠道（S3/Box 链接），无注册。
- 用途建议：**ambient 评估集候选**——远场段落天然是"语音 + 真实背景"混合流，≥6 s 原始片段可流式拼接；与 DiPCo（多人对话）互补，覆盖"单人在真实房间里说话"的误唤醒面。
- 接入归档要求：manifest 记 `dataset="VOiCES"`、`license_url="https://creativecommons.org/licenses/by/4.0/"`、`source_url` 指向官网、`author="Lab41/IQT Labs, SRI International"`；ATTRIBUTION 引用 Richey et al. 2018。

### 2.12 DEMAND（Diverse Environments Multichannel Acoustic Noise Database）

- 主源：Zenodo record 1227121（<https://zenodo.org/records/1227121>）。**Zenodo 网站对本机网络返回 403（"unusual traffic"）**，许可经其**出版方存档的 DataCite 元数据**核实：`rightsList: [{rights: "Creative Commons Attribution 4.0", rightsUri: "https://creativecommons.org/licenses/by/4.0"}]`（API：<https://api.datacite.org/dois/10.5281/zenodo.1227121>）。描述原文："A database of 16-channel environmental noise recordings"。
- **许可结论：CC BY 4.0，可商用（主源结论已核，核实途径为 DataCite 出版方存档元数据；Zenodo 页面原文未能直接抓取，建议接入前人工打开一次页面复核）。**
- 内容类型：真实室内环境噪声（家庭、办公室等 16 类环境），16 通道阵列（含双耳/一阶 Ambisonics 等配置），每环境 3 个约 5 分钟片段（论文口径，未逐字核实）。
- 规模：每通道配置约 4 h（16 环境 × 3 × 5 min）。
- 下载形态：Zenodo 分卷 zip，无注册。
- 用途建议：**训练背景池推荐**（真实室内环境声，与 FSD50K 的事件型 clips 互补）；单通道可切段 ≥6 s 供背景混音。ambient 评估集也可少量取用，但注意其片段多为"无语音纯背景"，与真实误唤醒场景的分布差异需评估。
- 接入归档要求：manifest 记 `dataset="DEMAND"`、`license_url="https://creativecommons.org/licenses/by/4.0/"`、`source_url` 指向 Zenodo record、`author="University of Surrey (DEMAND contributors)"`；`sources/` 存 Zenodo 页面与许可快照（接入时人工下载）。

### 2.13 Freesound 按 CC0 筛选的 API 方案

- 主源：Freesound API Terms of Use <https://freesound.org/docs/api/terms_of_use.html>（其内文链接正式条款页 <https://freesound.org/help/tos_api/>）。原文（已核）："**You can use the Freesound API for free only for non-commercial purposes. To use the Freesound API for commercial purposes, please contact us using the Freesound contact form and we will talk about licensing options.**"
- 声音内容许可：Freesound 逐 clip 由上传者选择 CC 许可；API 检索可用 `filter=license:"Creative Commons 0"` 只取 CC0 声音（API sounds 文档 <https://freesound.org/docs/api/>）。
- **许可结论：CC0 声音内容本身可商用（主源：https://freesound.org/help/tos_api/ 的声音许可框架 + 逐 clip CC0 声明）；但 API 服务的免费使用仅限非商用，批量商用取数需先联系 Freesound（主源已核：https://freesound.org/docs/api/terms_of_use.html）。**
- 用途建议：**训练背景池扩充方案**：注册 API key（免费）→ `filter=license:"Creative Commons 0"` + `duration:[6 TO *]` 检索 → 下载原始音频 → 按 `licensed_data.py` lineage 记录逐 clip 归因。若商用条款谈不拢，可改为人工逐条从网页下载 CC0 声音（网页下载不受 API ToU 约束，但仍受逐 clip CC 许可约束）。
- 接入归档要求：manifest 记 `dataset="Freesound"`、`license_url` 为该 clip 的精确 CC0 URL、`author`（上传者）、`title`、`source_url=freesound.org/s/<id>/`；与 FSD50K 白名单模式完全一致。

## 3. 排除清单及原因（简要）

- **AudioSet**：官方页 <https://research.google.com/audioset/index.html> 原文——"2,084,320 human-labeled 10-second sound clips **drawn from YouTube videos**"；官方只分发元数据，音频依赖 YouTube 播放。批量下载音频违反 YouTube ToS 风险高、逐 clip 许可混杂 → 排除。
- **TAU/DCASE 城市场景系列**（TUT/TAU Urban Acoustic Scenes 2018/2019/2020/2022，Zenodo 1228142 / 2589280 / 3670167 / 6337421 等）：Zenodo 对本机 403 拒访；DataCite 存档元数据仅标 "Open Access"、**无标准许可字段**；DCASE 官网任务页不含数据许可声明 → **未能核实一手来源**，既不能断言 research-only 也不能断言可商用；核实前不引入。
- **UrbanSound8K**：预期 CC BY-NC（任务既有口径）；官方 weebly 页面抓取成功但许可文本未能定位（动态渲染），逐字核实待补 → 排除维持。
- **VoxCeleb**：官方页 <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/> 抓取成功但"research purposes only"原文未能在本次抓取文本中定位；该口径为社区共识 → 排除维持（不可再分发）。
- **TED-LIUM 3**：LIUM 官网改版后原页面不可达（`lium.univ-lemans.fr/ted-lium3/` 404，`projets-lium.univ-lemans.fr` 本网络 TLS 被拒）→ **页面未能核实一手来源**；但底流 TED Talks 官方 Usage Policy 明确 **CC BY-NC-ND 4.0**（<https://www.ted.com/about/our-organization/our-policies-and-terms-of-use/ted-talks-usage-policy>，原文明确 NC 与 ND 含义及商用需许可）→ 排除。
- **CHiME-5（旧口径）**：旧"注册 + EULA、仅限挑战使用"已被取代——官方页明确 2024-01-01 起为 CC BY-SA 4.0（见 §2.3）→ **不再排除**，按 SLR150 CHiME-6 接入。
- **ESC-50**：整体 CC BY-NC 3.0（§2.8）→ 排除。
- **WHAM!**：CC BY-NC 4.0（§2.2）→ 排除。
- **AISHELL-2**：官方页 <https://www.aishelltech.com/aishell_2> 为中文动态页，许可条款未能定位（免费版限非商用的口径待逐字核实）；且为普通话，语言不匹配 → 排除。
- **GigaSpeech**：不算硬排除，但商用产品数据源排除（§2.6 + §5）。

## 4. `documentation/data_sources.md` 三条声称的核实结果

1. **WHAM! = CC BY-NC 4.0：正确**（主源 https://wham.whisper.ai/ License 区块，NC 徽章 + 全文链接；不是 BY）。
2. **VOiCES = CC BY 4.0：正确**（主源 https://iqtlabs.github.io/voices/，两次明确 "free for commercial, academic, and government use"）。
3. **DiPCo = CDLA-Permissive 1.0：未能核实一手来源**——Amazon Science 页面无许可声明，原分发 S3 桶已下线。建议文档改指向 SLR150 CHiME-6 的 **CC BY-SA 4.0**（已核），并注明 DiPCo 原许可声明已不可考。

## 5. 商用风险提示（宽松声明 + YouTube/第三方来源）

- **GigaSpeech**：`LICENSE`（Apache 2.0）只覆盖 GitHub 仓库里的**代码与标注整理**；README 没有数据集级许可声明。音频来自爬取的 Podcast 与 YouTube 内容（官方表格），YouTube 音频受 YouTube 服务条款约束（禁止未授权下载与再分发），播客音频版权归节目方；Test 子集还含"manually collected through other channels"的来源不明片段。**把 GigaSpeech 音频再分发（含打包进训练/评估数据集对外发布）在商用语境下风险不可接受**；内部使用也应只走官方申请渠道并保留凭证。
- **VoxPopuli**：数据 CC0 声明明确，但底流是欧洲议会视听材料；README 自己也要求同时查看 **European Parliament legal notice**。CC0 + 官方开放重用政策使风险较低，但接入归档应同时引用 EP legal notice，避免仅凭 CC0 单点声明。
- **The People's Speech**：**CC-BY 与 CC-BY-SA 子集混合**。CC-BY 部分只需归因；**CC-BY-SA 部分的衍生数据集若对外分发必须整体按 SA 传播**，对商用产品的数据集发布策略有传染性影响；且其来源含政府/法律程序录音，归因依赖上游 per-file manifest 的完整性。
- **聚合型"整体 CC BY"语料（MUSAN、FSD50K 模式）**：数据集级声明 ≠ 逐源 cleared。FSD50K eval 中约 18% clips 为 NC/Sampling+（已核上游元数据），MUSAN 内含逐 track 许可不一的 FMA 音乐。**仓库现有的 fail-closed 白名单 + 逐文件 manifest/ATTRIBUTION 是必要防线，任何新数据源都应复制该模式。**
- **Freesound API**：声音 CC0 可商用 ≠ API 服务可免费商用；批量取数前先解决 API ToU（联系官方）或改用页面人工下载。

## 6. 待核清单（本次未能核实一手来源的项目）

1. **MUSAN 语料内 `LICENSE` 逐源许可**（OpenSLR 页面整体声明已核；本次尝试对 `musan.tar.gz` 做部分下载流式解包未能在时限内到达 `LICENSE` 成员，保持待核）。
2. **Common Voice 音频数据集的 CC0 声明**（官网页面 JS 渲染无法抓取许可行；站点 JS 主包仅为加载器；HF 官方卡 gated）。
3. **UrbanSound8K / VoxCeleb / AISHELL-2 官方页许可原文**（页面可达但许可文本未能定位，动态渲染）。
4. **TAU/DCASE 系列 Zenodo 许可**（Zenodo 拒绝本机网络；DataCite 存档元数据无标准许可字段）。
5. **DiPCo 原发布 CDLA-Permissive 1.0 声明**（原分发渠道已下线；已用 SLR150 CC BY-SA 4.0 替代核实）。
6. DEMAND 的 Zenodo 页面原文复核（许可已经 DataCite 出版方存档元数据核实为 CC BY 4.0，建议接入前人工打开页面确认一次）。

## 7. 接入建议（优先级与归档模式）

**优先级 1 —— ambient 评估集扩充（原始 ≥6 s、可流式拼接）：**

| 来源 | 内容类型 | 许可 | 备注 |
|---|---|---|---|
| CHiME-6 (SLR150) | 家庭 4 人晚餐多人对话（远场阵列） | CC BY-SA 4.0 | 首选；取远场单通道；同 session 不跨 split；SA 义务只影响对外分发 |
| AMI Meeting Corpus | 办公室多人会议（近讲 + 远场阵列） | CC BY 4.0 | 与现有 LibriSpeech CC BY 4.0 白名单一致，接入摩擦最小 |
| VOiCES | 真实房间远场语音 + HVAC 等真实背景 | CC BY 4.0 | 覆盖"单人在真实房间说话"面；引用 Richey et al. 2018 |

**优先级 2 —— 训练背景池扩充：**

| 来源 | 内容类型 | 许可 | 备注 |
|---|---|---|---|
| DEMAND | 16 类真实室内环境噪声 | CC BY 4.0 | 与 FSD50K 事件型 clips 互补；接入前人工复核 Zenodo 页面许可 |
| Freesound CC0 API 筛选 | 任意环境声（≥6 s 过滤） | CC0（逐 clip） | API 免费使用限非商用，商用批量取数先联系官方 |
| VoxPopuli | 演讲/媒体人声 | CC0（数据） | 补"电视/演讲媒体"分布；不用其 NC 模型 |
| MUSAN（待核后） | 音乐/噪声 | 整体 CC BY 4.0 + 逐源待审 | 仅在解包审完 `LICENSE` 并排除不明 track 后接入 |

**归档要求（参照 `microwakeword/audio/licensed_data.py` 现行 lineage 模式）：**

1. 逐文件 manifest（`manifest.json`/`manifest.csv`）字段保持：`path`、`sha256`、`dataset`、`version`、`upstream_path`、`source_url`、`license_url`（精确到许可证版本 URL）、`author`、`title`、`split`、`modifications`。
2. `sources/` 存原始许可文本快照：Creative Commons 许可用 cc-legal-tools-data 的 `legalcode.txt` 现行做法；非 CC 许可（Apache 2.0、CDLA、站点条款）存官方页面/文件原文快照，记录下载 URL 与 sha256。
3. `ATTRIBUTION.md` 汇总署名与引用义务：CHiME-6 → Barker et al. 2018；VOiCES → Richey et al. 2018；AMI → AMI consortium（CC BY 4.0）；DEMAND → Surrey/zenodo DOI。
4. fail-closed 白名单扩展：CC BY-SA 数据单独打标（`sharealike: true`），避免无意进入对外分发的产物；许可不在白名单（NC/ND/unknown/无 URL）一律进 `excluded_*.csv`。
5. split 防泄漏：沿用"同源不跨 split"原则——FSD50K 按 uploader 分桶，新增数据按 **session/会议 ID** 分桶（AMI/CHiME 同一会议的所有通道锁在同一 split）。
6. 下载形态记录进 `manifest.json` 的 `archives` 列表（现行做法）：URL、文件名、sha256、发布方 MD5（SLR150 未公布校验值，以 ZIP/tar CRC 与 sha256 兜底）。
