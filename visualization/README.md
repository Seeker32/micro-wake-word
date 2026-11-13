# MixedNet 架构可视化（2D + 3D）

把 [`microwakeword/mixednet.py`](../microwakeword/mixednet.py) 定义的 MixedNet 结构做成两个可交互页面：

- **2D 结构图** `index.html`：逐层展开张量形状、参数量、MixConv 分支与 ring buffer，可在「训练（非流式）」和「流式推理」之间切换，参数可实时编辑重算。
- **3D 架构图** `3d.html`：每层一块“特征板”叠成塔，支持拖拽旋转、滚轮缩放、悬停/点击查看层信息。

两个页面共用 `viz-core.js`（结构推导）和 `theme.css`（主题令牌），所以数字和配色不会各算各的。

页面不加载 Keras / TFLite，也不发起任何网络请求：所有数值都由源码里的结构公式在前端重新推导，可离线查看。

## 打开方式

```bash
open visualization/index.html   # 2D 结构图
open visualization/3d.html      # 3D 架构图
```

也可以直接用浏览器打开文件（`file://` 亦可）。请保持 `index.html`、`3d.html`、`viz-core.js`、`theme.css` 在同一目录。

自检（仅 2D 页面）：

```text
file:///.../visualization/index.html?selftest=1
```

带 `?selftest=1` 时，页面顶部渲染一张 16 项断言表，逐项对照
`train-result/hi_circle_v1/trained_models/hi_circle_licensed_inputs_clean100_v1_hey_hi/model_summary.txt`
与同目录下 `stream_state_internal/model_summary.txt` 的实测值。

## 目录结构

```text
visualization/
├── index.html    # 2D 结构图：HTML + 内联 CSS + 内联 JS
├── 3d.html       # 3D 架构图：HTML + 内联 CSS + 内联 JS
├── viz-core.js   # 共用核心：预设、源码公式、buildGraph、自检（无 DOM 依赖）
├── theme.css     # 共用主题令牌：暗色默认，html[data-theme="light"] 覆盖
└── README.md     # 本文件
```

- 不新增 npm 依赖、不需要 `pip install`、没有构建步骤；脚本用普通 `<script src>` 引入，`file://` 下可直接工作。
- 改动只涉及本目录，不会影响 `microwakeword/` 的训练与推理代码。
- 建议用 Chrome / Safari / Firefox 的较新版本（ES2020、CSS 3D 变换）。

## 主题（暗色 / 明亮）

两个页面右上角都有「暗色 / 明亮」分段按钮，默认暗色：

- 选择写入 `localStorage`（键名 `mww-viz-theme`）与 URL 的 `theme` 参数：`?theme=light` / `?theme=dark`，两个页面共享。
- `<head>` 里的内联脚本会在首屏渲染前应用主题，避免刷新时的白/黑闪烁；`localStorage` 不可用（如部分浏览器的 `file://` 场景）时自动降级为暗色。
- 两套主题共用 `theme.css` 里的 CSS 变量；明亮模式采用白底蓝图风，并把强调色整体压暗以保证对比度（例如琥珀 `#9a5b00`、青绿 `#0b7d75`）。

## 2D 页面（`index.html`）

| 区域 | 内容 |
| --- | --- |
| 顶栏 | 权重参数、可训练 / 不可训练参数、fp32 体积；流式视角下改为显示运行时状态参数 |
| 左栏 · 预设 | 「当前训练配置」（真实训练结果）、「轻量示例」、复制分享链接、重置 |
| 左栏 · 参数 | `spectrogram_length`、`batch_size`、`clip_duration_ms`、`window_step_ms`、`first_conv_*`、`stride`、四个编码器列表、`pooled` / `max_pool` / `spatial_attention` |
| 主区 | 纵向结构图：Input → ExpandDims → 首层卷积 + ReLU → N 个 MixConv 块 → 上下文缓冲 / 注意力 → 池化 → Flatten → Dense |
| 详情抽屉 | 点击任意节点或 MixConv 分支，查看作用说明、形状公式、参数推导、源码片段与流式行为 |
| 对照表 | 仅「当前训练配置」预设可用，逐层对比 `model_summary.txt` |

键盘与可访问性：视角与预设按钮可用 Tab 聚焦、回车触发；`Esc` 关闭详情抽屉。顶栏「3D 视图」跳转到三维页面。

## 3D 页面（`3d.html`）

每层是一块“特征板”，用几何尺寸表达张量：

| 维度 | 编码 | 备注 |
| --- | --- | --- |
| 板宽 | `100 + 250 × √(通道数 / 最大通道数)` | 平方根压缩，避免 Flatten 的 1088 维压扁其它层 |
| 板深 | `60 + 115 × √(时间帧 / 最大帧数)` | 反映时间上下文长度 |
| 板厚 | `6 + 10 × (参数量 / 最大参数量)^0.4` | 卷积/BN 越重越厚，ReLU、缓冲几乎为薄片 |

尺寸是**可读性缩放**，不是等比例模型体积，页面里也不声称是。

交互：

- 拖拽旋转（左右改变偏航、上下改变俯仰，俯仰限制在 −86°…−8°）
- 滚轮缩放、双击重置视角、方向键微调、`+` / `-` 缩放（先 `Tab` 聚焦画面）
- 点击任意板面或左侧「图层」列表选中一层，右侧抽屉显示形状、参数量、ring buffer、推导与源码
- 「展开层距」拉大层间隔，便于看清被压住的层
- 「显示标签」控制顶层标签；子层标签在悬停或选中时自动出现；同侧标签会纵向避让，避免叠字
- 「自动旋转」持续绕 Y 轴旋转，再次点击停止

默认展示训练（非流式）结构；流式 ring buffer 的帧数和状态参数写在每层详情里，完整的流式视角请切到 2D 页面。预设与主题会写进 URL（`?preset=light&theme=light`）。

## 数据来源

「当前训练配置」预设取自：

```text
train-result/hi_circle_v1/trained_models/hi_circle_licensed_inputs_clean100_v1_hey_hi/training_config.yaml
  first_conv_filters: 32      first_conv_kernel_size: 5      stride: 3
  pointwise_filters: 64,64,64,64
  repeat_in_block: 1,1,1,1
  mixconv_kernel_sizes: [5], [7,11], [9,15], [23]
  residual_connection: 0,0,0,0
  pooled: 0   max_pool: 0   spatial_attention: 0
```

输入帧数 204 与 `model_summary.txt` 的 `(128, 204, 40)` 一致。页面同时展示
`load_config()` 的推导结果：

```text
1 + ⌊(clip_duration_ms − 30) / (stride × window_step_ms)⌋ + spectrogram_slices_dropped = 204
1 + ⌊(1500 − 30) / (3 × 10)⌋ + 154 = 204
```

`spectrogram_slices_dropped = (first_conv_kernel_size − 1) + Σ repeat × (max(kernel) − 1) × stride`
= `4 + 3 × (4 + 10 + 14 + 22)` = 154。

## 计算规则（与源码保持一致）

时间维 `T_out = ⌊(T_in − k) / s⌋ + 1`，频率维固定 40。

| 节点 | 参数与形状 |
| --- | --- |
| 首层 Conv2D | `k × 40 × filters`，`use_bias=False`，`valid` 填充 |
| ChannelSplit | `C // n`，余数并入第 0 组（`_split_channels`） |
| DepthwiseConv2D | `k × C_in + C_in`（Keras 默认带 bias，实测 192 / 256 / 384 / 320 / 512 / 1536） |
| 1×1 Conv2D | `C_in × filters`，`use_bias=False` |
| BatchNormalization | `4C`（`2C` 可训练 + `2C` 不可训练） |
| 残差支路 | `C_in × filters + 4 × filters` |
| SpatialAttention | `4 × 2 = 8`，形状不变 |
| 池化 | 0 参数，`T → 1` |
| Flatten | 展平 `T × 1 × C` |
| Dense | `展平维度 + 1`，`sigmoid` |

「当前训练配置」合计 **26,049** 参数（可训练 25,537 / 不可训练 512），与 `model_summary.txt` 完全一致。

流式视角的 ring buffer：

| 位置 | 帧数 | 状态参数 |
| --- | --- | --- |
| 首层卷积 | `max(k − stride, 0) = 2` | `2 × 40 = 80` |
| 块 1（k=5） | 4 | `4 × 32 = 128` |
| 块 2（k=11） | 10 | `10 × 64 = 640` |
| 块 3（k=15） | 14 | `14 × 64 = 896` |
| 块 4（k=23） | 22 | `22 × 64 = 1408` |
| 上下文 Identity 缓冲 | `T − 1 = 16` | `16 × 64 = 1024` |
| 合计 | | **4,176** |

流式模型以 `batch_size=1` 编译，2D 页面切到流式视角后形状首维显示为 1。状态参数单列「运行时状态」，不计入权重参数量。

## 与源码的映射

| 页面节点 | 源码位置 |
| --- | --- |
| 输入 / expand_dims / 首层卷积 | `mixednet.model()` |
| MixConv 分支与 `StridedKeep` / `StridedDrop` | `mixednet.MixConv.__call__`、`layers/strided_drop.py` |
| 1×1 卷积、BN、残差、ReLU | `mixednet.model()` 的 encoder 循环 |
| 上下文缓冲 | `stream.Stream(cell=Identity(), ring_buffer_size_in_time_dim=T−1)` |
| 空间注意力 | `mixednet.SpatialAttention.__call__` |
| 池化 / Flatten / Dense | `mixednet.model()` 结尾 |
| 丢弃帧数 | `mixednet.spectrogram_slices_dropped()` |
| 输入帧数推导 | `model_train_eval.load_config()` |

## 已知坑

1. **`mixednet.py` 的 argparse 默认值自相矛盾**：`residual_connection` 默认 `"0,0,0,0,0"`（5 项），而 `pointwise_filters` 等默认是 4 项，直接跑默认值会在
   `if len(pointwise_filters) != len(list): raise ValueError` 处报错。页面因此没有提供「源码默认值」预设，并会主动做同样的等长校验。
2. **深度卷积带 bias**：`MixConv` 里的 `DepthwiseConv2D` 没有设置 `use_bias=False`，参数量是 `k×C + C` 而不是 `k×C`。这也是页面对得上 192 / 256 / 384 等数值的关键。
3. **BN 参数属于「总量 4C」**：Keras 的 `BatchNormalization` 一半可训练、一半不可训练，页面分别计入两个总数。
4. **流式状态不是权重**：`stream.Stream` 的 ring buffer 在 `.tflite` 里会变成状态变量，页面单独展示，避免与权重参数量混淆。
5. **2D 对照表只对内置预设生效**：自定义配置下对照表替换为一句说明，避免误导。
6. **3D 尺寸是示意**：板宽/板深用平方根压缩、板厚用 0.4 次幂，只表达相对大小；绝对尺寸请以 2D 页面和 `model_summary.txt` 为准。
7. **CSS 3D 的遮挡是近似的**：相邻大板在投影上重叠时，浏览器按元素深度排序，个别角度可能出现几像素的遮挡误差；旋转一下即可确认层的真实顺序。

## 修改指引

- **新增预设**：在 `viz-core.js` 的 `PRESETS` 里加一项 `{ label, cfg }`，并在两个页面的左栏各加一个 `data-preset="<key>"` 按钮。
- **改公式**：只改 `viz-core.js` 的 `buildGraph()` / `slicesDropped()` / `derivedFrames()`；两个页面的渲染层不需要动。
- **加自检项**：在 `viz-core.js` 的 `runSelfTest()` 里追加 `add(label, actual, expected)`。
- **调 3D 外观**：`3d.html` 里的 `PALETTE`（每类层的颜色）、`renderFaces`（板面尺寸/厚度）、`layout`（层距）、`syncLabels`（标签避让）。

## 验证记录

- 2D `?selftest=1`：16 / 16 通过；用 Node 直接执行 `viz-core.js` + 页面脚本，同样 16 / 16。
- 2D 手工检查：训练 / 流式切换、详情抽屉、残差开关（26,049 → 41,409，非可训练 512 → 1,024）、`pooled`（26,049 → 25,025）、`spatial_attention`（+8）、关闭首层卷积、列表长度不一致时的错误提示，以及 500px 窄屏无横向溢出。
- 3D 手工检查：23 层、26,049 参数；轻量预设 19 层、13,057 参数；拖拽/缩放/双击重置、展开层距、隐藏标签、点击选中（抽屉与图层行同步高亮）、悬停子层时标签出现，均在 Chrome 无头截图中确认。
- 主题：两个页面的暗色 / 明亮切换、`?theme=light` 直开、`localStorage` 记忆、分享链接带着主题往返。
