/* microWakeWord 可视化 · 模型结构核心
 * 无 DOM 依赖：预设、源码公式、形状/参数量推导与自检。
 * index.html（2D）与 3d.html（3D）共用，避免两份实现漂移。 */
(function (global) {
  "use strict";
  /* ================= 常量 ================= */

  var PRESETS = {
    trained: {
      label: '当前训练配置',
      cfg: {
        inputFrames: 204,
        batch: 128,
        clipMs: 1500,
        windowStepMs: 10,
        firstConvFilters: 32,
        firstConvKernel: 5,
        stride: 3,
        pointwise: [64, 64, 64, 64],
        repeat: [1, 1, 1, 1],
        kernels: [[5], [7, 11], [9, 15], [23]],
        residual: [0, 0, 0, 0],
        pooled: 0,
        maxPool: 0,
        attention: 0
      }
    },
    light: {
      label: '轻量示例',
      cfg: {
        inputFrames: 176,
        batch: 128,
        clipMs: 1500,
        windowStepMs: 10,
        firstConvFilters: 32,
        firstConvKernel: 3,
        stride: 1,
        pointwise: [32, 32, 32],
        repeat: [1, 1, 1],
        kernels: [[5], [9], [15]],
        residual: [0, 0, 0],
        pooled: 0,
        maxPool: 0,
        attention: 0
      }
    }
  };

  var EXPECTED = {
    weights: 26049,
    trainable: 25537,
    nonTrainable: 512,
    conv0Params: 6400,
    conv0T: 67,
    depthwise: '192,256,384,320,512,1536',
    pointwise: '2048,4096,4096,4096',
    bn: '256,256,256,256',
    flatDim: 1088,
    denseParams: 1089,
    postT: 17,
    buffers: '2,4,10,14,22,16',
    state: 4176,
    derived: 204
  };

  var KIND = {
    io: { label: 'input', color: 'var(--muted)' },
    reshape: { label: 'reshape', color: 'var(--muted)' },
    conv: { label: 'conv', color: 'var(--amber)' },
    act: { label: 'relu', color: 'var(--green)' },
    mixconv: { label: 'mixconv', color: 'var(--teal)' },
    branch: { label: 'branch', color: 'var(--teal)' },
    bn: { label: 'bn', color: 'var(--blue)' },
    residual: { label: 'residual', color: 'var(--violet)' },
    add: { label: 'add', color: 'var(--violet)' },
    buffer: { label: 'buffer', color: 'var(--buf)' },
    attention: { label: 'attention', color: 'var(--violet)' },
    pool: { label: 'pool', color: 'var(--blue)' },
    flatten: { label: 'flatten', color: 'var(--muted)' },
    dense: { label: 'dense', color: 'var(--red)' },
    block: { label: 'block', color: 'var(--teal)' }
  };

  var CODE = {
    input: 'input_audio = tf.keras.layers.Input(shape=shape, batch_size=batch_size)',
    expand: 'net = tf.keras.ops.expand_dims(net, axis=2)  # [batch, time, 1, feature]',
    conv0: 'net = stream.Stream(\n  cell=tf.keras.layers.Conv2D(\n    flags.first_conv_filters, (flags.first_conv_kernel_size, 1),\n    strides=(flags.stride, 1), padding="valid", use_bias=False),\n  use_one_step=False, pad_time_dim=None, pad_freq_dim="valid")(net)\nnet = tf.keras.layers.Activation("relu")(net)',
    mixconv: 'net = stream.Stream(cell=tf.keras.layers.Identity(),\n  ring_buffer_size_in_time_dim=max(kernel_size) - 1,\n  use_one_step=False)(inputs)\n\nif len(kernel_sizes) == 1:\n    return tf.keras.layers.DepthwiseConv2D((ks, 1), padding="valid")(net)\n\nsplits = _split_channels(filters, len(kernel_sizes))\nx_splits = ChannelSplit(splits, axis=-1)(net)\nfor x, ks in zip(x_splits, kernel_sizes):\n    fit = strided_drop.StridedKeep(ks)(x)\n    x_outputs.append(DepthwiseConv2D((ks, 1), padding="valid")(fit))\n# StridedDrop 对齐到最短分支\nnet = tf.keras.layers.concatenate(x_outputs, axis=-1)',
    pointwise: 'net = tf.keras.layers.Conv2D(filters, kernel_size=1, use_bias=False, padding="same")(net)',
    bn: 'net = tf.keras.layers.BatchNormalization()(net)',
    relu: 'net = tf.keras.layers.Activation("relu")(net)',
    residual: 'residual = tf.keras.layers.Conv2D(filters, kernel_size=1, use_bias=False, padding="same")(net)\nresidual = tf.keras.layers.BatchNormalization()(residual)\n...\nresidual = strided_drop.StridedDrop(residual.shape[1] - net.shape[1])(residual)\nnet = net + residual',
    attention: 'tranposed = tf.keras.ops.transpose(inputs, axes=[0, 1, 3, 2])\nchannel_avg = AveragePooling2D(pool_size=(1, tranposed.shape[2]))(tranposed)\nchannel_max = MaxPooling2D(pool_size=(1, tranposed.shape[2]))(tranposed)\npooled = Concatenate(axis=-1)([channel_avg, channel_max])\nattention = stream.Stream(\n  cell=Conv2D(1, (4, 1), use_bias=False, activation="sigmoid"))(pooled)\nreturn net * attention',
    buffer: 'net = stream.Stream(\n  cell=tf.keras.layers.Identity(),\n  ring_buffer_size_in_time_dim=net.shape[1] - 1,\n  use_one_step=False)(net)',
    poolAvg: 'net = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)',
    poolMax: 'net = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)',
    flatten: 'net = tf.keras.layers.Flatten()(net)',
    dense: 'net = tf.keras.layers.Dense(1, activation="sigmoid")(net)'
  };

  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function fmt(n) { return Number(n).toLocaleString('en-US'); }

  function splitChannels(total, groups) {
    var base = Math.floor(total / groups);
    var out = new Array(groups).fill(base);
    out[0] += total - base * groups;
    return out;
  }

  function parseNumList(text) {
    var cleaned = String(text).replace(/[[\]]/g, ' ');
    var parts = cleaned.split(/[,\s]+/).filter(Boolean);
    return parts.map(function (p) { return Number(p); });
  }

  function parseKernelList(text) {
    var raw = String(text);
    var groups = raw.match(/\[[^\]]*\]/g);
    if (groups && groups.length) return groups.map(parseNumList);
    return raw.split(';').map(parseNumList).filter(function (a) { return a.length; });
  }

  function isInt(v, min) { return Number.isInteger(v) && v >= min; }

  /* ================= 源码公式 ================= */

  // mixednet.spectrogram_slices_dropped(flags)
  function slicesDropped(cfg) {
    var dropped = cfg.firstConvFilters > 0 ? cfg.firstConvKernel - 1 : 0;
    var n = Math.min(cfg.repeat.length, cfg.kernels.length);
    for (var i = 0; i < n; i++) {
      dropped += cfg.repeat[i] * (Math.max.apply(null, cfg.kernels[i]) - 1) * cfg.stride;
    }
    return dropped;
  }

  // model_train_eval.load_config(): spectrogram_length_final_layer + slices dropped
  function derivedFrames(cfg) {
    var sampleRate = 16000;
    var desired = sampleRate * cfg.clipMs / 1000;
    var windowSize = sampleRate * 30 / 1000;
    var step = cfg.stride * sampleRate * cfg.windowStepMs / 1000;
    var finalLayer = desired - windowSize < 0
      ? 0
      : 1 + Math.floor((desired - windowSize) / step);
    return finalLayer + slicesDropped(cfg);
  }

  function validate(cfg) {
    var errs = [];
    if (!isInt(cfg.batch, 1)) errs.push('batch_size 必须是正整数');
    if (!isInt(cfg.inputFrames, 1)) errs.push('输入帧数必须是正整数');
    if (!isInt(cfg.clipMs, 31)) errs.push('clip_duration_ms 必须大于 30ms');
    if (!isInt(cfg.windowStepMs, 1)) errs.push('window_step_ms 必须是正整数');
    if (!isInt(cfg.stride, 1)) errs.push('stride 必须是正整数');
    if (!isInt(cfg.firstConvKernel, 1)) errs.push('first_conv_kernel_size 必须是正整数');
    if (!isInt(cfg.firstConvFilters, 0)) errs.push('first_conv_filters 必须是 ≥ 0 的整数');

    var numeric = [
      ['pointwise_filters', cfg.pointwise, 1],
      ['repeat_in_block', cfg.repeat, 1],
      ['residual_connection', cfg.residual, 0]
    ];
    numeric.forEach(function (row) {
      var name = row[0], arr = row[1], min = row[2];
      if (!arr.length) errs.push(name + ' 不能为空');
      else if (arr.some(function (v) { return !isInt(v, min); })) {
        errs.push(name + ' 只能包含 ≥ ' + min + ' 的整数');
      }
    });

    if (!cfg.kernels.length) errs.push('mixconv_kernel_sizes 不能为空');
    cfg.kernels.forEach(function (ks, i) {
      if (!ks.length) errs.push('mixconv_kernel_sizes 第 ' + (i + 1) + ' 组为空');
      else if (ks.some(function (k) { return !isInt(k, 1); })) {
        errs.push('mixconv_kernel_sizes 第 ' + (i + 1) + ' 组必须是正整数');
      }
    });

    var lens = [cfg.pointwise.length, cfg.repeat.length, cfg.kernels.length, cfg.residual.length];
    if (new Set(lens).size > 1) {
      errs.push('pointwise_filters / repeat_in_block / mixconv_kernel_sizes / residual_connection 长度必须一致（mixednet.py 会抛 ValueError）');
    }
    return errs;
  }

  /* ================= 构图 ================= */

  function buildGraph(cfg) {
    var batch = cfg.batch;
    var shapeText = function (t, c) { return '(' + batch + ', ' + t + ', 1, ' + c + ')'; };
    var seq = [];
    var index = new Map();
    var stats = { weights: 0, trainable: 0, nonTrainable: 0, state: 0, buffers: [] };
    var errors = [];
    var warnings = [];
    var T = cfg.inputFrames;
    var C = 40;
    var stopped = false;

    var addWeights = function (p) { stats.weights += p; stats.trainable += p; };
    var addBN = function (c) {
      stats.weights += 4 * c;
      stats.trainable += 2 * c;
      stats.nonTrainable += 2 * c;
    };
    var addBuffer = function (frames, channels, label) {
      if (!(frames > 0)) return null;
      var params = frames * channels;
      stats.state += params;
      var buf = { frames: frames, channels: channels, params: params, label: label };
      stats.buffers.push(buf);
      return buf;
    };
    var push = function (node) { seq.push(node); index.set(node.id, node); return node; };
    var pushBlock = function (block) {
      seq.push(block);
      index.set(block.id, block);
      block.children.forEach(function (c) { index.set(c.id, c); });
    };

    var actNode = function (id, t, c) {
      return {
        id: id, kind: 'act',
        title: 'ReLU 激活', type: 'Activation("relu")',
        timeIn: t, timeOut: t,
        shapeIn: { train: shapeText(t, c), stream: shapeText(1, c) },
        shapeOut: { train: shapeText(t, c), stream: shapeText(1, c) },
        params: 0,
        desc: '逐元素非线性，所有卷积之后都会接一个 ReLU。',
        formula: 'y = max(0, x)',
        code: CODE.relu,
        streamNote: '流式时形状 (1, 1, 1, ' + c + ')。'
      };
    };

    push({
      id: 'input', kind: 'io',
      title: '音频特征输入', type: 'InputLayer',
      timeIn: null, timeOut: T,
      shapeIn: { train: '—', stream: '—' },
      shapeOut: {
        train: '(' + batch + ', ' + T + ', 40)',
        stream: '(' + batch + ', ' + cfg.stride + ', 40)'
      },
      params: 0,
      desc: '每 10ms 一帧、40 维的 micro_speech 预处理特征。训练用完整上下文，流式每次只喂 stride 帧（预设 3 帧 = 30ms）。',
      formula: 'modes.get_input_data_shape()：训练 → (spectrogram_length, 40)；流式 → (stride, 40)',
      code: CODE.input,
      streamNote: '流式模型每 30ms 调用一次，输入 （3, 40）。'
    });

    push({
      id: 'expand', kind: 'reshape',
      title: '扩展空间轴', type: 'ExpandDims',
      timeIn: T, timeOut: T,
      shapeIn: {
        train: '(' + batch + ', ' + T + ', 40)',
        stream: '(' + batch + ', ' + cfg.stride + ', 40)'
      },
      shapeOut: {
        train: shapeText(T, 40),
        stream: shapeText(cfg.stride, 40)
      },
      params: 0,
      desc: '把 [batch, time, feature] 变成 [batch, time, 1, feature]，后面全部按 2D 卷积（核在时间维上）处理。',
      formula: 'rank 3 → rank 4：在两个卷积轴之间插入长度 1 的空间轴',
      code: CODE.expand,
      streamNote: '流式时形状为 (1, stride, 1, 40)。'
    });

    if (cfg.firstConvFilters > 0) {
      var k0 = cfg.firstConvKernel;
      var tOut0 = Math.floor((T - k0) / cfg.stride) + 1;
      if (tOut0 <= 0) {
        errors.push('首层卷积时间维耗尽：T=' + T + '，kernel=' + k0 + '，stride=' + cfg.stride + '，后续层不再构建。');
        stopped = true;
      } else {
        var p0 = k0 * 40 * cfg.firstConvFilters;
        addWeights(p0);
        var frames0 = Math.max(k0 - cfg.stride, 0);
        var buf0 = addBuffer(frames0, 40, '首层卷积 ring buffer');
        push({
          id: 'conv0', kind: 'conv',
          title: '首层时间卷积 · ' + cfg.firstConvFilters + '@' + k0,
          type: 'Conv2D(k×1, stride, valid)',
          timeIn: T, timeOut: tOut0,
          shapeIn: { train: shapeText(T, 40), stream: shapeText(cfg.stride, 40) },
          shapeOut: { train: shapeText(tOut0, cfg.firstConvFilters), stream: shapeText(1, cfg.firstConvFilters) },
          params: p0,
          buffer: buf0,
          desc: '在时间维做 valid 卷积，stride 让模型每 ' + (cfg.stride * 10) + 'ms 推理一次；use_bias=False。',
          formula: 'T_out = ⌊(' + T + ' − ' + k0 + ') / ' + cfg.stride + '⌋ + 1 = ' + tOut0 + '；参数 = k × 40 × filters = ' + k0 + ' × 40 × ' + cfg.firstConvFilters + ' = ' + fmt(p0),
          code: CODE.conv0,
          streamNote: '流式 ring buffer = max(kernel − stride, 0) = ' + frames0 + ' 帧 × 40 通道 = ' + fmt(frames0 * 40) + ' 个状态变量；1 新帧 + 缓冲 → 输出 1 帧。'
        });
        T = tOut0;
        C = cfg.firstConvFilters;
        push(actNode('conv0.act', T, C));
      }
    }

    for (var bi = 0; bi < cfg.pointwise.length && !stopped; bi++) {
      var filters = cfg.pointwise[bi];
      var ks = cfg.kernels[bi];
      var reps = cfg.repeat[bi];
      var useRes = !!cfg.residual[bi];
      var inT = T;
      var inC = C;
      var children = [];
      var blockParams = 0;
      var resT = T;

      if (useRes) {
        var rp = C * filters + 4 * filters;
        addWeights(C * filters);
        addBN(filters);
        blockParams += rp;
        children.push({
          id: 'block' + bi + '.res', kind: 'residual',
          title: '残差支路 · 1×1 Conv + BN', type: 'Conv2D(1×1) + BatchNormalization',
          timeIn: T, timeOut: T,
          shapeIn: { train: shapeText(T, C), stream: shapeText(1, C) },
          shapeOut: { train: shapeText(T, filters), stream: shapeText(1, filters) },
          params: rp,
          desc: '块入口先做一个 1×1 卷积 + BN 得到残差；每次卷积后按时间维差用 StridedDrop 对齐再加回主干。',
          formula: '参数 = C_in × filters + 4 × filters = ' + C + ' × ' + filters + ' + 4 × ' + filters + ' = ' + fmt(rp),
          code: CODE.residual,
          streamNote: '流式下残差支路只保留 1 帧，StridedDrop 不丢弃数据。'
        });
      }

      for (var r = 0; r < reps && !stopped; r++) {
        var ksMax = Math.max.apply(null, ks);

        if (ksMax > 1) {
          var cin = C;
          var raw = ks.map(function (k) { return T - (k - 1); });
          var aligned = Math.min.apply(null, raw);
          if (aligned <= 0) {
            errors.push('块 ' + (bi + 1) + ' 的 MixConv（最大 kernel ' + ksMax + '）在 T=' + T + ' 下时间维耗尽，后续层不再构建。');
            stopped = true;
            break;
          }
          var splits = ks.length > 1 ? splitChannels(cin, ks.length) : [cin];
          var branches = ks.map(function (k, i) {
            return {
              k: k,
              channels: splits[i],
              tRaw: raw[i],
              tIn: T,
              tOut: aligned,
              drop: raw[i] - aligned,
              keep: k,
              params: k * splits[i] + splits[i]
            };
          });
          var mixId = 'block' + bi + '.r' + r + '.mix';
          branches.forEach(function (b) {
            index.set(mixId + '.k' + b.k, {
              id: mixId + '.k' + b.k,
              kind: 'branch',
              title: '分支 k=' + b.k + ' · ' + b.channels + ' 通道',
              type: 'StridedKeep + DepthwiseConv2D',
              params: b.params,
              desc: 'MixConv 把输入通道按 _split_channels 切开，每条分支用不同 kernel 做 depthwise 卷积，再对齐时间维后拼接。',
              formula: '通道 ' + b.channels + '；参数 = k × C + C（Keras depthwise 默认带 bias）= ' + b.k + ' × ' + b.channels + ' + ' + b.channels + ' = ' + fmt(b.params) + '；非流式时间维 ' + b.tRaw + ' → 对齐到 ' + b.tOut + '（丢弃 ' + b.drop + ' 帧）',
              code: 'fit = strided_drop.StridedKeep(' + b.k + ')(x)\nx = tf.keras.layers.DepthwiseConv2D((' + b.k + ', 1), strides=1, padding="valid")(fit)',
              streamNote: '流式：从共享 ring buffer 取最后 ' + b.k + ' 帧做 valid 卷积 → 1 帧输出。'
            });
          });
          var pMix = branches.reduce(function (a, b) { return a + b.params; }, 0);
          addWeights(pMix);
          blockParams += pMix;
          var bufMix = addBuffer(ksMax - 1, cin, '块 ' + (bi + 1) + ' MixConv ring buffer');
          children.push({
            id: mixId, kind: 'mixconv',
            title: 'MixConv · ' + ks.map(function (k) { return 'k=' + k; }).join(' + '),
            type: 'ChannelSplit + DepthwiseConv2D',
            timeIn: T, timeOut: aligned,
            shapeIn: { train: shapeText(T, cin), stream: shapeText(1, cin) },
            shapeOut: { train: shapeText(aligned, cin), stream: shapeText(1, cin) },
            params: pMix,
            branches: branches,
            buffer: bufMix,
            desc: '混合深度可分离卷积：把通道分组，每组用不同 kernel 在时间维上卷积，再裁剪到最短分支对齐拼接。',
            formula: '分支输出 T − (k−1)，取最小值 ' + aligned + '；参数合计 ' + fmt(pMix),
            code: CODE.mixconv,
            streamNote: '流式：共享 ring buffer ' + (ksMax - 1) + ' 帧 × ' + cin + ' 通道 = ' + fmt((ksMax - 1) * cin) + ' 个状态变量。'
          });
          T = aligned;
        }

        var cin2 = C;
        var pPw = cin2 * filters;
        addWeights(pPw);
        blockParams += pPw;
        children.push({
          id: 'block' + bi + '.r' + r + '.pw', kind: 'conv',
          title: '1×1 点卷积 ' + filters, type: 'Conv2D(1×1, use_bias=False)',
          timeIn: T, timeOut: T,
          shapeIn: { train: shapeText(T, cin2), stream: shapeText(1, cin2) },
          shapeOut: { train: shapeText(T, filters), stream: shapeText(1, filters) },
          params: pPw,
          desc: '在通道维做线性组合，把深度卷积的输出扩展到 filters 个通道。',
          formula: '参数 = C_in × filters = ' + cin2 + ' × ' + filters + ' = ' + fmt(pPw),
          code: CODE.pointwise,
          streamNote: '流式下只处理 1 帧，参数不变。'
        });
        C = filters;

        addBN(C);
        blockParams += 4 * C;
        children.push({
          id: 'block' + bi + '.r' + r + '.bn', kind: 'bn',
          title: '批归一化', type: 'BatchNormalization',
          timeIn: T, timeOut: T,
          shapeIn: { train: shapeText(T, C), stream: shapeText(1, C) },
          shapeOut: { train: shapeText(T, C), stream: shapeText(1, C) },
          params: 4 * C,
          desc: '每个通道 4 个参数（gamma / beta / moving mean / moving variance），其中 2C 可训练、2C 不可训练。',
          formula: '参数 = 4 × C = 4 × ' + C + ' = ' + fmt(4 * C),
          code: CODE.bn,
          streamNote: '推理时折叠为常量，不改变形状。'
        });

        if (useRes) {
          var drop = resT - T;
          children.push({
            id: 'block' + bi + '.r' + r + '.add', kind: 'add',
            title: '残差相加', type: 'StridedDrop + Add',
            timeIn: T, timeOut: T,
            shapeIn: { train: shapeText(T, C), stream: shapeText(1, C) },
            shapeOut: { train: shapeText(T, C), stream: shapeText(1, C) },
            params: 0,
            desc: '残差支路按时间维差裁剪后与主干相加。',
            formula: 'StridedDrop(' + drop + ') 对齐后逐元素相加，0 参数',
            code: 'residual = strided_drop.StridedDrop(residual.shape[1] - net.shape[1])(residual)\nnet = net + residual',
            streamNote: '流式下两者时间维都是 1，直接相加。'
          });
          resT = T;
        }

        children.push(actNode('block' + bi + '.r' + r + '.act', T, C));
      }

      if (children.length) {
        pushBlock({
          id: 'block' + bi, kind: 'block',
          title: 'MixConv 编码块 ' + (bi + 1),
          filters: filters, kernels: ks, repeat: reps, residual: useRes,
          timeIn: inT, timeOut: T,
          shapeIn: { train: shapeText(inT, inC), stream: shapeText(1, inC) },
          shapeOut: { train: shapeText(T, C), stream: shapeText(1, C) },
          params: blockParams,
          children: children
        });
      }
    }

    if (!stopped) {
      var postT = T;
      if (postT > 1) {
        if (cfg.attention) {
          var pAttn = 4 * 2;
          addWeights(pAttn);
          var bufAttn = addBuffer(3, 2, '注意力卷积 ring buffer');
          var bufCtx = addBuffer(postT - 1, C, '注意力上下文 ring buffer');
          push({
            id: 'post.attn', kind: 'attention',
            title: '空间注意力（CBAM 风格）', type: 'SpatialAttention',
            timeIn: postT, timeOut: postT,
            shapeIn: { train: shapeText(postT, C), stream: shapeText(1, C) },
            shapeOut: { train: shapeText(postT, C), stream: shapeText(1, C) },
            params: pAttn,
            buffers: [bufAttn, bufCtx],
            desc: '把时间轴转置后做通道平均/最大池化，拼成 2 通道描述符，再用时间核为 4 的卷积生成 0–1 注意力并乘回特征。',
            formula: '注意力卷积参数 = 4 × 1 × 2 × 1 = ' + pAttn + '（use_bias=False）；形状不变',
            code: CODE.attention,
            streamNote: '流式：注意力卷积缓冲 3 帧 × 2 通道，上下文缓冲 ' + (postT - 1) + ' 帧 × ' + C + ' 通道。'
          });
        } else {
          var bufPost = addBuffer(postT - 1, C, '上下文 ring buffer');
          push({
            id: 'post.buffer', kind: 'buffer',
            title: '上下文缓冲（Identity）', type: 'Stream(Identity)',
            timeIn: postT, timeOut: postT,
            shapeIn: { train: shapeText(postT, C), stream: shapeText(1, C) },
            shapeOut: { train: shapeText(postT, C), stream: shapeText(1, C) },
            params: 0,
            buffer: bufPost,
            desc: '非流式时是直通；流式时用 ring buffer 攒满 ' + (postT - 1) + ' 帧历史，让分类头看到完整上下文。',
            formula: 'ring_buffer_size_in_time_dim = T − 1 = ' + (postT - 1),
            code: CODE.buffer,
            streamNote: '状态变量 = ' + (postT - 1) + ' 帧 × ' + C + ' 通道 = ' + fmt((postT - 1) * C) + '。'
          });
        }

        if (cfg.pooled) {
          push({
            id: 'post.pool', kind: 'pool',
            title: cfg.maxPool ? '全局最大池化' : '全局平均池化', type: cfg.maxPool ? 'MaxPooling2D' : 'AveragePooling2D',
            timeIn: postT, timeOut: 1,
            shapeIn: { train: shapeText(postT, C), stream: shapeText(1, C) },
            shapeOut: { train: shapeText(1, C), stream: shapeText(1, C) },
            params: 0,
            desc: 'pool_size 覆盖整个时间维，把 ' + postT + ' 帧压成 1 帧，0 参数。',
            formula: 'pool_size = (' + postT + ', 1)，T → 1',
            code: cfg.maxPool ? CODE.poolMax : CODE.poolAvg,
            streamNote: '流式下时间维已经是 1，池化相当于直通。'
          });
          T = 1;
        }
      } else {
        warnings.push('编码器输出时间维已为 1，mixednet.py 的 if net.shape[1] > 1 分支会被跳过（不建注意力 / 上下文缓冲 / 池化）。');
      }

      var flatDim = T * C;
      var pDense = flatDim + 1;
      addWeights(pDense);
      push({
        id: 'flatten', kind: 'flatten',
        title: '展平 ' + flatDim + ' 维', type: 'Flatten',
        timeIn: T, timeOut: T,
        shapeIn: { train: shapeText(T, C), stream: shapeText(1, C) },
        shapeOut: { train: '(' + batch + ', ' + flatDim + ')', stream: '(' + batch + ', ' + flatDim + ')' },
        params: 0,
        flatDim: flatDim,
        desc: '把 [batch, T, 1, C] 压成 [batch, T×C]，0 参数。',
        formula: '展平维度 = T × 1 × C = ' + T + ' × ' + C + ' = ' + flatDim,
        code: CODE.flatten,
        streamNote: '流式下为 [batch, C]，但训练图已按 ' + flatDim + ' 维固定权重。'
      });

      push({
        id: 'dense', kind: 'dense',
        title: '单输出分类头', type: 'Dense(1, sigmoid)',
        timeIn: 1, timeOut: 1,
        shapeIn: { train: '(' + batch + ', ' + flatDim + ')', stream: '(' + batch + ', ' + flatDim + ')' },
        shapeOut: { train: '(' + batch + ', 1)', stream: '(' + batch + ', 1)' },
        params: pDense,
        desc: '全连接 + sigmoid，输出唤醒词概率；随后由应用层按滑动窗口判定。',
        formula: '参数 = ' + flatDim + ' + 1（bias）= ' + fmt(pDense),
        code: CODE.dense,
        streamNote: '每次推理输出一个概率，滑动窗口大小由应用配置（如 sliding_window_size=5）。'
      });
    }

    return {
      seq: seq,
      index: index,
      stats: stats,
      errors: errors,
      warnings: warnings,
      dropped: slicesDropped(cfg),
      derived: derivedFrames(cfg),
      stopped: stopped
    };
  }

  function runSelfTest() {
    var rows = [];
    var add = function (label, actual, expected) {
      rows.push({
        label: label,
        actual: String(actual),
        expected: String(expected),
        pass: String(actual) === String(expected)
      });
    };

    var g = buildGraph(clone(PRESETS.trained.cfg));
    var conv0 = g.index.get('conv0');
    var mix = [], pw = [], bn = [];
    g.seq.forEach(function (n) {
      if (n.kind !== 'block') return;
      n.children.forEach(function (c) {
        if (c.kind === 'mixconv') c.branches.forEach(function (b) { mix.push(b.params); });
        if (c.kind === 'conv') pw.push(c.params);
        if (c.kind === 'bn') bn.push(c.params);
      });
    });
    var flatten = g.index.get('flatten');
    var dense = g.index.get('dense');
    var post = g.index.get('post.buffer');

    add('总参数 = ' + fmt(EXPECTED.weights), g.stats.weights, EXPECTED.weights);
    add('可训练参数 = ' + fmt(EXPECTED.trainable), g.stats.trainable, EXPECTED.trainable);
    add('不可训练参数 = ' + EXPECTED.nonTrainable, g.stats.nonTrainable, EXPECTED.nonTrainable);
    add('首层卷积参数 = ' + fmt(EXPECTED.conv0Params), conv0.params, EXPECTED.conv0Params);
    add('首层输出时间维 = ' + EXPECTED.conv0T, conv0.timeOut, EXPECTED.conv0T);
    add('depthwise 参数序列', mix.join(','), EXPECTED.depthwise);
    add('1×1 卷积参数序列', pw.join(','), EXPECTED.pointwise);
    add('BN 参数序列', bn.join(','), EXPECTED.bn);
    add('Flatten 维度 = ' + fmt(EXPECTED.flatDim), flatten.flatDim, EXPECTED.flatDim);
    add('Dense 参数 = ' + fmt(EXPECTED.denseParams), dense.params, EXPECTED.denseParams);
    add('编码后时间维 = ' + EXPECTED.postT, post.timeIn, EXPECTED.postT);
    add('ring buffer 帧序列', g.stats.buffers.map(function (b) { return b.frames; }).join(','), EXPECTED.buffers);
    add('流式状态参数 = ' + fmt(EXPECTED.state), g.stats.state, EXPECTED.state);
    add('推导输入帧数 = ' + EXPECTED.derived, derivedFrames(PRESETS.trained.cfg), EXPECTED.derived);

    var light = buildGraph(clone(PRESETS.light.cfg));
    add('轻量预设可构建（权重 > 0）', light.stats.weights > 0, true);
    add('轻量预设无结构错误', light.errors.length, 0);
    return rows;
  }


  global.MWWViz = {
    PRESETS: PRESETS,
    EXPECTED: EXPECTED,
    KIND: KIND,
    CODE: CODE,
    clone: clone,
    fmt: fmt,
    parseNumList: parseNumList,
    parseKernelList: parseKernelList,
    splitChannels: splitChannels,
    slicesDropped: slicesDropped,
    derivedFrames: derivedFrames,
    validate: validate,
    buildGraph: buildGraph,
    runSelfTest: runSelfTest
  };
})(typeof window !== "undefined" ? window : globalThis);
