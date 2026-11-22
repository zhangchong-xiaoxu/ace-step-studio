# ace-step-studio · 一张消费级显卡的本地写歌与发行工坊

不充音乐会员、不调付费云 API，在一张 **16GB 消费级显卡**上基于开源 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)（MIT）跑通完整闭环：**一句话方向 → 45 秒洗脑短样快筛 → 保留原声逐比特无损扩写整首单曲 → 本地离线实唱校准 `.lrc` → 流媒体母带与跨平台配乐双轨交付**。

这是我（张翀）用 RTX 4060 Ti 16GB 在本地写歌并上架汽水音乐（首曲提审当日即过审上线）、同时为抖音 + 小红书视频自制无版权冲突配乐的实战 Skill 与脚本工具包。

```text
一句话方向 ─▶ demos.json ──generate_demos.py (LM 进程 → DiT 进程)──▶ 3 首 45s 候选小样 (人耳盲听挑片)
                                                                              │
                     ┌────────────────────────────────────────────────────────┘
                     ▼
        inspect-envelope 扫 0.5s 包络找切点 ──expand_song.py──▶ A 轨：5Hz 前缀自回归续写 + 0..切点逐比特无损拼接整首
                                                           └──▶ B 轨（可选）：独立 seed 移调纯伴奏（垫双平台视频防撞独家指纹）
                                                                              │
                     ┌────────────────────────────────────────────────────────┘
                     ▼
        check_lyrics_asr.py (本地滑窗校词) + master (-12.8 LUFS) ──package_release.py──▶ 正式 WAV/MP3 + LRC + 质检清单
```

---

## 为什么不只用 ACE-Step 官方自带的 `.claude/skills`？

ACE-Step 1.5 官方仓库内置了 `acestep` 等技能，但在「16GB 本地显卡 + 完整单曲上架」场景下有四个实测硬坑，本仓库专门解决它们：

| 痛点 | 官方默认做法 | 本 Skill（`ace-step-studio`）解法 |
|---|---|---|
| **Windows 16GB 显存崩溃** | 起单进程 API 服务同载 `5Hz LM (1.7B)` + `DiT` → 初始化直接报 `0xC0000005` 退出 | **强制双进程接力**：进程 1 跑 5Hz LM 落盘 `lm.json` 后退出释放显存，进程 2 单独跑 DiT 渲染立体声 |
| **短样满意后扩写整首翻车** | 调 `task_type="repaint"` 向右扩写 → 源码强制关闭 5Hz LM（`thinking=False`），导致重唱第一段或前 25 秒唱完后空转纯伴奏 | **5Hz 前缀自回归续写 + 原曲无损拼接**：扫 0.5s RMS 包络避开尾音衰减 → `cond/uncond` 双侧注入原曲前缀音频码续写后半首 → `reference_audio` 锁声线 → 相位对齐等功率交叉淡化，**前半首 `0..切点` 逐比特零误差保留原曲（`max_diff = 0.0`）** |
| **歌词对齐与封面依赖付费 API** | 歌词转写需配 OpenAI Whisper 或 ElevenLabs API Key，封面需配 Gemini API Key | **全本地离线可跑**：支持本地 `sherpa-onnx` FireRed CTC 滑窗 + 可选 `faster-whisper` 双核对（防长音频漏字与电音前奏幻觉），`package_release.py` 纯标准库校验封面与静音门禁 |
| **流媒体独家与多平台视频撞版权** | 未区分发行轨与视频内嵌轨 | **A+B 双轨工作流**：A 轨带词单曲按 `-12.8 LUFS / -1.0 dBTP` 投流媒体独家；B 轨另起独立 seed、微移调式与 BPM 出纯器乐（`-16 LUFS` 口播垫音 + `-14 LUFS` 6 秒片尾锚点），安全焊进多平台视频 |

---

## 安装与使用方式

### 方式一：作为 Agent Skill 安装（Claude Code / Copilot / Cursor / Codex / DSH）

支持 `skills` CLI 的环境可一键安装：

```bash
npx skills add zhangchong-xiaoxu/ace-step-studio
```

或直接克隆到你的工作区 / 全局技能目录：

```bash
# VS Code GitHub Copilot 工作区技能
git clone https://github.com/zhangchong-xiaoxu/ace-step-studio .github/skills/ace-step-studio

# Claude Code 技能目录
git clone https://github.com/zhangchong-xiaoxu/ace-step-studio .claude/skills/ace-step-studio
```

### 方式二：克隆仓库交给 AI 助手直接开干

在任意带 AI 助手的 IDE 里，把仓库地址和下面这句话发给它：

> 克隆这个仓库，先读 AGENTS.md 和 SKILL.md，运行 `python scripts/doctor.py --json` 检查我的机器，缺什么先告诉我、我同意再装，然后帮我做一首本地原创歌。

`doctor.py` 只用 Python 标准库、只读不安装：自动检测 Python、`ffmpeg`、显卡显存、本机 `ACE-Step-1.5` 路径与权重、以及可选的本地离线语音识别模型。

---

## 需要准备

| 组件 | 用途 | 备注 |
|---|---|---|
| NVIDIA 显卡（建议 12–16GB+ 显存） | 本地推理 5Hz LM 与 DiT | 已在 Windows + RTX 4060 Ti 16GB 完整实测（45s 小样 LM 约 10s + DiT 约 4.2s，不含模型加载） |
| [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 本地克隆与权重 | 核心生成引擎（MIT） | 含 `.venv` 环境及 `checkpoints/acestep-5Hz-lm-1.7B`、`acestep-v15-turbo`。可通过环境变量 `ACESTEP_ROOT` 指定路径 |
| Python 3.10+ | 运行 `doctor.py`、`package_release.py` | 生成脚本直接复用 `ACE-Step-1.5/.venv` 内的 Python（自带 `torch`、`soundfile`、`numpy`） |
| `ffmpeg` | 母带 `loudnorm`、`ebur128` 响度检测与 MP3 转码 | 需在 `PATH` 中 |
| 本地 ASR（可选） | 实唱词句滑窗核对（防漏唱拒审） | 可选 `sherpa-onnx`（FireRed `zh_en` CTC 模型，Apache-2.0）或 `faster-whisper`；未装时可跳过，靠人耳对词 |

---

## 五步实操（命令行速览）

假设 `doctor.py` 给出的 ACE-Step Python 路径存在变量 `$ACE_PY`（Linux/macOS 为 `$ACESTEP_ROOT/.venv/bin/python`，Windows PowerShell 为 `& "$env:ACESTEP_ROOT\.venv\Scripts\python.exe"`）：

### ① 环境自检

```bash
python scripts/doctor.py
```

### ② 分进程抽 2–3 首 45 秒洗脑副歌小样

参考 [`examples/demos-zh.json`](examples/demos-zh.json) 或 [`examples/demos-en.json`](examples/demos-en.json) 写好词曲规格（写词与提示词避坑见 [`references/songwriting-and-prompts.md`](references/songwriting-and-prompts.md)）：

```bash
$ACE_PY -B scripts/generate_demos.py lm --spec examples/demos-zh.json --out-dir output/demos
$ACE_PY -B scripts/generate_demos.py dit --spec examples/demos-zh.json --out-dir output/demos
```

### ③ 扫包络选切点 + 5Hz 前缀无损扩写整首（含可选 B 轨纯伴奏）

人耳试听选出最满意的一首小样后，先扫 0.5 秒窗包络，避开末尾提前衰减静音：

```bash
$ACE_PY -B scripts/generate_demos.py inspect-envelope --wav output/demos/demo2-cyber-go/demo2-cyber-go.wav
```

按推荐切点写好扩写规格（参考 [`examples/expand-full.example.json`](examples/expand-full.example.json)），分两步生成多组候选并完成逐比特无损拼接：

```bash
$ACE_PY -B scripts/expand_song.py lm-continue --spec expand.json --out-dir output/full
$ACE_PY -B scripts/expand_song.py dit-stitch --spec expand.json --out-dir output/full
```

### ④ 流媒体响度母带与静音门禁

挑出最满意的候选 seed，做平滑尾奏淡出与 `-12.8 LUFS / -1.0 dBTP` 母带（自动输出 `ebur128` 读数与 `< -42 dBFS` 静音窗检查）：

```bash
$ACE_PY -B scripts/expand_song.py master \
  --src-wav output/full/rendered_candidates/seed_20261179/stitched_seed_20261179.wav \
  --out-dir output/full/final \
  --out-stem takeA-s20261179 \
  --fade-start 119.2 \
  --total-end 123.2
```

### ⑤ 实唱校词与流媒体发行打包

用可选本地滑窗 ASR（或人耳听审）核对实唱词句，整理为 [`examples/synced-lyrics.example.json`](examples/synced-lyrics.example.json) 格式后一键打包并做上线前质检（封面与独家版权策略见 [`references/streaming-release-and-copyright.md`](references/streaming-release-and-copyright.md)）：

```bash
python scripts/package_release.py \
  --wav output/full/final/takeA-s20261179.wav \
  --mp3 output/full/final/takeA-s20261179.mp3 \
  --synced-json examples/synced-lyrics.example.json \
  --title "Neon Sky" \
  --artist "Your Artist Name" \
  --out-dir output/full/release
```

---

## 踩过的真坑

1. **Windows 16GB 同进程必崩（`0xC0000005`）**：哪怕把 5Hz LM 设到 CPU，只要和 DiT 在同一进程初始化就会触发访问违例。拆成 `lm` 与 `dit` 两个独立进程后 100% 稳定。
2. **45 秒小样末尾常提前 3–4 秒衰减静音**：如果直接从 `45.0s` 往后接，整首歌中间会留出 `>3s` 死寂断档，不仅难听还会触发音乐平台机审退回。扩写前先跑 `inspect-envelope` 把切点前移到尾音尚饱满处（如 `39.3–42.8s`）。
3. **英文电子曲 `Minimal Tech-House` + `groove bed` 漏唱陷阱**：`caption` 同时出现这两个词时，5Hz LM 唱完前 14 秒 Hook 就会把后 31 秒排成纯器乐 Club Drop（滑窗 ASR 可直接抓出后半段全 `<sil>`）。需要完整人声时请改写为 `Melodic House Pop ... no long instrumental drop`。
4. **整首丢进 CTC 会漏字，Whisper 遇纯电音前奏会幻觉**：140 秒歌曲一次塞进 CTC 模型会吞尾词，必须按 `≤18s` 滑窗并给首窗补 `0.3s` 前导零静音；英文曲用 `faster-whisper` 查词尾辅音极准，但遇到高能量纯合成器前奏易幻觉出 `Thanks for watching!`，须对照 CTC `<sil>` 剔除未实际起唱的前奏词句。
5. **音乐平台封面 ≠ 短视频封面**：单曲封面要求 `3000×3000` sRGB（`CMYK` 必拒）、`≤10 MB`，且**严禁**印 `AI生成`、`ACE-Step`、年份、歌词或社媒号（会被判含无关说明文字或第三方品牌驳回）。AIGC 身份请在上传后台表单如实勾选。

---

## 本地测试

不依赖显卡或大模型，2 秒内跑完核心拼接、包络分析、LRC 与封面头校验单元测试：

```bash
python -m unittest discover -s tests -v
```

---

## 许可与合规

- **本仓库代码与文档**：[MIT License](LICENSE)。
- **上游引擎与可选模型**（均不包含在本仓库内，使用前请核对各自许可）：
  - [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)：MIT License。
  - [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 及 FireRed ASR 模型：Apache-2.0 License。
  - FFmpeg：外部命令行调用（LGPL/GPL，随你本机安装发行版而定）。
- **原创与标识合规**：仅使用原创歌词，不抄袭受版权保护的词曲、不冒用特定歌手姓名或声线；在中国大陆及各大流媒体/社媒平台发布 AI 生成音乐时，请遵守《人工智能生成合成内容标识办法》（2025-09-01 起施行）及平台规则主动勾选 AIGC 声明。
