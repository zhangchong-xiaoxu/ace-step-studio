# Progress

更新日期：2026-11-22

## 当前状态

v0.1.0 首版就绪：根级标准 [SKILL.md](../SKILL.md) + 5 个无硬编码路径的参数化 Python CLI 脚本（`doctor.py`、`generate_demos.py`、`expand_song.py`、`check_lyrics_asr.py`、`package_release.py`）+ 词曲与流媒体发行参考卡 + 中英文示例规格与确定性单元测试全部就位。

## 下一步

1. 在下一次真实单曲或视频配乐创作中直接使用 `scripts/generate_demos.py` 与 `scripts/expand_song.py` 跑通端到端生成，若发现新跨平台边缘情况（如 macOS MPS / Linux ROCm 推理参数差异），补充到 `SKILL.md` 的 `Pitfalls Log` 并补对应单测（无需额外批准）。
2. 若社区反馈需要直接从粗略文本自动对齐逐行时间戳（无需手调 `synced-lyrics.json`），评估在 `check_lyrics_asr.py` 中增加 CTC 帧级强制对齐导出子命令。

## 进行中 / 卡住

- 无。

## 怎么验证还能用

```bash
python -B -m unittest discover -s tests -v    # 预期：全部单测 OK（<2s，不依赖 GPU 或大模型）
python -B scripts/doctor.py --json            # 预期：输出合法 JSON 环境检测报告，退出码 0
```

## 资料入口

- [文档地图与决策索引](README.md)
- [Skill 主说明](../SKILL.md)
- [写词与提示词规范](../references/songwriting-and-prompts.md)
- [流媒体母带、封面与 A+B 版权隔离](../references/streaming-release-and-copyright.md)

## 交接日志

- 2026-11-22：从三首实测单曲（《沿着灯光》《跑起来才算数》《Neon Sky》及独立 Synthwave 视频 B 轨）提炼并通用化为开源 MIT Skill 仓库 `ace-step-studio`，移除本机硬编码路径，补齐 `doctor.py`、纯标准库封面/静音/LRC 质检门禁与单元测试。
