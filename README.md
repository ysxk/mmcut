# mmcut (mm_cut)

> **ASR 定时间，多模态听音频定取舍，信号能量定切点。**  
> 专为口播视频设计的 AI 全自动智能粗剪工具与 Agent Skill。

---

## 💡 为什么做 mmcut？

现在市面上大部分 AI 剪口播工具（以及相关的 Agent 提示词/Skills）普遍采用「剪映式逻辑」：
**先用 ASR 把语音识别成文字，再让大模型基于文字去挑删减。**

实际落地中，这种纯文本方案常常导致**剪完后依然需要大量人工复审**，甚至比纯手工剪还要更耗精力：

1. **ASR 逐字时间不准或均分**：许多 ASR 引擎给出的字时间是机械均分的。例如一句 15 个字的话，每个字均分 0.18s，中途说错改口时，AI 只能基于均分时间盲猜下刀位置，极易造成吞字或留下尾音杂音；
2. **文字丢弃了最关键的声学信息**：口播剪辑的灵魂在于**语气、语调、重音和停顿**。
   - 例如：“HR*一个一个*去聊”是**强调语气**，ASR 看着像重复口误；
   - 又如：“只花了8分钟他就联系了9个人，*并给其中的两位发……*”，看文字是完整半句话，但一听音频就知道语调在“发”字上卡住了、后面重录了完整版，这是典型的**残句**；
3. **ASR 会“自作聪明”地过度纠错**：例如说英文卡壳“Cla... Claude”，ASR 会直接输出一个“Claude”，导致卡顿的那半个词在文本中根本不存在，根本无法被剔除。

---

## 🎯 mmcut 的核心哲学：三权分立

为了彻底解决上述痛点，`mmcut` 提出了**分工明确的剪辑流水线**：

```text
┌────────────────┐      ┌────────────────────────┐      ┌────────────────┐
│   内网 ASR     │      │   多模态 Audio LLM     │      │  信号能量 (VAD) │
│ (Qwen3-ASR等)  │      │  (具备原生听觉的 Agent) │      │  (Audio Energy)│
├────────────────┤      ├────────────────────────┤      ├────────────────┤
│ • 80ms 逐字时间戳│      │ • 听音频判语气、取舍   │      │ • 捕捉真实开口/收音│
│ • 生成句子编号  │ ──►  │ • 识别残句、重说、纠正 │ ──►  │ • 接缝收紧到   │
│ • 只定时，不决策│      │ • 不报时间，只定 S 编号 │      │   0.08~0.11s   │
└────────────────┘      └────────────────────────┘      └────────────────┘
```

- 🕒 **ASR 定时间**：逐字物理时间戳，80ms 一格，与人工满意版切点误差 $\le$ 0.02s；
- 👂 **模型定取舍**：多模态 Agent 直接听 60s 听音切片，按句子编号（S001、S002...）决定删留，模型不需要算时间，杜绝时间戳幻觉；
- ⚡ **能量定切点**：上一句收音后留 0.06s，下一句开口前留 0.04s，中间的静音、叹气、鼻息与吞咽杂音一并收紧切除，生成呼吸自然的无缝接缝。

> **实战成效**：在实录 4分21秒 的口播原片中，mmcut 全自动跑完，剪出 1分19秒 成片，精准剔除 28 处口误与残句（其中 21 处为断句重来），无需人工二审。

---

## ✨ 核心特性

- 🎬 **全自动一口气成片**：从多段源视频合并、提取音频、听音切块、ASR 对齐、多模态决策、音频粗剪复听、音画拼接到剪后字幕生成，一气呵成。
- 🎧 **音频驱动画面**：模型不直接看高负担的视频，画面严格跟随剪辑验收后的音频时间轴自动下刀，杜绝音画不同步。
- ⚡ **HDR 自动无损转 SDR**：针对 iPhone HLG / HDR 录制的视频源，自动采用原生 AVFoundation 处理为标准 BT.709 SDR，彻底告别画面灰白发灰问题。
- ⏩ **原生倍速与字幕对齐**：支持一键变速（如 1.2x，自动保调音高），EDL 和 SRT 字幕时间轴自动联动换算。
- 📦 **零三方 Python 依赖**：`scripts/mmcut.py` **完全基于 Python 3.9+ 标准库编写**，除 FFmpeg 外无需 `pip install` 任何重型三方库。
- 🤖 **标准 Agent Skill 兼容**：完美支持 Claude Code、Codex、PI-Desktop 以及所有遵循 Agent Skills 协议的智能体环境。

---

## 🛠️ 环境要求

| 依赖组件 | 版本与要求 | 说明 |
|---|---|---|
| **Python** | 3.9+ | 仅使用内置标准库，无需安装 pip 依赖 |
| **FFmpeg + ffprobe** | 4.4+ | 基础音视频处理、合并、静音检测与编码 |
| **Qwen3-ASR** | 内网或本地部署 | 默认端点 `http://192.168.1.7:8900`（可用环境变量 `MMCUT_ASR_URL` 覆盖） |
| **多模态 Agent** | 支持原生音频理解 | 能够直接接收/读取 mp3/wav 附件的 Agent 或模型（如 Claude 3.5 Sonnet / Gemini 等） |
| **Xcode Command Line Tools** | macOS 可选 | 编译 HDR 转 SDR 辅助工具（初次运行时自动编译至 `~/.cache/mmcut/`） |

---

## 🚀 快速上手

### 1. 作为 Agent Skill 安装

直接将本仓库克隆到你的 Agent Skills 目录下：

```bash
# 适用于 PI-Desktop / Agent 框架
git clone https://github.com/ysxk/mmcut.git .agents/skills/mm_cut

# 适用于 Claude Code
git clone https://github.com/ysxk/mmcut.git .claude/skills/mm_cut
```

装好后，对你的 Agent 直接发送指令即可触发：
> “帮我剪这段口播视频”  
> “用 mm_cut 处理刚录好的 3 段口播原片”

---

### 2. CLI 命令行调用

`scripts/mmcut.py` 也可以作为独立命令行工具调用：

```bash
# 查看所有子命令帮助
python3 scripts/mmcut.py --help

# 1. 准备阶段：多段合并、提取音频与分块
python3 scripts/mmcut.py prepare temp/2026-11-03_task/mm_cut video1.mp4 video2.mp4

# 2. ASR 识别阶段：生成逐字时间轴与可读稿 readable.txt
python3 scripts/mmcut.py asr \
  --pack temp/2026-11-03_task/mm_cut/1_音频 \
  --out temp/2026-11-03_task/mm_cut/2_听写 \
  --terms "Claude,OpenAI,Agent"

# 3. 候选吸附：将 Agent 标记的句子编号/词语吸附到音频能量切点
python3 scripts/mmcut.py snap \
  --work temp/2026-11-03_task/mm_cut \
  --candidates temp/2026-11-03_task/mm_cut/3_分析/candidates.json

# 4. 音频渲染：先剪出音频母带并收紧接缝
python3 scripts/mmcut.py render \
  --stage audio \
  --work temp/2026-11-03_task/mm_cut \
  --version v1

# 5. 画面渲染：复听验收通过后拼合画面（可加 --speed 1.2）
python3 scripts/mmcut.py render \
  --stage video \
  --work temp/2026-11-03_task/mm_cut \
  --version v1 \
  --accepted

# 6. 字幕生成：基于剪后音频生成精确 srt 字幕草稿
python3 scripts/mmcut.py srt \
  --transcript temp/2026-11-03_task/mm_cut/5_字幕/v1/transcript.json \
  --edl temp/2026-11-03_task/mm_cut/4_剪辑/v1/edl.json \
  --fixes temp/2026-11-03_task/mm_cut/5_字幕/v1/fixes.json \
  --out temp/2026-11-03_task/mm_cut/5_字幕/v1/video.srt

# 7. 最终交付：归档成片与交接清单到 output 目录
python3 scripts/mmcut.py finalize \
  --work temp/2026-11-03_task/mm_cut \
  --dest output/2026-11-03_task \
  --version v1
```

---

## 📂 项目结构

```text
mmcut/
├── README.md               # 项目详细说明文档
├── SKILL.md                # 规范的 Agent Skill 流程定义
├── LICENSE                 # MIT 开源许可证
├── .gitignore              # 媒体与中间产物忽略规则
├── scripts/
│   └── mmcut.py            # 核心流水线处理引擎（纯标准库）
└── references/
    ├── 剪辑规则.md          # 唯一的剪辑决策原则库（删前保后、接缝尺度、风险分层等）
    └── 听音频提示词.md      # 给多模态 Audio LLM 的分阶段结构化 Prompt
```

---

## 📋 运行流水线与中间产物

执行过程中产生的所有中间文件统一存放于 `temp/<任务名>/mm_cut/`，定稿前不污染输出目录：

```text
temp/<日期_任务名>/
└── mm_cut/
    ├── 0_源/       source 原片或多段 merged.mp4
    ├── 1_音频/     master.wav 母带、60s 听音切片 (chunks/)、静音能量图
    ├── 2_听写/     transcript.json 逐字稿、readable.txt 带 S 编号的可读稿
    ├── 3_分析/     candidates.json 决策候选、口误分析.md、可删建议.md
    ├── 4_剪辑/vN/  cut_master.wav 剪后音频、edl.json、splices.txt、验收.md、cut.mp4
    └── 5_字幕/vN/  剪后 ASR 逐字稿、fixes.json 模型校字、video.srt
```

---

## 📄 开源许可证

本项目采用 [MIT License](LICENSE) 开源协议。
