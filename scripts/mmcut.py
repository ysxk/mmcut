#!/usr/bin/env python3
"""
mm_cut 剪口播工具：ASR 定时间，多模态定取舍，能量定切点。

分工：内网 Qwen3-ASR 给逐字时间轴和初稿；具备原生音频理解能力的 agent 听音频，
按句子编号决定删哪里、校正字幕文字，不报时间；本脚本用信号能量把切点修到真实的
开口/收音位置，并做全部 ffmpeg 媒体处理。

子命令（按流程顺序）:
  prepare   多段先合并 → 提取音频 → 静音图 → 切成听音块
  asr       听音块逐块送内网 ASR → 逐字时间轴 + 分句 → transcript.json、readable.txt
  merge     （备用：ASR 不可用时）合并模型自己听写的 JSON → 对齐检查
  clip      导出放大复听片段（只出音频）
  snap      删除候选（句子编号/字/时间）解析成字范围 + 自动补静音候选
  render    --stage audio：按候选剪音频，接缝收紧到开口/收音；--stage video：按音频时间轴拼画面（可倍速、HDR 自动转 SDR）
  srt       剪后音频的 ASR 逐字时间 → 断行 → 套用模型校字 → video.srt
  finalize  用户定稿后复制到 output/ 并写交接文件

每个子命令 --help 查看参数。一律在项目根用 python3 调用。
"""
from __future__ import annotations

import argparse
import array
import bisect
import datetime as _dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MAX_MB_CEILING = 100.0   # 多模态模型文件输入上限（硬性）
SAFETY = 0.95            # 留 5% 余量
STD_FPS = {
    23.976: 24000 / 1001, 24.0: 24.0, 25.0: 25.0, 29.97: 30000 / 1001, 30.0: 30.0,
    48.0: 48.0, 50.0: 50.0, 59.94: 60000 / 1001, 60.0: 60.0,
}
FPS_EXPR = {24000 / 1001: "24000/1001", 30000 / 1001: "30000/1001", 60000 / 1001: "60000/1001"}
FILLERS = set("嗯啊哎诶呃额唉哦噢呀欸")
ASR_URL = os.environ.get("MMCUT_ASR_URL", "http://192.168.1.7:8900")
HDR_TRC = ("arib-std-b67", "smpte2084")
SENT_END = set("。！？!?")


# ───────────────────────── 通用 ─────────────────────────

def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"⚠️  {msg}", flush=True)


def info(msg: str) -> None:
    print(msg, flush=True)


def fin(p) -> str:
    """路径含冒号时 ffmpeg 会误判为协议，统一加 file: 前缀。"""
    return f"file:{p}"


def run(cmd, capture=False) -> subprocess.CompletedProcess:
    r = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        die(f"命令失败: {' '.join(str(c) for c in cmd[:8])} ...\n{r.stderr[-2000:]}")
    return r


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p, data) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


# ───────────────────────── 探测 ─────────────────────────

def probe(path) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", fin(path)],
            capture=True)
    return json.loads(r.stdout)


def vstream(p: dict):
    for s in p.get("streams", []):
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic", 0):
            return s
    return None


def astream(p: dict):
    for s in p.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None


def frac(s: str) -> float:
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        return 0.0


def rotation(vs: dict) -> int:
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    t = (vs.get("tags") or {}).get("rotate")
    return int(t) % 360 if t else 0


def display_size(vs: dict):
    w, h = int(vs["width"]), int(vs["height"])
    return (h, w) if rotation(vs) in (90, 270) else (w, h)


def pick_fps(vs: dict, override: float | None = None) -> float:
    if override:
        return STD_FPS.get(round(override, 3), override)
    a = frac(vs.get("avg_frame_rate", "0/1"))
    r = frac(vs.get("r_frame_rate", "0/1"))
    base = a or r or 30.0
    best = min(STD_FPS, key=lambda f: abs(f - base))
    return STD_FPS[best] if abs(best - base) < 1.0 else round(base, 3)


def fps_expr(f: float) -> str:
    for k, v in FPS_EXPR.items():
        if abs(k - f) < 1e-6:
            return v
    return f"{f:g}"


def is_vfr(vs: dict) -> bool:
    a, r = frac(vs.get("avg_frame_rate", "0/1")), frac(vs.get("r_frame_rate", "0/1"))
    return bool(a and r and abs(a - r) > 0.5)


def stream_dur(s: dict | None, fallback: float) -> float:
    try:
        return float(s["duration"]) if s and s.get("duration") else fallback
    except Exception:
        return fallback


def media_duration(path) -> float:
    return float(probe(path)["format"]["duration"])


# ─────────────────────── 信号分析（非 ASR） ───────────────────────

def mean_volume(path) -> float:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", fin(path), "-af", "volumedetect",
                        "-f", "null", "-"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else -25.0


def detect_silences(path, noise_db: float, min_dur: float, total: float):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", fin(path), "-af",
                        f"silencedetect=noise={noise_db:.1f}dB:d={min_dur}", "-f", "null", "-"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    out, cur = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            cur = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and cur is not None:
            out.append([round(cur, 3), round(float(m.group(1)), 3)])
            cur = None
    if cur is not None:
        out.append([round(cur, 3), round(total, 3)])
    return out


def islands_from(silences, total: float, min_len=0.05):
    """有声区间 = 静音的补集。"""
    out, cursor = [], 0.0
    for s, e in silences:
        if s - cursor >= min_len:
            out.append([round(cursor, 3), round(s, 3)])
        cursor = max(cursor, e)
    if total - cursor >= min_len:
        out.append([round(cursor, 3), round(total, 3)])
    return out


def read_pcm(wav_path, start: float, dur: float):
    with wave.open(str(wav_path), "rb") as w:
        sr, nch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        a = max(0, min(n - 1, int(start * sr)))
        w.setpos(a)
        buf = array.array("h")
        buf.frombytes(w.readframes(max(1, int(dur * sr))))
    if sys.byteorder == "big":
        buf.byteswap()
    if nch > 1:
        buf = array.array("h", buf[::nch])
    return buf, sr, a / sr


def energy_valley(wav_path, t: float, radius=0.2, frame=0.01):
    """t 附近能量最低的 10ms 帧中心（秒）及其 dBFS。"""
    buf, sr, t0 = read_pcm(wav_path, max(0.0, t - radius), 2 * radius)
    fl = max(1, int(frame * sr))
    best_i, best = 0, None
    for i in range(0, max(1, len(buf) - fl + 1), max(1, fl // 2)):
        seg = buf[i:i + fl]
        if not seg:
            break
        rms = math.sqrt(sum(x * x for x in seg) / len(seg))
        if best is None or rms < best:
            best, best_i = rms, i
    db = 20 * math.log10((best or 0) / 32768 + 1e-9)
    return round(t0 + (best_i + fl / 2) / sr, 3), round(db, 1)


def frames_db(wav_path, t0: float, t1: float, frame=0.01):
    """[t0, t1) 的逐帧 dBFS。返回 (第 0 帧起点秒, [dB…])，第 k 帧覆盖 [起点 + k·frame, 起点 + (k+1)·frame)。"""
    t0 = max(0.0, t0)
    buf, sr, s = read_pcm(wav_path, t0, max(frame, t1 - t0))
    fl = max(1, int(frame * sr))
    out = []
    for i in range(0, len(buf) - fl + 1, fl):
        seg = buf[i:i + fl]
        out.append(20 * math.log10(math.sqrt(sum(x * x for x in seg) / fl) / 32768 + 1e-9))
    return s, out


def refine_onset(wav_path, t, thr, lo, hi, frame=0.01):
    """ASR 给的字头 t → 真实开口时间。只在 [lo, hi] 内、t 前 0.2s 到后 0.15s 里找：
    t 处已有声就往前找到这段声音的起点；t 处还是静音就往后找第一帧有声。"""
    w0, w1 = max(lo, t - 0.2), min(hi, t + 0.15)
    if w1 - w0 < 2 * frame:
        return round(t, 3)
    s, db = frames_db(wav_path, w0, w1, frame)
    if not db:
        return round(t, 3)
    k = min(len(db) - 1, max(0, int((t - s) / frame)))
    if db[k] >= thr:
        while k > 0 and db[k - 1] >= thr:
            k -= 1
        return round(s + k * frame, 3)
    for j in range(k, len(db)):
        if db[j] >= thr:
            return round(s + j * frame, 3)
    return round(t, 3)


def refine_offset(wav_path, t, thr, lo, hi, frame=0.01):
    """ASR 给的字尾 t → 真实收音时间（尾音衰减到阈值以下）。在 [lo, hi] 内、t 前 0.15s 到后 0.25s 里找。"""
    w0, w1 = max(lo, t - 0.15), min(hi, t + 0.25)
    if w1 - w0 < 2 * frame:
        return round(t, 3)
    s, db = frames_db(wav_path, w0, w1, frame)
    if not db:
        return round(t, 3)
    k = min(len(db) - 1, max(0, int((t - s) / frame) - 1))
    if db[k] >= thr:
        while k + 1 < len(db) and db[k + 1] >= thr:
            k += 1
        return round(s + (k + 1) * frame, 3)
    for j in range(k, -1, -1):
        if db[j] >= thr:
            return round(s + (j + 1) * frame, 3)
    return round(t, 3)


# ───────────────────────── 听音包 ─────────────────────────

def plan_chunks(total: float, silences, chunk_sec: float):
    if total <= chunk_sec * 1.2:
        return [(0.0, total)], []
    cuts, notes = [0.0], []
    while total - cuts[-1] > chunk_sec * 1.2:
        target = cuts[-1] + chunk_sec
        lo, hi = cuts[-1] + chunk_sec * 0.6, cuts[-1] + chunk_sec * 1.1
        cand = [s for s in silences if s[0] >= lo and s[1] <= hi and s[1] - s[0] >= 0.3]
        if cand:
            best = max(cand, key=lambda s: (s[1] - s[0]) - abs((s[0] + s[1]) / 2 - target) / 30)
            point = (best[0] + best[1]) / 2
        else:
            point = target
            notes.append(f"{fmt_t(target)} 附近没有 ≥0.3s 的静音，硬切；该处的话可能被切成两半，听写时注意块首块尾")
        cuts.append(round(point, 3))
    cuts.append(total)
    return list(zip(cuts[:-1], cuts[1:])), notes


def build_listen_pack(src_media, outdir: Path, *, chunk_sec: float, max_mb: float, fmt: str,
                      bitrate: str, sr: int, noise_db: float | None, label: str,
                      island_min: float = 0.2) -> dict:
    """从任意媒体生成给模型听的音频块（只有音频），以及本地分析用的 analysis.wav。"""
    if max_mb > MAX_MB_CEILING:
        die(f"--max-mb 不能超过 {MAX_MB_CEILING:g}（多模态模型文件输入上限）")
    outdir.mkdir(parents=True, exist_ok=True)
    chunks_dir = outdir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir()

    analysis = outdir / "analysis.wav"   # 16k 单声道，仅本地信号分析和切放大片段，不发给模型
    run(["ffmpeg", "-y", "-v", "error", "-i", fin(src_media), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", fin(analysis)])
    total = media_duration(analysis)

    if noise_db is None:
        mv = mean_volume(analysis)
        noise_db = max(-50.0, min(-28.0, mv - 14.0))
        info(f"🔈 平均响度 {mv:.1f} dB → 静音阈值 {noise_db:.1f} dB（可用 --noise-db 覆盖）")
    coarse = detect_silences(analysis, noise_db, island_min, total)
    fine = detect_silences(analysis, noise_db, 0.05, total)
    islands = islands_from(coarse, total)
    write_json(outdir / "silences.json", {"noise_db": noise_db, "duration": total, "coarse_min_dur": island_min,
                                          "coarse": coarse, "fine_min_dur": 0.05, "fine": fine,
                                          "islands": islands})

    spans, notes = plan_chunks(total, coarse, chunk_sec)
    limit = max_mb * SAFETY
    chunks = []
    for i, (a, b) in enumerate(spans):
        name = f"chunk_{i:03d}.{fmt}"
        dst = chunks_dir / name
        codec = ["-c:a", "pcm_s16le", "-ar", str(sr)] if fmt == "wav" else \
            ["-c:a", "libmp3lame", "-b:a", bitrate, "-ar", str(sr)]
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", fin(analysis),
             "-ac", "1", *codec, fin(dst)])
        size_mb = dst.stat().st_size / 1024 / 1024
        if size_mb > limit:
            die(f"{name} 有 {size_mb:.1f}MB，超过 {limit:.1f}MB（{max_mb:g}MB×{SAFETY}）。把 --chunk-sec 调小重跑")
        local = [[round(max(s, a) - a, 3), round(min(e, b) - a, 3)] for s, e in islands if e > a and s < b]
        ids = [k + 1 for k, (s, e) in enumerate(islands) if e > a and s < b]
        with open(chunks_dir / f"chunk_{i:03d}.islands.txt", "w", encoding="utf-8") as f:
            f.write(f"# {name} 的有声区间（静音检测得到，时间为块内秒数，块起点 = 整条 {a:.3f}s）\n")
            for k, (s, e) in zip(ids, local):
                f.write(f"#{k:04d}  {s:8.3f} - {e:8.3f}  ({e - s:.2f}s)\n")
        chunks.append({"file": f"chunks/{name}", "islands_hint": f"chunks/chunk_{i:03d}.islands.txt",
                       "start": round(a, 3), "end": round(b, 3), "duration": round(b - a, 3),
                       "size_mb": round(size_mb, 2)})

    manifest = {"label": label, "source_media": str(Path(src_media).resolve()), "duration": round(total, 3),
                "analysis_wav": "analysis.wav", "noise_db": noise_db, "format": fmt, "sample_rate": sr,
                "bitrate": bitrate if fmt == "mp3" else None, "max_mb": max_mb, "chunk_sec": chunk_sec,
                "chunks": chunks, "notes": notes, "created": now_iso()}
    write_json(outdir / "manifest.json", manifest)
    for n in notes:
        warn(n)
    info(f"🎧 听音包: {len(chunks)} 块，最大 {max(c['size_mb'] for c in chunks):.2f}MB（上限 {limit:.1f}MB）→ {outdir}")
    return manifest


# ───────────────────────── prepare ─────────────────────────

def merge_videos(files, out: Path, fps_override, force_reencode: bool) -> dict:
    infos = []
    for f in files:
        p = probe(f)
        vs, a_s = vstream(p), astream(p)
        if not vs:
            die(f"{f} 没有视频流")
        if not a_s:
            die(f"{f} 没有音频流，口播素材不能没声音")
        fd = float(p["format"]["duration"])
        infos.append({"path": str(Path(f).resolve()), "p": p, "vs": vs, "as": a_s, "fmt_dur": fd,
                      "vdur": stream_dur(vs, fd), "adur": stream_dur(a_s, fd)})

    def sig(x):
        vs, a_s = x["vs"], x["as"]
        return (vs.get("codec_name"), display_size(vs), vs.get("pix_fmt"), vs.get("r_frame_rate"), rotation(vs),
                a_s.get("codec_name"), a_s.get("sample_rate"), a_s.get("channels"))

    same = all(sig(x) == sig(infos[0]) for x in infos)
    av_ok = all(abs(x["vdur"] - x["adur"]) < 0.1 for x in infos)
    first_vs = infos[0]["vs"]
    fps = pick_fps(first_vs, fps_override)

    if same and av_ok and not force_reencode:
        method = "stream-copy"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as lf:
            for x in infos:
                lf.write("file '" + x["path"].replace("'", "'\\''") + "'\n")
            listfile = lf.name
        try:
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile,
                 "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-movflags", "+faststart", fin(out)])
        finally:
            os.unlink(listfile)
        seg_lens = [x["fmt_dur"] for x in infos]
    else:
        method = "re-encode"
        reason = []
        if not same:
            reason.append("各段编码/分辨率/帧率/音频参数不一致")
        if not av_ok:
            reason.append("有分段音视频时长相差 ≥0.1s，直接拼会累积音画不同步")
        if force_reencode:
            reason.append("--force-reencode")
        info(f"🔧 统一参数重编码合并（{'；'.join(reason)}）")
        W, H = display_size(first_vs)
        W, H = W - W % 2, H - H % 2
        cmd = ["ffmpeg", "-y", "-v", "error"]
        for x in infos:
            cmd += ["-i", fin(x["path"])]
        parts = []
        for i in range(len(infos)):
            parts.append(f"[{i}:v:0]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                         f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps_expr(fps)},format=yuv420p[v{i}];"
                         f"[{i}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
        chain = "".join(f"[v{i}][a{i}]" for i in range(len(infos)))
        fc = ";".join(parts) + f";{chain}concat=n={len(infos)}:v=1:a=1[v][a]"
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium",
                "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                fin(out)]
        subprocess.run([str(c) for c in cmd], check=True)
        seg_lens = [max(x["vdur"], x["adur"]) for x in infos]

    merged_dur = media_duration(out)
    offsets, acc = [], 0.0
    for L in seg_lens:
        offsets.append(round(acc, 3))
        acc += L
    if abs(merged_dur - acc) > 0.5:
        warn(f"合并后时长 {merged_dur:.2f}s 与分段之和 {acc:.2f}s 相差 {merged_dur - acc:+.2f}s，接缝位置按比例校正")
        offsets = [round(o * merged_dur / acc, 3) for o in offsets]
    return {"method": method, "fps": fps,
            "files": [{"path": x["path"], "duration": round(L, 3), "offset": o}
                      for x, L, o in zip(infos, seg_lens, offsets)],
            "seams": offsets[1:], "duration": round(merged_dur, 3)}


def cmd_prepare(a):
    if a.max_mb > MAX_MB_CEILING:
        die(f"--max-mb 不能超过 {MAX_MB_CEILING:g}（多模态模型文件输入上限）")
    work = Path(a.work)
    src_dir, aud_dir = work / "0_源", work / "1_音频"
    src_dir.mkdir(parents=True, exist_ok=True)
    for d in ("2_听写", "3_分析/clips", "4_剪辑", "5_字幕"):
        (work / d).mkdir(parents=True, exist_ok=True)
    videos = [str(Path(v).expanduser()) for v in a.videos]
    for v in videos:
        if not os.path.isfile(v):
            die(f"找不到视频: {v}")
    if len(videos) > 1 and not a.merge:
        die("给了多个视频但没加 --merge。先问用户：是同一期分多次录的（加 --merge 按顺序合并），"
            "还是几条独立视频（每条单独建任务目录分别跑）")

    if len(videos) == 1:
        target = Path(videos[0]).resolve()
        link = src_dir / f"source{target.suffix.lower() or '.mp4'}"
        for old in src_dir.glob("source.*"):
            old.unlink()
        link.symlink_to(target)
        p = probe(target)
        vs = vstream(p)
        if not vs or not astream(p):
            die("源视频必须同时有画面和声音")
        meta = {"mode": "single", "source": str(link.resolve()), "source_link": str(link),
                "files": [{"path": str(target), "duration": round(float(p["format"]["duration"]), 3),
                           "offset": 0.0}], "seams": [], "fps": pick_fps(vs, a.fps), "vfr": is_vfr(vs),
                "merge_method": None}
    else:
        out = src_dir / "merged.mp4"
        info(f"🔗 按给定顺序合并 {len(videos)} 段：")
        for i, v in enumerate(videos, 1):
            info(f"   {i}. {v}")
        m = merge_videos(videos, out, a.fps, a.force_reencode)
        meta = {"mode": "merged", "source": str(out.resolve()), "source_link": str(out),
                "files": m["files"], "seams": m["seams"], "fps": m["fps"], "vfr": False,
                "merge_method": m["method"]}
        info(f"✅ 合并完成（{m['method']}）：{out}，{m['duration']:.2f}s，接缝 {[fmt_t(s) for s in m['seams']]}")

    src = meta["source"]
    p = probe(src)
    vs, a_s = vstream(p), astream(p)
    meta.update({"width": display_size(vs)[0], "height": display_size(vs)[1], "vcodec": vs.get("codec_name"),
                 "pix_fmt": vs.get("pix_fmt"), "color_transfer": vs.get("color_transfer"),
                 "duration": round(float(p["format"]["duration"]), 3), "prepared": now_iso()})
    if vs.get("color_transfer") in HDR_TRC:
        warn("源视频是 HDR（HLG/PQ）；render --stage video 会自动转成 SDR（BT.709）输出，不会发灰")

    master = aud_dir / "master.wav"   # 原采样率无损母带，剪音频用，不发给模型
    aud_dir.mkdir(parents=True, exist_ok=True)
    ch = min(int(a_s.get("channels") or 2), 2)
    asr = int(a_s.get("sample_rate") or 48000)
    run(["ffmpeg", "-y", "-v", "error", "-i", fin(src), "-map", "0:a:0", "-vn", "-ac", str(ch), "-ar", str(asr),
         "-c:a", "pcm_s16le", fin(master)])
    mdur = media_duration(master)
    vdur = stream_dur(vs, meta["duration"])
    if abs(mdur - vdur) > 0.2:
        warn(f"音频 {mdur:.2f}s 与画面 {vdur:.2f}s 相差 {mdur - vdur:+.2f}s；以音频时间轴为准，尾部画面可能缺帧或多余")
    meta["master_wav"] = str(master)
    meta["audio_duration"] = round(mdur, 3)
    write_json(src_dir / "sources.json", meta)

    build_listen_pack(master, aud_dir, chunk_sec=a.chunk_sec, max_mb=a.max_mb, fmt=a.format,
                      bitrate=a.bitrate, sr=a.sr, noise_db=a.noise_db, label="原始音频")
    info("下一步：跑 asr 出逐字时间轴和 readable.txt，再拿着 readable.txt 逐块听 1_音频/chunks/")


# ───────────────────────── 模型 JSON 读取 ─────────────────────────

def parse_time(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if re.fullmatch(r"-?[\d.]+", s):
        return float(s)
    parts = s.split(":")
    try:
        sec = 0.0
        for p in parts:
            sec = sec * 60 + float(p)
        return sec
    except ValueError:
        die(f"看不懂的时间格式: {v!r}")


def load_model_json(p: Path):
    txt = p.read_text(encoding="utf-8").strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
    data = json.loads(txt)
    if isinstance(data, dict):
        for k in ("segments", "lines", "subtitles", "items"):
            if k in data:
                return data[k]
        die(f"{p} 里找不到 segments/lines 数组")
    return data


def nearest_island_edge(t, islands, which: int, tol: float):
    best, bd = None, tol
    for isl in islands:
        d = abs(isl[which] - t)
        if d <= bd:
            best, bd = isl[which], d
    return best


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def load_chunk_segments(pack: Path, listen: Path, kinds_required=True):
    man = read_json(pack / "manifest.json")
    segs, problems = [], []
    for i, ch in enumerate(man["chunks"]):
        jp = listen / f"chunk_{i:03d}.json"
        if not jp.exists():
            die(f"缺少 {jp}（对应 {ch['file']}）。每个听音块都要听完")
        for raw in load_model_json(jp):
            try:
                s, e = parse_time(raw["start"]), parse_time(raw["end"])
            except KeyError:
                die(f"{jp} 有条目缺 start/end: {raw}")
            text = str(raw.get("text", "")).strip()
            if e <= s:
                problems.append(f"chunk_{i:03d} 条目结束不晚于开始：{raw}")
                e = s + 0.05
            if s < -0.2 or e > ch["duration"] + 0.5:
                problems.append(f"chunk_{i:03d} 条目超出块时长 {ch['duration']}s：{raw}")
            s, e = max(0.0, s), min(ch["duration"], e)
            segs.append({"start": round(s + ch["start"], 3), "end": round(e + ch["start"], 3), "text": text,
                         "kind": raw.get("kind", "speech"), "note": raw.get("note", ""),
                         "flags": raw.get("flags", []), "chunk": i})
    segs.sort(key=lambda x: (x["start"], x["end"]))
    return man, segs, problems


# ───────────────────────── merge ─────────────────────────

def cmd_merge(a):
    pack, listen, out = Path(a.pack), Path(a.listen), Path(a.out or a.listen)
    man, segs, problems = load_chunk_segments(pack, listen)
    sil = read_json(pack / "silences.json")
    islands = sil["islands"]
    total = sil["duration"]

    in_silence, snapped_s, deltas_by_chunk = [], 0, {}
    for k, sg in enumerate(segs, 1):
        sg["id"] = f"S{k:03d}"
        ov = sum(overlap(sg["start"], sg["end"], i0, i1) for i0, i1 in islands)
        dur = sg["end"] - sg["start"]
        if sg["kind"] != "noise" and ov < min(0.08, dur * 0.3):
            in_silence.append(sg)
        ns = nearest_island_edge(sg["start"], islands, 0, 0.5)
        ne = nearest_island_edge(sg["end"], islands, 1, 0.5)
        sg["snap"] = {"start": "island" if ns is not None else "model",
                      "end": "island" if ne is not None else "model"}
        if ns is not None:
            deltas_by_chunk.setdefault(sg["chunk"], []).append(sg["start"] - ns)
            sg["start"] = ns
            snapped_s += 1
        if ne is not None and ne > sg["start"]:
            sg["end"] = ne

    for x, y in zip(segs, segs[1:]):
        if y["start"] < x["end"] - 0.05:
            problems.append(f"{x['id']} 与 {y['id']} 时间重叠 {x['end'] - y['start']:.2f}s")

    big = [i for i in islands if i[1] - i[0] >= 0.15]
    uncovered = [i for i in big if not any(overlap(i[0], i[1], s["start"], s["end"]) > 0 for s in segs)]
    coverage = 1 - len(uncovered) / len(big) if big else 1.0
    drift = {}
    for c, ds in deltas_by_chunk.items():
        ds = sorted(ds)
        drift[c] = round(ds[len(ds) // 2], 3)
    bad_chunks = [c for c, d in drift.items() if abs(d) > 0.3]
    silence_ratio = len(in_silence) / max(1, len(segs))
    passed = coverage >= 0.9 and silence_ratio <= 0.05 and not bad_chunks

    stats = {"segments": len(segs), "islands_ge_0.15s": len(big), "island_coverage": round(coverage, 3),
             "segments_in_silence": len(in_silence), "start_snapped_ratio": round(snapped_s / max(1, len(segs)), 3),
             "median_start_offset_by_chunk": drift, "pass": passed}
    seams = []
    if a.sources and Path(a.sources).exists():
        seams = read_json(a.sources).get("seams", [])
    write_json(out / "transcript.json", {"duration": total, "segments": segs,
                                         "uncovered_islands": uncovered, "seams": seams, "stats": stats,
                                         "problems": problems, "created": now_iso()})

    lines = ["# 听写合并稿（整条音频时间轴，秒）。[静 x.xxs] = 静音检测得到的停顿；kind≠speech 的标在方括号里",
             "# 这是分析口误用的工作稿，不是字幕。"]
    seam_q = sorted(seams)
    prev_end = 0.0
    for sg in segs:
        while seam_q and seam_q[0] <= sg["start"] + 0.01:
            lines.append(f"===== 接缝 @ {seam_q.pop(0):.2f}s（下一个视频从这里开始）=====")
        gap = sg["start"] - prev_end
        if gap >= 0.2:
            lines.append(f"      [静 {gap:.2f}s]")
        tag = "" if sg["kind"] == "speech" else f"[{sg['kind']}] "
        flags = f"  ⟨{','.join(sg['flags'])}⟩" if sg.get("flags") else ""
        lines.append(f"{sg['id']} | {sg['start']:.2f}-{sg['end']:.2f} | {tag}{sg['text']}{flags}")
        prev_end = max(prev_end, sg["end"])
    if total - prev_end >= 0.2:
        lines.append(f"      [尾部 {prev_end:.2f}-{total:.2f}，{total - prev_end:.2f}s 无听写]")
    (out / "readable.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rep = [f"# 听写对齐检查  {now_iso()}", "",
           f"- 结论：{'✅ 通过' if passed else '❌ 不通过'}",
           f"- 有声区间覆盖率：{coverage:.1%}（≥90% 才算没漏听）",
           f"- 落在静音里的条目：{len(in_silence)} / {len(segs)}（≤5% 才算时间戳可信）",
           f"- 各块起点中位偏移：{drift}（|偏移|>0.3s 说明该块整体时间轴漂了）", ""]
    if uncovered:
        rep.append("## 有声但没听写的区间（逐个放大复听：是话就补听写，是杂音就在 JSON 里标 kind=noise）")
        rep += [f"- {i[0]:.2f}-{i[1]:.2f}（{i[1] - i[0]:.2f}s）" for i in uncovered]
        rep.append("")
    if in_silence:
        rep.append("## 时间戳落在静音里的条目（时间可能给错了，放大复听重定位）")
        rep += [f"- {s['id']} {s['start']:.2f}-{s['end']:.2f} {s['text']}" for s in in_silence]
        rep.append("")
    if bad_chunks:
        rep.append(f"## 时间轴漂移的块：{bad_chunks} → 用更小的 --chunk-sec 重新 prepare 或单独重听这些块")
        rep.append("")
    if problems:
        rep.append("## 其他问题")
        rep += [f"- {p}" for p in problems]
    (out / "check.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    info("\n".join(rep[:5]))
    info(f"→ {out / 'transcript.json'}、readable.txt、check.md")
    if not passed:
        sys.exit(2)


# ───────────────────────── asr ─────────────────────────

def asr_health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def asr_call(wav: Path, url: str, context: str, timeout: float = 900) -> dict:
    b = "----mmcut" + os.urandom(8).hex()
    parts = [f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
             for k, v in (("language", "zh"), ("context", context), ("return_timestamps", "true"),
                          ("response_format", "verbose_json"))]
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{wav.name}\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode() + wav.read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url.rstrip("/") + "/transcribe", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        die(f"ASR 请求失败（{url}）：{e}")


def asr_chars(res: dict, offset: float):
    """ASR 逐字结果 → [[字, 起, 止, 字后标点]]（整条时间）。标点从带标点的全文里对回来，对不上就不带。"""
    words = res.get("words") or [w for sg in res.get("segments", []) for w in sg.get("words", [])]
    out = []
    for w in words:
        t = re.sub(r"\s+", "", str(w.get("text", w.get("word", ""))))
        if t:
            out.append([t, round(float(w["start"]) + offset, 3), round(float(w["end"]) + offset, 3), ""])
    text = re.sub(r"\s+", "", str(res.get("text", "")))
    i = 0
    for ch in out:
        j = text.find(ch[0], i)
        if j < 0 or j - i > 3:
            continue
        i = j + len(ch[0])
        while i < len(text) and unicodedata.category(text[i]).startswith("P"):
            ch[3] += text[i]
            i += 1
    return out


def drop_hallucinated(chars, gap=0.3):
    """没人说话的地方 ASR 会把热词表"念"出来，特征是一串字时长几乎为 0。
    按停顿分组：≥3 字的组里过半是零时长，或 1–2 字的组全是零时长，整组丢掉。"""
    keep, dropped, grp = [], [], []

    def flush():
        z = sum(1 for c in grp if c[2] - c[1] <= 0.01)
        bad = (len(grp) >= 3 and z * 2 > len(grp)) or (len(grp) <= 2 and z == len(grp))
        (dropped if bad else keep).extend(grp)

    for c in chars:
        if grp and c[1] - grp[-1][2] >= gap:
            flush()
            grp = []
        grp.append(c)
    if grp:
        flush()
    return keep, dropped


def transcribe_span(wav: Path, s: float, e: float, tmp: Path, url: str, context: str):
    part = tmp / f"span_{s:09.3f}.wav"
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", fin(wav),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", fin(part)])
    return drop_hallucinated(asr_chars(asr_call(part, url, context), s))


def chars_text(chars, k0: int, k1: int) -> str:
    return "".join(c[0] + c[3] for c in chars[k0:k1])


def cmd_asr(a):
    pack, out = Path(a.pack), Path(a.out)
    man = read_json(pack / "manifest.json")
    sil = read_json(pack / "silences.json")
    wav, total = pack / man["analysis_wav"], sil["duration"]
    if not asr_health(a.url):
        die(f"ASR 服务连不上：{a.url}/health。停下来告诉用户，等服务恢复或改用备用的纯听写流程；不准换别的 ASR")
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mmcut_asr_", dir=str(out)))
    chars, chunk_of, notes = [], [], []
    try:
        for i, ch in enumerate(man["chunks"]):
            s0, e0 = ch["start"], ch["end"]
            got, bad = transcribe_span(wav, s0, e0, tmp, a.url, a.terms or "")
            # 长文件尾部偶尔丢字：块尾还有成段的声音却没出字，就把尾巴单独再识别一次（不带热词，防止念热词）
            last = got[-1][2] if got else s0
            tail = [x for x in sil["islands"] if x[0] >= last + 0.05 and x[1] <= e0 + 0.01]
            if sum(x[1] - x[0] for x in tail) >= 0.4:
                again, bad2 = transcribe_span(wav, max(s0, tail[0][0] - 0.15), e0, tmp, a.url, "")
                bad += bad2
                again = [c for c in again if c[1] >= last - 0.01]
                if again:
                    notes.append(f"chunk_{i:03d} 尾部 {last:.2f}-{e0:.2f}s 第一次没出字，单独重识别补回 {len(again)} 字："
                                 + "".join(c[0] for c in again))
                got += again
            if bad:
                notes.append(f"chunk_{i:03d} 丢掉疑似幻觉的字（时长≈0）：" + "".join(c[0] for c in bad)
                             + f"（{bad[0][1]:.2f}s 起）")
            chars += got
            chunk_of += [i] * len(got)
            info(f"   ASR chunk_{i:03d}（{s0:.1f}-{e0:.1f}s）→ {len(got)} 字")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not chars:
        die("ASR 一个字都没出，检查音频和服务")
    order = sorted(range(len(chars)), key=lambda k: chars[k][1])
    chars, chunk_of = [chars[k] for k in order], [chunk_of[k] for k in order]
    for x, y in zip(chars, chars[1:]):
        if y[1] < x[2]:
            y[1] = x[2] = round((y[1] + x[2]) / 2, 3)

    groups, cur = [], []
    for k, c in enumerate(chars):
        if cur:
            p = chars[cur[-1]]
            gap = c[1] - p[2]
            if gap >= a.gap or chunk_of[k] != chunk_of[cur[-1]] or (p[3][-1:] in SENT_END and gap >= 0.12):
                groups.append(cur)
                cur = []
        cur.append(k)
    groups.append(cur)
    segs = []
    for n, g in enumerate(groups, 1):
        k0, k1 = g[0], g[-1] + 1
        text = chars_text(chars, k0, k1)
        plain = re.sub(r"[^\w]", "", text)
        segs.append({"id": f"S{n:03d}", "start": chars[k0][1], "end": chars[k1 - 1][2], "text": text,
                     "kind": "filler" if plain and all(x in FILLERS for x in plain) else "speech",
                     "c0": k0, "c1": k1, "chunk": chunk_of[k0], "flags": []})

    starts = [c[1] for c in chars]

    def covered(s, e):
        k = bisect.bisect_left(starts, s - 1.0)
        while k < len(chars) and chars[k][1] <= e + 0.05:
            if chars[k][2] >= s - 0.05:
                return True
            k += 1
        return False
    uncovered = [i for i in sil["islands"] if i[1] - i[0] >= 0.15 and not covered(i[0], i[1])]
    odd = []
    for sg in segs:
        dur = sg["end"] - sg["start"]
        n = len(re.sub(r"[^\w]", "", sg["text"]))
        if dur >= 1.0 and n / dur > 9:
            odd.append(f"{sg['id']} {n} 字 / {dur:.2f}s（>9 字/秒，时间可能被压缩或多出字）：{sg['text']}")
        elif dur >= 2.0 and n / dur < 1.5:
            odd.append(f"{sg['id']} {n} 字 / {dur:.2f}s（<1.5 字/秒，可能漏识别了话）：{sg['text']}")

    seams = read_json(a.sources).get("seams", []) if a.sources and Path(a.sources).exists() else []
    stats = {"chars": len(chars), "segments": len(segs), "uncovered_islands": len(uncovered),
             "chunks": len(man["chunks"])}
    write_json(out / "transcript.json", {"source": "asr", "asr_url": a.url, "terms": a.terms or "",
                                         "duration": total, "chars": chars, "segments": segs,
                                         "uncovered_islands": uncovered, "seams": seams, "stats": stats,
                                         "notes": notes, "created": now_iso()})

    lines = ["# ASR 稿（整条音频时间，秒）。删除候选按 S 编号写；时间只作参考，不用抄。",
             "# [静 x.xxs] = 停顿；[未识别] = 有声音但 ASR 没出字（呼吸、杂音，或 ASR 漏掉的话，听的时候判断）。",
             "# 这是分析口误用的工作稿，不是字幕；专名可能识别错，以口播稿为准。"]
    events = [(sg["start"], 1, sg) for sg in segs] + [(u[0], 0, u) for u in uncovered]
    events += [(s, -1, s) for s in seams]
    events.sort(key=lambda x: (x[0], x[1]))
    ch_bounds = [(c["start"], c["end"]) for c in man["chunks"]]
    cur_ch, prev_end = -1, 0.0
    for t, kind, obj in events:
        ci = next((k for k, (s, e) in enumerate(ch_bounds) if s <= t < e), len(ch_bounds) - 1)
        if ci != cur_ch:
            cur_ch = ci
            s, e = ch_bounds[ci]
            lines.append(f"\n## chunk_{ci:03d}（整条 {s:.2f}-{e:.2f}s；块内时间 = 整条 − {s:.2f}）")
        if kind == -1:
            lines.append(f"===== 接缝 @ {obj:.2f}s（下一个视频从这里开始）=====")
            continue
        if kind == 0:
            lines.append(f"      [未识别 {obj[0]:.2f}-{obj[1]:.2f}（{obj[1] - obj[0]:.2f}s）]")
            prev_end = max(prev_end, obj[1])
            continue
        gap = obj["start"] - prev_end
        if gap >= 0.2:
            lines.append(f"      [静 {gap:.2f}s]")
        tag = "" if obj["kind"] == "speech" else f"[{obj['kind']}] "
        lines.append(f"{obj['id']} | {obj['start']:.2f}-{obj['end']:.2f} | {tag}{obj['text']}")
        prev_end = max(prev_end, obj["end"])
    (out / "readable.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rep = [f"# ASR 检查  {now_iso()}", "", f"- {len(man['chunks'])} 块，{len(chars)} 字，{len(segs)} 句",
           f"- 有声但没出字的区间：{len(uncovered)} 个（多数是呼吸/杂音；≥0.3s 的会变成 U 候选，必须复听）"]
    rep += [f"- {n}" for n in notes]
    if odd:
        rep += ["", "## 字速异常的句子（放大复听，核对 ASR 的字和时间）"] + [f"- {x}" for x in odd]
    (out / "check.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    info("\n".join(rep))
    info(f"→ {out / 'transcript.json'}、readable.txt、check.md")


# ───────────────────────── clip ─────────────────────────

def cmd_clip(a):
    pack = Path(a.pack)
    man = read_json(pack / "manifest.json")
    s = max(0.0, a.start - a.pad)
    e = min(man["duration"], a.end + a.pad)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    codec = ["-c:a", "pcm_s16le"] if out.suffix == ".wav" else ["-c:a", "libmp3lame", "-b:a", "96k"]
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", fin(pack / man["analysis_wav"]),
         *codec, fin(out)])
    info(f"🎧 {out}：片段内 0s = 整条 {s:.3f}s，片段长 {e - s:.2f}s。模型给的片段内时间 + {s:.3f} = 整条时间")


# ───────────────────────── snap ─────────────────────────

def snap_boundary(t, side, fine, wav, pad, noise_db):
    near = [s for s in fine if s[1] >= t - 0.35 and s[0] <= t + 0.35]
    if near:
        s = min(near, key=lambda x: 0 if x[0] <= t <= x[1] else min(abs(x[0] - t), abs(x[1] - t)))
        L = s[1] - s[0]
        if L <= 2 * pad:
            p = (s[0] + s[1]) / 2
        else:
            p = s[0] + pad if side == "start" else s[1] - pad
        return round(p, 3), "silence", None
    v, db = energy_valley(wav, t, 0.2)
    w = None if db <= noise_db + 12 else f"{'起点' if side == 'start' else '终点'}落在连续语音里（谷底 {db}dBFS），放大复听确认"
    return v, "valley", w


def expand_sids(spec, sid2seg):
    """"S012" / ["S012","S014"] / "S012-S015" → 句子列表。"""
    out = []
    for x in (spec if isinstance(spec, list) else [spec]):
        m = re.fullmatch(r"\s*S(\d+)\s*[-~–]\s*S?(\d+)\s*", str(x))
        ids = [f"S{n:03d}" for n in range(int(m.group(1)), int(m.group(2)) + 1)] if m else [str(x).strip()]
        for i in ids:
            if i not in sid2seg:
                die(f"transcript 里没有句子 {i}")
            out.append(sid2seg[i])
    return out


def find_chars(seg, chars, quote: str, nth: int = 1):
    """在句子 seg 的 ASR 字里找第 nth 次出现的 quote（忽略标点空格），返回字下标列表。"""
    want = re.sub(r"[\W_]+", "", quote)
    s, owner = "", []
    for k in range(seg["c0"], seg["c1"]):
        t = re.sub(r"[\W_]+", "", chars[k][0])
        s += t
        owner += [k] * len(t)
    pos, start = -1, 0
    for _ in range(max(1, nth)):
        pos = s.find(want, start)
        if pos < 0:
            die(f"{seg['id']}（{seg['text']}）里找不到第 {nth} 个“{quote}”。按 readable.txt 里的 ASR 原字写")
        start = pos + 1
    hit = sorted(set(owner[pos:pos + len(want)]))
    partial = [chars[k][0] for k in hit if owner.count(k) > sum(1 for j in owner[pos:pos + len(want)] if j == k)]
    return hit, partial


def to_ranges(idx):
    out = []
    for k in sorted(idx):
        if out and k == out[-1][1]:
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1])
    return out


def cmd_snap_asr(a, pack, sil, tr):
    """ASR 模式：候选按句子编号（sids）/ 句内原字（chars）/ 时间（start-end，ASR 没出字的地方才用）写，
    统一解析成字下标范围。真正的切点在 render 时按最终删哪些字、并吸附到真实开口/收音上算出来。"""
    chars, segs, total = tr["chars"], tr["segments"], sil["duration"]
    sid2seg = {s["id"]: s for s in segs}
    raw = read_json(a.candidates) if a.candidates and Path(a.candidates).exists() else []
    raw = raw.get("items", raw) if isinstance(raw, dict) else raw
    items, seen = [], set()
    for c in raw:
        for k in ("id", "type"):
            if k not in c:
                die(f"候选缺字段 {k}: {c}")
        if c["id"] in seen:
            die(f"候选 id 重复: {c['id']}")
        seen.add(c["id"])
        idx, rawt, warns = set(), None, []
        if c.get("sids"):
            for sg in expand_sids(c["sids"], sid2seg):
                idx.update(range(sg["c0"], sg["c1"]))
        for sp in (c["chars"] if isinstance(c.get("chars"), list) else [c["chars"]] if c.get("chars") else []):
            sg = expand_sids(sp["sid"], sid2seg)[0]
            hit, partial = find_chars(sg, chars, sp["del"], int(sp.get("nth", 1)))
            idx.update(hit)
            for t in partial:
                warns.append(f"ASR 把“{t}”识别成一个词，只能整词删；只想删一半就改按时间写并放大复听")
        if "start" in c and "end" in c:
            rawt = [parse_time(c["start"]), parse_time(c["end"])]
            idx.update(k for k, ch in enumerate(chars) if rawt[0] <= (ch[1] + ch[2]) / 2 <= rawt[1])
            warns.append("按时间删（ASR 没给字的地方），切点只吸附到静音/能量谷，必须放大复听")
        if not idx and rawt is None:
            die(f"{c['id']} 没写 sids / chars / start-end，不知道删哪里")
        rng = to_ranges(idx)
        for r0, r1 in rng:
            if r0 > 0 and chars[r0][1] - chars[r0 - 1][2] < 0.1:
                warns.append(f"起点在连续语音里（“{chars[r0 - 1][0]}|{chars[r0][0]}”之间几乎没停顿），放大复听确认没吃字")
            if r1 < len(chars) and chars[r1][1] - chars[r1 - 1][2] < 0.1:
                warns.append(f"终点在连续语音里（“{chars[r1 - 1][0]}|{chars[r1][0]}”之间几乎没停顿），放大复听确认没吃字")
        if rawt:
            s1, _, w1 = snap_boundary(rawt[0], "start", sil["fine"], pack / "analysis.wav", a.pad, sil["noise_db"])
            e1, _, w2 = snap_boundary(rawt[1], "end", sil["fine"], pack / "analysis.wav", a.pad, sil["noise_db"])
            rawt = [s1, e1]
            warns += [w for w in (w1, w2) if w]
        st = min([chars[r0][1] for r0, _ in rng] + ([rawt[0]] if rawt else []))
        en = max([chars[r1 - 1][2] for _, r1 in rng] + ([rawt[1]] if rawt else []))
        text = c.get("text") or " / ".join(chars_text(chars, r0, r1) for r0, r1 in rng)
        items.append({"id": c["id"], "type": c["type"], "risk": c.get("risk", "high"),
                      "delete": bool(c.get("delete", True)), "text": text, "keep": c.get("keep", ""),
                      "reason": c.get("reason", ""), "sids": c.get("sids"), "chars": rng, "raw": rawt,
                      "start": round(st, 3), "end": round(en, 3), "warnings": warns, "auto": False})

    if not a.no_auto:
        first, last = chars[0][1], chars[-1][2]
        g = 0
        for s, e in sil["coarse"]:
            if s < first or e > last or e - s < 0.2:
                continue
            ds, de = round(s + a.pad, 3), round(e - a.pad, 3)
            if de - ds < 0.05:
                continue
            g += 1
            L = e - s
            items.append({"id": f"G{g:03d}", "type": "静音", "risk": "low", "delete": True,
                          "text": f"[静 {L:.2f}s]", "keep": "", "reason": "≤1s 句间卡顿" if L <= 1 else "＞1s 明显停顿",
                          "chars": [], "raw": [ds, de], "start": ds, "end": de, "warnings": [], "auto": True})
        u = 0
        for s, e in tr.get("uncovered_islands", []):
            if first < s < last and e - s >= 0.3:
                u += 1
                items.append({"id": f"U{u:03d}", "type": "未识别有声段", "risk": "high", "delete": False,
                              "text": "", "keep": "", "reason": "有声音但 ASR 没出字：是话就决定删不删，是杂音就随相邻删除一起处理",
                              "chars": [], "raw": [s, e], "start": s, "end": e,
                              "warnings": ["ASR 没出字的有声段，听一下是什么"], "auto": True})
    ids = [x["id"] for x in items]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        die(f"候选 id 与自动候选重复：{dup}。G/H/T/U 开头的编号留给脚本，换个前缀")
    items.sort(key=lambda x: (x["start"], x["end"]))
    for x in items:
        x["overlaps"] = [y["id"] for y in items if y is not x and overlap(x["start"], x["end"], y["start"], y["end"]) > 0]
    write_json(a.out, {"mode": "asr", "transcript": str(Path(a.transcript).resolve()),
                       "analysis_wav": str((pack / "analysis.wav").resolve()), "noise_db": sil["noise_db"],
                       "pad": a.pad, "head_pad": a.head_pad, "tail_pad": a.tail_pad, "duration": total,
                       "items": items, "created": now_iso()})
    by = {}
    for x in items:
        by[x["type"]] = by.get(x["type"], 0) + 1
    info(f"✅ {len(items)} 个候选 → {a.out}")
    info("   " + "，".join(f"{k} {v}" for k, v in by.items()))
    flagged = [x for x in items if x["warnings"] and not x["auto"]]
    if flagged:
        info(f"⚠️  {len(flagged)} 个候选需要放大复听：" + "、".join(x["id"] for x in flagged[:30]))
    us = [x["id"] for x in items if x["id"].startswith("U")]
    if us:
        info(f"🔎 {len(us)} 个 ASR 没出字的有声段（U，默认不删）：夹在删除之间的会被接缝收紧自动带走，"
             f"夹在保留内容中间的要听一下是不是漏识别的话")


def cmd_snap(a):
    pack = Path(a.pack)
    man = read_json(pack / "manifest.json")
    sil = read_json(pack / "silences.json")
    wav = pack / man["analysis_wav"]
    total, fine, noise_db, pad = sil["duration"], sil["fine"], sil["noise_db"], a.pad
    tr = read_json(a.transcript)
    if tr.get("chars"):
        return cmd_snap_asr(a, pack, sil, tr)
    segs = tr["segments"]
    speech = [s for s in segs if s.get("kind", "speech") in ("speech", "filler")]
    raw = read_json(a.candidates) if a.candidates and Path(a.candidates).exists() else []
    raw = raw.get("items", raw) if isinstance(raw, dict) else raw

    items, seen = [], set()
    for c in raw:
        for k in ("id", "type", "start", "end"):
            if k not in c:
                die(f"候选缺字段 {k}: {c}")
        if c["id"] in seen:
            die(f"候选 id 重复: {c['id']}")
        seen.add(c["id"])
        s0, e0 = parse_time(c["start"]), parse_time(c["end"])
        s1, m1, w1 = snap_boundary(s0, "start", fine, wav, pad, noise_db)
        e1, m2, w2 = snap_boundary(e0, "end", fine, wav, pad, noise_db)
        warns = [w for w in (w1, w2) if w]
        for name, o, n in (("起点", s0, s1), ("终点", e0, e1)):
            crossed = sum(overlap(min(o, n), max(o, n), i0, i1) for i0, i1 in sil["islands"])
            if crossed > 0.03:
                warns.append(f"{name}吸附时跨过了 {crossed:.2f}s 有声段（{o:.2f}→{n:.2f}），复听确认没吃字/没漏删")
        if e1 - s1 < 0.05:
            warn(f"{c['id']} 吸附后长度不足 0.05s，丢弃（原 {s0:.2f}-{e0:.2f}）")
            continue
        items.append({"id": c["id"], "type": c["type"], "risk": c.get("risk", "high"),
                      "delete": bool(c.get("delete", True)), "text": c.get("text", ""),
                      "keep": c.get("keep", ""), "reason": c.get("reason", ""), "orig": [s0, e0],
                      "start": s1, "end": e1, "snap": [m1, m2], "warnings": warns, "auto": False})

    if not a.no_auto and speech:
        first, last = speech[0]["start"], max(s["end"] for s in speech)
        if first > 0.3:
            items.append({"id": "H0", "type": "开头静音/杂音", "risk": "low", "delete": True, "text": "",
                          "keep": "", "reason": "第一句话之前", "orig": [0.0, first],
                          "start": 0.0, "end": round(max(0.0, first - pad), 3), "snap": ["edge", "silence"],
                          "warnings": [], "auto": True})
        if total - last > 0.3:
            items.append({"id": "T0", "type": "结尾静音/杂音", "risk": "low", "delete": True, "text": "",
                          "keep": "", "reason": "最后一句话之后（收尾、关录制的动静）", "orig": [last, total],
                          "start": round(last + pad, 3), "end": total, "snap": ["silence", "edge"],
                          "warnings": [], "auto": True})
        g = 0
        for s, e in sil["coarse"]:
            if s < first or e > last or e - s < 0.2:
                continue
            ds, de = round(s + pad, 3), round(e - pad, 3)
            if de - ds < 0.05:
                continue
            g += 1
            L = e - s
            items.append({"id": f"G{g:03d}", "type": "静音", "risk": "low", "delete": True,
                          "text": f"[静 {L:.2f}s]", "keep": "",
                          "reason": "≤1s 句间卡顿" if L <= 1 else "＞1s 明显停顿",
                          "orig": [s, e], "start": ds, "end": de, "snap": ["silence", "silence"],
                          "warnings": [], "auto": True})
        for k, (s, e) in enumerate(tr.get("uncovered_islands", []), 1):
            if first < s < last:
                items.append({"id": f"U{k:03d}", "type": "未听写有声段", "risk": "high", "delete": False,
                              "text": "", "keep": "", "reason": "有声音但听写里没有，复听后再决定",
                              "orig": [s, e], "start": s, "end": e, "snap": ["island", "island"],
                              "warnings": ["没听写的有声段，必须复听"], "auto": True})

    ids = [x["id"] for x in items]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        die(f"候选 id 与自动候选重复：{dup}。G/H/T/U 开头的编号留给脚本，换个前缀")
    items.sort(key=lambda x: (x["start"], x["end"]))
    for x in items:
        x["overlaps"] = [y["id"] for y in items if y is not x and overlap(x["start"], x["end"], y["start"], y["end"]) > 0]
    write_json(a.out, {"pad": pad, "duration": total, "items": items, "created": now_iso()})
    by = {}
    for x in items:
        by[x["type"]] = by.get(x["type"], 0) + 1
    del_sec = sum(x["end"] - x["start"] for x in items if x["delete"])
    info(f"✅ {len(items)} 个候选 → {a.out}")
    info("   " + "，".join(f"{k} {v}" for k, v in by.items()))
    info(f"   标记删除约 {del_sec:.1f}s（重叠部分重复计算）")
    flagged = [x for x in items if x["warnings"]]
    if flagged:
        info(f"⚠️  {len(flagged)} 个候选需要放大复听：" + "、".join(x["id"] for x in flagged[:30]))


# ───────────────────────── render ─────────────────────────

def resolve_selection(cand, include, exclude):
    """最终删除清单 = 候选里 delete=true 的项 + --include − --exclude（用户口头调整时用）。"""
    ids = {c["id"] for c in cand["items"]}
    unknown = [i for i in include + exclude if i not in ids]
    if unknown:
        die(f"候选里没有这些 id：{unknown}")
    sel = [c["id"] for c in cand["items"] if (c["delete"] or c["id"] in include) and c["id"] not in exclude]
    return sel


def build_edl(cand, selected, total, merge_gap):
    items = {c["id"]: c for c in cand["items"]}
    dels = sorted([[items[i]["start"], items[i]["end"], [i]] for i in selected])
    merged = []
    for s, e, ids in dels:
        s, e = max(0.0, s), min(total, e)
        if merged and s <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] += ids
        else:
            merged.append([s, e, list(ids)])
    keeps, cursor = [], 0.0
    for s, e, _ in merged:
        if s - cursor >= 0.04:
            keeps.append([cursor, s])
        cursor = max(cursor, e)
    if total - cursor >= 0.04:
        keeps.append([cursor, total])
    if not keeps:
        die("全删光了，检查 candidates.json 的 delete 标记")
    return merged, keeps


def joint_cut(chars, left, right, r0, r1, wav, thr, head_pad, tail_pad, total):
    """删掉字 r0..r1-1 后，左右保留字之间的接缝收紧：
    左边保留的字收音后留 tail_pad，右边保留的字开口前留 head_pad，中间的静音、呼吸、杂音一起删。
    中间没有停顿（连续语音）就切在能量谷，带警告。"""
    warns = []
    d_s, d_e = chars[r0][1], chars[r1 - 1][2]
    if left is None:
        L = 0.0
    else:
        c = chars[left]
        if d_s - c[2] < 0.1:
            L, db = energy_valley(wav, (c[2] + d_s) / 2, 0.08)
            if db > thr + 12:
                warns.append(f"“{c[0]}|{chars[r0][0]}”之间是连续语音，切在能量谷 {L:.2f}s（{db}dBFS）")
        else:
            off = refine_offset(wav, c[2], thr, c[1], d_s)
            d_on = refine_onset(wav, d_s, thr, off, d_e)
            L = off + min(tail_pad, max(0.0, (d_on - off) / 2))
    if right is None:
        R = total
    else:
        c = chars[right]
        if c[1] - d_e < 0.1:
            R, db = energy_valley(wav, (d_e + c[1]) / 2, 0.08)
            if db > thr + 12:
                warns.append(f"“{chars[r1 - 1][0]}|{c[0]}”之间是连续语音，切在能量谷 {R:.2f}s（{db}dBFS）")
        else:
            on = refine_onset(wav, c[1], thr, d_e, c[2])
            d_off = refine_offset(wav, d_e, thr, d_s, on)
            R = on - min(head_pad, max(0.0, (on - d_off) / 2))
    return round(L, 3), round(R, 3), warns


def build_edl_asr(cand, selected, total, merge_gap, long_gap, protect):
    """ASR 模式：按最终要删的字算接缝。开头第一句前、结尾最后一句后总是删掉（H0/T0）。
    两句保留的话之间停了 ≥ long_gap 秒（中间是呼吸、杂音、没识别出的嘀咕），也按接缝收紧（P），
    除非那里有 protect 里的 U 段（模型听出来是 ASR 漏掉的话），那就绕开它。"""
    tr = read_json(cand["transcript"])
    chars, wav, thr = tr["chars"], Path(cand["analysis_wav"]), cand["noise_db"]
    hp, tp = cand["head_pad"], cand["tail_pad"]
    items = {c["id"]: c for c in cand["items"]}
    dead, owner, ivals, warns = set(), {}, [], []
    for i in selected:
        it = items[i]
        for r0, r1 in it.get("chars") or []:
            for k in range(r0, r1):
                dead.add(k)
                owner.setdefault(k, []).append(i)
        if it.get("raw"):
            ivals.append([it["raw"][0], it["raw"][1], [i]])
    for u in protect:
        if u not in items or not u.startswith("U"):
            die(f"--protect 只接受 U 开头的候选 id，{u} 不对")
    kept_u = [items[u]["raw"] for u in protect]
    p = 0
    for k in range(len(chars) - 1):
        if k in dead or k + 1 in dead or chars[k + 1][1] - chars[k][2] < long_gap:
            continue
        a_, b_ = chars[k], chars[k + 1]
        anchors = [(a_[1], refine_offset(wav, a_[2], thr, a_[1], b_[1]))]
        anchors += [(s, e) for s, e in kept_u if a_[2] < s and e < b_[1]]
        anchors.append((refine_onset(wav, b_[1], thr, a_[2], b_[2]), b_[2]))
        for (_, x_end), (y_start, _) in zip(anchors, anchors[1:]):
            L, R = x_end + tp, y_start - hp
            if R - L > 0.04:
                p += 1
                ivals.append([round(L, 3), round(R, 3), [f"P{p:03d}"]])
    alive = [k for k in range(len(chars)) if k not in dead]
    if not alive:
        die("全删光了，检查 candidates.json 的 delete 标记")
    runs = to_ranges(dead)
    for r0, r1 in runs:
        left = r0 - 1 if r0 > 0 else None
        right = r1 if r1 < len(chars) else None
        L, R, w = joint_cut(chars, left, right, r0, r1, wav, thr, hp, tp, total)
        ids = sorted({x for k in range(r0, r1) for x in owner.get(k, [])})
        warns += [f"{'/'.join(ids)}：{x}" for x in w]
        if R > L:
            ivals.append([L, R, ids])
    if alive[0] == 0:   # 第一个字没被删，开头静音单独切；被删了的话上面的删除段已经从 0 开始
        on = refine_onset(wav, chars[alive[0]][1], thr, 0.0, chars[alive[0]][2])
        if on - hp > 0.04:
            ivals.append([0.0, round(on - hp, 3), ["H0"]])
    if alive[-1] == len(chars) - 1:
        c = chars[alive[-1]]
        off = refine_offset(wav, c[2], thr, c[1], total)
        if total - (off + tp) > 0.04:
            ivals.append([round(off + tp, 3), total, ["T0"]])
    merged = []
    for s, e, ids in sorted(ivals):
        s, e = max(0.0, s), min(total, e)
        if e <= s:
            continue
        if merged and s <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] += [x for x in ids if x not in merged[-1][2]]
        else:
            merged.append([s, e, list(ids)])
    keeps, cursor = [], 0.0
    for s, e, _ in merged:
        if s - cursor >= 0.04:
            keeps.append([cursor, s])
        cursor = max(cursor, e)
    if total - cursor >= 0.04:
        keeps.append([cursor, total])
    return merged, keeps, warns, chars


def keep_text(chars, s, e):
    return "".join(c[0] + c[3] for c in chars if s <= (c[1] + c[2]) / 2 <= e)


def cut_wav(src: Path, dst: Path, keeps, fade=0.008):
    out = []
    with wave.open(str(src), "rb") as r, wave.open(str(dst), "wb") as w:
        nch, sw, sr, n = r.getnchannels(), r.getsampwidth(), r.getframerate(), r.getnframes()
        if sw != 2:
            die("master.wav 不是 16bit PCM")
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        cut_cursor = 0
        for s, e in keeps:
            a, b = max(0, round(s * sr)), min(n, round(e * sr))
            r.setpos(a)
            buf = array.array("h")
            buf.frombytes(r.readframes(b - a))
            if sys.byteorder == "big":
                buf.byteswap()
            frames = len(buf) // nch
            nf = min(int(fade * sr), frames // 2)
            for i in range(nf):
                g = i / nf
                for c in range(nch):
                    buf[i * nch + c] = int(buf[i * nch + c] * g)
                    j = (frames - 1 - i) * nch + c
                    buf[j] = int(buf[j] * g)
            if sys.byteorder == "big":
                buf.byteswap()
            w.writeframes(buf.tobytes())
            out.append({"src_start": round(a / sr, 6), "src_end": round(b / sr, 6),
                        "cut_start": round(cut_cursor / sr, 6), "cut_end": round((cut_cursor + frames) / sr, 6),
                        "samples": frames})
            cut_cursor += frames
    return out, sr


def cmd_render(a):
    work = Path(a.work)
    vdir = work / "4_剪辑" / a.version
    vdir.mkdir(parents=True, exist_ok=True)
    src = read_json(work / "0_源" / "sources.json")
    edl_path = vdir / "edl.json"

    if a.stage == "audio":
        cand = read_json(work / "3_分析" / "candidates.snapped.json")
        include = [x for x in (a.include or "").split(",") if x]
        exclude = [x for x in (a.exclude or "").split(",") if x]
        selected = resolve_selection(cand, include, exclude)
        master = Path(src["master_wav"])
        total = media_duration(master)
        warns, chars = [], None
        protect = [x for x in (a.protect or "").split(",") if x]
        if cand.get("mode") == "asr":
            merged, keeps, warns, chars = build_edl_asr(cand, selected, total, a.merge_gap, a.long_gap, protect)
        else:
            merged, keeps = build_edl(cand, selected, total, a.merge_gap)
        segs, sr = cut_wav(master, vdir / "cut_master.wav", keeps)
        cut_total = segs[-1]["cut_end"]
        id2item = {c["id"]: c for c in cand["items"]}
        id2item.update({"H0": {"type": "开头静音/杂音"}, "T0": {"type": "结尾静音/杂音"}})
        id2item.update({i: {"type": "长停顿（含呼吸杂音）"} for d in merged for i in d[2] if i.startswith("P")})
        if chars:
            for k, x in enumerate(segs, 1):
                x["id"] = f"K{k:02d}"
                x["text"] = keep_text(chars, x["src_start"], x["src_end"])
        splices = []
        for k, (x, y) in enumerate(zip(segs, segs[1:]), 1):
            gone = [d for d in merged if d[0] >= x["src_end"] - 0.05 and d[1] <= y["src_start"] + 0.05]
            ids = [i for d in gone for i in d[2]]
            jump = y["src_start"] - x["src_end"]
            splices.append({"n": k, "cut_t": round(x["cut_end"], 3), "src_from": round(x["src_end"], 3),
                            "src_to": round(y["src_start"], 3), "src_jump": round(jump, 3),
                            "cross_take": jump >= a.cross_take, "deleted_ids": ids,
                            "deleted_types": sorted({id2item[i]["type"] for i in ids if i in id2item})})
        # 拼接链：连续 ≥2 个接缝都跳过 ≥8s 原片，说明这段话是从几遍不同录制里拼出来的
        chains, cur = [], []
        for sp in splices:
            if sp["src_jump"] >= 8:
                cur.append(sp)
            else:
                if len(cur) >= 2:
                    chains.append(cur)
                cur = []
        if len(cur) >= 2:
            chains.append(cur)
        chain_info = [{"splices": [sp["n"] for sp in ch], "pieces": len(ch) + 1,
                       "src_span": round(ch[-1]["src_to"] - ch[0]["src_from"], 1)} for ch in chains]
        edl = {"version": a.version, "source": src["source"], "master_wav": str(master), "sample_rate": sr,
               "mode": cand.get("mode", "listen"), "include": include, "exclude": exclude, "selected": selected,
               "merge_gap": a.merge_gap,
               "deletions": [{"start": round(s, 3), "end": round(e, 3), "ids": ids} for s, e, ids in merged],
               "keeps": segs, "splices": splices, "chains": chain_info, "joint_warnings": warns,
               "source_duration": round(total, 3), "cut_duration": round(cut_total, 3), "speed": 1.0,
               "stage": "audio", "created": now_iso()}
        write_json(edl_path, edl)
        if chars:
            jumps = {sp["n"]: sp for sp in splices}
            lines = [f"# {a.version} 保留稿：成片里的话按播放顺序。K = 保留段；≡ 接缝（原片跳过多少秒）",
                     "# 用户说“把 XX 那句删掉”时，在这里按原话找到 K 段，再对回 readable.txt 的 S 编号"]
            for k, x in enumerate(segs, 1):
                lines.append(f"{x['id']} | 成片 {x['cut_start']:.2f}-{x['cut_end']:.2f} | 原片 {x['src_start']:.2f}-"
                             f"{x['src_end']:.2f} | {x['text']}")
                sp = jumps.get(k)
                if sp:
                    mark = "⚠️ 跨 take" if sp["cross_take"] else ""
                    lines.append(f"   ≡ #{sp['n']:03d} 跳过原片 {sp['src_jump']:.2f}s {mark}")
            (vdir / "保留稿.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        pack = build_listen_pack(vdir / "cut_master.wav", vdir / "听音包", chunk_sec=a.chunk_sec, max_mb=a.max_mb,
                                 fmt=a.format, bitrate=a.bitrate, sr=a.sr,
                                 noise_db=read_json(work / "1_音频" / "silences.json")["noise_db"],
                                 label=f"剪后音频 {a.version}", island_min=0.1)
        with open(vdir / "splices.txt", "w", encoding="utf-8") as f:
            f.write(f"# {a.version} 剪后音频的接缝。整条 = 剪后整条时间；块内 = 该听音块里的时间。验收时逐个听前后 2 秒\n")
            for ch in pack["chunks"]:
                f.write(f"\n## {ch['file']}（剪后 {ch['start']:.2f}-{ch['end']:.2f}s）\n")
                for sp in splices:
                    if not (ch["start"] <= sp["cut_t"] < ch["end"]):
                        continue
                    main = [i for i in sp["deleted_ids"] if i[:1] not in "GP"]
                    n_sil = len(sp["deleted_ids"]) - len(main)
                    texts = [id2item[i].get("text", "") for i in main if i in id2item and id2item[i].get("text")]
                    mark = f"  ⚠️ 跨 take（原片跳过 {sp['src_jump']:.1f}s，专听语气/音量接不接得上）" if sp["cross_take"] else ""
                    f.write(f"#{sp['n']:03d}  块内 {sp['cut_t'] - ch['start']:.2f}s（整条 {sp['cut_t']:.2f}s）  "
                            f"删了 {'/'.join(sp['deleted_types']) or '?'}（{', '.join(main)}"
                            f"{'，' if main and n_sil else ''}{f'静音/停顿 {n_sil} 处' if n_sil else ''}）  "
                            f"{' | '.join(texts)[:80]}{mark}\n")
            if chain_info:
                f.write("\n## 拼接链（一段话由几遍不同录制拼成，最容易听着碎）\n")
                for c in chain_info:
                    f.write(f"- 接缝 {c['splices']}：{c['pieces']} 段拼成，原片横跨 {c['src_span']}s\n")
            if warns:
                f.write("\n## 切在连续语音里的接缝（放大复听有没有吞字）\n" + "".join(f"- {w}\n" for w in warns))
        info(f"✂️  音频剪完：{total:.2f}s → {cut_total:.2f}s（删 {total - cut_total:.2f}s，{len(splices)} 个接缝，"
             f"跨 take {sum(sp['cross_take'] for sp in splices)} 个，拼接链 {len(chain_info)} 条）")
        info(f"   {vdir / 'cut_master.wav'}；验收听 {vdir / '听音包/chunks'}，接缝清单 {vdir / 'splices.txt'}")
        info("下一步：模型复听验收接缝。通过后才能 --stage video 拼画面")
        return

    # ---- stage video：按音频时间轴拼画面 ----
    if not edl_path.exists():
        die(f"没有 {edl_path}，先跑 --stage audio 并完成复听验收")
    edl = read_json(edl_path)
    if not a.accepted and not edl.get("audio_accepted"):
        die("剪后音频还没验收。模型复听接缝没问题后，加 --accepted 再拼画面（会记进 edl.json）")
    edl["audio_accepted"] = edl.get("audio_accepted") or now_iso()
    inp = Path(edl["source"])
    p = probe(inp)
    vs = vstream(p)
    fps = pick_fps(vs, a.fps or src.get("fps"))
    br = vs.get("bit_rate") or p["format"].get("bit_rate") or "6000000"
    bitrate_k = int(br) // 1000 if str(br).isdigit() else 6000
    if not vs.get("bit_rate") and p["format"].get("bit_rate"):
        bitrate_k = max(1000, bitrate_k - 192)
    prof = {"high": "high", "main": "main", "baseline": "baseline"}.get(str(vs.get("profile", "high")).lower(), "high")
    speed = a.speed or 1.0
    if not 0.5 <= speed <= 2.0:
        die("--speed 只支持 0.5–2.0")
    hdr = vs.get("color_transfer") in HDR_TRC and not a.keep_hdr
    info(f"📹 源：{inp.name}，目标 {fps_expr(fps)}fps CFR，{bitrate_k}kbps，倍速 {speed:g}x"
         + ("，HDR → 先出 10bit 中间片再转 SDR" if hdr else f"，profile={prof}，yuv420p"))

    # 画面跟着音频走：成片第 f 帧显示的是"音频此刻对应的源时间"那一帧（误差 ≤ 半帧，不累积）。
    # 倍速时成片时间 t 对应剪后音频 t·speed。
    jobs, cum_frames = [], 0
    for i, k in enumerate(edl["keeps"]):
        f0, f1 = cum_frames, round(k["cut_end"] / speed * fps)
        n = f1 - f0
        cum_frames = f1
        if n > 0:
            ss = k["src_start"] + (f0 / fps * speed - k["cut_start"]) - 0.5 * speed / fps
            jobs.append((i, max(0.0, ss), n))
    vf = (f"setpts=(PTS-STARTPTS)/{speed:g}," if speed != 1.0 else "") + f"fps={fps_expr(fps)}"
    if hdr:
        color = [x for k_, v in (("-color_primaries", vs.get("color_primaries")), ("-color_trc", vs.get("color_transfer")),
                                 ("-colorspace", vs.get("color_space"))) if v for x in (k_, v)]
        venc = ["-vf", vf + ",format=p010le", "-c:v", "hevc_videotoolbox", "-profile:v", "main10",
                "-b:v", f"{bitrate_k}k", "-tag:v", "hvc1", *color]
    else:
        venc = ["-vf", vf + ",format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-profile:v", prof,
                "-b:v", f"{bitrate_k}k", "-maxrate", f"{bitrate_k * 13 // 10}k", "-bufsize", f"{bitrate_k * 2}k"]
    tmp = Path(tempfile.mkdtemp(prefix="mmcut_", dir=str(vdir)))
    try:
        def enc(job):
            i, s, n = job
            out = tmp / f"seg_{i:05d}.mp4"
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.6f}", "-i", fin(inp), "-an", *venc,
                 "-frames:v", str(n), "-r", fps_expr(fps), fin(out)])
            return out
        workers = 4 if hdr else (os.cpu_count() or 4)
        info(f"🎞️  按音频时间轴逐段取画面：{len(jobs)} 段（并行 {workers}）")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(enc, j) for j in jobs]):
                fut.result()
                done += 1
                print(f"\r   {done}/{len(jobs)}", end="", flush=True)
        print()
        listfile = tmp / "concat.txt"
        seg_files = [str((tmp / f"seg_{i:05d}.mp4").resolve()).replace("'", "'\\''") for i, _, _ in jobs]
        listfile.write_text("".join(f"file '{x}'\n" for x in seg_files), encoding="utf-8")
        vonly = tmp / "video_only.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", fin(vonly)])
        if hdr:
            info("🎨 HDR → SDR（AVFoundation 转 BT.709）")
            sdr = tmp / "video_sdr.mp4"
            r = subprocess.run([str(hdr2sdr_bin()), str(vonly), str(sdr)], capture_output=True, text=True)
            if r.returncode != 0 or not sdr.exists():
                die(f"HDR 转 SDR 失败：{r.stdout[-500:]}{r.stderr[-500:]}。可加 --keep-hdr 先出 HDR 版")
            vonly = sdr
        out = vdir / "cut.mp4"
        af = ["-af", f"atempo={speed:g}"] if speed != 1.0 else []   # 保调变速，单级 atempo 支持 0.5–2.0
        run(["ffmpeg", "-y", "-v", "error", "-i", fin(vonly), "-i", fin(vdir / "cut_master.wav"),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *af, "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", fin(out)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    q = probe(out)
    vd = stream_dur(vstream(q), 0.0)
    ad = stream_dur(astream(q), 0.0)
    target = edl["cut_duration"] / speed
    ok = abs(vd - target) <= 1.5 / fps and abs(ad - target) <= 0.05
    edl.update({"stage": "video", "video": str(out), "fps": fps, "speed": speed,
                "output_duration": round(target, 3), "hdr_to_sdr": hdr,
                "color_transfer_out": vstream(q).get("color_transfer"), "video_duration": round(vd, 3),
                "audio_duration_in_mp4": round(ad, 3), "sync_ok": ok, "rendered": now_iso()})
    write_json(edl_path, edl)
    info(f"✅ {out}")
    info(f"   成片应为 {target:.3f}s（剪后音频 {edl['cut_duration']:.3f}s ÷ {speed:g}）｜画面 {vd:.3f}s｜成片音轨 {ad:.3f}s"
         f" → {'同步 OK' if ok else '⚠️ 时长不一致，检查源视频尾部'}")
    if speed != 1.0:
        info(f"   edl.json 的 keeps 时间仍是 1.0x 剪后音频时间；成片时间 = 剪后时间 ÷ {speed:g}，srt 会自动换算")


HDR2SDR_SWIFT = r'''
import AVFoundation
let args = CommandLine.arguments
let src = URL(fileURLWithPath: args[1]), dst = URL(fileURLWithPath: args[2])
try? FileManager.default.removeItem(at: dst)
let asset = AVURLAsset(url: src)
let sema = DispatchSemaphore(value: 0)
var ok = false
Task {
  do {
    let comp = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset)
    comp.colorPrimaries = AVVideoColorPrimaries_ITU_R_709_2
    comp.colorTransferFunction = AVVideoTransferFunction_ITU_R_709_2
    comp.colorYCbCrMatrix = AVVideoYCbCrMatrix_ITU_R_709_2
    guard let ex = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else { print("no session"); sema.signal(); return }
    ex.videoComposition = comp
    ex.outputURL = dst; ex.outputFileType = .mp4
    await ex.export()
    ok = ex.status == .completed
    print("status", ex.status.rawValue, ex.error?.localizedDescription ?? "")
  } catch { print("err", error) }
  sema.signal()
}
sema.wait()
exit(ok ? 0 : 1)
'''


def hdr2sdr_bin() -> Path:
    """本机 ffmpeg 没有 zscale/libplacebo 做不了正确的 HLG 色调映射，用 AVFoundation 导出转 BT.709。首次用时编译。"""
    d = Path.home() / ".cache" / "mmcut"
    exe, src = d / "hdr2sdr", d / "hdr2sdr.swift"
    if exe.exists() and src.exists() and src.read_text(encoding="utf-8") == HDR2SDR_SWIFT:
        return exe
    if shutil.which("swiftc") is None:
        die("HDR 转 SDR 需要 swiftc（Xcode 命令行工具）。没有的话加 --keep-hdr 先出 HDR 版，并告诉用户")
    d.mkdir(parents=True, exist_ok=True)
    src.write_text(HDR2SDR_SWIFT, encoding="utf-8")
    r = subprocess.run(["swiftc", "-O", str(src), "-o", str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"编译 hdr2sdr 失败：{r.stderr[-1500:]}")
    return exe


# ───────────────────────── srt ─────────────────────────

def srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def subtitle_tokens(chars):
    """字 → 字幕最小单位：连续的英文字母/数字并成一个词，不拆到两行；语气词去掉。"""
    toks = []
    for c in chars:
        t, s, e, p = c
        if t in FILLERS:
            continue
        word = bool(re.fullmatch(r"[A-Za-z0-9.+#'-]+", t))
        if toks and word and toks[-1]["word"] and s - toks[-1]["e"] < 0.12 and not toks[-1]["p"]:
            toks[-1].update(t=toks[-1]["t"] + t, e=e, p=p)
        else:
            toks.append({"t": t, "s": s, "e": e, "p": p, "word": word})
    return toks


def build_sub_lines(chars, max_chars: int, gap: float):
    """断行：先按标点和真停顿（≥gap）切成意群；超长的意群在中间附近停顿最长的字缝处对半切
    （停顿多落在词与词之间，不容易把词拆开）；≤3 字的碎片并到相邻行。"""
    toks = subtitle_tokens(chars)

    def L(ts):
        return sum(len(x["t"]) for x in ts)

    phrases, cur = [], []
    for t in toks:
        if cur and t["s"] - cur[-1]["e"] >= gap:
            phrases.append(cur)
            cur = []
        cur.append(t)
        if any(ch in "，,。.！!？?；;：:、" for ch in t["p"]):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)

    def split(ph):
        n = L(ph)
        if n <= max_chars or len(ph) < 2:
            return [ph]
        best = None
        for i in range(1, len(ph)):
            left = L(ph[:i])
            if min(left, n - left) < max(3, n * 0.3):
                continue
            score = (ph[i]["s"] - ph[i - 1]["e"]) - abs(left - n / 2) * 0.01
            if best is None or score > best[0]:
                best = (score, i)
        i = best[1] if best else len(ph) // 2
        return split(ph[:i]) + split(ph[i:])

    out = []
    for ph in (x for p_ in phrases for x in split(p_)):
        prev = out[-1] if out else None
        if prev and min(L(prev), L(ph)) <= 3 and L(prev) + L(ph) <= max_chars \
                and ph[0]["s"] - prev[-1]["e"] < gap and prev[-1]["p"][-1:] not in SENT_END:
            out[-1] = prev + ph
        else:
            out.append(ph)
    return [{"start": ts[0]["s"], "end": ts[-1]["e"], "text": join_tokens(ts)} for ts in out]


def join_tokens(ts) -> str:
    """英文词和中文之间加空格以外，标点一律不上字幕（引号书名号保留）。"""
    out = ""
    for x in ts:
        if out and x["word"] and re.search(r"[A-Za-z0-9]$", out):
            out += " "
        out += x["t"]
        out += "".join(ch for ch in x["p"] if ch in "“”‘’《》\"")
    return out.strip()


def cmd_srt_asr(a):
    tr = read_json(a.transcript)
    out = Path(a.out or Path(a.transcript).parent)
    total = tr["duration"]
    speed = a.speed
    if speed is None and a.edl and Path(a.edl).exists():
        speed = read_json(a.edl).get("speed") or 1.0
    speed = speed or 1.0
    lines = build_sub_lines(tr["chars"], a.max_chars, a.gap)
    for x, y in zip(lines, lines[1:]):
        x["end"] = y["start"] if y["start"] - x["end"] < 0.3 else x["end"] + 0.15
    if lines:
        lines[-1]["end"] = min(total, lines[-1]["end"] + 0.3)
    for i, x in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else total
        if x["end"] - x["start"] < a.min_dur:
            x["end"] = min(nxt, x["start"] + a.min_dur)
        x["n"] = i + 1
    out.mkdir(parents=True, exist_ok=True)
    draft = ["# 字幕草稿（时间来自剪后音频的 ASR，已定死；只校文字）。剪后音频时间，1.0x。",
             "# 校字结果写 fixes.json：{\"fixes\":[{\"n\":12,\"text\":\"改后的整行\"}]}，只列要改的行"]
    draft += [f"{x['n']:03d} | {x['start']:.2f}-{x['end']:.2f} | {x['text']}" for x in lines]
    (out / "lines.txt").write_text("\n".join(draft) + "\n", encoding="utf-8")
    fixed = 0
    if a.fixes and Path(a.fixes).exists():
        fx = read_json(a.fixes)
        for f_ in (fx.get("fixes", fx) if isinstance(fx, dict) else fx):
            n = int(f_["n"])
            if not 1 <= n <= len(lines):
                die(f"fixes 里第 {n} 行不存在（共 {len(lines)} 行）")
            lines[n - 1]["text"] = str(f_["text"]).strip()
            fixed += 1
    lines = [x for x in lines if x["text"]]
    issues = []
    for x in lines:
        plain = re.sub(r"\s+", "", x["text"])
        if len(plain) > a.max_chars:
            issues.append(f"第 {x['n']} 行 {len(plain)} 字，超过 {a.max_chars}：{x['text']}")
        if x["end"] - x["start"] < 0.3:
            issues.append(f"第 {x['n']} 行只有 {x['end'] - x['start']:.2f}s，太短")
    unc = [u for u in tr.get("uncovered_islands", []) if u[1] - u[0] >= 0.3]
    body = "".join(f"{i}\n{srt_ts(x['start'] / speed)} --> {srt_ts(x['end'] / speed)}\n{x['text']}\n\n"
                   for i, x in enumerate(lines, 1))
    (out / "video.srt").write_text(body, encoding="utf-8")
    rep = [f"# 字幕检查  {now_iso()}", "",
           f"- 共 {len(lines)} 行；时间来自剪后音频 ASR 逐字时间轴；校字 {fixed} 行；倍速 {speed:g}x（srt 已换算成成片时间）",
           f"- 剪后音频里有声但 ASR 没出字的区间（≥0.3s）：{len(unc)} 个，听一下是不是漏了字幕"]
    rep += [f"  - {s:.2f}-{e:.2f}" for s, e in unc]
    if issues:
        rep += ["", "## 需要处理"] + [f"- {x}" for x in issues]
    (out / "srt_check.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    info(f"✅ {out / 'video.srt'}（{len(lines)} 行，校字 {fixed} 行，{speed:g}x）；草稿 {out / 'lines.txt'}；检查 {out / 'srt_check.md'}")
    if issues:
        warn(f"{len(issues)} 条问题，看 srt_check.md")


def cmd_srt(a):
    if a.transcript:
        return cmd_srt_asr(a)
    if not (a.pack and a.listen):
        die("ASR 模式给 --transcript；备用模式给 --pack 和 --listen")
    pack, listen, out = Path(a.pack), Path(a.listen), Path(a.out or a.listen)
    man, lines, problems = load_chunk_segments(pack, listen)
    sil = read_json(pack / "silences.json")
    islands, total = sil["islands"], sil["duration"]
    lines = [x for x in lines if x["text"] and x.get("kind", "speech") != "noise"]
    for x in lines:
        ns = nearest_island_edge(x["start"], islands, 0, 0.3)
        ne = nearest_island_edge(x["end"], islands, 1, 0.3)
        if ns is not None:
            x["start"] = ns
        if ne is not None and ne > x["start"]:
            x["end"] = ne
    lines.sort(key=lambda x: x["start"])
    for x, y in zip(lines, lines[1:]):
        if x["end"] > y["start"]:
            x["end"] = y["start"]
    for i, x in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else total
        if x["end"] - x["start"] < a.min_dur:
            x["end"] = min(nxt, x["start"] + a.min_dur)
        x["end"] = min(x["end"], total)
    issues = list(problems)
    for i, x in enumerate(lines, 1):
        plain = re.sub(r"\s+", "", x["text"])
        if len(plain) > a.max_chars:
            issues.append(f"第 {i} 条 {len(plain)} 字，超过 {a.max_chars}：{x['text']}")
        if plain and all(ch in FILLERS for ch in plain):
            issues.append(f"第 {i} 条只有语气词，字幕里应去掉：{x['text']}")
        if x["end"] - x["start"] < 0.3:
            issues.append(f"第 {i} 条只有 {x['end'] - x['start']:.2f}s，太短")
    uncovered = [isl for isl in islands if isl[1] - isl[0] >= 0.3 and
                 not any(overlap(isl[0], isl[1], x["start"], x["end"]) > 0 for x in lines)]
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{i}\n{srt_ts(x['start'])} --> {srt_ts(x['end'])}\n{x['text']}\n\n" for i, x in enumerate(lines, 1))
    (out / "video.srt").write_text(body, encoding="utf-8")
    rep = [f"# 字幕检查  {now_iso()}", "", f"- 共 {len(lines)} 条，音频 {total:.2f}s",
           f"- 没有字幕覆盖的有声区间：{len(uncovered)} 个"]
    rep += [f"  - {s:.2f}-{e:.2f}" for s, e in uncovered]
    if issues:
        rep += ["", "## 需要处理"] + [f"- {x}" for x in issues]
    (out / "srt_check.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    info(f"✅ {out / 'video.srt'}（{len(lines)} 条）；检查 → {out / 'srt_check.md'}")
    if uncovered or issues:
        warn(f"{len(uncovered)} 个有声区间没字幕，{len(issues)} 条问题，看 srt_check.md")


# ───────────────────────── finalize ─────────────────────────

def cmd_finalize(a):
    work = Path(a.work)
    vdir = work / "4_剪辑" / a.version
    video, srt = vdir / "cut.mp4", work / "5_字幕" / a.version / "video.srt"
    for f in (video, srt):
        if not f.exists():
            die(f"缺少 {f}")
    edl = read_json(vdir / "edl.json")
    if not edl.get("sync_ok"):
        warn("edl.json 里 sync_ok 不是 true，确认音画同步后再交付")
    dest = Path(a.dest)
    if "output" not in dest.parts:
        warn(f"{dest} 不在 output/ 下；工作台规定定稿交付物放 output/<YYYY-MM-DD_任务名>/")
    if (dest / "source_cut.mp4").exists() and not a.overwrite:
        die(f"{dest} 已有定稿文件；确认要覆盖加 --overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, dest / "source_cut.mp4")
    shutil.copy2(srt, dest / "subtitles.srt")

    src = read_json(work / "0_源" / "sources.json")
    cand = read_json(work / "3_分析" / "candidates.snapped.json")
    id2 = {c["id"]: c for c in cand["items"]}
    by = {}
    for i in edl["selected"]:
        t = id2.get(i, {}).get("type", "?")
        by[t] = by.get(t, 0) + 1
    vd = float(probe(dest / "source_cut.mp4")["format"]["duration"])
    speed = edl.get("speed") or 1.0
    asr_mode = edl.get("mode") == "asr"
    hand = {
        "skill": "mm_cut", "finalized_at": now_iso(), "version": a.version,
        "deliverables": {"video": "source_cut.mp4", "subtitles": "subtitles.srt"},
        "video_duration": round(vd, 3), "fps": edl.get("fps"), "size": [src.get("width"), src.get("height")],
        "source": {"mode": src["mode"], "files": src["files"], "seams": src["seams"],
                   "merge_method": src.get("merge_method")},
        "cut": {"source_duration": edl["source_duration"], "cut_duration": edl["cut_duration"],
                "deleted_by_type": by, "splices": len(edl["splices"]),
                "cross_take_splices": sum(1 for s in edl["splices"] if s.get("cross_take"))},
        "speed": speed, "hdr_to_sdr": edl.get("hdr_to_sdr", False),
        "timeline_truth": "音频时间轴。画面按音频剪辑表逐段取帧拼接，字幕时间轴 = 成片时间轴"
                          + (f"（成片已 {speed:g} 倍速；edl.json 的 keeps 是 1.0x 剪后时间，÷{speed:g} = 成片时间）"
                             if speed != 1.0 else ""),
        "subtitles_method": ("剪后音频送内网 Qwen3-ASR 得逐字时间轴，多模态模型复听校字（时间不改）"
                             if asr_mode else "具备原生音频理解的多模态模型复听剪后音频直接产出"),
        "script": a.script, "terms": [t.strip() for t in (a.terms or "").split(",") if t.strip()],
        "notes": a.notes or "",
        "next": "封装：由另一个 agent 以本目录 source_cut.mp4 + subtitles.srt 为输入继续（分镜、包装、导出）",
        "work_dir": str(work.resolve()),
    }
    write_json(dest / "handoff.json", hand)
    src_desc = "单条视频" if src["mode"] == "single" else \
        f"{len(src['files'])} 段同一期视频合并（{src.get('merge_method')}）"
    md = ["# 交接：剪口播定稿 → 封装", "",
          f"- 定稿时间：{hand['finalized_at']}（版本 {a.version}）",
          f"- 成片素材：`source_cut.mp4`（{vd:.2f}s，{src.get('width')}x{src.get('height')}，{fps_expr(edl.get('fps') or 30)}fps"
          + (f"，{speed:g} 倍速" if speed != 1.0 else "") + ("，HDR 已转 SDR" if edl.get("hdr_to_sdr") else "") + "）",
          f"- 字幕：`subtitles.srt`（{hand['subtitles_method']}）",
          f"- 来源：{src_desc}",
          f"- 剪掉：{edl['source_duration']:.1f}s → {edl['cut_duration']:.1f}s；"
          + "，".join(f"{k} {v}" for k, v in by.items()),
          f"- 时间轴真相：{hand['timeline_truth']}",
          f"- 口播稿：{a.script or '无'}",
          f"- 专名：{'、'.join(hand['terms']) or '无'}"]
    if a.notes:
        md.append(f"- 备注：{a.notes}")
    md += ["", "## 下一步", "",
           "由封装 agent 接手，输入就是本目录的 `source_cut.mp4` 和 `subtitles.srt`。",
           "字幕时间轴与视频时间轴一致，可直接使用；封装的中间产物写 `temp/`，成片定稿后再进 `output/`。",
           "剪辑工作目录（听写、候选、验收记录）在：", "", f"`{hand['work_dir']}`", ""]
    (dest / "交接.md").write_text("\n".join(md), encoding="utf-8")
    info(f"✅ 定稿交付 → {dest}（source_cut.mp4、subtitles.srt、handoff.json、交接.md）")


# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def pack_opts(p):
        p.add_argument("--chunk-sec", type=float, default=60,
                       help="每个听音块目标时长（默认 60s：块太长模型后半段会听漏，ASR 也容易丢尾巴）")
        p.add_argument("--max-mb", type=float, default=100, help="单块大小上限，≤100（默认 100，留 5%% 余量）")
        p.add_argument("--format", choices=["mp3", "wav"], default="mp3")
        p.add_argument("--bitrate", default="64k")
        p.add_argument("--sr", type=int, default=16000)

    p = sub.add_parser("prepare", help="合并（可选）→ 提取音频 → 听音包")
    p.add_argument("--work", required=True, help="temp/<YYYY-MM-DD_任务名>/mm_cut")
    p.add_argument("--merge", action="store_true", help="多个视频是同一期分段录制，按给定顺序合并")
    p.add_argument("--force-reencode", action="store_true")
    p.add_argument("--fps", type=float)
    p.add_argument("--noise-db", type=float)
    pack_opts(p)
    p.add_argument("videos", nargs="+")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("asr", help="听音块逐块送内网 ASR → 逐字时间轴、分句、readable.txt")
    p.add_argument("--pack", required=True, help="1_音频（原片）或 4_剪辑/vN/听音包（剪后，出字幕用）")
    p.add_argument("--out", required=True, help="2_听写 或 5_字幕/vN")
    p.add_argument("--terms", help="专名/热词，逗号分隔，传给 ASR 的 context（如 Claude,Codex,剪映）")
    p.add_argument("--url", default=ASR_URL, help=f"ASR 服务地址（默认 {ASR_URL}，可用环境变量 MMCUT_ASR_URL 改）")
    p.add_argument("--gap", type=float, default=0.3, help="字间停顿 ≥ 此值就断句（默认 0.3s）")
    p.add_argument("--sources", help="0_源/sources.json，用于标多段合并的接缝")
    p.set_defaults(fn=cmd_asr)

    p = sub.add_parser("merge", help="（备用）合并模型自己听写的 JSON 并做对齐检查")
    p.add_argument("--pack", required=True)
    p.add_argument("--listen", required=True)
    p.add_argument("--out")
    p.add_argument("--sources", help="0_源/sources.json，用于标接缝")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("clip", help="导出放大复听片段（只出音频）")
    p.add_argument("--pack", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--pad", type=float, default=1.5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_clip)

    p = sub.add_parser("snap", help="删除候选边界吸附 + 自动静音/头尾候选")
    p.add_argument("--pack", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--candidates")
    p.add_argument("--out", required=True)
    p.add_argument("--pad", type=float, default=0.08, help="句内删静音时两侧各留的呼吸（秒）")
    p.add_argument("--head-pad", type=float, default=0.04, help="ASR 模式：删除接缝后，下一句开口前留多少（秒）")
    p.add_argument("--tail-pad", type=float, default=0.06, help="ASR 模式：删除接缝前，上一句收音后留多少（秒）")
    p.add_argument("--no-auto", action="store_true")
    p.set_defaults(fn=cmd_snap)

    p = sub.add_parser("render", help="先剪音频（audio），验收后按音频时间轴拼画面（video）")
    p.add_argument("--work", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--stage", choices=["audio", "video"], required=True)
    p.add_argument("--include", help="额外要删的候选 id，逗号分隔（覆盖 delete=false）")
    p.add_argument("--exclude", help="不删的候选 id，逗号分隔（覆盖 delete=true）")
    p.add_argument("--merge-gap", type=float, default=0.12, help="两段删除间隔小于此值时连同中间一起删")
    p.add_argument("--long-gap", type=float, default=1.0,
                   help="ASR 模式：两句保留的话之间停顿 ≥ 此秒数，按接缝收紧（中间呼吸杂音一起删）")
    p.add_argument("--protect", help="ASR 模式：长停顿里要保住的 U 段 id（模型听出是 ASR 漏掉的话），逗号分隔")
    p.add_argument("--accepted", action="store_true", help="video 阶段：确认剪后音频已复听验收")
    p.add_argument("--fps", type=float)
    p.add_argument("--speed", type=float, default=1.0, help="video 阶段：成片倍速（保调，如 1.2）；edl/srt 自动按它换算")
    p.add_argument("--keep-hdr", action="store_true", help="video 阶段：HDR 源不转 SDR（默认转 BT.709，防发灰）")
    p.add_argument("--cross-take", type=float, default=30, help="接缝两侧在原片相隔 ≥ 此秒数就标“跨 take”（默认 30）")
    pack_opts(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("srt", help="剪后音频 ASR 逐字时间 → 断行 → 套用校字 → video.srt")
    p.add_argument("--transcript", help="ASR 模式：5_字幕/vN/transcript.json（对剪后听音包跑 asr 得到）")
    p.add_argument("--fixes", help="ASR 模式：模型校字结果 fixes.json")
    p.add_argument("--edl", help="4_剪辑/vN/edl.json，读倍速换算成片时间")
    p.add_argument("--speed", type=float, help="手动指定倍速（默认读 --edl）")
    p.add_argument("--gap", type=float, default=0.35, help="ASR 模式：字间停顿 ≥ 此值算真停顿，必断行")
    p.add_argument("--pack", help="备用模式：4_剪辑/vN/听音包")
    p.add_argument("--listen", help="备用模式：5_字幕/vN（放 chunk_XXX.json）")
    p.add_argument("--out")
    p.add_argument("--max-chars", type=int, default=16)
    p.add_argument("--min-dur", type=float, default=0.6)
    p.set_defaults(fn=cmd_srt)

    p = sub.add_parser("finalize", help="用户定稿后复制到 output/ 并写交接")
    p.add_argument("--work", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--dest", required=True, help="output/<YYYY-MM-DD_任务名>")
    p.add_argument("--script", help="口播稿路径（交接用）")
    p.add_argument("--terms", help="专名，逗号分隔")
    p.add_argument("--notes")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_finalize)

    a = ap.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        die("需要 ffmpeg 和 ffprobe")
    a.fn(a)


if __name__ == "__main__":
    main()
