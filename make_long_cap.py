#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, shutil, subprocess
from pathlib import Path
from typing import List
from PIL import Image, ImageOps, ImageDraw

# ===== 可调参数 =====
FRAME_WIDTH = 1280
CANVAS_BG = (0, 0, 0)
GAP_PX = 6
KEEP_TOP_SHADOW = True
TOP_SHADOW_H = 120

SUB_BAND_RATIO = 0.20       # 底部取 20% 高度当“字幕条”
SIDE_MARGIN_RATIO = 0.05    # 左右各裁 5%
ROUND_CORNERS = 8

# 去重参数
DEDUP_ENABLE = True
DEDUP_METRIC = "mse"        # "mse" / "sad"
DEDUP_THRESH = 0.006        # 0.004~0.010 调
DEDUP_MIN_GAP = 1           # 至少隔 N 条才允许再保留一次
# ====================

def parse_hms(s: str) -> float:
    parts = [int(p) for p in s.split(":")]
    return parts[0]*60 + parts[1] if len(parts)==2 else parts[0]*3600 + parts[1]*60 + parts[2]

def ensure_tools():
    for t in ["ffmpeg", "yt-dlp"]:
        if shutil.which(t) is None:
            print(f"请先安装 {t} 并确保在 PATH 中。"); sys.exit(1)

def download_video(url, outdir) -> str:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "yt-dlp","-f","mp4/best","--restrict-filenames","-o",str(outdir/"%(id)s.%(ext)s"),url
    ], check=True)
    vids = sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
    if not vids: print("未找到下载的视频文件"); sys.exit(1)
    return str(vids[0])

def extract_interval_frames(video_path, start_sec, end_sec, interval_s, outdir):
    """固定间隔导出帧；每次调用都会先清空旧文件。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 清空旧的 shot_*.jpg
    for f in outdir.glob("shot_*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass

    # 2) 用 -t 指定“持续时长”，避免 -to 整体时间轴边界误差
    duration = max(0.01, end_sec - start_sec)

    # 3) 用分数表达 fps，避免某些版本对 0.5 的解析差异
    #    例如 interval=2.0 -> fps=1/2
    fps_expr = f"fps=1/{int(round(interval_s))}" if abs(interval_s - round(interval_s)) < 1e-6 \
               else f"fps=1/{interval_s}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.2f}",
        "-t",  f"{duration:.2f}",
        "-i", video_path,
        "-vf", fps_expr,
        "-q:v", "2",
        str(outdir / "shot_%05d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def resize_w(img: Image.Image, w: int) -> Image.Image:
    if img.width==w: return img
    r = w/img.width
    return img.resize((w,int(img.height*r)), Image.LANCZOS)

def make_top_shadow(img: Image.Image, h_px: int) -> Image.Image:
    if h_px<=0: return img
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    for i in range(h_px):
        alpha = int(180*(i/h_px))
        d.line([(0,img.height-h_px+i),(img.width,img.height-h_px+i)], fill=(0,0,0,alpha))
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")

# --- 去重工具 ---
import numpy as np
def to_gray_np(im: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(im), dtype=np.uint8)

def strip_similarity(a: Image.Image, b: Image.Image, metric="mse") -> float:
    # 居中裁到相同尺寸比较
    w = min(a.width, b.width); h = min(a.height, b.height)
    if w<=0 or h<=0: return 1e9
    ax0=(a.width-w)//2; ay0=(a.height-h)//2
    bx0=(b.width-w)//2; by0=(b.height-h)//2
    A = to_gray_np(a.crop((ax0,ay0,ax0+w,ay0+h))).astype(np.int16)
    B = to_gray_np(b.crop((bx0,by0,bx0+w,by0+h))).astype(np.int16)
    if metric=="sad":
        return float(np.mean(np.abs(A-B)))/255.0
    diff = (A-B).astype(np.float32)
    return float(np.mean(diff*diff))/(255.0*255.0)

def main():
    if len(sys.argv)<2:
        print("用法：python make_longcap.py <YouTube_URL> [--start 7:53] [--end 9:13] [--interval 2.0] [--band 0.20] [--side 0.05]")
        sys.exit(1)

    url = sys.argv[1]
    start="7:53"; end="9:13"; interval=2.0
    band=SUB_BAND_RATIO; side=SIDE_MARGIN_RATIO

    args=sys.argv[2:]; i=0
    while i<len(args):
        if args[i]=="--start" and i+1<len(args): start=args[i+1]; i+=2
        elif args[i]=="--end" and i+1<len(args): end=args[i+1]; i+=2
        elif args[i]=="--interval" and i+1<len(args): interval=float(args[i+1]); i+=2
        elif args[i]=="--band" and i+1<len(args): band=float(args[i+1]); i+=2
        elif args[i]=="--side" and i+1<len(args): side=float(args[i+1]); i+=2
        else: i+=1

    s=parse_hms(start); e=parse_hms(end)
    if e<=s: print("结束时间必须大于开始时间"); sys.exit(1)
    if interval<=0: print("interval 必须 > 0"); sys.exit(1)

    ensure_tools()
    work=Path("work"); work.mkdir(exist_ok=True)
    video=download_video(url, work)
    shots=work/"shots"; shots.mkdir(exist_ok=True)
    extract_interval_frames(video, s, e, interval, shots)

    paths=sorted(shots.glob("shot_*.jpg"))
    print(f"[INFO] 抽帧总数: {len(paths)}")
    if not paths: print("没有帧"); sys.exit(1)

    # 第一帧：整图
    first=Image.open(paths[0]).convert("RGB")
    first=resize_w(first, FRAME_WIDTH)
    if KEEP_TOP_SHADOW: first=make_top_shadow(first, TOP_SHADOW_H)

    # 其余帧 -> 只裁字幕条（只做一次）
    raw_strips: List[Image.Image]=[]
    for p in paths[1:]:                         # 注意：明确跳过第1帧
        im=Image.open(p).convert("RGB")
        im=resize_w(im, FRAME_WIDTH)
        w,h=im.size
        y0=max(0, int(h*(1.0-band))); y1=h
        strip=im.crop((0,y0,w,y1))
        dx=int(w*side)
        if dx>0 and 2*dx<strip.width:
            strip=strip.crop((dx,0,strip.width-dx,strip.height))
        if ROUND_CORNERS>0:
            m=Image.new("L", strip.size, 0)
            d=ImageDraw.Draw(m)
            d.rounded_rectangle([0,0,strip.width,strip.height], radius=ROUND_CORNERS, fill=255)
            strip=strip.convert("RGBA"); strip.putalpha(m)
        raw_strips.append(strip)

    print(f"[INFO] 字幕条原始数: {len(raw_strips)}")

    # 去重（只保留变化明显的字幕条）——只做一次，结果覆盖 raw_strips
    strips: List[Image.Image]=raw_strips
    if DEDUP_ENABLE and strips:
        kept=[]; last=None; gap=999
        for s_im in strips:
            if last is None:
                kept.append(s_im); last=s_im; gap=0; continue
            score=strip_similarity(last, s_im, DEDUP_METRIC)
            if score>=DEDUP_THRESH or gap>=DEDUP_MIN_GAP:
                kept.append(s_im); last=s_im; gap=0
            else:
                gap+=1
        strips=kept

    print(f"[INFO] 字幕条去重后: {len(strips)}")

    # 画布尺寸（只用“去重后”的 strips 计算）
    total_h = first.height + GAP_PX + sum(s.height + GAP_PX for s in strips)
    max_w   = max([first.width] + [s.width for s in strips])  # 为防左右裁边导致宽度略小
    canvas = Image.new("RGB", (max_w, total_h), CANVAS_BG)

    # 拼接（只拼一次）
    y=0
    canvas.paste(first, (0,y)); y += first.height + GAP_PX
    for s_im in strips:
        x=(max_w - s_im.width)//2
        if s_im.mode=="RGBA":
            canvas.paste(s_im, (x,y), s_im)
        else:
            canvas.paste(s_im, (x,y))
        y += s_im.height + GAP_PX

    out="long_subtitle_strip_dedup.jpg"
    canvas.save(out, quality=95)
    print(f"[DONE] 已生成: {out}")

if __name__=="__main__":
    main()