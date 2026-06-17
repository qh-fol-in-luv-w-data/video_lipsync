"""
LatentSync 1.5 — Lip-sync video từ ảnh + audio
Hỗ trợ: Windows (CUDA GPU) | macOS (MPS / CPU)

Cài đặt:
    pip install huggingface_hub diffusers accelerate omegaconf \
                ffmpeg-python face-alignment mediapipe

Dùng:
    python lipsync.py --image inputs/images/avatar.jpg \
                      --audio inputs/audios/question.mp3 \
                      --out   outputs/result.mp4
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
REPO_DIR    = Path(__file__).parent / "LatentSync"
HF_REPO     = "ByteDance/LatentSync-1.5"
CKPT_DIR    = REPO_DIR / "checkpoints"
UNET_CONFIG = REPO_DIR / "configs" / "unet" / "second_stage.yaml"
UNET_CKPT   = CKPT_DIR / "latentsync_unet.pt"

# Các file checkpoint cần download từ HuggingFace
REQUIRED_CKPTS = [
    "checkpoints/latentsync_unet.pt",
    "checkpoints/whisper/small.pt",
    "checkpoints/auxiliary/dwpose/yolox_l.onnx",
    "checkpoints/auxiliary/dwpose/dw-ll_ucoco_384.onnx",
]

IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
        if not IS_WINDOWS and torch.backends.mps.is_available():
            print("  Device: Apple MPS")
            return "mps"
    except ImportError:
        pass
    print("  Device: CPU (chậm)")
    return "cpu"


# ---------------------------------------------------------------------------
# Setup: clone repo
# ---------------------------------------------------------------------------

def setup_repo():
    if REPO_DIR.exists():
        print("[Setup] LatentSync repo đã có.")
        return

    print("[Setup] Cloning LatentSync từ GitHub...")
    subprocess.run(
        ["git", "clone", "https://github.com/bytedance/LatentSync.git", str(REPO_DIR)],
        check=True,
    )

    req = REPO_DIR / "requirements.txt"
    if req.exists():
        print("[Setup] Cài requirements.txt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=True,
        )


def download_checkpoints():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Lỗi] pip install huggingface_hub")
        sys.exit(1)

    print("[Setup] Kiểm tra checkpoints...")
    for repo_path in REQUIRED_CKPTS:
        out_path = REPO_DIR / repo_path
        if out_path.exists():
            print(f"  [OK]  {repo_path}")
            continue
        print(f"  [DL]  {repo_path} ...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=repo_path,
            local_dir=str(REPO_DIR),
            local_dir_use_symlinks=False,
        )


# ---------------------------------------------------------------------------
# Bước 1: Ảnh → video loop (cùng độ dài audio)
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def image_to_video(image_path: Path, audio_path: Path,
                   out_path: Path, fps: int = 25) -> None:
    duration = get_audio_duration(audio_path)
    print(f"[Bước 1] Tạo video loop ({duration:.1f}s, {fps} fps)...")

    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-framerate", str(fps),
        "-i", str(image_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        # Kích thước phải chẵn và ≥ 256 cho LatentSync
        "-vf", "scale='if(gt(iw,ih),512,trunc(512*iw/ih/2)*2)':'if(gt(iw,ih),trunc(512*ih/iw/2)*2,512)'",
        "-shortest",
        str(out_path),
    ], check=True)


# ---------------------------------------------------------------------------
# Bước 2: LatentSync inference
# ---------------------------------------------------------------------------

def run_lipsync(input_video: Path, audio_path: Path,
                output_video: Path, steps: int, guidance: float) -> None:
    print(f"[Bước 2] LatentSync 1.5 inference (steps={steps}, cfg={guidance})...")

    device = get_device()

    # Tìm script inference
    for candidate in [
        REPO_DIR / "scripts" / "inference.py",
        REPO_DIR / "inference.py",
    ]:
        if candidate.exists():
            script = candidate
            break
    else:
        print("[Lỗi] Không tìm thấy inference.py trong repo LatentSync.")
        sys.exit(1)

    cmd = [
        sys.executable, str(script),
        "--unet_config_path",    str(UNET_CONFIG),
        "--inference_ckpt_path", str(UNET_CKPT),
        "--inference_steps",     str(steps),
        "--guidance_scale",      str(guidance),
        "--video_path",          str(input_video),
        "--audio_path",          str(audio_path),
        "--video_out_path",      str(output_video),
    ]

    env = os.environ.copy()
    if device == "mps":
        # fallback cho các CUDA op chưa có trên MPS
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif device == "cuda":
        # Tắt TF32 để tránh artifact trên một số GPU Ampere
        env["NVIDIA_TF32_OVERRIDE"] = "0"

    print(f"  Running: {script.name}")
    result = subprocess.run(cmd, cwd=str(REPO_DIR), env=env)

    if result.returncode != 0:
        print("[Lỗi] Inference thất bại. Xem output ở trên.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Batch mode: xử lý nhiều cặp ảnh-audio cùng lúc
# ---------------------------------------------------------------------------

def batch_run(image_dir: Path, audio_dir: Path,
              out_dir: Path, args) -> None:
    pairs = []
    for img in sorted(image_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        # Ghép với audio cùng tên (nếu có), không thì dùng audio đầu tiên
        audio = next(
            (a for a in audio_dir.glob(f"{img.stem}.*")
             if a.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}),
            next(
                (a for a in sorted(audio_dir.glob("*"))
                 if a.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}),
                None,
            ),
        )
        if audio:
            pairs.append((img, audio))

    if not pairs:
        print("[Batch] Không tìm thấy cặp ảnh-audio nào.")
        return

    print(f"[Batch] Tìm thấy {len(pairs)} cặp. Bắt đầu xử lý...")
    for i, (img, audio) in enumerate(pairs, 1):
        out = out_dir / f"{img.stem}_{audio.stem}.mp4"
        print(f"\n[{i}/{len(pairs)}] {img.name} + {audio.name} → {out.name}")
        if out.exists():
            print("  Đã có, bỏ qua.")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            loop_video = Path(tmp) / "loop.mp4"
            image_to_video(img, audio, loop_video, fps=args.fps)
            run_lipsync(loop_video, audio, out, args.steps, args.guidance)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LatentSync 1.5 — Lip-sync từ ảnh + audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Một cặp ảnh + audio
  python lipsync.py --image inputs/images/avatar.jpg \\
                    --audio inputs/audios/q01.mp3 \\
                    --out   outputs/q01.mp4

  # Batch: toàn bộ ảnh trong inputs/images + audios trong inputs/audios
  python lipsync.py --batch

  # Bỏ qua bước setup (đã cài sẵn)
  python lipsync.py --image ... --audio ... --skip-setup
        """,
    )
    p.add_argument("--image",    help="Ảnh đầu vào (jpg/png)")
    p.add_argument("--audio",    help="Audio đầu vào (mp3/wav)")
    p.add_argument("--out",      default="outputs/result.mp4", help="Video đầu ra")
    p.add_argument("--batch",    action="store_true",
                   help="Batch mode: xử lý toàn bộ inputs/images + inputs/audios")
    p.add_argument("--steps",    type=int,   default=20,  help="Denoising steps (20–40)")
    p.add_argument("--guidance", type=float, default=2.0, help="Guidance scale (1.5–3.0)")
    p.add_argument("--fps",      type=int,   default=25,  help="FPS video đầu ra")
    p.add_argument("--skip-setup", action="store_true",
                   help="Bỏ qua bước clone repo và download checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    # Đảm bảo folders tồn tại
    for d in ["inputs/images", "inputs/audios", "outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LatentSync 1.5 — Lip-sync Video Generator")
    print("=" * 60)

    if not args.skip_setup:
        setup_repo()
        download_checkpoints()

    # --- Batch mode ---
    if args.batch:
        batch_run(
            Path("inputs/images"), Path("inputs/audios"), Path("outputs"), args
        )
        return

    # --- Single mode ---
    if not args.image or not args.audio:
        print("[Lỗi] Cần --image và --audio, hoặc dùng --batch")
        sys.exit(1)

    image_path  = Path(args.image).resolve()
    audio_path  = Path(args.audio).resolve()
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"[Lỗi] Không tìm thấy ảnh: {image_path}")
        sys.exit(1)
    if not audio_path.exists():
        print(f"[Lỗi] Không tìm thấy audio: {audio_path}")
        sys.exit(1)

    print(f"  Ảnh   : {image_path.name}")
    print(f"  Audio : {audio_path.name}")
    print(f"  Output: {output_path.name}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        loop_video = Path(tmp) / "loop_input.mp4"
        image_to_video(image_path, audio_path, loop_video, fps=args.fps)
        run_lipsync(loop_video, audio_path, output_path,
                    args.steps, args.guidance)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print("\n" + "=" * 60)
    print(f"Hoàn tất! → {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
