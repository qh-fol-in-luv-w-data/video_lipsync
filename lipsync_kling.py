"""
Kling AI Lip-sync — image + audio → video
API: https://api.klingai.com/v1/videos/lip-sync

Cài đặt: pip install requests python-dotenv

Dùng:
    python lipsync_kling.py --image inputs/images/avatar.jpg \
                            --audio inputs/audios/q01.mp3 \
                            --out   outputs/q01.mp4

    python lipsync_kling.py --batch   # toàn bộ inputs/
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL   = "https://api.klingai.com"
API_KEY    = os.environ.get("KLING_API_KEY", "")
POLL_INTERVAL = 5    # giây
MAX_WAIT      = 300  # tối đa 5 phút chờ 1 task

# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------

def auth_headers() -> dict:
    if not API_KEY:
        print("[Lỗi] Chưa set KLING_API_KEY. Xem hướng dẫn trong README.")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }


# ---------------------------------------------------------------------------
# Bước 1: image → video ngắn (Kling image2video)
# Kling lip-sync nhận VIDEO input, không nhận ảnh trực tiếp
# ---------------------------------------------------------------------------

def image_to_video_kling(image_path: Path) -> str:
    """Upload ảnh lên Kling image2video, trả về video_id."""
    print(f"  [1/3] Tạo video từ ảnh: {image_path.name}")

    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    suffix  = image_path.suffix.lower().lstrip(".")
    mime    = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"

    payload = {
        "model_name": "kling-v1",
        "image":       f"data:{mime};base64,{img_b64}",
        "duration":    5,       # giây — đủ để lipsync overlay sau
        "cfg_scale":   0.5,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/videos/image2video",
        headers=auth_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    task_id = data["data"]["task_id"]
    print(f"       image2video task: {task_id} — đang chờ...")
    return _poll_video_task(task_id, endpoint="image2video")


def _poll_video_task(task_id: str, endpoint: str) -> str:
    """Poll cho đến khi video xong, trả về video_id."""
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        resp = requests.get(
            f"{BASE_URL}/v1/videos/{endpoint}/{task_id}",
            headers=auth_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data   = resp.json()["data"]
        status = data.get("task_status", "")

        if status == "succeed":
            video_id = data["task_result"]["videos"][0]["id"]
            print(f"       Video ID: {video_id}")
            return video_id
        if status == "failed":
            print(f"[Lỗi] Task {task_id} thất bại: {data}")
            sys.exit(1)

        print(f"       Trạng thái: {status} — chờ {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

    print(f"[Lỗi] Timeout sau {MAX_WAIT}s")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bước 2: lip-sync video + audio
# ---------------------------------------------------------------------------

def create_lipsync(video_id: str, audio_path: Path) -> str:
    """Gửi lip-sync task, trả về task_id."""
    print(f"  [2/3] Lip-sync: video_id={video_id} + {audio_path.name}")

    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
    suffix    = audio_path.suffix.lower().lstrip(".")
    mime      = "audio/mpeg" if suffix == "mp3" else f"audio/{suffix}"

    payload = {
        "input": {
            "video_id":   video_id,
            "audio_type": "file",
            "audio_file": f"data:{mime};base64,{audio_b64}",
        }
    }
    resp = requests.post(
        f"{BASE_URL}/v1/videos/lip-sync",
        headers=auth_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    task_id = resp.json()["data"]["task_id"]
    print(f"       Lip-sync task: {task_id}")
    return task_id


def poll_lipsync(task_id: str) -> str:
    """Poll lip-sync task, trả về URL video kết quả."""
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        resp = requests.get(
            f"{BASE_URL}/v1/videos/lip-sync/{task_id}",
            headers=auth_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data   = resp.json()["data"]
        status = data.get("task_status", "")

        if status == "succeed":
            url = data["task_result"]["videos"][0]["url"]
            print(f"       Video URL nhận được.")
            return url
        if status == "failed":
            print(f"[Lỗi] Lip-sync task thất bại: {data}")
            sys.exit(1)

        print(f"       Trạng thái: {status} — chờ {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

    print(f"[Lỗi] Timeout sau {MAX_WAIT}s")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bước 3: download video kết quả
# ---------------------------------------------------------------------------

def download_video(url: str, out_path: Path) -> None:
    print(f"  [3/3] Download → {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"       Đã lưu ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Single & Batch
# ---------------------------------------------------------------------------

def process_one(image_path: Path, audio_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [BỎ QUA] {out_path.name} đã có.")
        return

    print(f"\n{'='*55}")
    print(f"  Ảnh : {image_path.name}")
    print(f"  Audio: {audio_path.name}")
    print(f"  Output: {out_path.name}")
    print(f"{'='*55}")

    video_id = image_to_video_kling(image_path)
    task_id  = create_lipsync(video_id, audio_path)
    url      = poll_lipsync(task_id)
    download_video(url, out_path)


def batch_run(image_dir: Path, audio_dir: Path, out_dir: Path) -> None:
    images = sorted(f for f in image_dir.glob("*")
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    audios = sorted(f for f in audio_dir.glob("*")
                    if f.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"})

    if not images:
        print("[Lỗi] Không có ảnh trong", image_dir); sys.exit(1)
    if not audios:
        print("[Lỗi] Không có audio trong", audio_dir); sys.exit(1)

    # Ghép: audio tên trùng ảnh → ưu tiên; còn lại dùng ảnh đầu tiên
    pairs = []
    for audio in audios:
        img = next((i for i in images if i.stem == audio.stem), images[0])
        out = out_dir / f"{img.stem}_{audio.stem}.mp4"
        pairs.append((img, audio, out))

    print(f"[Batch] {len(pairs)} cặp cần xử lý")
    done = 0
    for img, audio, out in pairs:
        try:
            process_one(img, audio, out)
            done += 1
        except Exception as e:
            print(f"  [LỖI] {audio.name}: {e}")

    print(f"\nHoàn tất: {done}/{len(pairs)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Kling AI Lip-sync")
    ap.add_argument("--image",  help="Ảnh đầu vào")
    ap.add_argument("--audio",  help="Audio đầu vào")
    ap.add_argument("--out",    default="outputs/result.mp4")
    ap.add_argument("--batch",  action="store_true",
                    help="Batch mode: inputs/images/ + inputs/audios/")
    args = ap.parse_args()

    for d in ["inputs/images", "inputs/audios", "outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.batch:
        batch_run(Path("inputs/images"), Path("inputs/audios"), Path("outputs"))
    else:
        if not args.image or not args.audio:
            ap.print_help(); sys.exit(1)
        process_one(Path(args.image), Path(args.audio), Path(args.out))


if __name__ == "__main__":
    main()
