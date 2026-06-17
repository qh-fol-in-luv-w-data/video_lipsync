#!/bin/bash
set -e

REPO_DIR="/app/LatentSync"
HF_REPO="ByteDance/LatentSync-1.5"

# ── Màu log ─────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo ""
echo "══════════════════════════════════════════"
echo "  LatentSync 1.5 — Startup"
echo "══════════════════════════════════════════"

# ── 1. Clone LatentSync nếu chưa có ─────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    warn "Chưa có LatentSync repo — đang clone..."
    git clone https://github.com/bytedance/LatentSync.git "$REPO_DIR"
    log "Clone xong."

    # Cài thêm requirements của LatentSync (nếu có file riêng)
    if [ -f "$REPO_DIR/requirements.txt" ]; then
        warn "Cài requirements của LatentSync..."
        pip install --no-cache-dir -q -r "$REPO_DIR/requirements.txt" || true
        log "Requirements xong."
    fi
else
    log "LatentSync repo đã có."
fi

# ── 2. Download checkpoints nếu chưa có ─────────────────────
UNET_CKPT="$REPO_DIR/checkpoints/latentsync_unet.pt"
if [ ! -f "$UNET_CKPT" ]; then
    warn "Chưa có checkpoint — đang download từ HuggingFace (~4GB)..."
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

files = [
    "checkpoints/latentsync_unet.pt",
    "checkpoints/whisper/small.pt",
    "checkpoints/auxiliary/dwpose/yolox_l.onnx",
    "checkpoints/auxiliary/dwpose/dw-ll_ucoco_384.onnx",
]

for f in files:
    out = f"/app/LatentSync/{f}"
    if os.path.exists(out):
        print(f"  [OK] {f}")
        continue
    print(f"  [DL] {f} ...")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.5",
        filename=f,
        local_dir="/app/LatentSync",
        local_dir_use_symlinks=False,
    )
    print(f"  [OK] {f}")
PYEOF
    log "Checkpoints đã sẵn sàng."
else
    log "Checkpoints đã có."
fi

# ── 3. Kiểm tra GPU ─────────────────────────────────────────
python3 - <<'PYEOF'
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\033[0;32m[✓]\033[0m GPU: {name} ({vram:.1f} GB VRAM)")
else:
    print("\033[1;33m[!]\033[0m GPU không khả dụng — chạy trên CPU (chậm hơn)")
PYEOF

echo ""
echo "══════════════════════════════════════════"
echo ""

# ── 4. Chạy lệnh truyền vào ─────────────────────────────────
exec python3 /app/lipsync.py "$@"
