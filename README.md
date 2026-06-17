# LatentSync 1.5 — Audio → Lip-sync Video

Tạo video lip-sync từ **ảnh chân dung + file audio**, chạy trên Windows GPU.

---

## Cấu trúc thư mục

```
ats_phongvan/
├── inputs/
│   ├── images/       ← Bỏ ảnh avatar vào đây  (.jpg / .png)
│   └── audios/       ← Bỏ audio câu hỏi vào đây (.mp3 / .wav)
├── outputs/          ← Video lip-sync xuất ra đây (.mp4)
│
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements-lipsync.txt
└── lipsync.py
```

---

## Yêu cầu hệ thống

| | |
|---|---|
| OS | Windows 10/11 với WSL2 |
| GPU | NVIDIA, VRAM ≥ 6GB (khuyến nghị 8GB+) |
| Driver | NVIDIA ≥ 527 |
| Docker | Docker Desktop 4.x (WSL2 backend) |

**Cài NVIDIA Container Toolkit cho Docker Desktop (một lần):**
```powershell
# Trong WSL2 terminal
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Hướng dẫn sử dụng

### Bước 1 — Clone và build (một lần, ~10–15 phút)

```bat
git clone <repo-url>
cd ats_phongvan
docker compose build
```

> Image ~8GB (PyTorch CUDA + tất cả dependencies).

### Bước 2 — Chuẩn bị file

Bỏ file vào đúng thư mục:

```
inputs/
├── images/
│   └── avatar.jpg          ← ảnh chân dung, nhìn thẳng, đủ sáng
└── audios/
    ├── 01_Technical.mp3
    ├── 02_Technical.mp3
    └── ...
```

**Tip ảnh tốt:**
- Chụp thẳng mặt, không nghiêng
- Ánh sáng đều, không bóng ngược
- Kích thước ≥ 512×512 px
- Nền đơn giản, ít chi tiết

**Ghép tự động theo tên file:**
```
images/q01.jpg  ↔  audios/q01.mp3   → outputs/q01_q01.mp4
images/q02.jpg  ↔  audios/q02.mp3   → outputs/q02_q02.mp4
```
Nếu chỉ có **1 ảnh + nhiều audio**, tất cả audio sẽ ghép với ảnh đó.

### Bước 3 — Chạy

```bat
:: Batch — xử lý toàn bộ inputs/ (khuyến nghị)
docker compose run --rm lipsync

:: Một cặp cụ thể
docker compose run --rm lipsync ^
  --image inputs/images/avatar.jpg ^
  --audio inputs/audios/01_Technical.mp3 ^
  --out   outputs/q01.mp4

:: Chất lượng cao hơn (chậm hơn ~2x)
docker compose run --rm lipsync --batch --steps 35 --guidance 2.5
```

Lần đầu chạy sẽ tự động:
1. Clone LatentSync từ GitHub
2. Download checkpoint ~4GB từ HuggingFace

→ Lưu vào Docker volume `ats_latentsync_repo`, **không cần download lại** những lần sau.

---

## Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--steps` | 20 | Denoising steps — nhiều hơn = mịn hơn, chậm hơn |
| `--guidance` | 2.0 | Guidance scale (1.5–3.0) — cao hơn = khớp audio hơn |
| `--fps` | 25 | FPS video output |
| `--batch` | — | Xử lý toàn bộ inputs/ |
| `--skip-setup` | — | Bỏ qua clone/download (đã có sẵn) |

---

## Quản lý

```bat
:: Xem log khi đang chạy
docker compose logs -f lipsync

:: Dừng
docker compose down

:: Xoá checkpoint để download lại
docker volume rm ats_latentsync_repo

:: Xoá image (giải phóng ~8GB)
docker rmi ats-lipsync:latest
```

---

## Troubleshooting

| Lỗi | Giải pháp |
|---|---|
| `could not select device driver nvidia` | Cài NVIDIA Container Toolkit, restart Docker Desktop |
| `CUDA out of memory` | Giảm `--steps 15` hoặc dùng ảnh nhỏ hơn |
| `face not detected` | Dùng ảnh rõ mặt hơn, đủ ánh sáng, nhìn thẳng |
| Download HuggingFace chậm/lỗi | Thử lại hoặc dùng VPN |
| Video output bị blur | Tăng `--steps 30 --guidance 2.5` |
