"""
Text-to-Speech cho câu hỏi phỏng vấn (tiếng Việt)
Backend: edge-tts — pip install edge-tts

Giọng: vi-VN-NamMinhNeural (nam)
"""

import asyncio
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
QUESTIONS_DIR = "Generated_Questions"
AUDIO_DIR     = "Generated_Audio"

VOICE  = "vi-VN-NamMinhNeural"
RATE   = "+0%"
PITCH  = "+0Hz"
FORMAT = "mp3"

# ---------------------------------------------------------------------------
# Xử lý từ chuyên ngành
#
# Nguyên tắc:
#   - Acronym (viết hoa toàn bộ): đánh vần từng chữ cái  → SQL = S Q L
#   - Dạng XX/XX (CI/CD, UI/UX):  đánh vần từng phần      → C I C D
#   - B2B / B2C:                  đọc dạng "B to B"
#   - Từ tiếng Anh thông thường (Docker, RESTful, GraphQL, Kubernetes...):
#     giữ nguyên để TTS đọc theo chuẩn tiếng Anh
# ---------------------------------------------------------------------------

# Các từ trông giống acronym nhưng TTS đọc đúng khi để nguyên
_READ_AS_IS = {
    "SOLID", "REST", "CORS", "SAML", "ACID", "MEAN", "MERN",
    "LAMP", "CRUD", "NULL", "TRUE", "FALSE", "GET", "POST",
    "PUT", "PATCH", "DELETE", "SCRUM", "LEAN",
}


def _spell(word: str) -> str:
    """Đánh vần từng chữ cái, cách nhau dấu cách."""
    return " ".join(word)


def preprocess(text: str) -> str:
    # 1. Dạng XX/XX  (CI/CD, UI/UX, R&D, P&L ...)
    def slash_or_amp(m):
        parts = re.split(r"[/&]", m.group(0))
        return " ".join(_spell(p) if p.isupper() else p for p in parts)

    text = re.sub(r"\b[A-Z]{1,6}(?:[/&][A-Z]{1,6})+\b", slash_or_amp, text)

    # 2. BxC / BxB  (B2B, B2C, G2G ...)
    text = re.sub(
        r"\b([A-Z])\d([A-Z])\b",
        lambda m: f"{m.group(1)} to {m.group(2)}",
        text,
    )

    # 3. Acronym thuần (≥ 2 chữ hoa liên tiếp) chưa nằm trong whitelist
    def spell_acronym(m):
        word = m.group(0)
        if word in _READ_AS_IS:
            return word
        return _spell(word)

    text = re.sub(r"\b[A-Z]{2,}\b", spell_acronym, text)

    return text


# ---------------------------------------------------------------------------
# TTS engine
# ---------------------------------------------------------------------------

async def synthesize_async(text: str, out_path: Path):
    import edge_tts
    processed = preprocess(text)
    communicate = edge_tts.Communicate(processed, VOICE, rate=RATE, pitch=PITCH)
    await communicate.save(str(out_path))


def synthesize(text: str, out_path: Path):
    asyncio.run(synthesize_async(text, out_path))


# ---------------------------------------------------------------------------
# Parser câu hỏi từ Markdown
# ---------------------------------------------------------------------------

def parse_questions(md_path: Path) -> list[dict]:
    pattern = re.compile(r"^(\d+)\.\s+\*\*\[([^\]]+)\]\*\*\s+(.+)", re.MULTILINE)
    content = md_path.read_text(encoding="utf-8")
    return [
        {"index": int(m.group(1)), "tag": m.group(2).strip(), "text": m.group(3).strip()}
        for m in pattern.finditer(content)
    ]


def sanitize(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


# ---------------------------------------------------------------------------
# Xử lý từng file
# ---------------------------------------------------------------------------

def process_md_file(md_path: Path, out_base: Path):
    questions = parse_questions(md_path)
    if not questions:
        print(f"  [BỎ QUA] Không tìm thấy câu hỏi: {md_path.name}")
        return 0, 0

    audio_dir = out_base / sanitize(md_path.stem)
    audio_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    print(f"\n  {md_path.name}  ({len(questions)} câu)")

    for q in questions:
        out_path = audio_dir / f"{q['index']:02d}_{sanitize(q['tag'])}.{FORMAT}"

        if out_path.exists():
            print(f"    [{q['index']:02d}] Đã có — bỏ qua")
            skipped += 1
            continue

        print(f"    [{q['index']:02d}] [{q['tag']}] {q['text'][:55]}...")
        try:
            synthesize(q["text"], out_path)
            print(f"          => {out_path.name}")
            done += 1
        except Exception as e:
            print(f"          [LỖI] {e}")

    return done, skipped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_deps():
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("[LỖI] pip install edge-tts")
        sys.exit(1)


def main():
    check_deps()

    questions_root = Path(QUESTIONS_DIR)
    audio_root     = Path(AUDIO_DIR)

    if not questions_root.exists():
        print(f"[LỖI] Không tìm thấy thư mục: {QUESTIONS_DIR}")
        sys.exit(1)

    md_files = sorted(questions_root.rglob("*.md"))
    if not md_files:
        print("Không tìm thấy file .md nào.")
        sys.exit(0)

    print(f"Giọng : {VOICE}")
    print(f"Nguồn : {QUESTIONS_DIR}/  ({len(md_files)} files)")
    print(f"Đích  : {AUDIO_DIR}/")
    print("=" * 60)

    total_done = total_skipped = 0
    for md_path in md_files:
        rel = md_path.parent.relative_to(questions_root)
        done, skipped = process_md_file(md_path, audio_root / rel)
        total_done    += done
        total_skipped += skipped

    print("\n" + "=" * 60)
    print(f"Hoàn tất — Đã tạo: {total_done} | Bỏ qua: {total_skipped}")
    print(f"Audio: {AUDIO_DIR}/")


if __name__ == "__main__":
    main()
