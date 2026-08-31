from pathlib import Path

from PIL import Image

src = Path(
    r"C:\Users\subha\.cursor\projects\c-ate-frondend\assets"
    r"\c__Users_subha_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"c158ceba5ecba11d7977f518fb9b6d3c_images_"
    r"ChatGPT_Image_Aug_13__2026__05_14_32_PM-81fd264a-bb57-4051-9564-ee644d1f5911.png"
)
out_dir = Path(__file__).resolve().parents[1] / "public" / "branding"
out_dir.mkdir(parents=True, exist_ok=True)

im = Image.open(src).convert("RGBA")
w, h = im.size
print("size", w, h)
im.save(out_dir / "verilumen-ate-logo.png", optimize=True)

pixels = im.load()


def is_content(x: int, y: int) -> bool:
    r, g, b, a = pixels[x, y]
    if a < 20:
        return False
    return (r + g + b) > 40


rows = [y for y in range(h) if any(is_content(x, y) for x in range(0, w, 2))]
first, last = rows[0], rows[-1]
print("content y", first, last)

empty_runs: list[tuple[int, int, int]] = []
in_empty = False
start = 0
for y in range(h):
    has = any(is_content(x, y) for x in range(0, w, 2))
    if not has and not in_empty:
        in_empty = True
        start = y
    elif has and in_empty:
        empty_runs.append((start, y - 1, y - start))
        in_empty = False

cut = None
for s, _e, length in empty_runs:
    if s > first + int(h * 0.15) and length >= 4:
        cut = s
        break
if cut is None:
    cut = int(h * 0.48)

xs = [x for x in range(w) for y in range(first, cut, 2) if is_content(x, y)]
left, right = max(0, min(xs) - 8), min(w, max(xs) + 8)
top, bottom = max(0, first - 8), cut
mark = im.crop((left, top, right, bottom))
mark.save(out_dir / "verilumen-mark.png", optimize=True)
print("mark", mark.size, "cut", cut, "box", (left, top, right, bottom))
print("wrote", sorted(p.name for p in out_dir.iterdir()))
