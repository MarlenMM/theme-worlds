"""Build the README and GitHub Pages media from a folder of notebook outputs.

    python scripts/make_media.py PATH/TO/outputs

Writes, for the curated set below:

* ``docs/media/panoramas/<slug>.jpg`` — each panorama at full resolution
  (JPEG q90; the lossless PNGs are attached to the GitHub release instead);
* ``docs/media/panoramas/index.js`` — titles, seeds and measured seam ratios,
  read by the viewer in ``docs/index.html``;
* ``docs/media/loop.webp`` — one panorama scrolling sideways forever.

Pixels are only resized, re-encoded and rolled. The single thing drawn on top
of a generated image is the pair of small notches in ``loop.webp``, which mark
where the PNG's left and right edges meet — without them there is no way to
tell where the file ends, which is the point being made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.notebook_code import REPO, load  # noqa: E402

# (slug, title, source file). Chosen for range: two single themes, eight
# two-theme worlds, both canvas shapes the notebook supports.
GALLERY = [
    ("volcanic-flooded", "Volcanic lava world + flooded metropolis",
     "volcanic_lava_world+flooded_metropolis_0910-0542_seed201.png"),
    ("arctic-village", "Arctic ice cave + cozy village in snowfall",
     "arctic_ice_cave+cozy_village_snowfall_0908-1757_seed1002.png"),
    ("fantasy-medieval", "Floating-island fantasy + medieval kingdom",
     "fantasy+medieval_0909-0248_seed1001.png"),
    ("cyberpunk-underwater", "Cyberpunk + underwater",
     "cyberpunk+underwater_0909-0122_seed1001.png"),
    ("giant-tree-biopunk", "Giant tree civilization + biopunk organic city",
     "giant_tree_civilization+biopunk_organic_city_0910-0636_seed1001.png"),
    ("ink-wash-dragon", "Ink wash mountains + dragon spires",
     "ink_wash_mountains+dragon_spire_0910-0514_seed201.png"),
    ("gothic-amusement", "Gothic cathedral courtyard + abandoned amusement park",
     "gothic_cathedral_courtyard+abandoned_amusement_park_0910-0725_seed455.png"),
    ("alien-desert", "Alien jungle planet + desert planet",
     "alien_jungle_planet+desert_planet_0909-0607_seed1000.png"),
    ("cyberpunk", "Cyberpunk megacity",
     "cyberpunk_0908-1736_seed1000.png"),
    ("stained-glass", "Stained glass realm",
     "stained_glass_realm_0909-0539_seed1000.png"),
]

LOOP_SOURCE = "arctic-village"
# 120 frames x 50 ms = one full revolution every 6 s, in ~3.6 MB. 960 px wide
# at 160 frames looked marginally smoother and cost 9 MB, too much for a README.
LOOP_SIZE = (800, 200)
LOOP_FRAMES = 120
LOOP_FRAME_MS = 50
LOOP_QUALITY = 70


def notch(draw: ImageDraw.ImageDraw, x: float, height: int, size: int = 9) -> None:
    """A small white triangle at the top and bottom edge, pointing inwards."""
    for tip_y, base_y in ((size, 0), (height - 1 - size, height - 1)):
        points = [(x - size, base_y), (x + size, base_y), (x, tip_y)]
        draw.polygon(points, fill=(255, 255, 255), outline=(20, 20, 24))


def build_loop(image: Image.Image, out: Path) -> None:
    source = np.asarray(image.convert("RGB"))
    width = source.shape[1]
    frames = []
    for k in range(LOOP_FRAMES):
        shift = round(k * width / LOOP_FRAMES)
        frame = Image.fromarray(np.roll(source, -shift, axis=1)).resize(LOOP_SIZE, Image.LANCZOS)
        # Column 0 of the original file now sits at (width - shift) % width.
        edge = ((width - shift) % width) * LOOP_SIZE[0] / width
        notch(ImageDraw.Draw(frame), edge, LOOP_SIZE[1])
        frames.append(frame)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=LOOP_FRAME_MS,
                   loop=0, quality=LOOP_QUALITY, method=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("outputs", type=Path, help="folder the notebook wrote PNGs into")
    args = parser.parse_args()

    seam_ratio = load("seam_ratio", np=np)["seam_ratio"]
    media = REPO / "docs" / "media"
    panoramas = media / "panoramas"
    panoramas.mkdir(parents=True, exist_ok=True)

    index = []
    for slug, title, filename in GALLERY:
        image = Image.open(args.outputs / filename).convert("RGB")
        image.save(panoramas / f"{slug}.jpg", quality=90, optimize=True, progressive=True)
        themes, rest = filename.rsplit("_seed", 1)[0].rsplit("_", 1)[0], filename.rsplit("_seed", 1)[1]
        index.append({
            "slug": slug,
            "title": title,
            "themes": themes.split("+"),
            "seed": int(rest.removesuffix(".png")),
            "width": image.width,
            "height": image.height,
            "seam_ratio": round(seam_ratio(image), 2),
            "source_png": filename,
        })
        print(f"{slug:<22} {image.width}x{image.height}  seam {index[-1]['seam_ratio']:.2f}  "
              f"{(panoramas / f'{slug}.jpg').stat().st_size / 1e3:.0f} KB")
        if slug == LOOP_SOURCE:
            build_loop(image, media / "loop.webp")
            print(f"{'loop.webp':<22} {(media / 'loop.webp').stat().st_size / 1e6:.2f} MB")

    # A script rather than JSON, so the viewer also works opened straight from disk.
    (panoramas / "index.js").write_text(
        "// Generated by scripts/make_media.py.\nwindow.PANORAMAS = "
        + json.dumps(index, indent=2) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    main()
