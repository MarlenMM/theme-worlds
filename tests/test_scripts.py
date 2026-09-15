"""The measurement script, on a synthetic folder of outputs."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def test_measure_seams_counts_a_byte_identical_rerun_once(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rng = np.random.default_rng(0)
    first = Image.fromarray(rng.integers(0, 255, (64, 256, 3), dtype=np.uint8))
    second = Image.fromarray(rng.integers(0, 255, (64, 256, 3), dtype=np.uint8))
    first.save(outputs / "cyberpunk_0908-1736_seed1000.png")
    first.save(outputs / "cyberpunk_0909-0037_seed1000.png")      # same seed, second session
    second.save(outputs / "fantasy+medieval_0909-0248_seed1001.png")
    first.save(outputs / "cyberpunk_0908-1736_seamcheck.png")      # a figure, not a panorama

    results = tmp_path / "results"
    subprocess.run([sys.executable, str(REPO / "scripts" / "measure_seams.py"), str(outputs),
                    "--results", str(results)], check=True, capture_output=True)

    with open(results / "seam_ratios.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["identical_to"] for row in rows] == ["", "cyberpunk_0908-1736_seed1000.png", ""]
    assert "| Panoramas (byte-identical copies counted once) | **2** |" in (
        results / "seam_ratios.md").read_text(encoding="utf-8")
