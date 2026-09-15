"""The notebook's own definitions, wired to a CPU-sized stub pipeline.

Nothing here downloads a model. The stub stands in for SD v1.5's tokenizer,
text encoder, UNet and VAE with a few deterministic tensor operations that keep
the two properties the sampler depends on:

* the UNet has a spatial receptive field with **zero padding**, so a window's
  border is treated differently from its interior — exactly the behaviour that
  makes a naive tiled panorama show seams;
* its prediction **depends on the conditioning**, so the fusion modes are
  distinguishable.

The scheduler is the real ``diffusers.DDIMScheduler`` with SD v1.5's settings.
"""

from __future__ import annotations

import math
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.notebook_code import load  # noqa: E402

HIDDEN = 8
SD15_SCHEDULER = dict(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False, steps_offset=1, prediction_type="epsilon",
)
BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16


class StubTokenizer:
    model_max_length = 77

    def __call__(self, text, padding=None, max_length=None, truncation=False, return_tensors=None):
        ids = [49406] + [zlib.crc32(word.encode()) % 49000 for word in text.split()] + [49407]
        if truncation and len(ids) > max_length:
            ids = ids[: max_length - 1] + [49407]
        if padding == "max_length":
            ids += [49407] * (max_length - len(ids))
        return SimpleNamespace(input_ids=torch.tensor([ids]))


class StubTextEncoder:
    config = SimpleNamespace(hidden_size=HIDDEN)

    def __init__(self):
        self.table = torch.randn(49408, HIDDEN, generator=torch.Generator().manual_seed(0))

    def __call__(self, input_ids):
        return (self.table[input_ids],)


class StubUNet:
    config = SimpleNamespace(in_channels=4)
    kernel = BLUR.expand(4, 1, 3, 3).contiguous()

    def __call__(self, x, t, encoder_hidden_states):
        spatial = F.conv2d(x.float(), self.kernel, padding=1, groups=4)
        cond = torch.tanh(encoder_hidden_states.float().mean(dim=1)[:, :4])[:, :, None, None]
        return SimpleNamespace(sample=(0.8 * spatial + 0.2 * cond).to(x.dtype))


class StubVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=0.18215)
    kernel = BLUR.expand(3, 1, 3, 3).contiguous()

    def decode(self, latents, return_dict=False):
        rgb = torch.tanh(F.conv2d(latents[:, :3].float(), self.kernel, padding=1, groups=3))
        return (F.interpolate(rgb, scale_factor=8, mode="nearest"),)

    def enable_tiling(self):
        pass


NOTEBOOK_NAMES = (
    "VAE_SCALE", "WINDOW", "THEME_PROMPTS", "QUALITY_SUFFIX", "NEGATIVE_PROMPT",
    "build_views", "theme_weights", "effective_view_batch", "count_tokens", "encode_prompt",
    "multidiffusion", "decode_canvas", "seam_ratio", "roll_half", "fused_prompt",
    "build_conditioning", "render_world",
)


@pytest.fixture(scope="session")
def nb():
    """Namespace holding the notebook's definitions, as if Steps 1-6 had run."""
    pipe = SimpleNamespace(
        tokenizer=StubTokenizer(), text_encoder=StubTextEncoder(), unet=StubUNet(),
        vae=StubVAE(), scheduler=DDIMScheduler(**SD15_SCHEDULER),
    )
    namespace = load(
        *NOTEBOOK_NAMES,
        torch=torch, np=np, math=math, time=time, Image=Image, ImageDraw=ImageDraw,
        ImageFont=ImageFont, pipe=pipe, DEVICE="cpu", DTYPE=torch.float32,
        # Step 6 settings that render_world falls back to.
        NUM_STEPS=3, GUIDANCE_SCALE=7.5, VIEW_STRIDE=32, VIEW_BATCH_SIZE=4,
        FUSION_MODE="compose", FUSION_LAMBDA=0.5, BLEND_SHARPNESS=2.0,
    )
    return SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith("__")})
