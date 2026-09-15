"""Properties of the MultiDiffusion sampler, tested on the notebook's own code."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

LATENT_H, LATENT_W = 64, 128          # a 1024x512 canvas: the notebook's control size


def coverage(views, width):
    counts = np.zeros(width, dtype=int)
    for _, _, w0, w1 in views:
        counts[np.arange(w0, w1) % width] += 1
    return counts


def embed(nb, text, n_views):
    return nb.encode_prompt(text).expand(n_views, -1, -1)


def sample(nb, views, cond_sets, cond_weights, seed=0):
    uncond = nb.encode_prompt(nb.NEGATIVE_PROMPT)
    return nb.multidiffusion(cond_sets, cond_weights, uncond, views, LATENT_H, LATENT_W,
                             steps=3, guidance=7.5, seed=seed, view_batch_size=2,
                             progress_every=0)


# --- windows -----------------------------------------------------------------

def test_circular_windows_cover_every_column_equally(nb):
    views = nb.build_views(64, 256, stride=16, circular=True)
    assert len(views) == 16                       # 2048x512 at stride 16, as in the write-up
    assert set(coverage(views, 256)) == {4}       # no column is special, the join included


def test_strip_windows_stay_inside_the_canvas_and_thin_out_at_the_ends(nb):
    views = nb.build_views(64, 100, stride=16, circular=False)
    assert all(w0 >= 0 and w1 <= 100 for _, _, w0, w1 in views)
    counts = coverage(views, 100)
    assert counts.min() >= 1
    assert counts[0] < counts[50]                 # a strip has ends; a loop does not


def test_taller_canvases_are_fully_covered_by_several_rows_of_windows(nb):
    views = nb.build_views(96, 128, stride=16, circular=True)
    grid = np.zeros((96, 128), dtype=int)
    for h0, h1, w0, w1 in views:
        grid[h0:h1, np.arange(w0, w1) % 128] += 1
    assert len({h0 for h0, _, _, _ in views}) > 1
    assert grid.min() >= 1


def test_a_canvas_smaller_than_one_window_is_rejected(nb):
    with pytest.raises(ValueError):
        nb.build_views(32, 256)


def test_region_weights_give_each_theme_one_side_of_the_loop(nb):
    views = nb.build_views(64, 256, stride=16, circular=True)
    weights = nb.theme_weights(views, 256, sharpness=2.0)
    weight_at = dict(zip((w0 for _, _, w0, _ in views), weights, strict=True))
    assert all(0.0 <= w <= 1.0 for w in weight_at.values())
    assert weight_at[224] == 1.0                  # window centred on the join: pure theme A
    assert weight_at[96] == 0.0                   # window centred opposite it: pure theme B
    for w0, w in weight_at.items():               # mirror-symmetric around the loop
        assert w == pytest.approx(weight_at[(192 - w0) % 256])


# --- the seam metric ---------------------------------------------------------

def periodic_texture(width=512, height=64, seed=0):
    """A sum of whole-period sinusoids: exactly periodic, so it genuinely loops."""
    rng = np.random.default_rng(seed)
    x = np.arange(width)[None, :, None] / width
    image = np.full((height, width, 3), 128.0)
    for k in range(1, 12):
        amplitude = rng.uniform(4, 16, size=(height, 1, 3))
        phase = rng.uniform(0, 2 * np.pi, size=(height, 1, 3))
        image += amplitude * np.sin(2 * np.pi * k * x + phase)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def test_seam_ratio_separates_a_loop_from_a_cropped_strip(nb):
    loop = periodic_texture()
    strip = loop.crop((0, 0, 384, 64))            # three quarters of a period: the ends disagree
    assert nb.seam_ratio(loop) < 1.5
    assert nb.seam_ratio(strip) > 5.0


def test_rolling_by_half_moves_the_join_to_the_centre(nb):
    image = Image.fromarray(np.arange(8, dtype=np.uint8)[None, :].repeat(2, axis=0))
    assert np.asarray(nb.roll_half(image))[0].tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


# --- sampling ----------------------------------------------------------------

def test_a_seed_reproduces_exactly_and_a_different_seed_does_not(nb):
    views = nb.build_views(LATENT_H, LATENT_W, stride=32)
    cond, weights = [embed(nb, "neon megacity", len(views))], torch.ones(1, len(views))
    first, again, other = (sample(nb, views, cond, weights, seed=s) for s in (7, 7, 8))
    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


@pytest.mark.parametrize("circular", [True, False])
def test_only_circular_sampling_treats_every_column_alike(nb, monkeypatch, circular):
    """Roll the starting noise sideways and the result rolls with it, exactly.

    That is what "the loop is a property of the sampling" means: with circular
    windows no column of the canvas is special, including the one where the
    image file happens to start. Without them, the ends of the strip behave
    differently and the property fails; the ``False`` case proves the test
    can tell the difference.
    """
    views = nb.build_views(LATENT_H, LATENT_W, stride=32, circular=circular)
    cond, weights = [embed(nb, "ice cave", len(views))], torch.ones(1, len(views))
    noise = torch.randn(1, 4, LATENT_H, LATENT_W, generator=torch.Generator().manual_seed(0))

    def from_noise(initial):
        monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: initial.clone())
        return sample(nb, views, cond, weights)

    base = from_noise(noise)
    shifted = from_noise(torch.roll(noise, 32, dims=-1))
    assert torch.allclose(shifted, torch.roll(base, 32, dims=-1), atol=1e-4) is circular


def test_composing_a_prompt_with_itself_is_ordinary_guidance(nb):
    views = nb.build_views(LATENT_H, LATENT_W, stride=32)
    n = len(views)
    a = embed(nb, "floating islands", n)
    single = sample(nb, views, [a], torch.ones(1, n))
    composed = sample(nb, views, [a, a], torch.full((2, n), 0.5))
    assert torch.allclose(single, composed, atol=1e-5)


def test_composition_weights_move_between_the_two_themes(nb):
    views = nb.build_views(LATENT_H, LATENT_W, stride=32)
    n = len(views)
    a, b = embed(nb, "floating islands", n), embed(nb, "medieval castle town", n)
    only_a = sample(nb, views, [a], torch.ones(1, n))
    all_a = sample(nb, views, [a, b], torch.tensor([[1.0] * n, [0.0] * n]))
    half = sample(nb, views, [a, b], torch.full((2, n), 0.5))
    assert torch.allclose(only_a, all_a, atol=1e-5)
    assert not torch.allclose(only_a, half, atol=1e-3)


# --- conditioning ------------------------------------------------------------

def test_the_four_fusion_modes_build_the_conditioning_they_describe(nb):
    themes = ["cyberpunk", "medieval"]
    views = nb.build_views(64, 256, stride=16)
    n = len(views)
    a = nb.encode_prompt(f"{nb.THEME_PROMPTS['cyberpunk']}, {nb.QUALITY_SUFFIX}")[0]
    b = nb.encode_prompt(f"{nb.THEME_PROMPTS['medieval']}, {nb.QUALITY_SUFFIX}")[0]

    def build(mode):
        return nb.build_conditioning(themes, views, 256, mode, 0.5, 2.0)

    prompts, cond_sets, weights, mode = build("compose")
    assert (len(prompts), len(cond_sets), mode) == (2, 2, "compose")
    assert torch.allclose(weights.sum(dim=0), torch.ones(n))
    assert torch.equal(cond_sets[0][5], a) and torch.equal(cond_sets[1][5], b)

    _, cond_sets, _, mode = build("blend")
    assert (len(cond_sets), mode) == (1, "blend")
    assert torch.allclose(cond_sets[0][0], (a + b) / 2, atol=1e-6)

    _, cond_sets, _, mode = build("regions")
    order = [w0 for _, _, w0, _ in views]
    assert (len(cond_sets), mode) == (1, "regions")
    assert torch.allclose(cond_sets[0][order.index(224)], a)       # pure theme A at the join
    assert torch.allclose(cond_sets[0][order.index(96)], b)        # pure theme B opposite

    prompts, cond_sets, _, mode = build("single_prompt")
    assert (len(prompts), len(cond_sets), mode) == (1, 1, "single_prompt")
    assert all(nb.THEME_PROMPTS[t] in prompts[0] for t in themes)

    assert nb.build_conditioning(["cyberpunk"], views, 256, "compose", 0.5, 2.0)[3] == "single_theme"
    with pytest.raises(ValueError):
        build("average_everything")


def test_render_world_runs_end_to_end_and_crops_the_decoder_padding_off(nb):
    result = nb.render_world(["arctic_ice_cave", "cozy_village_snowfall"], 1024, 512, seed=3,
                             steps=2, stride=32, quiet=True)
    assert result["image"].size == (1024, 512)
    assert result["fusion"] == "compose" and result["conditioning_branches"] == 2
    assert result["view_batch_size"] == 2          # halved so the real UNet batch stays constant
    assert result["num_views"] == 4
    assert result["seam_ratio"] == pytest.approx(nb.seam_ratio(result["image"]), abs=1e-3)
