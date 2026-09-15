"""Structural checks on the notebook: what it loads, and how it was saved."""

from __future__ import annotations

import ast

from scripts.notebook_code import NOTEBOOK, code_cells, definitions, read_notebook

# The project rules allowed exactly one generative checkpoint. CLIP is the
# permitted "pretrained reward model": it scores finished images and nothing more.
ALLOWED_WEIGHTS = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "openai/clip-vit-large-patch14",
}


def parsed_cells():
    return [ast.parse(source) for source in code_cells()]


def test_every_code_cell_is_valid_python():
    assert len(parsed_cells()) > 0


def test_only_sd15_and_the_clip_scorer_are_ever_loaded():
    constants, model_ids = {}, []
    for tree in parsed_cells():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                constants.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"):
                model_ids.append(node.args[0])
    assert model_ids, "no from_pretrained calls found"
    loaded = set()
    for arg in model_ids:
        assert isinstance(arg, ast.Name), f"model id passed inline: {ast.unparse(arg)}"
        loaded.add(constants[arg.id])
    assert loaded == ALLOWED_WEIGHTS


def test_initial_noise_is_drawn_on_the_cpu_generator():
    # CPU RNG is identical across GPUs, which is what makes a seed reproduce
    # byte-for-byte on a different Colab machine.
    assert 'torch.Generator(device="cpu")' in definitions()["multidiffusion"]


def test_theme_menu_has_fifty_themes_plus_a_custom_slot():
    menu = ast.literal_eval(definitions()["THEME_PROMPTS"].split("=", 1)[1].strip())
    assert len(menu) == 51 and "custom" in menu
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in menu.values())


def test_notebook_was_saved_after_a_single_top_to_bottom_run():
    cells = [cell for cell in read_notebook()["cells"] if cell["cell_type"] == "code"]
    assert [cell["execution_count"] for cell in cells] == list(range(1, len(cells) + 1))


def test_notebook_stays_small_enough_for_github_to_render():
    assert NOTEBOOK.stat().st_size < 5_000_000
    for cell in read_notebook()["cells"]:
        for output in cell.get("outputs", []):
            assert not {"image/png", "image/jpeg"} <= set(output.get("data", {}))
