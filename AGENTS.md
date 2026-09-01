# Repository Guidelines

## Project Structure & Module Organization

Application code uses a `src` layout under `src/comictranslate/`. Keep pipeline orchestration in `pipeline.py`, CLI parsing in `cli.py`, configuration in `config.py`, and stage-specific logic in modules such as `detection.py`, `masking.py`, `inpainting.py`, `qwen.py`, and `rendering.py`. `main.py` is a thin development entry point; the installed `comictranslate` command maps to `comictranslate.cli:main`. Tests live in `tests/` and generally mirror the module under test. Third-party attribution and license texts belong in `THIRD_PARTY_NOTICES.md` and `LICENSES/`.

## Build, Test, and Development Commands

- `uv sync --all-groups` installs the exact Python 3.10.18 environment from `uv.lock`, including pytest.
- `uv run pytest` runs the fast unit suite; real model inference is skipped by default.
- `RUN_MODEL_INTEGRATION=1 uv run pytest tests/test_model_integration.py` runs RT-DETR, ONNX, and LaMa integration checks using local model files.
- `uv run python main.py INPUT -o OUTPUT` runs the CLI from the checkout. Add absolute `--detector-model`, `--qwen-model`, `--text-mask-model`, and `--lama-model` paths when defaults are unavailable.
- `uv build` creates wheel and source distributions through Hatchling.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase module constants. Add type hints and `from __future__ import annotations` to new modules. Prefer small, stage-focused functions, `pathlib.Path`, frozen/slotted dataclasses for value objects, and domain exceptions from `errors.py`. No formatter or linter is configured; keep imports and wrapping consistent with nearby code and avoid unrelated formatting changes.

## Testing Guidelines

Use pytest and name files `test_<module>.py` and tests `test_<behavior>()`. Put shared setup in `tests/conftest.py`; use `tmp_path`, fixtures, and mocks so unit tests never require model downloads or mutate source assets. Mark model-backed tests with `@pytest.mark.integration` and gate them behind `RUN_MODEL_INTEGRATION=1`. There is no numeric coverage gate; every behavior change should include a focused regression test, especially for image dimensions, alpha preservation, region ordering, and failure cleanup.

## Commit & Pull Request Guidelines

History follows Conventional Commits, often with a scope and concise Chinese summary: `fix(rendering): 修正竖排文字对齐`. Use one logical change per commit (`feat`, `fix`, `test`, `docs`, `refactor`). Pull requests should explain intent and user-visible effects, list commands run and skipped integration checks, link related issues, and include before/after images for rendering or pipeline-output changes. Never commit local model files, translated pages, debug output, or machine-specific absolute paths.

## Working Principles

1. **Understand before modifying**: Read the relevant code and call chain before making changes. If anything is ambiguous, state your assumptions explicitly; if it cannot be determined, ask rather than guess.
2. **Prefer simplicity**: Solve the current problem with the minimum amount of code. Do not add features that were not requested. Avoid premature abstraction for one-off logic and unnecessary encapsulation.
3. **Make surgical changes**: Touch only what is necessary for the task. Do not opportunistically beautify, refactor, or modify unrelated code, comments, or formatting.
4. **Follow project conventions**: Project conventions take precedence over personal preferences. If two implementation patterns conflict, use the newer or more explicitly established one as the primary pattern and explain why. Flag the other for future cleanup rather than mixing the two.
5. **Execute toward the goal**: Define the success criteria first, then validate continuously. After each significant step, state what has been completed, what has been verified, and what remains.
6. **Expose uncertainty explicitly**: Do not hide failures, skipped tests, or uncertain conclusions. Clearly call out anything that cannot be confirmed.
7. **Test the intended behavior**: Tests should express business intent, not merely cover implementation behavior. When business logic changes, tests should be updated to reflect the new rules. If tests are skipped, do not claim that “tests passed.” If tests were not run, state that explicitly.
8. **Test proportionally**: Keep validation proportional to the scope and risk of the change. Prefer the smallest targeted test set that can verify the intended behavior and likely regressions. Do not run project-wide test suites, full builds, end-to-end tests, or unrelated checks by default for localized changes. Broaden the test scope only when the change affects shared infrastructure, cross-cutting behavior, public interfaces, multiple modules, or when targeted tests are insufficient to establish confidence. Avoid repeating tests that provide no additional validation.
