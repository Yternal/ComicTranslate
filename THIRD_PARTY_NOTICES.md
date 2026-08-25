# Third-party notices

## comic-text-detector

- Upstream: <https://github.com/dmMaze/comic-text-detector>
- Vendored commit: `440b978563c71b758e31aaa315d100faba1efa2f`
- License: GNU General Public License v3.0
- License text: `LICENSES/comic-text-detector-GPL-3.0.txt`

The code in `src/comictranslate/vendor/comic_text_detector.py` is a modified,
inference-only adaptation. Training, dataset generation, experiment tracking,
visualization, YOLO block detection, `wandb`, and `trdg` integration were
removed. The retained color-threshold and connected-component mask refinement
was adjusted in 2026 to recover thin Latin glyphs while excluding ROI-edge
components such as speech-bubble borders.
