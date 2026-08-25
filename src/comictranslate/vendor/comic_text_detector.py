"""Inference-only mask refinement adapted from comic-text-detector.

Upstream: https://github.com/dmMaze/comic-text-detector
Commit: 440b978563c71b758e31aaa315d100faba1efa2f
License: GPL-3.0 (see LICENSES/comic-text-detector-GPL-3.0.txt)

This 2026 adaptation removes training, logging, dataset, YOLO block-detection,
and visualization code. It retains the color/threshold based refined-mask
selection used after the ONNX segmentation output.
"""

from __future__ import annotations

import cv2
import numpy as np


def _min_xor_threshold(thresholded: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, int]:
    inverted = 255 - thresholded
    inverted_score = int(cv2.bitwise_xor(inverted, predicted).sum())
    normal_score = int(cv2.bitwise_xor(thresholded, predicted).sum())
    return (inverted, inverted_score) if inverted_score < normal_score else (thresholded, normal_score)


def _candidate_masks(image: np.ndarray, predicted: np.ndarray) -> list[tuple[np.ndarray, int]]:
    candidates: list[tuple[np.ndarray, int]] = []
    for channel in cv2.split(image):
        _, thresholded = cv2.threshold(
            channel, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY
        )
        candidates.append(_min_xor_threshold(thresholded, predicted))
    candidates.sort(key=lambda item: item[1])
    candidates = candidates[:1]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded = cv2.erode(predicted, np.ones((3, 3), np.uint8), iterations=1)
    pixels = gray[eroded > 127]
    if pixels.size:
        histogram, edges = np.histogram(pixels, bins=255, range=(0, 255))
        total = int(histogram.sum())
        selected: list[float] = []
        for index in np.argsort(histogram)[::-1]:
            if histogram[index] < total * 0.001:
                break
            color = float((edges[index] + edges[index + 1]) / 2)
            if all(abs(color - existing) > 10 for existing in selected):
                selected.append(color)
            if len(selected) == 3:
                break
        for color in selected:
            lower = max(0, round(color - 30))
            upper = min(255, round(color + 30))
            thresholded = cv2.inRange(gray, lower, upper)
            candidates.append(_min_xor_threshold(thresholded, predicted))
    return candidates


def _try_add_component(
    merged: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray,
    label_index: int,
    stat: np.ndarray,
) -> None:
    x, y, width, height, _area = (int(value) for value in stat)
    if width * height < 3:
        return
    local_labels = labels[y : y + height, x : x + width]
    component = np.zeros((height, width), np.uint8)
    component[local_labels == label_index] = 255
    current = merged[y : y + height, x : x + width]
    proposal = cv2.bitwise_or(current, component)
    predicted_local = predicted[y : y + height, x : x + width]
    if cv2.bitwise_xor(proposal, predicted_local).sum() < cv2.bitwise_xor(
        current, predicted_local
    ).sum():
        merged[y : y + height, x : x + width] = proposal


def _recover_thin_glyphs(
    merged: np.ndarray,
    candidates: list[tuple[np.ndarray, int]],
    predicted: np.ndarray,
) -> None:
    """Recover glyph components missed by the conservative XOR criterion."""
    predicted_y, predicted_x = np.where(predicted > 0)
    if not predicted_x.size:
        return
    image_height, image_width = predicted.shape
    margin = max(8, round(min(image_height, image_width) * 0.05))
    envelope = (
        max(0, int(predicted_x.min()) - margin),
        max(0, int(predicted_y.min()) - margin),
        min(image_width, int(predicted_x.max()) + margin + 1),
        min(image_height, int(predicted_y.max()) + margin + 1),
    )
    maximum_area = max(32, round(image_height * image_width * 0.04))
    for candidate, _score in candidates:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate, 8, cv2.CV_16U
        )
        for label_index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label_index])
            center_x, center_y = x + width / 2, y + height / 2
            touches_edge = (
                x <= 1
                or y <= 1
                or x + width >= image_width - 1
                or y + height >= image_height - 1
            )
            inside_prediction_envelope = (
                envelope[0] <= center_x <= envelope[2]
                and envelope[1] <= center_y <= envelope[3]
            )
            text_sized = (
                3 <= area <= maximum_area
                and width < image_width * 0.45
                and height < image_height * 0.45
            )
            if touches_edge or not inside_prediction_envelope or not text_sized:
                continue
            local_labels = labels[y : y + height, x : x + width]
            target = merged[y : y + height, x : x + width]
            target[local_labels == label_index] = 255


def refine_mask(image: np.ndarray, predicted_mask: np.ndarray) -> np.ndarray:
    """Turn a soft ONNX text mask into a pixel-level binary text mask."""
    predicted = np.ascontiguousarray(predicted_mask.astype(np.uint8))
    _, predicted = cv2.threshold(predicted, 30, 255, cv2.THRESH_BINARY)
    # The upstream erosion is tuned for dense Japanese glyphs. It removes most
    # of the thin italic Latin strokes present in the target model's English
    # pages. A one-pixel gate expansion keeps refinement anchored to the ONNX
    # prediction while allowing Otsu components to recover complete glyphs.
    selection_gate = cv2.dilate(predicted, np.ones((3, 3), np.uint8), iterations=1)
    merged = np.zeros_like(predicted)

    candidates = _candidate_masks(image, selection_gate)
    for candidate, _score in candidates:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate, 8, cv2.CV_16U
        )
        for label_index in range(1, count):
            _try_add_component(
                merged, selection_gate, labels, label_index, stats[label_index]
            )
    _recover_thin_glyphs(merged, candidates, predicted)

    inverse = 255 - merged
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        inverse, 8, cv2.CV_16U
    )
    if count > 1:
        areas = np.sort(stats[:, -1])
        hole_limit = areas[-2] if len(areas) > 1 else areas[-1]
        for label_index in range(count):
            if stats[label_index, -1] < hole_limit:
                _try_add_component(
                    merged, selection_gate, labels, label_index, stats[label_index]
                )
    return np.where(merged > 127, 255, 0).astype(np.uint8)
