# src/pdfscribe/pipeline/prefilter.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class HeuristicConfig:
    min_w: int = 32  # Minimum width in pixels to consider
    min_h: int = 32  # Minimum height in pixels to consider
    max_aspect: float = (
        8.0  # maximum allowed aspect ratio (to skip very wide/tall images)
    )
    entropy_thresh: float = 2.0  # Minimum entropy required (Low entropy means they're just a meaningless gradatio or something uninformative)
    edge_density_thresh: float = 0.004  # Minimum edge-desntiy required
    sample_downscale: int = 512  # process on at most 512px max side


@dataclass
class HeuristicScore:
    entropy: float
    edge_density: float
    decision: str  # "KEEP" or "SKIP"
    reason: str  # Textual reason for the decision


def _to_small_gray(im: Image.Image, max_side: int) -> Image.Image:
    """
    - Converts an image to grayscale.
    - Downscale it so the largest side is at mode max_side (if needed).
    - Uses bilinear downsampling to avoid moire patterns.
    """
    im = ImageOps.exif_transpose(im)
    im = im.convert("L")  # to grayscale
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR
        )
    return im


def _entropy(gray: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a grayscale image.

    Low entropy means the image is uniform or simple,
    so I assume this may be blank or gradient.
    High entropy means the image has a lot of detail/texture.
    so I assume this may be a text, detailed photo, diagrams, etc.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # remove zeros
    return float(-np.sum(hist * np.log2(hist)))


def _convolve2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    A simple 2D convolution implementation with edge padding
    It's used for edge detection.
    """
    kh, kw = k.shape
    pad_h, pad_w = kh // 2, kw // 2
    ap = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(a, dtype=float)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            out[y, x] = float(np.sum(ap[y : y + kh, x : x + kw] * k))
    return out


def _edge_density(gray: np.ndarray) -> float:
    """
    Use the Sobel operator to find edges int he given grayscale image.
    - Computes the gradient magnitude at each pixel.
    - Counts the number of pixels with edge strength in the top 10% (strongest edges)
    - Returns the fraction of such edge pixels (edge density)
    """
    gx = np.array(
        [
            [1, 0, -1],  # Sobel kernel (horizontal)
            [2, 0, -2],
            [1, 0, -1],
        ],
        dtype=float,  # float for precision
    )
    gy = gx.T
    gxx = _convolve2d(gray, gx)
    gyy = _convolve2d(gray, gy)
    mag = np.hypot(gxx, gyy)
    edges = (mag > np.percentile(mag, 90)).sum()  # crude top-10% edges
    return float(edges) / gray.size


def score_image(path: str, cfg: HeuristicConfig) -> HeuristicScore:
    """
    Score the image at the given path using simple heuristics to decide
    whether to keep or skip it.

    If both entropy and edge density are below thresholds, skips the image as likely uninformative.
    Otherwise, keeps it.
    """
    with Image.open(path) as im:
        w, h = im.size
        if w < cfg.min_w or h < cfg.min_h:
            return HeuristicScore(0.0, 0.0, "SKIP", "too_small")
        ar = max(w / h, h / w)
        if ar > cfg.max_aspect:
            return HeuristicScore(0.0, 0.0, "SKIP", "extreme_aspect")

        sim = _to_small_gray(im, cfg.sample_downscale)
        g = np.asarray(sim, dtype=np.float32)
        ent = _entropy(g)
        ed = _edge_density(g)

        if ent < cfg.entropy_thresh and ed < cfg.edge_density_thresh:
            return HeuristicScore(ent, ed, "SKIP", "low_entropy_low_edges")

        return HeuristicScore(ent, ed, "KEEP", "passes_thresholds")
