"""Shared helpers for the Try this experiments. They embed the five sample claims with CLIP in this
process and rank them with cosine similarity, the metric the Weaviate collection uses, so no Docker
is needed. With five claims, an exact ranking gives the same order Weaviate's index would."""

import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claims import config, store  # noqa: E402
from claims.embeddings import embed_image, embed_text  # noqa: E402

CLAIMS = store.load_claims()


def images(field: str) -> np.ndarray:
    """One row per claim: the embedding of its X-ray (xray_path) or pet photo (photo_path)."""
    return np.array([embed_image(config.ROOT / c[field]) for c in CLAIMS])


def texts(values) -> np.ndarray:
    return np.array([embed_text(v) for v in values])


def rank_scores(similarity: np.ndarray):
    """similarity[i, j]: query i against claim j, where claim i is the right answer. Returns top-1
    accuracy, the mean margin (right claim minus the best wrong one) and the spread of each row."""
    n = len(similarity)
    top1 = float(np.mean(similarity.argmax(axis=1) == np.arange(n)))
    margin = float(np.mean([similarity[i, i] - np.delete(similarity[i], i).max() for i in range(n)]))
    spread = float(np.mean(similarity.max(axis=1) - similarity.min(axis=1)))
    return top1, margin, spread
