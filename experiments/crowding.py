"""Try this 3: how far apart are the claims in each modality? (make try-crowding)

A search can only tell claims apart if their vectors are far apart. This measures the average
cosine similarity between different claims' X-rays, pet photos, claim notes and vet notes, then
edits each image (crop, shrink, grey) and checks how much room is left between the right claim
and the next one. No Docker needed.

Change EDITS (the same edits claims/evaluate.py uses) and run it again.
"""

import sys

import numpy as np
from PIL import Image

from common import CLAIMS, config, embed_image, images, texts

sys.path.insert(0, str(config.ROOT))
from claims.evaluate import EDITS  # noqa: E402


def between(vectors: np.ndarray) -> float:
    s = vectors @ vectors.T
    return float(s[~np.eye(len(s), dtype=bool)].mean())


def main() -> int:
    vectors = {"X-ray": images("xray_path"), "pet photo": images("photo_path"),
               "claim note": texts([c["claim_note"] for c in CLAIMS]),
               "vet note": texts([c["vet_note"] for c in CLAIMS])}
    print("Average similarity between two different claims (1.0 = identical):")
    for name, v in vectors.items():
        print(f"  {name:<11} {between(v):.3f}")

    print(f"\nEach image edited, then searched against the originals ({len(CLAIMS)} claims x {len(EDITS)} edits):")
    print(f"{'modality':<10}  {'edit':<16}  {'right first':>11}  {'margin':>7}")
    for name, field in (("X-ray", "xray_path"), ("pet photo", "photo_path")):
        stored = vectors[name]
        for edit_name, edit in EDITS.items():
            hits, margins = 0, []
            for i, c in enumerate(CLAIMS):
                query = np.array(embed_image(edit(Image.open(config.ROOT / c[field]).convert("RGB"))))
                s = stored @ query
                hits += int(np.argmax(s) == i)
                margins.append(s[i] - np.delete(s, i).max())
            print(f"{name:<10}  {edit_name:<16}  {hits:>6} of {len(CLAIMS)}  {np.mean(margins):>+7.3f}")
    print("\nMargin: similarity to the right claim minus the best wrong one, averaged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
