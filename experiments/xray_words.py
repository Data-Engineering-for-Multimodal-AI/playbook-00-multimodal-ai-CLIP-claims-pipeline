"""Try this 2: can words find an X-ray? (make try-xray-words)

CLIP was trained on photos and their captions from the web, not on radiographs. Each sample claim
says which part of the body was X-rayed. This searches the five X-rays with a sentence naming that
body part, and does the same for the photos with the breed, for comparison. No Docker needed.

Change the sentences in BODY_PARTS and run it again.
"""

import sys

import numpy as np

from common import CLAIMS, images, rank_scores, texts

# From each claim note (data/ClaimNotes/) and vet note (data/VetNotes/)
BODY_PARTS = {
    "CLM001": "an X-ray of a dog's hip",
    "CLM002": "an X-ray of a dog's shoulder",
    "CLM003": "an X-ray of a dog's hind legs",
    "CLM004": "a chest X-ray of a dog",
    "CLM005": "an X-ray of a dog's lower spine",
}


def main() -> int:
    xrays = images("xray_path")
    photos = images("photo_path")
    rows = [
        ("body part -> X-ray", [BODY_PARTS[c["claim_id"]] for c in CLAIMS], xrays),
        ("breed -> pet photo", [f"a photo of a {c['breed']}" for c in CLAIMS], photos),
    ]
    print(f"{'search':<20}  {'right claim first':>17}  {'margin':>7}  {'spread':>6}")
    for name, queries, targets in rows:
        top1, margin, spread = rank_scores(texts(queries) @ targets.T)
        print(f"{name:<20}  {top1 * 5:>11.0f} of 5  {margin:>+7.3f}  {spread:>6.3f}")
    print("\nWhat each sentence found first among the X-rays:")
    similarity = texts([BODY_PARTS[c["claim_id"]] for c in CLAIMS]) @ xrays.T
    for c, row in zip(CLAIMS, similarity):
        best = CLAIMS[int(np.argmax(row))]
        print(f"  {BODY_PARTS[c['claim_id']]:<31} -> {best['pet_name']:<5} (expected {c['pet_name']}; "
              f"similarities {row.min():.3f} to {row.max():.3f})")
    print("\nSpread: the highest similarity in a row minus the lowest, averaged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
