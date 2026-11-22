"""Try this 1: which words find a pet's photo? (make try-prompts)

Text and images share CLIP's vector space, so a sentence can search the pet photos. This builds one
query per claim from a template (the breed, the pet's name, the claim note, ...) and checks whether
it ranks that claim's photo first among the five. No Docker needed.

Add your own templates to TEMPLATES and run it again. {breed}, {name} and {note} are filled in from
data/StructuredData/policy_details.csv and data/ClaimNotes/.
"""

import sys
import time

from common import CLAIMS, images, rank_scores, texts

TEMPLATES = [
    "{breed}",
    "a photo of a {breed}",
    "a photo of a dog",
    "{name}",
    "a photo of {name}, a dog",
    "{note}",
]


def main() -> int:
    started = time.perf_counter()
    photos = images("photo_path")
    print(f"{len(CLAIMS)} pet photos; one query per claim; similarity is cosine (1.0 = same direction)\n")
    print(f"{'query template':<26}  {'right photo first':>17}  {'margin':>7}  example query")
    for template in TEMPLATES:
        queries = [template.format(breed=c["breed"], name=c["pet_name"], note=c["claim_note"]) for c in CLAIMS]
        top1, margin, _ = rank_scores(texts(queries) @ photos.T)
        example = queries[2] if len(queries[2]) < 40 else queries[2][:37] + "..."
        print(f"{template:<26}  {top1 * 5:>11.0f} of 5  {margin:>+7.3f}  {example}")
    print(f"\nMargin: similarity to the right photo minus the best wrong one, averaged ({time.perf_counter() - started:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
