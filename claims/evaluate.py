"""
How well does the search find the right claim? Each test uses queries that differ from what is stored:

  vet note -> claim       the vet's notes (different words, same case) against the claim descriptions;
                          also with BM25 keyword search, for comparison
  breed -> photo          "a photo of a <breed>" against the pet photos (text finds images)
  edited photo -> photo   each photo cropped, flipped, in grey, rotated or shrunk
  edited X-ray -> X-ray   the same edits on each X-ray

    python -m claims.evaluate
"""
import json
from typing import Callable, Dict, List

from PIL import Image, ImageOps

from claims import config, search, store

EDITS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "crop 70%": lambda im: im.crop((int(im.width * .15), int(im.height * .15), int(im.width * .85), int(im.height * .85))),
    "flip": ImageOps.mirror,
    "grey": lambda im: ImageOps.grayscale(im).convert("RGB"),
    "rotate 10°": lambda im: im.rotate(10, expand=True, fillcolor=(0, 0, 0)),
    "shrink to 96 px": lambda im: im.resize((96, max(1, int(96 * im.height / im.width)))),
}


def _score(rows: List[Dict]) -> Dict:
    ranks = [r["rank"] for r in rows]
    return {"queries": len(rows), "top1": round(sum(r == 1 for r in ranks) / len(ranks), 2),
            "mean_reciprocal_rank": round(sum(1 / r for r in ranks if r) / len(ranks), 2), "results": rows}


def _rank(results: List[Dict], claim_id: str) -> int:
    ids = [r["claim_id"] for r in results]
    return ids.index(claim_id) + 1 if claim_id in ids else 0


def run(client) -> Dict:
    claims = store.load_claims()
    n = len(claims)
    out = {}
    rows, keyword_rows = [], []
    for c in claims:
        found = search.by_text(client, c["vet_note"], n)
        rows.append({"query": f"vet note {c['claim_id']}", "expected": c["claim_id"], "rank": _rank(found, c["claim_id"]),
                     "top": found[0]["claim_id"]})
        kw = search.keyword(client, c["vet_note"], n, fields=("claim_note",))   # not the stored vet notes
        keyword_rows.append({"query": f"vet note {c['claim_id']}", "expected": c["claim_id"],
                             "rank": _rank(kw, c["claim_id"]), "top": kw[0]["claim_id"] if kw else None})
    out["vet note -> claim (CLIP text)"] = _score(rows)
    out["vet note -> claim (BM25 keywords)"] = _score(keyword_rows)

    rows = []
    for c in claims:
        found = search.by_description(client, f"a photo of a {c['breed']}", n)
        rows.append({"query": f"a photo of a {c['breed']}", "expected": c["claim_id"], "rank": _rank(found, c["claim_id"]),
                     "top": found[0]["claim_id"]})
    out["breed -> photo (text to image)"] = _score(rows)

    for kind, field, fn in (("photo", "photo_path", search.by_photo), ("X-ray", "xray_path", search.by_xray)):
        rows = []
        for c in claims:
            original = Image.open(config.ROOT / c[field]).convert("RGB")
            for name, edit in EDITS.items():
                found = fn(client, edit(original), n)
                rows.append({"query": f"{c['claim_id']} {kind}, {name}", "expected": c["claim_id"],
                             "rank": _rank(found, c["claim_id"]), "top": found[0]["claim_id"],
                             "similarity": found[0]["similarity"]})
        out[f"edited {kind} -> {kind}"] = _score(rows)
    return out


def render(results: Dict) -> str:
    lines = [f"{'Test':<36}{'Queries':>8}{'Top-1':>8}{'MRR':>7}"]
    for name, r in results.items():
        lines.append(f"{name:<36}{r['queries']:>8}{r['top1']:>8.2f}{r['mean_reciprocal_rank']:>7.2f}")
    misses = [(name, row) for name, r in results.items() for row in r["results"] if row["rank"] != 1]
    if misses:
        lines.append("\nMisses (query: expected -> found first):")
        for name, row in misses:
            lines.append(f"  {row['query']}: {row['expected']} -> {row['top']} (rank {row['rank'] or 'not found'})")
    return "\n".join(lines)


def main() -> None:
    client = store.connect()
    try:
        if not client.collections.exists(config.COLLECTION):
            raise SystemExit("No claims loaded yet: python store_with_embedding.py")
        results = run(client)
    finally:
        client.close()
    (config.ROOT / "output").mkdir(exist_ok=True)
    with open(config.ROOT / "output" / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(render(results))


if __name__ == "__main__":
    main()
