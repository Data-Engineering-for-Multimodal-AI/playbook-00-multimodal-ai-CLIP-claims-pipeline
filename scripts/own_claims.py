#!/usr/bin/env python3
"""
Use your own claims: check a folder laid out like data/, then load it into Weaviate.

    python scripts/own_claims.py --dir my_claims --check-only   # check every file, send nothing
    python scripts/own_claims.py --dir my_claims                # check, embed and load (replaces the sample)

The folder has the same layout as data/ (see README, "Use your own data"). Loading replaces the claims
in the ClaimDocument collection; python store_with_embedding.py (or make run) puts the sample back.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POLICY_COLUMNS = ["claim_id", "policy_id", "pet_name", "breed", "clinic", "xray_covered", "surgery_covered",
                  "dental_covered", "deductible", "co_pay_percent"]
HISTORY_COLUMNS = ["claim_id", "date", "type", "amount"]
FLAGS = ["xray_covered", "surgery_covered", "dental_covered"]
NUMBERS = ["deductible", "co_pay_percent"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_SIDE = 32
CLIP_TOKENS = 77    # CLIP's text encoder reads at most 77 tokens; the rest is cut off


def _lfs(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(100).startswith(b"version https://git-lfs")


def _read_csv(path: Path, columns: List[str], name: str, problems: List[str]):
    if not path.is_file():
        problems.append(f"{name}: missing")
        return None
    if _lfs(path):
        problems.append(f"{name}: a Git LFS pointer, not the file (run git lfs pull)")
        return None
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        problems.append(f"{name}: not UTF-8 text")
        return None
    except Exception as e:  # noqa: BLE001
        problems.append(f"{name}: can't read as CSV ({type(e).__name__})")
        return None
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        problems.append(f"{name}: missing column{'s' if len(missing) > 1 else ''} {', '.join(missing)}")
        return None
    return frame


def _text(path: Path, name: str, problems: List[str], notes: List[str], tokenizer) -> None:
    if not path.is_file():
        problems.append(f"{name}: missing")
        return
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs"):
        problems.append(f"{name}: a Git LFS pointer, not the file (run git lfs pull)")
        return
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        problems.append(f"{name}: not UTF-8 text")
        return
    if not text:
        problems.append(f"{name}: empty")
        return
    if tokenizer is not None:
        tokens = len(tokenizer(text)["input_ids"])
        if tokens > CLIP_TOKENS:
            notes.append(f"{name}: {tokens} tokens; CLIP embeds only the first {CLIP_TOKENS}")


def _image(folder: Path, kind: str, pet: str, problems: List[str], notes: List[str]) -> None:
    rel = folder.name
    matches = sorted(p for p in folder.iterdir() if p.is_file() and p.name.lower().startswith(pet.lower() + "_")) \
        if folder.is_dir() else []
    if not matches:
        problems.append(f"{rel}/: no {kind} for {pet} (a file named {pet.lower()}_<anything>.png or .jpg)")
        return
    if len(matches) > 1:
        notes.append(f"{rel}/: {len(matches)} files for {pet}; only {matches[0].name} is used")
    path = matches[0]
    name = f"{rel}/{path.name}"
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        problems.append(f"{name}: not an image file type ({', '.join(sorted(IMAGE_SUFFIXES))})")
        return
    if _lfs(path):
        problems.append(f"{name}: a Git LFS pointer, not the image (run git lfs pull)")
        return
    try:
        with Image.open(path) as im:
            im.load()
            size = im.size
    except (UnidentifiedImageError, OSError) as e:
        problems.append(f"{name}: can't open as an image ({type(e).__name__})")
        return
    if min(size) < MIN_SIDE:
        problems.append(f"{name}: too small ({size[0]}x{size[1]}), needs at least {MIN_SIDE}x{MIN_SIDE}")


def scan(folder: Path, tokenizer=None) -> Tuple[List[Dict], List[str], List[str]]:
    """Check the folder. Returns (claims as rows of policy_details.csv, problems, notes)."""
    folder = Path(folder)
    problems: List[str] = []
    notes: List[str] = []
    if not folder.is_dir():
        return [], [f"{folder}: not a folder"], notes
    policies = _read_csv(folder / "StructuredData" / "policy_details.csv", POLICY_COLUMNS,
                         "StructuredData/policy_details.csv", problems)
    history = _read_csv(folder / "StructuredData" / "claim_history.csv", HISTORY_COLUMNS,
                        "StructuredData/claim_history.csv", problems)
    if policies is None:
        return [], problems, notes
    rows = policies.to_dict("records")
    if not rows:
        problems.append("StructuredData/policy_details.csv: no claims")
    seen_ids, seen_pets = set(), set()
    for line, row in enumerate(rows, start=2):
        where = f"StructuredData/policy_details.csv line {line}"
        claim_id, pet = row["claim_id"].strip(), row["pet_name"].strip()
        empty = [c for c in POLICY_COLUMNS if not str(row[c]).strip()]
        if empty:
            problems.append(f"{where}: empty {', '.join(empty)}")
            continue
        if claim_id in seen_ids:
            problems.append(f"{where}: claim_id {claim_id} appears twice")
        if pet.lower() in seen_pets:
            problems.append(f"{where}: pet_name {pet} appears twice; photos and X-rays are found by pet name")
        if "_" in pet or "/" in pet:
            problems.append(f"{where}: pet_name {pet!r} can't contain _ or /; files are named <pet>_<anything>")
        seen_ids.add(claim_id)
        seen_pets.add(pet.lower())
        for flag in FLAGS:
            if row[flag].strip() not in ("True", "False"):
                problems.append(f"{where}: {flag} is {row[flag]!r}, needs True or False")
        for number in NUMBERS:
            try:
                float(row[number])
            except ValueError:
                problems.append(f"{where}: {number} is {row[number]!r}, not a number")
        _text(folder / "ClaimNotes" / f"{claim_id}_description.txt", f"ClaimNotes/{claim_id}_description.txt",
              problems, notes, tokenizer)
        _text(folder / "VetNotes" / f"{claim_id}_vet_notes.txt", f"VetNotes/{claim_id}_vet_notes.txt",
              problems, notes, tokenizer)
        _image(folder / "PetXrays", "X-ray", pet, problems, notes)
        _image(folder / "PetPhotos", "photo", pet, problems, notes)
    if history is not None:
        for line, row in enumerate(history.to_dict("records"), start=2):
            where = f"StructuredData/claim_history.csv line {line}"
            if row["claim_id"].strip() not in seen_ids:
                problems.append(f"{where}: claim_id {row['claim_id']!r} isn't in policy_details.csv")
            try:
                float(row["amount"])
            except ValueError:
                problems.append(f"{where}: amount is {row['amount']!r}, not a number")
            if pd.isna(pd.to_datetime(row["date"], format="%Y-%m-%d", errors="coerce")):
                problems.append(f"{where}: date is {row['date']!r}, needs YYYY-MM-DD")
    return rows, problems, notes


def _tokenizer():
    """CLIP's tokenizer, to warn about notes it would cut off. None if the model isn't downloaded yet."""
    try:
        from transformers import CLIPTokenizer
        from transformers.utils import logging

        from claims import config
        logging.set_verbosity_error()
        return CLIPTokenizer.from_pretrained(config.CLIP_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path, help="a folder laid out like data/")
    parser.add_argument("--check-only", action="store_true", help="check the files and send nothing")
    args = parser.parse_args(argv)
    rows, problems, notes = scan(args.dir, _tokenizer())
    if problems:
        print(f"{len(problems)} problem{'s' if len(problems) > 1 else ''} in {args.dir}; nothing was loaded:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{args.dir}: {len(rows)} claims OK (note, vet note, X-ray and photo for each)")
    for n in notes:
        print(f"  note: {n}")
    if args.check_only:
        return 0

    from claims import config, store
    folder = args.dir.resolve()
    config.CLAIM_NOTES, config.VET_NOTES = folder / "ClaimNotes", folder / "VetNotes"
    config.XRAYS, config.PHOTOS = folder / "PetXrays", folder / "PetPhotos"
    config.POLICIES = folder / "StructuredData" / "policy_details.csv"
    config.HISTORY = folder / "StructuredData" / "claim_history.csv"
    client = store.connect()
    try:
        if not client.is_ready():
            sys.exit("Weaviate isn't ready: docker compose up -d --wait weaviate")
        print(f"Loading {args.dir} into {config.COLLECTION} (replaces the claims there)...")
        count = store.ingest(client)
        print(f"{count} claims in Weaviate. Search them with make search; make run puts the sample back.")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
