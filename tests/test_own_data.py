"""Checks for the "Use your own data" loader (scripts/own_claims.py). No Weaviate, no model download."""
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from own_claims import scan  # noqa: E402

DATA = REPO / "data"
HEADER = "claim_id,policy_id,pet_name,breed,clinic,xray_covered,surgery_covered,dental_covered,deductible,co_pay_percent\n"


def one_claim(folder: Path, pet: str = "Bella", claim: str = "CLM010", policy_row: str = None) -> Path:
    for sub in ("ClaimNotes", "VetNotes", "PetXrays", "PetPhotos", "StructuredData"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    (folder / "ClaimNotes" / f"{claim}_description.txt").write_text("Bella ate a sock.", encoding="utf-8")
    (folder / "VetNotes" / f"{claim}_vet_notes.txt").write_text("Radiograph taken.", encoding="utf-8")
    shutil.copy(next((DATA / "PetXrays").glob("emma_*")), folder / "PetXrays" / f"{pet.lower()}_hip.png")
    shutil.copy(next((DATA / "PetPhotos").glob("emma_*")), folder / "PetPhotos" / f"{pet.lower()}_beach.jpg")
    row = policy_row or f"{claim},POL-1010,{pet},Beagle,Oak Clinic,True,False,True,100,20\n"
    (folder / "StructuredData" / "policy_details.csv").write_text(HEADER + row, encoding="utf-8")
    (folder / "StructuredData" / "claim_history.csv").write_text(
        f"claim_id,pet_name,date,type,amount\n{claim},{pet},2024-03-15,Visit,80\n", encoding="utf-8")
    return folder


def test_sample_data_passes():
    rows, problems, notes = scan(DATA)
    assert problems == [] and notes == []
    assert [r["pet_name"] for r in rows] == ["Emma", "Jack", "Max", "Otis", "Rocky"]


def test_good_folder(tmp_path):
    rows, problems, _ = scan(one_claim(tmp_path))
    assert problems == [] and rows[0]["claim_id"] == "CLM010"


@pytest.mark.parametrize("path,content,expected", [
    ("VetNotes/CLM010_vet_notes.txt", b"\xe9t\xe9", "VetNotes/CLM010_vet_notes.txt: not UTF-8 text"),
    ("ClaimNotes/CLM010_description.txt", b"  \n", "ClaimNotes/CLM010_description.txt: empty"),
    ("PetPhotos/bella_beach.jpg", b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
    ("PetPhotos/bella_beach.jpg", b"not an image", "PetPhotos/bella_beach.jpg: can't open as an image"),
    ("StructuredData/claim_history.csv", b"claim_id,pet_name,date,type,amount\nCLM010,Bella,15/03/2024,Visit,80\n",
     "line 2: date is '15/03/2024', needs YYYY-MM-DD"),
    ("StructuredData/claim_history.csv", b"claim_id,pet_name,date,type,amount\nCLM099,Rex,2024-01-01,Visit,x\n",
     "line 2: claim_id 'CLM099' isn't in policy_details.csv"),
    ("StructuredData/claim_history.csv", b"claim_id,date\nCLM010,2024-01-01\n", "missing columns type, amount"),
])
def test_bad_files_are_reported(tmp_path, path, content, expected):
    one_claim(tmp_path)
    (tmp_path / path).write_bytes(content)
    _, problems, _ = scan(tmp_path)
    assert any(expected in p for p in problems), problems


def test_missing_files(tmp_path):
    one_claim(tmp_path)
    (tmp_path / "VetNotes" / "CLM010_vet_notes.txt").unlink()
    (tmp_path / "PetXrays" / "bella_hip.png").unlink()
    _, problems, _ = scan(tmp_path)
    assert problems == ["VetNotes/CLM010_vet_notes.txt: missing",
                        "PetXrays/: no X-ray for Bella (a file named bella_<anything>.png or .jpg)"]


def test_bad_policy_values(tmp_path):
    one_claim(tmp_path, policy_row="CLM010,POL-1010,Bella,Beagle,Oak Clinic,yes,False,True,100,abc\n")
    _, problems, _ = scan(tmp_path)
    assert problems == ["StructuredData/policy_details.csv line 2: xray_covered is 'yes', needs True or False",
                        "StructuredData/policy_details.csv line 2: co_pay_percent is 'abc', not a number"]


def test_duplicates_and_empty_fields(tmp_path):
    rows = ("CLM010,POL-1010,Bella,Beagle,Oak Clinic,True,False,True,100,20\n"
            "CLM010,POL-1011,bella,Beagle,Oak Clinic,True,False,True,100,20\n"
            "CLM012,POL-1012,Rex,,Oak Clinic,True,False,True,100,20\n")
    one_claim(tmp_path, policy_row=rows)
    _, problems, _ = scan(tmp_path)
    assert "StructuredData/policy_details.csv line 3: claim_id CLM010 appears twice" in problems
    assert any("pet_name bella appears twice" in p for p in problems)
    assert "StructuredData/policy_details.csv line 4: empty breed" in problems


def test_tiny_image_and_extra_files(tmp_path):
    one_claim(tmp_path)
    Image.new("RGB", (20, 20)).save(tmp_path / "PetXrays" / "bella_hip.png")
    shutil.copy(tmp_path / "PetPhotos" / "bella_beach.jpg", tmp_path / "PetPhotos" / "bella_park.jpg")
    _, problems, notes = scan(tmp_path)
    assert problems == ["PetXrays/bella_hip.png: too small (20x20), needs at least 32x32"]
    assert notes == ["PetPhotos/: 2 files for Bella; only bella_beach.jpg is used"]


def test_long_note_is_a_note(tmp_path):
    one_claim(tmp_path)
    (tmp_path / "ClaimNotes" / "CLM010_description.txt").write_text("word " * 100, encoding="utf-8")
    _, problems, notes = scan(tmp_path, tokenizer=lambda text: {"input_ids": text.split()})
    assert problems == []
    assert notes == ["ClaimNotes/CLM010_description.txt: 100 tokens; CLIP embeds only the first 77"]


def test_missing_policy_file(tmp_path):
    one_claim(tmp_path)
    (tmp_path / "StructuredData" / "policy_details.csv").unlink()
    assert scan(tmp_path)[1] == ["StructuredData/policy_details.csv: missing"]
