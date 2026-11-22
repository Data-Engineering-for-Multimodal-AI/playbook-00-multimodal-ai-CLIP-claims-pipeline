"""Tests: data and embeddings without services; search against Weaviate when it's running
(docker compose up -d --wait weaviate; CI starts it)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from claims import config, embeddings, evaluate, search, store  # noqa: E402


def test_every_claim_has_its_files():
    claims = store.load_claims()
    assert [c["pet_name"] for c in claims] == ["Emma", "Jack", "Max", "Otis", "Rocky"]
    for c in claims:
        assert (config.ROOT / c["xray_path"]).is_file() and (config.ROOT / c["photo_path"]).is_file()
        assert Path(c["photo_path"]).name.startswith(c["pet_name"].lower() + "_")
        assert c["total_claimed"] > 0


def test_embeddings_are_normalized_and_cross_modal():
    text = np.array(embeddings.embed_text("a photo of a Labrador Retriever"))
    photo = np.array(embeddings.embed_image(config.PHOTOS / "max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg"))
    other = np.array(embeddings.embed_image(config.PHOTOS / "jack_karsten-winegeart-oU6KZTXhuvk-unsplash.jpg"))
    assert text.shape == (512,) and abs(np.linalg.norm(text) - 1) < 1e-5
    assert text @ photo > text @ other     # the Labrador sentence is closer to the Labrador photo


@pytest.fixture(scope="module")
def client():
    try:
        db = store.connect()
    except Exception:
        pytest.skip("Weaviate isn't running")
    store.ingest(db, verbose=False)
    yield db
    db.close()


def test_search_by_each_modality(client):
    assert search.by_description(client, "a photo of a French Bulldog", 1)[0]["pet_name"] == "Jack"
    assert search.by_photo(client, config.PHOTOS / "otis_pauline-loroy-U3aF7hgUSrk-unsplash.jpg", 1)[0]["pet_name"] == "Otis"
    xray = next(config.XRAYS.glob("rocky_*"))
    assert search.by_xray(client, xray, 1)[0]["pet_name"] == "Rocky"
    assert search.keyword(client, "cough", 1)[0]["pet_name"] == "Otis"


def test_evaluation(client):
    results = evaluate.run(client)
    assert results["breed -> photo (text to image)"]["top1"] == 1.0
    assert results["edited photo -> photo"]["top1"] == 1.0
    assert results["edited X-ray -> X-ray"]["top1"] >= 0.8
