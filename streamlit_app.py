"""
Claims search UI: streamlit run streamlit_app.py  (needs Weaviate running and the claims loaded)

Four ways in: a text describing the claim, a description of the pet, an X-ray, or a pet photo.
Each result shows the claim, its X-ray and photo, the policy and the claim history.
"""
import json

import pandas as pd
import streamlit as st
from PIL import Image

from claims import config, search, store

st.set_page_config(page_title="Multimodal Claims Search", layout="wide")


@st.cache_resource
def client():
    return store.connect()


def show(result: dict, rank: int) -> None:
    score = f"similarity {result['similarity']:.3f}" if "similarity" in result else f"BM25 {result['score']:.2f}"
    with st.container(border=True):
        st.markdown(f"**{rank}. {result['pet_name']}** ({result['breed']}) · claim {result['claim_id']} · {score}")
        left, middle, right = st.columns([1, 1, 2])
        left.image(str(config.ROOT / result["photo_path"]), caption="Pet photo", width="stretch")
        middle.image(str(config.ROOT / result["xray_path"]), caption="X-ray", width="stretch")
        with right:
            st.write(f"*Claim:* {result['claim_note']}")
            st.write(f"*Vet:* {result['vet_note']}")
            covered = [k.replace("_covered", "").replace("xray", "X-ray")
                       for k in ("xray_covered", "surgery_covered", "dental_covered") if result[k]]
            st.write(f"*Policy {result['policy_id']}* at {result['clinic']}: covers {', '.join(covered) or 'none of these'}; "
                     f"deductible ${result['deductible']:.0f}, co-pay {result['co_pay_percent']:.0f}%")
            history = pd.DataFrame(json.loads(result["claim_history"]))
            if not history.empty:
                st.dataframe(history, hide_index=True, width="stretch")
                st.caption(f"Claimed so far: ${result['total_claimed']:.0f}")


def sample_picker(folder, label: str):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    upload = st.file_uploader(f"Upload {label}", type=["png", "jpg", "jpeg"])
    if upload is not None:
        return Image.open(upload).convert("RGB")
    choice = st.selectbox(f"Or use a sample {label}", files, format_func=lambda p: p.name.split("_")[0].title())
    return Image.open(choice).convert("RGB")


def main() -> None:
    st.title("Multimodal Claims Search")
    try:
        db = client()
        if not db.collections.exists(config.COLLECTION):
            st.error("No claims loaded yet. Run: python store_with_embedding.py")
            return
    except Exception as e:  # Weaviate not running
        st.error(f"Can't reach Weaviate ({e}). Start it: docker compose up -d --wait weaviate")
        return

    mode = st.radio("Search by", ["Claim text", "Describe the pet", "X-ray", "Pet photo"], horizontal=True)
    limit = st.slider("Results", 1, 5, 3)
    results = []
    if mode == "Claim text":
        text = st.text_input("Describe the claim", "limping after a fall")
        if text:
            results = search.by_text(db, text, limit)
    elif mode == "Describe the pet":
        text = st.text_input("Describe the pet (CLIP matches words to photos)", "a curly-haired dog on a bed")
        if text:
            results = search.by_description(db, text, limit)
    elif mode == "X-ray":
        image = sample_picker(config.XRAYS, "X-ray")
        st.image(image, width=240)
        results = search.by_xray(db, image, limit)
        st.caption("CLIP isn't a medical model: X-ray search finds X-rays that look alike, not the same diagnosis.")
    else:
        image = sample_picker(config.PHOTOS, "pet photo")
        st.image(image, width=240)
        results = search.by_photo(db, image, limit)
    for rank, result in enumerate(results, 1):
        show(result, rank)


main()
