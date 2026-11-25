# Data licenses

This file lists the datasets and models this playbook uses, where they come from, and their licenses. The code in this repository is MIT-licensed. That license does not cover third-party data or models; each one keeps its own license, listed below.

## Datasets

| Dataset | Source | License | In this repo |
|---|---|---|---|
| Pet photos (5 JPEGs) | [Unsplash](https://unsplash.com). One photo per pet; the photographer and photo ID are in each file name (see Notes) | [Unsplash License](https://unsplash.com/license) | `data/PetPhotos/` (11 MB) |
| Pet X-rays (5 PNGs) | Not stated | Not stated | `data/PetXrays/` (14 MB) |
| Claim notes, vet notes, policy and claim history CSVs | AI-generated for this playbook. The owners, policies and claims are fictional | MIT (this repository) | `data/ClaimNotes/`, `data/VetNotes/`, `data/StructuredData/` (48 KB) |
| Pet invoices (5 PNGs) | AI-generated for this playbook. The owners, policies and claims are fictional | MIT (this repository) | `data/PetInvoices/` (11 MB) |

## Models

| Model | Source | License | How it's used |
|---|---|---|---|
| CLIP ViT-B/32 | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | Not stated on the Hugging Face model card. OpenAI's [CLIP repository](https://github.com/openai/CLIP) is MIT-licensed | Embeds the claim text, X-rays and pet photos into one vector space. Downloaded from Hugging Face by `make setup` (about 600 MB). Set by `CLIP_MODEL` in `claims/config.py` |

## Notes

- The pet photos, by file name:
  - `emma_brooke-cagle-Ntm4C2lCWxQ-unsplash.jpg`: [unsplash.com/photos/Ntm4C2lCWxQ](https://unsplash.com/photos/Ntm4C2lCWxQ)
  - `jack_karsten-winegeart-oU6KZTXhuvk-unsplash.jpg`: [unsplash.com/photos/oU6KZTXhuvk](https://unsplash.com/photos/oU6KZTXhuvk)
  - `max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg`: [unsplash.com/photos/WX4i1Jq_o0Y](https://unsplash.com/photos/WX4i1Jq_o0Y)
  - `otis_pauline-loroy-U3aF7hgUSrk-unsplash.jpg`: [unsplash.com/photos/U3aF7hgUSrk](https://unsplash.com/photos/U3aF7hgUSrk)
  - `rocky_victor-g-N04FIfHhv_k-unsplash.jpg`: [unsplash.com/photos/N04FIfHhv_k](https://unsplash.com/photos/N04FIfHhv_k)
- Pet X-rays: the source and license are unclear. `data/README.md` says only "The X-ray images are anonymized sample medical images."

## Citations

- CLIP: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision", arXiv:2103.00020, 2021.
- The pet photos and X-rays: no citation given by the source.
