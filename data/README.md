# Sample data

Five fictional pet-insurance claims. The pets, owners' notes, vet notes, policies, clinics and claim
histories are made up; the clinic names are fictional.

| Claim | Pet | Breed | Claim note (`ClaimNotes/`) | Vet note (`VetNotes/`) |
|---|---|---|---|---|
| CLM001 | Emma | Goldendoodle | Limping after the park; X-ray of the hip | Reduced weight-bearing, right hind limb; no fracture |
| CLM002 | Jack | French Bulldog | Slipped on a hardwood floor; feared dislocated shoulder | Discomfort in the front right shoulder; no dislocation |
| CLM003 | Max | Labrador Retriever | Stiff, hesitant to jump; arthritis or strain? | Reduced mobility in hind limbs; possible early arthritis |
| CLM004 | Otis | Border Collie | Persistent cough; chest X-ray | Abnormal lung sounds; mild bronchitis suspected |
| CLM005 | Rocky | Jack Russell Terrier | Fall from the couch | Tenderness near the lumbar spine; soft-tissue injury |

- `StructuredData/policy_details.csv`: policy, clinic, what it covers, deductible and co-pay per claim.
- `StructuredData/claim_history.csv`: earlier payments per claim (date, treatment, amount).
- `PetInvoices/`: one generated invoice image per claim (fictional clinics); not used by the pipeline.

## X-rays (`PetXrays/`)

All five are lateral chest radiographs of dogs; each stands in for its claim's imaging, even where the
claim is about a hip, shoulder or spine. File names keep the scans' DICOM identifiers. Source: a
veterinary X-ray dataset on Kaggle; **which dataset, and its license, are still to be confirmed.**

## Pet photos (`PetPhotos/`)

From [Unsplash](https://unsplash.com) (Unsplash License: free to use, attribution appreciated). The
photo ID is in each file name:

| Pet | Photo by | Unsplash ID |
|---|---|---|
| Emma (Goldendoodle) | Brooke Cagle | Ntm4C2lCWxQ |
| Jack (French Bulldog) | Karsten Winegeart | oU6KZTXhuvk |
| Max (Labrador Retriever) | Taylor Kopel | WX4i1Jq_o0Y |
| Otis (Border Collie) | Pauline Loroy | U3aF7hgUSrk |
| Rocky (Jack Russell Terrier) | Victor G | N04FIfHhv_k |

## Adding claims

Add `ClaimNotes/CLM<ID>_description.txt`, `VetNotes/CLM<ID>_vet_notes.txt`, an X-ray and a photo whose
file names start with the pet's name in lower case and an underscore, and a row in each CSV. Then
`python store_with_embedding.py`.
