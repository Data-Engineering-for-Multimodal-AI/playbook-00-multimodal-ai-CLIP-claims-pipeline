# Sample Data for Multimodal Claims Pipeline

This directory contains sample data used by the Multimodal Claims Pipeline application. It demonstrates the different types of data the system can process and search.

## Directory Structure

- **ClaimNotes/** - Text descriptions of insurance claims
- **PetXrays/** - X-ray images for medical claims
- **PetPhotos/** - Photos of pets for pet identification
- **PetInvoices/** - Medical invoices related to claims (optional)
- **VetNotes/** - Veterinarian notes for claims (optional)
- **StructuredData/** - CSV files with structured policy and claim data

## Sample Data Contents

### ClaimNotes

Contains claim descriptions for 5 pets with different medical conditions:
- CLM001_description.txt - Emma (dental issues)
- CLM002_description.txt - Jack (digestive issues)
- CLM003_description.txt - Max (leg fracture)
- CLM004_description.txt - Otis (eye infection)
- CLM005_description.txt - Rocky (leg injury)

### PetXrays

Contains X-ray images for the pets:
- emma_1.2.410.200028.100.3.20190924.1430080534.21770.1.1.png
- jack_1.2.410.200028.100.3.20190927.1023440667.730.1.1.png
- max_1.2.410.200028.100.3.20190925.1056080334.32610.1.2.png
- otis_1.2.410.200028.100.3.20190926.1008510164.10580.1.2.png
- rocky_1.2.410.200028.100.3.20190927.1032580692.2539.1.1.png

### PetPhotos

Contains photos of the pets:
- emma_brooke-cagle-Ntm4C2lCWxQ-unsplash.jpg
- jack_victor-g-N04FIfHhv_k-unsplash.jpg
- max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg
- otis_karsten-winegeart-oU6KZTXhuvk-unsplash.jpg
- rocky_pauline-loroy-U3aF7hgUSrk-unsplash.jpg

### StructuredData

Contains structured data files:
- **policy_details.csv** - Information about insurance policies including coverage details
- **claim_history.csv** - Historical claim data with dates, amounts, and outcomes

## Adding Your Own Data

You can extend the system with your own data by following these guidelines:

### Adding New Claims

1. Add a claim note in `ClaimNotes/` with filename format `CLM[ID]_description.txt`
2. Add corresponding X-ray in `PetXrays/` with pet name prefix
3. Add pet photo in `PetPhotos/` with pet name prefix
4. Update `StructuredData/policy_details.csv` with policy information
5. Update `StructuredData/claim_history.csv` with claim history

### Format Requirements

- **Claim Notes**: Text files with description of the medical condition
- **X-ray Images**: PNG format, preferably DICOM-exported X-rays
- **Pet Photos**: JPG format, clear front-facing photos of pets
- **Policy Details CSV**: Must include all columns from the sample file
- **Claim History CSV**: Must follow the same format as the sample file

## Data Attribution

The sample pet photos used in this dataset are from Unsplash:
- "Emma" photo by Brooke Cagle 
- "Jack" photo by Victor G
- "Max" photo by Taylor Kopel
- "Otis" photo by Karsten Winegeart
- "Rocky" photo by Pauline Loroy

The X-ray images are anonymized sample medical images. 