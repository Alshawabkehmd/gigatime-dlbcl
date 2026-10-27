# Data

**No patient data is stored in this repository.** The DLBCL-Morph data belong to their authors.

## Where to get it
DLBCL-Morph (Vrabac et al., *Scientific Data* 2021): https://github.com/stanfordmlgroup/DLBCL-Morph
— follow the links there to the data download (figshare / Stanford Box).

## Files the notebook expects in this folder

| File | Used in | Source |
|---|---|---|
| `clinical_data_cleaned.csv` | `features` and `full` mode | DLBCL-Morph download, top-level folder |
| `gigatime_case_features_rich.csv` | `features` mode | produced by notebook Section 4 (`full` mode) |
| H&E patches `Patches/HE/<patient_id>/<patch_id>.png` | `full` mode | DLBCL-Morph download; point `PATCH_ROOT` at the `HE` folder |

`synthetic` mode needs nothing — it generates fake data with the same columns.

## Column definitions
See `DATA_DICTIONARY.md`. The notebook renames DLBCL-Morph columns as follows:
`patient_id→case_id`, `OS→os` (years), `Follow-up Status→event` (1 = death), `HANS→coo`,
`IPI Score→ipi_score`, `IPI Risk Group (4 Class)→ipi_group`, `MYC IHC→myc_ihc`, `BCL2 IHC→bcl2_ihc`.

## Feature table format
One row per patient, `case_id` plus 84 columns: for each of the 21 virtual channels
`<channel>_mean`, `<channel>_std`, `<channel>_p90`, `<channel>_posfrac`.
