# Data Dictionary — DLBCL-Morph clinical variables

Cohort: 209 patients (rows). Source: DLBCL-Morph (Vrabac et al., Scientific Data 2021).
Data are NOT included in this repository; obtain from https://github.com/stanfordmlgroup/DLBCL-Morph.
Confirm 0/1 coding direction for HANS, CD10, MUM1, and LDH against the original DLBCL-Morph column_description.csv.

| Variable | Description | Coding / units | Missing (of 209) |
|---|---|---|---|
| `patient_id` | Anonymized patient identifier | integer ID | 0 |
| `MYC IHC` | MYC protein expression by immunohistochemistry | % tumor cells positive (0-100) | 14 |
| `BCL2 IHC` | BCL2 protein expression by immunohistochemistry | % tumor cells positive (0-100) | 14 |
| `BCL6 IHC` | BCL6 protein expression by immunohistochemistry | % tumor cells positive (0-100) | 18 |
| `CD10 IHC` | CD10 protein by immunohistochemistry | 0 = negative, 1 = positive [verify direction] | 25 |
| `MUM1 IHC` | MUM1/IRF4 protein by immunohistochemistry | 0 = negative, 1 = positive [verify direction] | 24 |
| `HANS` | Cell-of-origin, Hans algorithm | 0 = GCB, 1 = non-GCB [verify direction] | 24 |
| `BCL6 FISH` | BCL6 gene rearrangement (FISH) | 0 = absent, 1 = present | 31 |
| `MYC FISH` | MYC gene rearrangement (FISH) | 0 = absent, 1 = present | 30 |
| `BCL2 FISH` | BCL2 gene rearrangement (FISH) | 0 = absent, 1 = present | 53 |
| `Age` | Age at diagnosis | years | 0 |
| `ECOG PS` | ECOG performance status | 0 (fully active) to 3 (limited) | 0 |
| `LDH` | Serum lactate dehydrogenase | 0 = normal, 1 = elevated [verify direction] | 4 |
| `EN` | Number of extranodal sites involved | count | 2 |
| `Stage` | Ann Arbor stage | 1-4 | 0 |
| `IPI Score` | International Prognostic Index score | 0-5 points | 0 |
| `IPI Risk Group (4 Class)` | IPI risk category | 0 = low to 3 = high | 1 |
| `RIPI Risk Group` | Revised IPI risk group | 0-2 | 0 |
| `OS` | Overall survival time | years | 0 |
| `PFS` | Progression-free survival time | years | 0 |
| `Follow-up Status` | Outcome event indicator | 0 = alive/censored, 1 = died | 0 |

## Derived variables used in the analysis
| Derived variable | Definition |
|---|---|
| Double-expressor (DEL) | MYC IHC >= 40% AND BCL2 IHC >= 50% |
| Double-hit-like (DH) | MYC FISH = 1 AND (BCL2 FISH = 1 OR BCL6 FISH = 1) |
| Event (survival) | Follow-up Status = 1 (death) |
| Clinical model covariates | Age, Stage, LDH, ECOG PS, HANS (cell-of-origin) |