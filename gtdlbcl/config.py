"""Fixed settings for the GigaTIME x DLBCL-Morph analysis.

Everything that affects a result lives here so a re-run uses the same choices.
"""

# Random seed used by every resampling / cross-validation step.
SEED = 17

# Seeds averaged for the out-of-fold GigaTIME survival signature.
SIGNATURE_SEEDS = (0, 7, 17, 42, 123)

# Cross-validation folds.
CV_FOLDS = 5

# Elastic-net Cox settings for the GigaTIME signature.
COX_PENALIZER = 0.1
COX_L1_RATIO = 0.5

# False-discovery-rate threshold (Benjamini-Hochberg).
FDR_ALPHA = 0.05

# Double-expressor definition (2016 WHO thresholds, immunohistochemistry %).
MYC_THRESHOLD = 40
BCL2_THRESHOLD = 50

# Published DLBCL-Morph morphology baseline (Vrabac et al., 2021).
# NOTE: this value is optimism-corrected in-sample, not out-of-fold.
MORPHOLOGY_BASELINE_CINDEX = 0.635

# GigaTIME source code, pinned for reproducibility.
GIGATIME_REPO = "https://github.com/prov-gigatime/GigaTIME.git"
GIGATIME_COMMIT = "acc49b2567fd15ce95b5a82caaab5f0a295b88f7"
GIGATIME_HF_REPO = "prov-gigatime/GigaTIME"
# Weights: set to the Hugging Face commit hash / SHA-256 recorded in outputs/model_provenance.json
# of the original run to lock a re-run to the same checkpoint (None = latest, recorded only).
GIGATIME_HF_REVISION = None
GIGATIME_WEIGHTS_SHA256 = None
# Physical scale: DLBCL-Morph TMA patches, 224 x 224 px at the native scanning magnification
# (microns per pixel were not distributed with the patches; see data/README.md).

# GigaTIME outputs 23 channels; TRITC and Cy5 are background channels and are dropped.
RAW_CHANNELS = [
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16",
    "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67", "Tryptase",
    "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin",
]
KEEP_IDX = [i for i, n in enumerate(RAW_CHANNELS) if n not in ("TRITC", "Cy5")]
CHANNELS = [RAW_CHANNELS[i] for i in KEEP_IDX]  # the 21 analytic channels

# Lineage groups used for colouring and for the co-expression summary.
IMMUNE = ["CD3", "CD4", "CD8", "PD-1", "T-bet", "CD68", "CD14", "CD16", "CD11c", "PD-L1", "Tryptase"]
TUMOR = ["DAPI", "CD138", "Ki67", "PHH3-B", "Caspase3-D", "CD20"]
T_CELL = ["CD3", "CD4", "CD8", "PD-1"]
MYELOID = ["CD68", "CD14", "CD16", "CD11c"]

# Clinical covariates in the comparison model.
CLINICAL_COVARIATES = ["age", "stage", "ldh", "ecog", "coo"]

# Robustness-test sizes (the notebook can lower these for a quick run).
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000
# Full-pipeline outcome permutations for the survival-signature null (slow: ~3 s each).
N_SURVIVAL_PERMUTATIONS = 300
# Full-procedure null for the change in C from adding GigaTIME to the clinical model (slow: ~65 s each).
N_ADDED_VALUE_PERMUTATIONS = 100
THRESHOLD_GRID = [(30, 50), (40, 40), (40, 50), (40, 60), (50, 50)]
