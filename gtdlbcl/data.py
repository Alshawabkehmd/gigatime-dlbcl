"""Loading and merging the clinical table and the GigaTIME feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# DLBCL-Morph column name -> short name used in the analysis.
CLINICAL_RENAME = {
    "patient_id": "case_id",
    "Follow-up Status": "event",
    "OS": "os",
    "Age": "age",
    "Stage": "stage",
    "LDH": "ldh",
    "ECOG PS": "ecog",
    "HANS": "coo",
    "IPI Score": "ipi_score",
    "IPI Risk Group (4 Class)": "ipi_group",
    "MYC IHC": "myc_ihc",
    "BCL2 IHC": "bcl2_ihc",
    "MYC FISH": "myc_fish",
    "BCL2 FISH": "bcl2_fish",
    "BCL6 FISH": "bcl6_fish",
}

REQUIRED_CLINICAL = list(CLINICAL_RENAME)


def load_clinical(path) -> pd.DataFrame:
    """Read clinical_data_cleaned.csv and rename columns to short names."""
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_CLINICAL if c not in raw.columns]
    if missing:
        raise ValueError(f"clinical file is missing columns: {missing}")
    df = raw.rename(columns=CLINICAL_RENAME)
    df["case_id"] = df["case_id"].astype(str)
    return df


def load_features(path) -> pd.DataFrame:
    """Read the per-patient GigaTIME feature table (one row per patient)."""
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        raise ValueError("feature file needs a case_id column")
    df["case_id"] = df["case_id"].astype(str)
    missing = [f"{c}_mean" for c in config.CHANNELS if f"{c}_mean" not in df.columns]
    if missing:
        raise ValueError(f"feature file is missing channel columns: {missing[:5]}...")
    return df


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """All numeric feature columns (everything except case_id)."""
    return [c for c in feat.columns if c != "case_id"]


def mean_columns() -> list[str]:
    """The 21 per-channel mean columns, in channel order."""
    return [f"{c}_mean" for c in config.CHANNELS]


def add_double_expressor(clin: pd.DataFrame, myc=None, bcl2=None) -> pd.DataFrame:
    """Add `del_status` (1 = double-expressor, 0 = not, NaN = not evaluable)."""
    myc = config.MYC_THRESHOLD if myc is None else myc
    bcl2 = config.BCL2_THRESHOLD if bcl2 is None else bcl2
    out = clin.copy()
    ok = out[["myc_ihc", "bcl2_ihc"]].notna().all(axis=1)
    out["del_status"] = np.where(
        ok, ((out["myc_ihc"] >= myc) & (out["bcl2_ihc"] >= bcl2)).astype(float), np.nan
    )
    return out


def merge(feat: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features and clinical data on case_id."""
    return feat.merge(clin, on="case_id", how="inner")


def cohort_flow(feat: pd.DataFrame, clin: pd.DataFrame) -> dict:
    """Patient counts at each step, so every n in the paper can be traced."""
    c = add_double_expressor(clin)
    m = merge(feat, c)
    del_cohort = m.dropna(subset=["del_status", "coo"])
    return {
        "clinical_rows": int(len(clin)),
        "with_gigatime_features": int(len(m)),
        "with_os_and_event": int(m.dropna(subset=["os", "event"]).shape[0]),
        "events_in_feature_cohort": int(m["event"].fillna(0).sum()),
        "double_expressor_evaluable_all": int(c["del_status"].notna().sum()),
        "double_expressor_cohort": int(len(del_cohort)),
        "double_expressor_positive": int(del_cohort["del_status"].sum()),
        "double_expressor_negative": int((1 - del_cohort["del_status"]).sum()),
    }
