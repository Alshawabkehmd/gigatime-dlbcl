"""Synthetic data with the same columns as the real files.

Used for the automated tests and for trying the notebook without the real data.
It is NOT DLBCL-Morph data and any figure made from it is labelled SYNTHETIC.
A double-expressor 'immune-cold' shift is planted on purpose so the tests can
check that the pipeline recovers a known signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def make_synthetic(n=170, seed=0, del_effect=0.35):
    rng = np.random.default_rng(seed)
    ids = [str(20000 + i) for i in range(n)]

    age = rng.normal(64, 12, n).clip(20, 90).round()
    stage = rng.integers(1, 5, n)
    ldh = rng.integers(0, 2, n)
    ecog = rng.choice([0, 1, 2, 3], n, p=[0.4, 0.35, 0.2, 0.05])
    en = rng.integers(0, 3, n)
    ipi = ((age > 60).astype(int) + (stage >= 3) + ldh + (ecog >= 2) + (en >= 2)).astype(int)
    ipi_group = np.select([ipi <= 1, ipi == 2, ipi == 3], [0, 1, 2], 3)
    coo = rng.integers(0, 2, n).astype(float)
    myc = rng.choice([0, 10, 20, 30, 40, 50, 60, 70], n, p=[.2, .15, .15, .15, .12, .1, .08, .05]).astype(float)
    bcl2 = rng.choice([0, 20, 40, 50, 70, 90], n, p=[.2, .15, .15, .15, .2, .15]).astype(float)
    del_status = ((myc >= config.MYC_THRESHOLD) & (bcl2 >= config.BCL2_THRESHOLD)).astype(int)

    # survival driven by IPI only (so GigaTIME features carry no outcome signal)
    hazard = 0.06 * np.exp(0.45 * ipi)
    t_event = rng.exponential(1 / hazard)
    t_cens = rng.uniform(2, 14, n)
    os_t = np.minimum(t_event, t_cens).round(2)
    event = (t_event <= t_cens).astype(int)

    clin = pd.DataFrame({
        "patient_id": ids, "MYC IHC": myc, "BCL2 IHC": bcl2, "BCL6 IHC": rng.integers(0, 90, n).astype(float),
        "CD10 IHC": rng.integers(0, 2, n).astype(float), "MUM1 IHC": rng.integers(0, 2, n).astype(float),
        "HANS": coo, "BCL6 FISH": rng.integers(0, 2, n).astype(float),
        "MYC FISH": (rng.random(n) < 0.08).astype(float), "BCL2 FISH": rng.integers(0, 2, n).astype(float),
        "Age": age, "ECOG PS": ecog, "LDH": ldh.astype(float), "EN": en.astype(float), "Stage": stage.astype(float),
        "IPI Score": ipi, "IPI Risk Group (4 Class)": ipi_group.astype(float),
        "RIPI Risk Group": np.minimum(ipi_group, 2), "OS": os_t, "PFS": os_t, "Follow-up Status": event,
    })

    # features: a shared cellularity factor + lineage factors + noise
    cell = rng.normal(0, 1, n)
    base = {c: 0.3 for c in config.CHANNELS}
    base.update({"DAPI": 0.7, "CD20": 0.55, "CD3": 0.5, "CD4": 0.45, "CK": 0.15, "CD68": 0.2})
    cols = {}
    for c in config.CHANNELS:
        v = base[c] + 0.05 * cell + 0.04 * rng.normal(0, 1, n)
        if c in ("CD3", "CD4", "T-bet", "CD11c", "CD68", "PD-L1"):
            v = v - del_effect * base[c] * del_status        # immune-cold in double-expressors
        if c in ("DAPI", "CD138", "Caspase3-D", "Ki67"):
            v = v + del_effect * 0.5 * base[c] * del_status   # more tumour / turnover
        v = np.clip(v, 0.01, 0.99)
        cols[f"{c}_mean"] = v
        cols[f"{c}_std"] = np.abs(0.1 + 0.02 * rng.normal(0, 1, n))
        cols[f"{c}_p90"] = np.clip(v + 0.15, 0, 1)
    for c in config.CHANNELS:
        cols[f"{c}_posfrac"] = np.clip(cols[f"{c}_mean"] - 0.1, 0, 1)
    feat = pd.DataFrame({"case_id": ids, **cols})
    return clin, feat
