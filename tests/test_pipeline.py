"""Tests run on synthetic data (no patient data needed), plus an optional real-data check.

    pytest -q                      # synthetic tests
    pytest -q -m real              # also checks published numbers if data/ has the real CSVs
"""

import json
import os

import numpy as np
import pytest

from gtdlbcl import config, data, stats, synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    clin_raw, feat = synthetic.make_synthetic(seed=config.SEED)
    p = tmp_path_factory.mktemp("d") / "clin.csv"
    clin_raw.to_csv(p, index=False)
    return data.load_clinical(p), feat


def test_schema(synth):
    clin, feat = synth
    assert feat.shape[1] == 1 + 4 * len(config.CHANNELS)
    assert len(config.CHANNELS) == 21 and "TRITC" not in config.CHANNELS
    assert {"os", "event", "coo", "ipi_score", "myc_ihc", "bcl2_ihc"} <= set(clin.columns)


def test_double_expressor_definition(synth):
    clin, _ = synth
    c = data.add_double_expressor(clin)
    expect = ((clin["myc_ihc"] >= 40) & (clin["bcl2_ihc"] >= 50)).astype(float)
    assert (c["del_status"] == expect).all()


def test_planted_signal_is_recovered(synth):
    """The synthetic data has an immune-cold double-expressor shift; the analysis must find it."""
    clin, feat = synth
    table, s = stats.double_expressor(feat, clin)
    assert s["n_fdr_significant_coo_adjusted"] >= 5
    t = table.set_index("channel")
    assert t.loc["CD3", "delta_x100"] < 0 and t.loc["CD138", "delta_x100"] > 0


def test_no_false_survival_signal(synth):
    """Synthetic survival depends on IPI only, so GigaTIME features must not look prognostic."""
    clin, feat = synth
    _, u = stats.univariable_cox(feat, clin)
    assert u["n_fdr_significant"] <= 2
    sig = stats.gigatime_signature(feat, clin, seeds=(0, 17))
    assert 0.35 < sig["cindex_mean"] < 0.65
    assert stats.clinical_and_ipi(feat, clin)["ipi_cindex"] > 0.6


def test_deterministic(synth):
    clin, feat = synth
    a = stats.gigatime_signature(feat, clin, seeds=(17,))
    b = stats.gigatime_signature(feat, clin, seeds=(17,))
    assert a == b
    r1 = stats.double_expressor_robustness(feat, clin, n_perm=20, n_boot=50)
    r2 = stats.double_expressor_robustness(feat, clin, n_perm=20, n_boot=50)
    assert r1["permutation_p"] == r2["permutation_p"]
    assert r1["bootstrap_ci_x100"] == r2["bootstrap_ci_x100"]


def test_cohort_flow_counts(synth):
    clin, feat = synth
    f = data.cohort_flow(feat, clin)
    assert f["double_expressor_positive"] + f["double_expressor_negative"] == f["double_expressor_cohort"]


REAL = all(os.path.exists(os.path.join(ROOT, "data", f))
           for f in ["clinical_data_cleaned.csv", "gigatime_case_features_rich.csv"])


@pytest.mark.real
@pytest.mark.skipif(not REAL, reason="real DLBCL-Morph files not in data/")
def test_reproduces_published_values():
    clin = data.load_clinical(os.path.join(ROOT, "data", "clinical_data_cleaned.csv"))
    feat = data.load_features(os.path.join(ROOT, "data", "gigatime_case_features_rich.csv"))
    exp = json.load(open(os.path.join(ROOT, "results", "expected_results.json")))["checks"]
    res = {
        "cohort_flow": data.cohort_flow(feat, clin),
        "univariable_cox": stats.univariable_cox(feat, clin)[1],
        "gigatime_signature": stats.gigatime_signature(feat, clin),
        "clinical_and_ipi": stats.clinical_and_ipi(feat, clin),
        "coo_classification": stats.coo_classification(feat, clin),
        "coexpression": stats.coexpression(feat)[1],
        "face_validity": stats.face_validity(feat),
        "double_expressor": stats.double_expressor(feat, clin)[1],
    }
    for key, spec in exp.items():
        top = key.split(".")[0]
        if top not in res:
            continue
        d = res
        for k in key.split("."):
            d = d[k]
        assert abs(d - spec["value"]) <= spec["tolerance"], f"{key}: {d} vs {spec['value']}"
