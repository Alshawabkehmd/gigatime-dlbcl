"""Statistical analyses. Every function is deterministic for a given seed."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from . import config
from .data import add_double_expressor, feature_columns, mean_columns, merge

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- helpers
def _survival_frame(feat, clin):
    m = merge(feat, clin).dropna(subset=["os", "event"]).reset_index(drop=True)
    m["event"] = m["event"].astype(int)
    return m


def _cindex(time, risk, event):
    """Harrell's C where a higher `risk` means shorter survival."""
    return float(concordance_index(time, -np.asarray(risk), event))


# ---------------------------------------------------------------- goal 1: is the output biologically plausible?
def face_validity(feat):
    """Cohort-average virtual expression per channel, plus CD20 rank."""
    means = feat[mean_columns()].mean()
    means.index = config.CHANNELS
    rank = int((means > means["CD20"]).sum()) + 1
    return {
        "channel_means": means.round(4).to_dict(),
        "cd20_mean": round(float(means["CD20"]), 3),
        "cd20_range": [round(float(feat["CD20_mean"].min()), 3), round(float(feat["CD20_mean"].max()), 3)],
        "ck_mean": round(float(means["CK"]), 3),
        "cd3_mean": round(float(means["CD3"]), 3),
        "dapi_mean": round(float(means["DAPI"]), 3),
        "cd20_rank_of_21": rank,
    }


def coexpression(feat):
    """Channel-by-channel Pearson correlation and within-lineage summaries."""
    M = feat[mean_columns()].copy()
    M.columns = config.CHANNELS
    C = M.corr()

    def within(group):
        vals = [C.loc[a, b] for i, a in enumerate(group) for b in group[i + 1:]]
        return round(float(np.mean(vals)), 2)

    summary = {
        "within_t_cell_r": within(config.T_CELL),
        "within_myeloid_r": within(config.MYELOID),
        "cd3_cd4_r": round(float(C.loc["CD3", "CD4"]), 2),
        "cd8_pd1_r": round(float(C.loc["CD8", "PD-1"]), 2),
        "cd14_cd11c_r": round(float(C.loc["CD14", "CD11c"]), 2),
        "cd3_ck_r": round(float(C.loc["CD3", "CK"]), 2),
        "cd20_ck_r": round(float(C.loc["CD20", "CK"]), 2),
        "ki67_phh3_r": round(float(C.loc["Ki67", "PHH3-B"]), 2),
    }
    return C, summary


def coo_classification(feat, clin, seed=config.SEED):
    """Can the virtual features predict Hans cell-of-origin? (out-of-fold AUC)"""
    m = merge(feat, clin).dropna(subset=["coo"]).reset_index(drop=True)
    y = m["coo"].astype(int).values
    X = m[feature_columns(feat)].values
    oof = np.zeros(len(y))
    skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return {"n": int(len(y)), "auc_oof": round(float(roc_auc_score(y, oof)), 3)}


# ---------------------------------------------------------------- goal 2: does it predict outcome?
def univariable_cox(feat, clin):
    """One Cox model per standardized feature, Benjamini-Hochberg corrected."""
    m = _survival_frame(feat, clin)
    rows = []
    for col in feature_columns(feat):
        x = m[col]
        sd = x.std()
        if not np.isfinite(sd) or sd == 0:
            continue
        d = pd.DataFrame({"x": (x - x.mean()) / sd, "os": m["os"], "event": m["event"]})
        try:
            cph = CoxPHFitter().fit(d, "os", "event")
        except Exception:
            cph = CoxPHFitter(penalizer=0.01).fit(d, "os", "event")
        rows.append({"feature": col, "hr_per_sd": float(np.exp(cph.params_["x"])),
                     "p": float(cph.summary.loc["x", "p"])})
    R = pd.DataFrame(rows)
    R["q_fdr"] = multipletests(R["p"], method="fdr_bh")[1]
    R = R.sort_values("p").reset_index(drop=True)
    return R, {
        "n_patients": int(len(m)),
        "n_events": int(m["event"].sum()),
        "n_features": int(len(R)),
        "n_fdr_significant": int((R["q_fdr"] < config.FDR_ALPHA).sum()),
        "top_feature": R.loc[0, "feature"],
        "top_p": round(float(R.loc[0, "p"]), 4),
        "top_q": round(float(R.loc[0, "q_fdr"]), 3),
    }


def oof_cox_scores(m, cols, seed, penalizer=config.COX_PENALIZER, l1_ratio=config.COX_L1_RATIO):
    """Out-of-fold elastic-net Cox risk scores (standardization fit inside each fold)."""
    oof = np.full(len(m), np.nan)
    for tr, te in KFold(config.CV_FOLDS, shuffle=True, random_state=seed).split(m):
        train = m.iloc[tr]
        mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
        d = ((train[cols] - mu) / sd).copy()
        d["os"], d["event"] = train["os"].values, train["event"].values
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio).fit(d, "os", "event")
        oof[te] = cph.predict_log_partial_hazard((m.iloc[te][cols] - mu) / sd).values
    return oof


def gigatime_signature(feat, clin, seeds=config.SIGNATURE_SEEDS):
    """Out-of-fold C-index of the GigaTIME elastic-net Cox signature, over several seeds."""
    m = _survival_frame(feat, clin)
    cols = feature_columns(feat)
    per_seed = {}
    for s in seeds:
        per_seed[int(s)] = round(_cindex(m["os"], oof_cox_scores(m, cols, s), m["event"]), 3)
    vals = list(per_seed.values())
    return {"n": int(len(m)), "events": int(m["event"].sum()),
            "cindex_mean": round(float(np.mean(vals)), 3),
            "cindex_min": min(vals), "cindex_max": max(vals), "per_seed": per_seed}


def clinical_and_ipi(feat, clin, seed=config.SEED):
    """IPI C-index and out-of-fold clinical-model C-index on the same patients."""
    m = _survival_frame(feat, clin)
    ipi = m.dropna(subset=["ipi_score"])
    cl = m.dropna(subset=config.CLINICAL_COVARIATES).reset_index(drop=True)
    oof = oof_cox_scores(cl, config.CLINICAL_COVARIATES, seed, penalizer=0.01, l1_ratio=0.0)
    return {
        "ipi_n": int(len(ipi)),
        "ipi_cindex": round(_cindex(ipi["os"], ipi["ipi_score"], ipi["event"]), 3),
        "clinical_n": int(len(cl)),
        "clinical_cindex_oof": round(_cindex(cl["os"], oof, cl["event"]), 3),
    }


def added_value(feat, clin, seeds=config.SIGNATURE_SEEDS):
    """Nested likelihood-ratio test: clinical model vs clinical + out-of-fold GigaTIME score.

    Repeated over several cross-validation seeds, because the out-of-fold score
    (and therefore the LRT) depends on how patients are split into folds. The
    hazard ratio of the GigaTIME score is reported too: a value below 1 means the
    score runs in the wrong direction (higher predicted risk, fewer deaths).
    """
    m = _survival_frame(feat, clin)
    base_cols = config.CLINICAL_COVARIATES
    per_seed = {}
    base_c = None
    for s in seeds:
        m["gt_score"] = oof_cox_scores(m, feature_columns(feat), s)
        cl = m.dropna(subset=base_cols).reset_index(drop=True)
        base = CoxPHFitter().fit(cl[base_cols + ["os", "event"]], "os", "event")
        full = CoxPHFitter().fit(cl[base_cols + ["gt_score", "os", "event"]], "os", "event")
        lr = 2 * (full.log_likelihood_ - base.log_likelihood_)
        base_c = round(float(base.concordance_index_), 3)
        per_seed[int(s)] = {"lrt_p": round(float(sps.chi2.sf(lr, 1)), 3),
                            "gigatime_hr": round(float(np.exp(full.params_["gt_score"])), 2),
                            "cindex_apparent": round(float(full.concordance_index_), 3)}
    ps = [v["lrt_p"] for v in per_seed.values()]
    hrs = [v["gigatime_hr"] for v in per_seed.values()]
    return {
        "n": int(len(cl)),
        "clinical_cindex_apparent": base_c,
        "lrt_p_median": round(float(np.median(ps)), 3),
        "lrt_p_range": [min(ps), max(ps)],
        "gigatime_hr_all_below_1": bool(all(h < 1 for h in hrs)),
        "per_seed": per_seed,
    }


def km_tests(feat, clin, seed=config.SEED):
    """Log-rank tests: IPI groups (positive control) and GigaTIME median split."""
    m = _survival_frame(feat, clin)
    ipi = m.dropna(subset=["ipi_group"])
    p_ipi = multivariate_logrank_test(ipi["os"], ipi["ipi_group"].astype(int), ipi["event"]).p_value
    m["gt_score"] = oof_cox_scores(m, feature_columns(feat), seed)
    hi = m["gt_score"] > m["gt_score"].median()
    p_gt = logrank_test(m.loc[hi, "os"], m.loc[~hi, "os"], m.loc[hi, "event"], m.loc[~hi, "event"]).p_value
    return {
        "ipi_logrank_p": float(f"{p_ipi:.3g}"),
        "gigatime_split_logrank_p": round(float(p_gt), 3),
        "gigatime_high_risk_death_rate": round(float(m.loc[hi, "event"].mean()), 2),
        "gigatime_low_risk_death_rate": round(float(m.loc[~hi, "event"].mean()), 2),
    }, m


# ---------------------------------------------------------------- double-expressor signature
def _composition(feat_rows):
    X = feat_rows[mean_columns()].values
    return X / X.sum(axis=1, keepdims=True)


def _del_frame(feat, clin, myc=None, bcl2=None):
    c = add_double_expressor(clin, myc, bcl2)
    return merge(feat, c).dropna(subset=["del_status", "coo"]).reset_index(drop=True)


def _channel_pvalues(frac, y, coo):
    ps = []
    for i in range(frac.shape[1]):
        try:
            X = sm.add_constant(np.column_stack([frac[:, i], coo]))
            ps.append(float(sm.Logit(y, X).fit(disp=0).pvalues[1]))
        except Exception:
            ps.append(1.0)
    return np.array(ps)


def _n_sig(frac, y, coo):
    return int((multipletests(_channel_pvalues(frac, y, coo), method="fdr_bh")[1] < config.FDR_ALPHA).sum())


def double_expressor(feat, clin, seed=config.SEED):
    """Composition shift in double-expressors, adjusted for cell-of-origin."""
    d = _del_frame(feat, clin)
    y = d["del_status"].astype(int).values
    coo = d["coo"].astype(float).values
    frac = _composition(d)
    p = _channel_pvalues(frac, y, coo)
    q = multipletests(p, method="fdr_bh")[1]
    delta = frac[y == 1].mean(0) - frac[y == 0].mean(0)
    table = pd.DataFrame({"channel": config.CHANNELS, "delta_x100": delta * 100, "p": p, "q_fdr": q})
    table["group"] = table["channel"].map(
        lambda c: "Immune" if c in config.IMMUNE else ("Tumor/proliferation" if c in config.TUMOR else "Other"))

    # case-level classifier (out-of-fold AUC)
    X = d[feature_columns(feat)].values
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed).split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]

    raw_total = d[mean_columns()].values.mean(1)
    summary = {
        "n": int(len(d)), "n_del": int(y.sum()), "n_non_del": int((1 - y).sum()),
        "n_fdr_significant_coo_adjusted": int((q < config.FDR_ALPHA).sum()),
        "significant_channels": table.loc[table["q_fdr"] < config.FDR_ALPHA, "channel"].tolist(),
        "global_signal_del": round(float(raw_total[y == 1].mean()), 3),
        "global_signal_non_del": round(float(raw_total[y == 0].mean()), 3),
        "global_signal_p": float(f"{sps.mannwhitneyu(raw_total[y == 1], raw_total[y == 0]).pvalue:.3g}"),
        "classifier_auc_oof": round(float(roc_auc_score(y, oof)), 3),
    }
    return table.sort_values("delta_x100").reset_index(drop=True), summary


def double_expressor_robustness(feat, clin, n_perm=config.N_PERMUTATIONS,
                                n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Permutation null, bootstrap CIs, leave-one-out, and threshold sensitivity."""
    rng = np.random.default_rng(seed)
    d = _del_frame(feat, clin)
    y = d["del_status"].astype(int).values
    coo = d["coo"].astype(float).values
    frac = _composition(d)
    observed = _n_sig(frac, y, coo)

    perm = np.array([_n_sig(frac, rng.permutation(y), coo) for _ in range(n_perm)])
    perm_p = (np.sum(perm >= observed) + 1) / (n_perm + 1)

    key = ["CD3", "CD4", "PD-L1", "CD68", "T-bet", "Ki67", "CD138", "Caspase3-D", "DAPI"]
    idx = {c: i for i, c in enumerate(config.CHANNELS)}
    boot = {k: [] for k in key}
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        ys, fs = y[s], frac[s]
        if ys.sum() < 3 or (1 - ys).sum() < 3:
            continue
        for k in key:
            boot[k].append((fs[ys == 1, idx[k]].mean() - fs[ys == 0, idx[k]].mean()) * 100)
    ci = {k: [round(float(np.percentile(v, 2.5)), 2), round(float(np.median(v)), 2),
              round(float(np.percentile(v, 97.5)), 2)] for k, v in boot.items()}

    loo = []
    for k in np.where(y == 1)[0]:
        keep = np.arange(len(y)) != k
        loo.append(_n_sig(frac[keep], y[keep], coo[keep]))

    thresholds = []
    for myc, bcl2 in config.THRESHOLD_GRID:
        dd = _del_frame(feat, clin, myc, bcl2)
        yy = dd["del_status"].astype(int).values
        thresholds.append({"myc": myc, "bcl2": bcl2, "n_del": int(yy.sum()),
                           "n_significant": _n_sig(_composition(dd), yy, dd["coo"].astype(float).values)})

    return {
        "observed_significant": observed,
        "permutation_n": int(n_perm),
        "permutation_null_median": float(np.median(perm)),
        "permutation_null_95th": float(np.percentile(perm, 95)),
        "permutation_p": round(float(perm_p), 4),
        "bootstrap_ci_x100": ci,
        "leave_one_out_min": int(min(loo)), "leave_one_out_max": int(max(loo)),
        "thresholds": thresholds,
        "_perm_distribution": perm.tolist(), "_loo_distribution": loo,
    }
