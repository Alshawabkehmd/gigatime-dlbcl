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
        z = cl[base_cols + ["os", "event"]].copy()
        z["gt_score_sd"] = (cl["gt_score"] - cl["gt_score"].mean()) / cl["gt_score"].std()
        hr_sd = float(np.exp(CoxPHFitter().fit(z, "os", "event").params_["gt_score_sd"]))
        per_seed[int(s)] = {"lrt_p": round(float(sps.chi2.sf(lr, 1)), 3),
                            "gigatime_hr": round(float(np.exp(full.params_["gt_score"])), 2),
                            "gigatime_hr_per_sd": round(hr_sd, 2),
                            "cindex_apparent": round(float(full.concordance_index_), 3)}
    ps = [v["lrt_p"] for v in per_seed.values()]
    hrs = [v["gigatime_hr"] for v in per_seed.values()]
    return {
        "n": int(len(cl)),
        "clinical_cindex_apparent": base_c,
        "lrt_p_median": round(float(np.median(ps)), 3),
        "lrt_p_range": [min(ps), max(ps)],
        "gigatime_hr_all_below_1": bool(all(h < 1 for h in hrs)),
        "gigatime_hr_per_sd_range": [min(v["gigatime_hr_per_sd"] for v in per_seed.values()),
                                     max(v["gigatime_hr_per_sd"] for v in per_seed.values())],
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


FIT_FAILURES = []   # (where, channel) of any logistic fit that failed; reported in results


def _channel_pvalues(frac, y, coo, where="primary"):
    ps = []
    for i in range(frac.shape[1]):
        try:
            X = sm.add_constant(np.column_stack([frac[:, i], coo]))
            ps.append(float(sm.Logit(y, X).fit(disp=0).pvalues[1]))
        except Exception:
            FIT_FAILURES.append((where, config.CHANNELS[i]))
            ps.append(1.0)   # counted as not significant; logged in FIT_FAILURES
    return np.array(ps)


def _n_sig(frac, y, coo):
    return int((multipletests(_channel_pvalues(frac, y, coo), method="fdr_bh")[1] < config.FDR_ALPHA).sum())


def double_expressor(feat, clin, seed=config.SEED):
    """Composition shift in double-expressors, adjusted for cell-of-origin."""
    d = _del_frame(feat, clin)
    y = d["del_status"].astype(int).values
    coo = d["coo"].astype(float).values
    frac = _composition(d)
    n_fail_before = len(FIT_FAILURES)
    p = _channel_pvalues(frac, y, coo)
    q = multipletests(p, method="fdr_bh")[1]
    n_failed = len(FIT_FAILURES) - n_fail_before
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
        "classifier_features": f"all {X.shape[1]} raw feature columns",
        "classifier_auc_oof_composition_only": round(float(roc_auc_score(y, _oof_logit(frac, y, seed))), 3),
        "n_failed_logistic_fits": int(n_failed),
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

    # labels are shuffled WITHIN cell-of-origin strata, so the null keeps the
    # double-expressor / cell-of-origin association (exchangeable under H0)
    perm = np.array([_n_sig(frac, _strat_perm(y, coo, rng), coo) for _ in range(n_perm)])
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
                           "same_patients_as_primary": bool(set(dd.loc[yy == 1, "case_id"]) == set(d.loc[y == 1, "case_id"])),
                           "n_significant": _n_sig(_composition(dd), yy, dd["coo"].astype(float).values)})

    return {
        "observed_significant": observed,
        "permutation_n": int(n_perm),
        "permutation_scheme": "labels shuffled within Hans cell-of-origin strata",
        "permutation_null_max": int(perm.max()),
        "permutation_null_median": float(np.median(perm)),
        "permutation_null_95th": float(np.percentile(perm, 95)),
        "permutation_p": round(float(perm_p), 4),
        "bootstrap_ci_x100": ci,
        "leave_one_out_min": int(min(loo)), "leave_one_out_max": int(max(loo)),
        "thresholds": thresholds,
        "_perm_distribution": perm.tolist(), "_loo_distribution": loo,
    }


# ---------------------------------------------------------------- additions after independent audit
def _strat_perm(y, strata, rng):
    """Shuffle labels y within each level of `strata`."""
    out = np.array(y).copy()
    for g in np.unique(strata):
        idx = np.where(strata == g)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def survival_null(feat, clin, n_perm=config.N_SURVIVAL_PERMUTATIONS, seed=config.SEED):
    """Full-pipeline outcome permutation for the GigaTIME signature C-index.

    Out-of-fold C-indices below 0.5 occur under a true null (cross-validation
    "anti-learning"), so the observed value is compared with C-indices obtained
    after shuffling (time, event) pairs and re-running the whole pipeline.
    """
    m = _survival_frame(feat, clin)
    cols = feature_columns(feat)

    def pooled_and_fold(mm):
        r = oof_cox_scores(mm, cols, seed)
        pooled = _cindex(mm["os"], r, mm["event"])
        fold = [_cindex(mm["os"].iloc[te], r[te], mm["event"].iloc[te])
                for _, te in KFold(config.CV_FOLDS, shuffle=True, random_state=seed).split(mm)]
        return pooled, float(np.mean(fold))

    obs_pooled, obs_fold = pooled_and_fold(m)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        mm = m.copy()
        mm[["os", "event"]] = m[["os", "event"]].values[rng.permutation(len(m))]
        null.append(pooled_and_fold(mm)[0])
    null = np.array(null)
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {
        "n_permutations": int(n_perm),
        "observed_cindex_pooled": round(obs_pooled, 3),
        "observed_cindex_per_fold_mean": round(obs_fold, 3),
        "null_median": round(float(np.median(null)), 3),
        "null_95_range": [round(float(lo), 3), round(float(hi), 3)],
        "null_fraction_below_0_5": round(float((null < 0.5).mean()), 2),
        "p_two_sided": round(float((np.sum(np.abs(null - 0.5) >= abs(obs_pooled - 0.5)) + 1) / (n_perm + 1)), 3),
        "events": int(m["event"].sum()),
        # Schoenfeld: detectable HR per SD at 80 % power, alpha 0.05, one standardized covariate
        "detectable_hr_per_sd_80pct_power": round(float(np.exp((1.959964 + 0.841621) / np.sqrt(m["event"].sum()))), 2),
        "_null": null.tolist(),
    }


def double_expressor_cohort(feat, clin):
    """Table 1 for the patients actually analysed (feature cohort with DE status and COO).

    Note: HANS is coded 1 = GCB, 0 = non-GCB (verified by re-deriving the Hans
    algorithm from CD10 / BCL6 / MUM1).
    """
    from lifelines import KaplanMeierFitter
    d = _del_frame(feat, clin)
    y = d["del_status"].astype(int).values

    def group(g):
        x = d[y == g]
        q = x["age"].quantile([0.25, 0.5, 0.75]).values
        km = KaplanMeierFitter().fit(x["os"], x["event"])
        med = km.median_survival_time_
        return {
            "n": int(len(x)),
            "age_median_iqr": [round(float(v), 1) for v in q[[1, 0, 2]]],
            "stage_3_4_pct": round(float((x["stage"] >= 3).mean() * 100), 1),
            "ipi_3_5_pct": round(float((x["ipi_score"] >= 3).mean() * 100), 1),
            "ldh_elevated_pct": round(float((x["ldh"] == 1).sum() / x["ldh"].notna().sum() * 100), 1),
            "non_gcb_pct": round(float((x["coo"] == 0).mean() * 100), 1),
            "deaths_pct": round(float(x["event"].mean() * 100), 1),
            "km_median_os_years": None if not np.isfinite(med) else round(float(med), 1),
        }

    tab = pd.crosstab(y, d["coo"].astype(int))
    hit = (d["myc_fish"] == 1) & ((d["bcl2_fish"] == 1) | (d["bcl6_fish"] == 1))
    return {
        "double_expressor": group(1), "other": group(0),
        "non_gcb_fisher_p": float(f"{sps.fisher_exact(tab.values)[1]:.3g}"),
        "double_hit_or_triple_hit_n": int(hit.sum()),
        "double_hit_among_double_expressors": int((hit & (y == 1)).sum()),
    }


def double_expressor_axis(feat, clin, n_perm=config.N_PERMUTATIONS, seed=config.SEED):
    """Is the double-expressor signal one axis or many channels, and what drives it?

    - centered log-ratio (CLR) re-analysis instead of shares;
    - PCA of the CLR data and a single global test on PC1 (COO-adjusted),
      with a within-COO-strata permutation p;
    - adjustment for each channel in turn (if adjusting for one channel removes
      the rest, the channels are one axis, not independent findings);
    - MYC and BCL2 as continuous variables;
    - non-GCB stratum only, and exclusion of double/triple-hit cases.
    """
    d = _del_frame(feat, clin)
    y = d["del_status"].astype(int).values
    coo = d["coo"].astype(float).values
    raw = d[mean_columns()].values
    frac = raw / raw.sum(1, keepdims=True)
    lr = np.log(np.clip(raw, 1e-4, None))
    clr = lr - lr.mean(1, keepdims=True)
    ch = config.CHANNELS

    def count(X, yy, cov, method="fdr_bh"):
        ps = []
        for i in range(X.shape[1]):
            try:
                M = sm.add_constant(np.column_stack([X[:, i], cov]))
                ps.append(float(sm.Logit(yy, M).fit(disp=0).pvalues[1]))
            except Exception:
                ps.append(1.0)
        return int((multipletests(ps, method=method)[1] < config.FDR_ALPHA).sum())

    Z = (clr - clr.mean(0)) / clr.std(0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U * sv
    if np.corrcoef(pcs[:, 0], frac[:, ch.index("CK")])[0, 1] < 0:   # orient PC1 so + = more CK share
        pcs[:, 0], Vt[0] = -pcs[:, 0], -Vt[0]
    var = sv ** 2 / np.sum(sv ** 2)

    def pc_p(k, yy=y):
        return float(sm.Logit(yy, sm.add_constant(np.column_stack([pcs[:, k], coo]))).fit(disp=0).pvalues[1])

    def pc1_t(yy):
        return abs(float(sm.Logit(yy, sm.add_constant(np.column_stack([pcs[:, 0], coo]))).fit(disp=0).tvalues[1]))

    rng = np.random.default_rng(seed)
    t_obs = pc1_t(y)
    t_null = np.array([pc1_t(_strat_perm(y, coo, rng)) for _ in range(n_perm)])

    single = {}
    for j, c in enumerate(ch):
        single[c] = count(np.delete(frac, j, 1), y, np.column_stack([coo, frac[:, j]]))

    myc = d["myc_ihc"].astype(float).values
    bcl2 = d["bcl2_ihc"].astype(float).values
    ck_raw = raw[:, ch.index("CK")]
    nongcb = coo == 0
    hit = ((d["myc_fish"] == 1) & ((d["bcl2_fish"] == 1) | (d["bcl6_fish"] == 1))).values

    def ols_p(v):
        return float(sm.OLS(pcs[:, 0], sm.add_constant(np.column_stack([v, coo]))).fit().pvalues[1])

    return {
        "n_significant_share_bh": count(frac, y, coo),
        "n_significant_share_by": count(frac, y, coo, "fdr_by"),
        "n_significant_clr_bh": count(clr, y, coo),
        "pca_variance_explained_pc1_3": [round(float(v), 2) for v in var[:3]],
        "pc1_top_loadings": [ch[i] for i in np.argsort(-np.abs(Vt[0]))[:6]],
        "pc1_coo_adjusted_p": float(f"{pc_p(0):.3g}"),
        "pc1_stratified_permutation_p": round(float((np.sum(t_null >= t_obs) + 1) / (n_perm + 1)), 4),
        "pc2_p": float(f"{pc_p(1):.3g}"), "pc3_p": float(f"{pc_p(2):.3g}"),
        "n_significant_after_adjusting_each_channel": single,
        "channels_whose_adjustment_removes_signal": [c for c, n in single.items() if n == 0],
        "n_significant_adjusted_for_ck": single["CK"],
        "ck_raw_mean_del": round(float(ck_raw[y == 1].mean()), 3),
        "ck_raw_mean_other": round(float(ck_raw[y == 0].mean()), 3),
        "ck_raw_p": float(f"{sps.mannwhitneyu(ck_raw[y == 1], ck_raw[y == 0]).pvalue:.3g}"),
        "ck_raw_vs_myc_spearman": round(float(sps.spearmanr(ck_raw, myc)[0]), 2),
        "pc1_vs_myc_continuous_p": float(f"{ols_p(myc):.3g}"),
        "pc1_vs_bcl2_continuous_p": float(f"{ols_p(bcl2):.3g}"),
        "non_gcb_only_n": [int(y[nongcb].sum()), int((1 - y[nongcb]).sum())],
        "non_gcb_only_n_significant": _n_sig_nocov(frac[nongcb], y[nongcb]),
        "excluding_double_hit_n_significant": count(frac[~hit], y[~hit], coo[~hit]),
        "_pc1_scores": pcs[:, 0].tolist(), "_y": y.tolist(), "_myc": myc.tolist(),
    }


def _n_sig_nocov(frac, y):
    ps = []
    for i in range(frac.shape[1]):
        try:
            ps.append(float(sm.Logit(y, sm.add_constant(frac[:, i])).fit(disp=0).pvalues[1]))
        except Exception:
            ps.append(1.0)
    return int((multipletests(ps, method="fdr_bh")[1] < config.FDR_ALPHA).sum())



def _oof_logit(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed).split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return oof


# ---------------------------------------------------------------- additions after second audit
def _firth_fit(X, y, fixed=None, maxit=200, tol=1e-9):
    """Firth-penalized logistic regression (Heinze & Schemper 2002).

    `fixed` = index of a coefficient held at 0; the penalty always uses the FULL design's
    information matrix, so the penalized likelihood-ratio test compares like with like.
    Returns (beta, penalized log-likelihood, covariance of the free parameters' fit).
    """
    from scipy.special import expit
    free = np.array([j for j in range(X.shape[1]) if j != fixed])
    b = np.zeros(X.shape[1])

    def pll(beta):
        pr = np.clip(expit(X @ beta), 1e-12, 1 - 1e-12)
        info = X.T @ (X * (pr * (1 - pr))[:, None])
        return float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)) + 0.5 * np.linalg.slogdet(info)[1])

    cur = pll(b)
    for _ in range(maxit):
        pr = expit(X @ b)
        w = pr * (1 - pr)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.inv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], inv, X * np.sqrt(w)[:, None])
        score = X.T @ (y - pr + h * (0.5 - pr))
        step = np.zeros_like(b)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        t = 1.0
        while t > 1e-8:
            new = pll(b + t * step)
            if new >= cur - 1e-12:
                break
            t /= 2
        b, prev, cur = b + t * step, cur, new
        if abs(cur - prev) < tol and np.max(np.abs(t * step)) < 1e-7:
            break
    pr = expit(X @ b)
    cov = np.linalg.inv(X.T @ (X * (pr * (1 - pr))[:, None]))
    return b, cur, cov


def firth_double_expressor(feat, clin):
    """Small-sample sensitivity: Firth logistic (penalized likelihood-ratio p) for each channel share,
    adjusted for Hans COO; odds ratios per SD with Wald 95 % CIs from the penalized fit."""
    d = _del_frame(feat, clin)
    y = d["del_status"].astype(float).values
    coo = d["coo"].astype(float).values
    raw = d[mean_columns()].values
    frac = raw / raw.sum(1, keepdims=True)
    rows = []
    for i, c in enumerate(config.CHANNELS):
        z = (frac[:, i] - frac[:, i].mean()) / frac[:, i].std()
        Xf = np.column_stack([np.ones(len(y)), z, coo])
        bf, lf, cov = _firth_fit(Xf, y)
        _, lr, _ = _firth_fit(Xf, y, fixed=1)
        p = float(sps.chi2.sf(2 * (lf - lr), 1))
        se = float(np.sqrt(cov[1, 1]))
        rows.append({"channel": c, "or_per_sd": float(np.exp(bf[1])),
                     "ci_low": float(np.exp(bf[1] - 1.96 * se)), "ci_high": float(np.exp(bf[1] + 1.96 * se)), "p": p})
    t = pd.DataFrame(rows)
    t["q_fdr"] = multipletests(t["p"], method="fdr_bh")[1]
    sig = t[t["q_fdr"] < config.FDR_ALPHA]
    key = t.set_index("channel").loc[["CK", "CD3", "CD20", "CD138", "DAPI", "Ki67"]]
    return t, {
        "n_significant_firth": int(len(sig)),
        "significant_channels_firth": sig["channel"].tolist(),
        "or_per_sd_key": {k: [round(r.or_per_sd, 2), round(r.ci_low, 2), round(r.ci_high, 2)] for k, r in key.iterrows()},
        "n_gcb_double_expressors": int(((y == 1) & (coo == 1)).sum()),
    }


def cohort_selection(feat, clin):
    """Patients with vs without usable image features; IPI reconstruction check."""
    c = clin.copy()
    inc = c["case_id"].isin(set(feat["case_id"]))

    def mw(col):
        a, b = c.loc[inc, col].dropna(), c.loc[~inc, col].dropna()
        return float(f"{sps.mannwhitneyu(a, b).pvalue:.3g}")

    en = c["EN"] if "EN" in c.columns else pd.Series(np.nan, index=c.index)
    recon = ((c["age"] > 60).astype(float) + (c["stage"] >= 3) + (c["ldh"] == 1) + (c["ecog"] >= 2) + (en >= 2))
    complete = c[["age", "stage", "ldh", "ecog", "ipi_score"]].notna().all(1) & en.notna()
    return {
        "included": int(inc.sum()), "excluded": int((~inc).sum()),
        "ipi_mean_included": round(float(c.loc[inc, "ipi_score"].mean()), 2),
        "ipi_mean_excluded": round(float(c.loc[~inc, "ipi_score"].mean()), 2),
        "ipi_mannwhitney_p": mw("ipi_score"),
        "age_median_included": float(c.loc[inc, "age"].median()), "age_median_excluded": float(c.loc[~inc, "age"].median()),
        "deaths_pct_included": round(float(c.loc[inc, "event"].mean() * 100), 1),
        "deaths_pct_excluded": round(float(c.loc[~inc, "event"].mean() * 100), 1),
        "ipi_reconstruction_complete_records": int(complete.sum()),
        "ipi_reconstruction_mismatches": int((complete & (recon != c["ipi_score"])).sum()),
    }


def follow_up_and_ph(feat, clin):
    """Follow-up (reverse Kaplan-Meier), KM median OS, and proportional-hazards checks."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
    m = _survival_frame(feat, clin)
    rev = KaplanMeierFitter().fit(m["os"], 1 - m["event"])
    km = KaplanMeierFitter().fit(m["os"], m["event"])
    cl = m.dropna(subset=config.CLINICAL_COVARIATES)
    cph = CoxPHFitter().fit(cl[config.CLINICAL_COVARIATES + ["os", "event"]], "os", "event")
    ph = proportional_hazard_test(cph, cl[config.CLINICAL_COVARIATES + ["os", "event"]], time_transform="rank")
    return {
        "median_follow_up_reverse_km_years": round(float(rev.median_survival_time_), 1),
        "km_median_os_years": round(float(km.median_survival_time_), 1),
        "median_observed_time_years": round(float(m["os"].median()), 1),
        "ph_test_p": {k: float(f"{v:.3g}") for k, v in zip(config.CLINICAL_COVARIATES, ph.p_value)},
        "os_time_origin": "end of treatment (Vrabac et al., 2021)",
    }


def matched_survival_comparison(feat, clin, seeds=config.SIGNATURE_SEEDS, n_boot=1000, seed=config.SEED,
                                bootstrap=True):
    """IPI, clinical, clinical + extranodal, GigaTIME and clinical + GigaTIME, all on the SAME
    patients and the SAME outer folds. The combined model is built entirely inside each outer
    training fold: the GigaTIME score for training patients comes from an inner 5-fold
    cross-validation, the clinical + score Cox model is fitted on those, and it is applied to the
    held-out fold with the GigaTIME model refitted on the whole training fold. Reports pooled
    out-of-fold C-indices per split and the change in C (combined minus clinical) with a
    patient-bootstrap 95 % CI of the out-of-fold predictions (does not include refitting variance).
    """
    m = _survival_frame(feat, clin)
    if "EN" not in m.columns:
        m["EN"] = np.nan
    m["en"] = m["EN"]
    clin_cols = config.CLINICAL_COVARIATES
    clin_en = clin_cols + ["en"]
    m = m.dropna(subset=clin_en + ["ipi_score"]).reset_index(drop=True)
    gcols = feature_columns(feat)
    rng = np.random.default_rng(seed)

    def cox_fit_predict(train, test, cols, pen=0.01, l1=0.0):
        mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
        d = ((train[cols] - mu) / sd).copy()
        d["os"], d["event"] = train["os"].values, train["event"].values
        cph = CoxPHFitter(penalizer=pen, l1_ratio=l1).fit(d, "os", "event")
        return cph.predict_log_partial_hazard((test[cols] - mu) / sd).values

    per_seed, deltas, preds = {}, [], []
    for s in seeds:
        pred = {k: np.full(len(m), np.nan) for k in ["clinical", "clinical_en", "gigatime", "combined"]}
        for tr, te in KFold(config.CV_FOLDS, shuffle=True, random_state=s).split(m):
            train, test = m.iloc[tr].reset_index(drop=True), m.iloc[te]
            pred["clinical"][te] = cox_fit_predict(train, test, clin_cols)
            pred["clinical_en"][te] = cox_fit_predict(train, test, clin_en)
            pred["gigatime"][te] = cox_fit_predict(train, test, gcols, config.COX_PENALIZER, config.COX_L1_RATIO)
            train = train.copy()
            train["gt"] = oof_cox_scores(train, gcols, s)          # inner CV, training fold only
            test = test.copy()
            test["gt"] = pred["gigatime"][te]
            pred["combined"][te] = cox_fit_predict(train, test, clin_cols + ["gt"])
        c = {k: _cindex(m["os"], v, m["event"]) for k, v in pred.items()}
        c["ipi"] = _cindex(m["os"], m["ipi_score"], m["event"])
        per_seed[int(s)] = {k: round(v, 3) for k, v in c.items()}
        preds.append(pred)
        deltas.append(c["combined"] - c["clinical"])
    # patient bootstrap of the change in C, averaged over all splits (refitting variance not included)
    boot = []
    for _ in range(n_boot if bootstrap else 0):
        i = rng.integers(0, len(m), len(m))
        t, e = m["os"].iloc[i], m["event"].iloc[i]
        boot.append(np.mean([_cindex(t, p["combined"][i], e) - _cindex(t, p["clinical"][i], e) for p in preds]))
    ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)] if boot else None
    mean = {k: round(float(np.mean([v[k] for v in per_seed.values()])), 3) for k in per_seed[config.SEED]}
    return {
        "n": int(len(m)), "events": int(m["event"].sum()),
        "cindex_mean_over_splits": mean,
        "delta_c_combined_minus_clinical_range": [round(float(min(deltas)), 3), round(float(max(deltas)), 3)],
        "delta_c_mean": round(float(np.mean(deltas)), 3),
        "delta_c_bootstrap_95ci": ci,
        "per_seed": per_seed,
    }


def added_value_null(feat, clin, n_perm=config.N_ADDED_VALUE_PERMUTATIONS, seed=config.SEED, seeds=config.SIGNATURE_SEEDS):
    """Full-procedure null for the change in C: GigaTIME feature rows are shuffled across patients
    (keeping the clinical-outcome relationships intact) and matched_survival_comparison is re-run."""
    obs = matched_survival_comparison(feat, clin, seeds=seeds, bootstrap=False)["delta_c_mean"]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        f = feat.copy()
        cols = feature_columns(feat)
        f[cols] = feat[cols].values[rng.permutation(len(feat))]
        null.append(matched_survival_comparison(f, clin, seeds=seeds, bootstrap=False)["delta_c_mean"])
    null = np.array(null)
    return {"n_permutations": int(n_perm), "observed_delta_c_mean": obs,
            "null_95_range": [round(float(np.percentile(null, 2.5)), 3), round(float(np.percentile(null, 97.5)), 3)],
            "p_one_sided": round(float((np.sum(null >= obs) + 1) / (n_perm + 1)), 3), "_null": null.tolist()}
