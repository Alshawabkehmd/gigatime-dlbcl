"""Publication figures. Every function saves a PNG (300 dpi) and returns its path."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from . import config

# Palette (validated: CVD-safe categorical pair + neutral)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8985"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
IPI_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordered risk: light -> dark
GROUP_COLOR = {"Immune": BLUE, "Tumor/proliferation": ORANGE, "Other": GREY}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.edgecolor": "#c9c8c3", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.6, "legend.frameon": False,
    })


def _save(fig, out_dir, name, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", fontsize=46, color="#d03b3b", alpha=0.18,
                 ha="center", va="center", rotation=25, fontweight="bold")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    return path


def channel_levels(feat, out_dir, synthetic=False):
    """Face validity: cohort-average level of each virtual channel."""
    means = {c: feat[f"{c}_mean"].mean() for c in config.CHANNELS}
    order = sorted(means, key=means.get)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    cols = [BLUE if c in config.IMMUNE else ORANGE if c in config.TUMOR else GREY for c in order]
    ax.barh(order, [means[c] for c in order], color=cols, height=0.7)
    for i, c in enumerate(order):
        if c in ("CD20", "CK"):
            ax.text(means[c] + 0.01, i, f"{means[c]:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Cohort-average virtual signal (0-1)")
    ax.set_title("Virtual channel levels across the cohort")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=BLUE, label="Immune"), Patch(color=ORANGE, label="Tumor / proliferation"),
                       Patch(color=GREY, label="Other")], loc="lower right")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_channel_levels.png", synthetic)


def coexpression_heatmap(C, out_dir, synthetic=False):
    d = 1 - C.values
    np.fill_diagonal(d, 0)
    order = dendrogram(linkage(squareform(d, checks=False), "average"), no_plot=True)["leaves"]
    Cw = C.iloc[order, order]
    labs = list(Cw.columns)
    fig, ax = plt.subplots(figsize=(8.6, 7.8))
    im = ax.imshow(Cw.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=90, fontsize=9)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=9)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        n = t.get_text()
        t.set_color(BLUE if n in config.IMMUNE else ORANGE if n in config.TUMOR else INK2)
    ax.grid(False)
    ax.set_title("Virtual channel co-expression (Pearson r, clustered)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_coexpression.png", synthetic)


def cindex_comparison(sig, clin_ipi, out_dir, synthetic=False):
    items = [
        ("GigaTIME\nsignature", sig["cindex_mean"], ORANGE),
        ("Published\nmorphology*", config.MORPHOLOGY_BASELINE_CINDEX, GREY),
        ("IPI score", clin_ipi["ipi_cindex"], BLUE),
        ("Clinical\nmodel", clin_ipi["clinical_cindex_oof"], BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhspan(0.35, 0.5, color="#d03b3b", alpha=0.06, zorder=0)
    lo, hi = sig["cindex_min"], sig["cindex_max"]
    for i, (lab, v, col) in enumerate(items):
        ax.bar(i, v, color=col, width=0.58, zorder=3)
        top = max(v, hi) if i == 0 else v
        ax.text(i, top + 0.008, f"{v:.2f}", ha="center", fontweight="bold", color=INK)
    ax.errorbar(0, sig["cindex_mean"], yerr=[[sig["cindex_mean"] - lo], [hi - sig["cindex_mean"]]],
                color=INK, capsize=4, lw=1.2, zorder=4)
    ax.axhline(0.5, color=INK2, ls="--", lw=1)
    ax.text(-0.45, 0.505, "chance (0.50)", ha="left", fontsize=9, color=INK2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i[0] for i in items])
    ax.set_ylim(0.35, 0.75)
    ax.set_ylabel("Overall-survival C-index")
    ax.set_title("Survival discrimination: GigaTIME vs clinical prognosticators")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, "GigaTIME: out-of-fold, mean (range) over 5 CV seeds. Clinical: out-of-fold. "
             "*Published value is optimism-corrected in-sample, not out-of-fold.", fontsize=8, color=INK2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_cindex.png", synthetic)


def km_ipi(m, p_value, out_dir, synthetic=False):
    d = m.dropna(subset=["ipi_group"])
    labels = {0: "Low", 1: "Low-intermediate", 2: "High-intermediate", 3: "High"}
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for g in sorted(d["ipi_group"].astype(int).unique()):
        s = d[d["ipi_group"].astype(int) == g]
        kmf.fit(s["os"], s["event"], label=f"{labels[g]} (n={len(s)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=IPI_RAMP[g], lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"Positive control: survival by IPI risk group (log-rank p = {p_value:.1e})")
    ax.legend(title="IPI risk", loc="lower left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_km_ipi.png", synthetic)


def km_gigatime(m, km, out_dir, synthetic=False):
    hi = m["gt_score"] > m["gt_score"].median()
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for mask, lab, col in [(~hi, "Predicted low risk", BLUE), (hi, "Predicted high risk", ORANGE)]:
        kmf.fit(m.loc[mask, "os"], m.loc[mask, "event"], label=f"{lab} (n={int(mask.sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=col, lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"GigaTIME signature, median split (log-rank p = {km['gigatime_split_logrank_p']:.2f})")
    ax.legend(loc="lower left")
    fig.text(0.01, -0.02, f"Death rate: predicted high risk {km['gigatime_high_risk_death_rate']:.0%} vs "
             f"low risk {km['gigatime_low_risk_death_rate']:.0%} - the signature does not rank risk correctly.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_km_gigatime.png", synthetic)


def double_expressor_bars(table, summary, out_dir, synthetic=False):
    fig, ax = plt.subplots(figsize=(8, 7))
    for j, r in table.iterrows():
        ax.barh(j, r["delta_x100"], color=GROUP_COLOR[r["group"]], height=0.7, zorder=3)
        q = r["q_fdr"]
        star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
        off = 0.08 if r["delta_x100"] >= 0 else -0.08
        ax.text(r["delta_x100"] + off, j, star, va="center", ha="left" if r["delta_x100"] >= 0 else "right",
                color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["channel"], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in share of total virtual signal (x100), double-expressor minus other")
    ax.set_title(f"Virtual composition shift in double-expressor tumors (n={summary['n_del']} vs {summary['n_non_del']})")
    ax.legend(handles=[Patch(color=BLUE, label="Immune"), Patch(color=ORANGE, label="Tumor / proliferation"),
                       Patch(color=GREY, label="Other")], loc="lower right")
    fig.text(0.01, -0.02, "* FDR q<0.05, ** q<0.01, *** q<0.001 (logistic regression adjusted for cell-of-origin). "
             "Shares sum to 1, so shifts are not independent; they form one axis (see fig_double_expressor_axis).", fontsize=8, color=INK2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_double_expressor.png", synthetic)


def robustness_panel(rob, out_dir, synthetic=False):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    perm = np.array(rob["_perm_distribution"])
    a.hist(perm, bins=np.arange(-0.5, max(perm.max(), rob["observed_significant"]) + 1.5), color=GREY)
    a.axvline(rob["observed_significant"], color=ORANGE, lw=2.5)
    a.set_title(f"A. Permutation null, {rob['permutation_n']} shuffles (p = {rob['permutation_p']:.3f})")
    a.set_xlabel("Significant channels, labels shuffled within cell-of-origin")
    a.set_ylabel("Count")
    b = ax[0, 1]
    ci = rob["bootstrap_ci_x100"]
    keys = list(ci)[::-1]
    for j, k in enumerate(keys):
        lo, md, hi = ci[k]
        col = BLUE if md < 0 else ORANGE
        b.plot([lo, hi], [j, j], color=col, lw=2.2)
        b.plot(md, j, "o", color=col, ms=6)
    b.axvline(0, color=INK, lw=1)
    b.set_yticks(range(len(keys)))
    b.set_yticklabels(keys)
    b.set_title("B. Bootstrap 95% CIs")
    b.set_xlabel("Composition shift (x100)")
    c = ax[1, 0]
    loo = np.array(rob["_loo_distribution"])
    c.hist(loo, bins=np.arange(loo.min() - 0.5, loo.max() + 1.5), color=BLUE, rwidth=0.85)
    c.set_xticks(range(int(loo.min()), int(loo.max()) + 1))
    c.set_title(f"C. Leave-one-out ({rob['leave_one_out_min']}-{rob['leave_one_out_max']} channels)")
    c.set_xlabel("Significant channels after dropping one case")
    c.set_ylabel("Count")
    d = ax[1, 1]
    th = rob["thresholds"]
    labs = [f"MYC>={t['myc']}\nBCL2>={t['bcl2']}" for t in th]
    d.bar(range(len(th)), [t["n_significant"] for t in th], color=BLUE, width=0.6)
    for i, t in enumerate(th):
        tag = "same 21" if t.get("same_patients_as_primary") else f"n={t['n_del']}"
        d.text(i, t["n_significant"] + 0.2, f"{t['n_significant']}\n({tag})", ha="center", fontsize=8)
    d.set_xticks(range(len(th)))
    d.set_xticklabels(labs, fontsize=8)
    d.set_title("D. Threshold sensitivity")
    d.set_ylabel("Significant channels")
    d.set_ylim(0, max(t["n_significant"] for t in th) + 3)
    fig.suptitle("Internal robustness of the double-expressor signature", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_robustness.png", synthetic)


def double_expressor_axis(axis, out_dir, synthetic=False):
    """A: PC1 by group. B: PC1 vs MYC IHC. C: significant channels after adjusting for each channel."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [0.8, 1.1, 1.4]})
    pc1, y, myc = np.array(axis["_pc1_scores"]), np.array(axis["_y"]), np.array(axis["_myc"])
    rng = np.random.default_rng(0)
    a = ax[0]
    for g, col, lab in [(0, GREY, "Other"), (1, ORANGE, "Double-expressor")]:
        v = pc1[y == g]
        a.scatter(g + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=col, alpha=0.7, zorder=3)
        a.plot([g - 0.25, g + 0.25], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    a.set_xticks([0, 1]); a.set_xticklabels([f"Other\n(n={(y == 0).sum()})", f"Double-expressor\n(n={(y == 1).sum()})"])
    a.set_ylabel("PC1 score (centered log-ratio)")
    a.set_title(f"A. One axis (PC1 p = {axis['pc1_coo_adjusted_p']:.1g})")
    b = ax[1]
    b.scatter(myc[y == 0] + rng.uniform(-1.5, 1.5, (y == 0).sum()), pc1[y == 0], s=14, color=GREY, alpha=0.7, label="Other")
    b.scatter(myc[y == 1] + rng.uniform(-1.5, 1.5, (y == 1).sum()), pc1[y == 1], s=18, color=ORANGE, alpha=0.85, label="Double-expressor")
    b.set_xlabel("MYC IHC (%)"); b.set_ylabel("PC1 score")
    b.set_title(f"B. Tracks MYC (p = {axis['pc1_vs_myc_continuous_p']:.0e}) > BCL2 (p = {axis['pc1_vs_bcl2_continuous_p']:.2f})")
    b.legend(loc="upper left", fontsize=9)
    c = ax[2]
    adj = axis["n_significant_after_adjusting_each_channel"]
    names = sorted(adj, key=lambda k: adj[k])
    c.barh(range(len(names)), [adj[k] for k in names], color=[ORANGE if adj[k] == 0 else BLUE for k in names], height=0.7, zorder=3)
    c.set_yticks(range(len(names))); c.set_yticklabels(names, fontsize=8)
    c.grid(axis="y", visible=False)
    c.set_xlabel("Significant channels after adjusting for this one")
    c.set_title("C. Adjusting for one axis channel removes all")
    fig.suptitle("The double-expressor shift is one MYC-associated axis, not many independent channels",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_double_expressor_axis.png", synthetic)


def survival_null(sn, out_dir, synthetic=False):
    """Null distribution of the out-of-fold C-index after shuffling outcomes."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    null = np.array(sn["_null"])
    ax.hist(null, bins=25, color=GREY, zorder=3)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.axvline(sn["observed_cindex_pooled"], color=ORANGE, lw=2.5)
    ax.text(sn["observed_cindex_pooled"], ax.get_ylim()[1] * 0.92, f" observed {sn['observed_cindex_pooled']:.2f}",
            color=ORANGE, ha="left", fontsize=9)
    ax.set_xlabel("Out-of-fold C-index with shuffled survival outcomes")
    ax.set_ylabel("Permutations")
    lo, hi = sn["null_95_range"]
    ax.set_title(f"C-index below 0.5 is within chance (null 95%: {lo:.2f}-{hi:.2f}; p = {sn['p_two_sided']:.2f})")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_survival_null.png", synthetic)


def mif_showcase(he, vir, out_dir):
    """H&E next to a composite and single virtual channels (needs full-mode output)."""
    idx = {n: i for i, n in enumerate(config.CHANNELS)}

    def norm(c):
        c = np.clip(c, 0, 1)
        return (c - c.min()) / (np.ptp(c) + 1e-8)

    def tint(c, rgb):
        return norm(c)[..., None] * np.array(rgb)

    comp = np.clip(tint(vir[..., idx["DAPI"]], (0.15, 0.35, 1.0)) * 0.9
                   + tint(vir[..., idx["CD20"]], (0.1, 1.0, 0.35))
                   + tint(vir[..., idx["CD3"]], (1.0, 0.15, 0.75)) * 0.8, 0, 1)
    panels = [("H&E (input)", he), ("Composite: DAPI / CD20 / CD3", comp),
              ("CD20 (B cell)", tint(vir[..., idx["CD20"]], (0.1, 1, 0.35))),
              ("CD3 (T cell)", tint(vir[..., idx["CD3"]], (1, 0.15, 0.75))),
              ("CD68 (macrophage)", tint(vir[..., idx["CD68"]], (1, 0.75, 0.1))),
              ("CK (negative control)", tint(vir[..., idx["CK"]], (0.9, 0.25, 0.2)))]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.6))
    for a, (t, im) in zip(axes, panels):
        a.imshow(im)
        a.set_title(t, fontsize=10, loc="center")
        a.axis("off")
    fig.suptitle("GigaTIME virtual multiplex immunofluorescence on a DLBCL H&E patch "
                 "(virtual channels only; no real-mIF ground truth in this cohort)", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_mif_showcase.png", False)
