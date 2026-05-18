"""Plot generation for the validation report.

Every function takes a results dict (from :mod:`mammoval.pipeline`) and returns
a base64-encoded PNG data URI, so :mod:`mammoval.report` can embed the figures
directly into a single self-contained HTML file — nothing to host, easy to
attach to a submission or share with a reviewer.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless: works in Colab, CI and on a server
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "roc_plot", "calibration_plot", "triage_plot", "risk_band_plot",
    "subgroup_forest_plot", "decision_curve_plot", "froc_plot",
]

_BLUE, _GREY, _RED, _GREEN = "#1f5fa8", "#888888", "#c0392b", "#2e8b57"


def _to_uri(fig):
    """Render a Matplotlib figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def roc_plot(results):
    """ROC curve with the AUC + CI and the target-specificity operating points."""
    disc = results["discrimination"]
    roc = disc["roc_curve"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(roc["fpr"], roc["tpr"], color=_BLUE, lw=2.2,
            label=f"AI  AUC={disc['auc']:.3f} "
                  f"({disc['auc_ci'][0]:.3f}-{disc['auc_ci'][1]:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=_GREY, lw=1, label="chance")

    for op in results.get("operating_points", {}).get("by_target_specificity", []):
        conf = op["confusion"]
        ax.plot(1 - conf["specificity"], conf["sensitivity"], "o",
                color=_RED, ms=7)
        ax.annotate(f"  spec {op['target_specificity']:.2f}\n  sens "
                    f"{conf['sensitivity']:.2f}",
                    (1 - conf["specificity"], conf["sensitivity"]),
                    fontsize=8, va="top")

    rc = results.get("reader_comparison", {})
    if "auc_reader" in rc:
        ax.plot([], [], " ", label=f"reference reader AUC={rc['auc_reader']:.3f}")

    ax.set_xlabel("1 - specificity  (false-positive rate)")
    ax.set_ylabel("sensitivity  (true-positive rate)")
    ax.set_title("ROC - standalone discrimination")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    return _to_uri(fig)


def calibration_plot(results):
    """Reliability diagram — observed cancer rate vs predicted probability."""
    cal = results.get("calibration", {})
    rows = cal.get("reliability_curve")
    if not rows:
        return None
    pred = [r["predicted_mean"] for r in rows]
    obs = [r["observed_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0, 1], [0, 1], "--", color=_GREY, lw=1, label="perfect calibration")
    ax.plot(pred, obs, "o-", color=_BLUE, lw=2, label="AI model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed cancer rate")
    slope = cal.get("intercept_slope", {})
    title = "Calibration (reliability diagram)"
    if "slope" in slope:
        title += f"\nslope={slope['slope']:.2f}, intercept={slope['intercept']:.2f}"
        if "brier_score" in cal:
            title += f", Brier={cal['brier_score']:.3f}"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _to_uri(fig)


def triage_plot(results):
    """Workload reduction vs sensitivity retained — the rule-out trade-off."""
    rows = results.get("screening", {}).get("triage_simulation")
    if not rows:
        return None
    wl = [r["workload_reduction"] * 100 for r in rows]
    sens = [r["sensitivity_retained"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(wl, sens, "-", color=_BLUE, lw=2.2)
    ax.fill_between(wl, sens, 100, color=_RED, alpha=0.08)
    ax.set_xlabel("workload reduction  (% of exams ruled out)")
    ax.set_ylabel("sensitivity retained  (%)")
    ax.set_title("AI rule-out / triage trade-off")
    ax.axhline(95, ls="--", color=_GREEN, lw=1)
    ax.annotate("95% sensitivity floor", (1, 95.4), fontsize=8, color=_GREEN)
    ax.grid(alpha=0.25)
    return _to_uri(fig)


def risk_band_plot(results):
    """Observed cancer rate per ordinal risk band, with Wilson CIs."""
    rows = results.get("screening", {}).get("risk_bands")
    if not rows:
        return None
    bands = [r["band"] for r in rows]
    rate = [r["cancer_rate"] * 100 for r in rows]
    lo = [(r["cancer_rate"] - r["cancer_rate_ci"][0]) * 100 for r in rows]
    hi = [(r["cancer_rate_ci"][1] - r["cancer_rate"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(bands, rate, color=_BLUE, alpha=0.85,
           yerr=[lo, hi], capsize=3, ecolor=_GREY)
    ax.set_xlabel("AI risk band (low -> high score)")
    ax.set_ylabel("observed cancer rate (%)")
    ax.set_title("Risk stratification across score bands")
    ax.grid(alpha=0.25, axis="y")
    return _to_uri(fig)


def subgroup_forest_plot(results):
    """Forest plot of per-subgroup AUC (point + CI) across all subgroups."""
    subs = results.get("subgroups", {})
    rows = []
    for col, block in subs.items():
        if not isinstance(block, dict) or "groups" not in block:
            continue
        for g in block["groups"]:
            rows.append((f"{col}: {g['group']}  (n={g['n']})",
                         g["auc"], g["ci_low"], g["ci_high"]))
    if not rows:
        return None
    rows.reverse()
    labels = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    err_lo = [r[1] - r[2] for r in rows]
    err_hi = [r[3] - r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.errorbar(aucs, y, xerr=[err_lo, err_hi], fmt="o", color=_BLUE,
                capsize=3, ms=6)
    ax.axvline(0.5, ls="--", color=_GREY, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("subgroup AUC (95% CI)")
    ax.set_title("Subgroup discrimination - effect-modifier check")
    ax.grid(alpha=0.25, axis="x")
    return _to_uri(fig)


def decision_curve_plot(results):
    """Decision-curve analysis: net benefit vs threshold probability."""
    rows = results.get("calibration", {}).get("decision_curve")
    if not rows:
        return None
    thr = [r["threshold"] for r in rows]
    model = [r["net_benefit_model"] for r in rows]
    treat_all = [r["net_benefit_treat_all"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(thr, model, color=_BLUE, lw=2.2, label="AI model")
    ax.plot(thr, treat_all, color=_GREY, lw=1.3, label="treat all")
    ax.axhline(0, color=_RED, lw=1.3, label="treat none")
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision-curve analysis (clinical utility)")
    ymax = max(model) if model else 0.1
    ax.set_ylim(min(-0.02, -0.2 * abs(ymax)), ymax * 1.15 + 1e-3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _to_uri(fig)


def froc_plot(localization_results):
    """FROC curve with the reported operating points marked."""
    if not localization_results:
        return None
    froc = localization_results["froc_curve"]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot(froc["fp_per_image"], froc["sensitivity"], color=_BLUE, lw=2.2)
    for label, sens in localization_results["sensitivity_at_fp"].items():
        fp = float(label.split("_")[1])
        ax.plot(fp, sens, "o", color=_RED, ms=6)
        ax.annotate(f" {sens:.2f}", (fp, sens), fontsize=8)
    mean = localization_results["mean_sensitivity"]
    ci = localization_results.get("mean_sensitivity_ci", [None, None])
    ax.set_xlabel("mean false marks per image")
    ax.set_ylabel("lesion-localisation sensitivity")
    title = f"FROC - lesion localisation\nmean sensitivity={mean:.3f}"
    if ci[0] is not None:
        title += f" ({ci[0]:.3f}-{ci[1]:.3f})"
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    return _to_uri(fig)
