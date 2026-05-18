"""Render a validation result into a self-contained HTML report.

One ``build_report`` call turns the nested dicts from :mod:`mammoval.pipeline`
into a single styled HTML file with embedded plots — the kind of artefact that
accompanies a regulatory submission or a model-release sign-off. No external
assets, so it can be emailed or version-controlled as-is.
"""
from __future__ import annotations

import datetime as _dt

from jinja2 import Environment

from . import plotting

__all__ = ["build_report"]


# --------------------------------------------------------------------------
# formatting helpers (registered as Jinja filters)
# --------------------------------------------------------------------------
def _num(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "&ndash;"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x, nd=1):
    try:
        return f"{100 * float(x):.{nd}f}%"
    except (TypeError, ValueError):
        return "&ndash;"


def _ci(pair, nd=3, scale=1.0):
    try:
        lo, hi = pair
        return f"{scale * lo:.{nd}f} &ndash; {scale * hi:.{nd}f}"
    except (TypeError, ValueError):
        return "&ndash;"


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   color:#1d2630;max-width:1080px;margin:0 auto;padding:32px 28px;line-height:1.5;}
 h1{font-size:25px;border-bottom:3px solid #1f5fa8;padding-bottom:8px;margin-bottom:4px;}
 h2{font-size:19px;color:#1f5fa8;margin-top:34px;border-bottom:1px solid #d8dee6;padding-bottom:4px;}
 h3{font-size:15px;margin-top:20px;color:#2c3e50;}
 .sub{color:#6b7785;font-size:13px;margin-top:0;}
 table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;}
 th,td{border:1px solid #d8dee6;padding:6px 9px;text-align:left;}
 th{background:#eef2f7;}
 td.n{text-align:right;font-variant-numeric:tabular-nums;}
 .kpi{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0;}
 .card{flex:1 1 168px;background:#f6f8fb;border:1px solid #d8dee6;border-radius:8px;padding:12px 14px;}
 .card .v{font-size:23px;font-weight:600;color:#1f5fa8;}
 .card .l{font-size:11px;color:#6b7785;text-transform:uppercase;letter-spacing:.4px;}
 .card .s{font-size:11px;color:#6b7785;}
 img{max-width:100%;border:1px solid #e3e8ee;border-radius:6px;margin:8px 0;}
 .row{display:flex;gap:18px;flex-wrap:wrap;}
 .row>div{flex:1 1 340px;}
 .verdict{padding:10px 14px;border-radius:6px;font-size:13px;margin:10px 0;}
 .ok{background:#e7f5ec;border-left:4px solid #2e8b57;}
 .warn{background:#fdf0e6;border-left:4px solid #e08e3c;}
 .note{background:#eef2f7;border-left:4px solid #1f5fa8;}
 .err{background:#fdecec;border-left:4px solid #c0392b;}
 footer{margin-top:40px;border-top:1px solid #d8dee6;padding-top:10px;
   color:#6b7785;font-size:11px;}
 code{background:#eef2f7;padding:1px 4px;border-radius:3px;font-size:12px;}
</style></head><body>

<h1>{{ title }}</h1>
<p class="sub">Generated {{ generated }} &middot; mammoval clinical validation pipeline</p>

<div class="note verdict">
 <b>Scope.</b> Standalone (black-box) validation of a breast-cancer AI model
 against a biopsy-proven reference standard. The model was measured, not
 modified or retrained. Read the <b>Limitations</b> section before quoting any
 number outside this cohort.
</div>

{# ---------- executive summary ---------- #}
<h2>1 &nbsp; Executive summary</h2>
<div class="kpi">
 <div class="card"><div class="l">ROC AUC</div>
  <div class="v">{{ d.auc|num }}</div>
  <div class="s">95% CI {{ d.auc_ci|ci }} (DeLong)</div></div>
 <div class="card"><div class="l">Cases</div>
  <div class="v">{{ meta.n_cases }}</div>
  <div class="s">{{ meta.n_malignant }} malignant &middot; prev {{ meta.prevalence|pct }}</div></div>
{% if primary_op %}
 <div class="card"><div class="l">Sensitivity @ spec {{ primary_op.target_specificity|num(2) }}</div>
  <div class="v">{{ primary_op.confusion.sensitivity|pct }}</div>
  <div class="s">spec achieved {{ primary_op.specificity_achieved|pct }}</div></div>
{% endif %}
{% if froc %}
 <div class="card"><div class="l">FROC mean sensitivity</div>
  <div class="v">{{ froc.mean_sensitivity|num }}</div>
  <div class="s">95% CI {{ froc.mean_sensitivity_ci|ci }}</div></div>
{% endif %}
</div>
{% if reader and reader.interpretation %}
<div class="verdict {{ 'ok' if reader.noninferiority.non_inferior else 'warn' }}">
 <b>AI vs reference reader.</b> {{ reader.interpretation }}</div>
{% endif %}

{# ---------- cohort ---------- #}
<h2>2 &nbsp; Validation cohort</h2>
<table>
 <tr><th>Dataset</th><td>{{ meta.dataset }}</td>
     <th>Cases analysed</th><td class="n">{{ meta.n_cases }}</td></tr>
 <tr><th>Patients</th><td class="n">{{ meta.n_patients }}</td>
     <th>Malignant / non-malignant</th>
     <td class="n">{{ meta.n_malignant }} / {{ meta.n_non_malignant }}</td></tr>
 <tr><th>Prevalence</th><td class="n">{{ meta.prevalence|pct }}</td>
     <th>Dropped (missing score/label)</th>
     <td class="n">{{ meta.n_dropped_missing }}</td></tr>
</table>

{# ---------- discrimination ---------- #}
<h2>3 &nbsp; Discrimination</h2>
<p>How well the continuous AI score separates cancer from non-cancer, before
 any threshold is chosen. The partial AUC isolates the high-specificity region
 a screening device actually operates in.</p>
<div class="row">
 <div><img src="{{ plots.roc }}" alt="ROC curve"></div>
 <div>
 <table>
  <tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>
  <tr><td>ROC AUC (DeLong)</td><td class="n">{{ d.auc|num }}</td>
      <td class="n">{{ d.auc_ci|ci }}</td></tr>
{% if d.auc_patient_cluster_ci and not d.auc_patient_cluster_ci.error %}
  <tr><td>AUC, patient-level cluster CI</td>
      <td class="n">{{ d.auc_patient_cluster_ci.estimate|num }}</td>
      <td class="n">{{ [d.auc_patient_cluster_ci.ci_low, d.auc_patient_cluster_ci.ci_high]|ci }}</td></tr>
{% endif %}
  <tr><td>Standardised partial AUC (spec 0.80&ndash;1.00)</td>
      <td class="n">{{ d.partial_auc_high_spec.pauc_standardized|num }}</td>
      <td class="n">&ndash;</td></tr>
  <tr><td>Average precision (PR-AUC)</td>
      <td class="n">{{ d.average_precision|num }}</td><td class="n">&ndash;</td></tr>
 </table>
 <p class="sub">The patient-level CI resamples whole patients, not images, so
  correlated multi-view exams do not understate uncertainty.</p>
 </div>
</div>

{# ---------- operating points ---------- #}
<h2>4 &nbsp; Operating points</h2>
<p>A screening device runs at one threshold. Each row fixes a clinically
 relevant specificity and reports the sensitivity achieved there.</p>
<table>
 <tr><th>Target spec.</th><th>Threshold</th><th>Sensitivity</th>
     <th>Sensitivity 95% CI</th><th>Spec. achieved</th>
     <th>TP / FP / TN / FN</th><th>PPV</th></tr>
{% for op in ops.by_target_specificity %}
 <tr><td class="n">{{ op.target_specificity|num(2) }}</td>
     <td class="n">{{ op.threshold|num }}</td>
     <td class="n">{{ op.confusion.sensitivity|pct }}</td>
     <td class="n">{% if op.sensitivity_bootstrap_ci and not op.sensitivity_bootstrap_ci.error %}
       {{ [op.sensitivity_bootstrap_ci.ci_low, op.sensitivity_bootstrap_ci.ci_high]|ci }}
       {% else %}&ndash;{% endif %}</td>
     <td class="n">{{ op.specificity_achieved|pct }}</td>
     <td class="n">{{ op.confusion.tp }} / {{ op.confusion.fp }} /
        {{ op.confusion.tn }} / {{ op.confusion.fn }}</td>
     <td class="n">{{ op.confusion.ppv|pct }}</td></tr>
{% endfor %}
 <tr><td>Youden-optimal</td><td class="n">{{ ops.youden.threshold|num }}</td>
     <td class="n">{{ ops.youden.sensitivity|pct }}</td><td class="n">&ndash;</td>
     <td class="n">{{ ops.youden.specificity|pct }}</td>
     <td class="n">&ndash;</td><td class="n">&ndash;</td></tr>
</table>

{# ---------- calibration ---------- #}
<h2>5 &nbsp; Calibration &amp; clinical utility</h2>
{% if cal and not cal.skipped and not cal.error %}
<div class="row">
 <div><img src="{{ plots.calibration }}" alt="calibration"></div>
 <div><img src="{{ plots.decision }}" alt="decision curve"></div>
</div>
<table>
 <tr><th>Brier score</th><td class="n">{{ cal.brier_score|num }}</td>
     <th>Expected calibration error</th>
     <td class="n">{{ cal.expected_calibration_error|num }}</td></tr>
 <tr><th>Calibration slope</th>
     <td class="n">{{ cal.intercept_slope.slope|num(2) }}</td>
     <th>Calibration intercept</th>
     <td class="n">{{ cal.intercept_slope.intercept|num(2) }}</td></tr>
</table>
<p class="sub">Slope 1.0 / intercept 0.0 is ideal. Slope &lt; 1 = over-confident
 scores; a non-zero intercept = systematic over/under-estimation of risk, the
 signature of a prevalence shift between training and deployment populations.</p>
{% else %}
<div class="note verdict">Calibration skipped &mdash;
 {{ cal.skipped or cal.error or 'scores are not probabilities' }}.</div>
{% endif %}

{# ---------- screening behaviour ---------- #}
<h2>6 &nbsp; Screening behaviour &amp; AI triage</h2>
{% if scr and not scr.error %}
<h3>6.1 &nbsp; Programme metrics at the primary operating point</h3>
<table>
 <tr><th>Recall rate</th><td class="n">{{ scr.at_primary_operating_point.recall_rate|pct }}</td>
     <th>Cancer detection rate</th>
     <td class="n">{{ scr.at_primary_operating_point.cancer_detection_rate_per_1000|num(2) }} /1000</td></tr>
 <tr><th>Sensitivity</th>
     <td class="n">{{ scr.at_primary_operating_point.sensitivity|pct }}</td>
     <th>Specificity</th>
     <td class="n">{{ scr.at_primary_operating_point.specificity|pct }}</td></tr>
 <tr><th>PPV of recall (PPV1)</th>
     <td class="n">{{ scr.at_primary_operating_point.ppv1_recall|pct }}</td>
     <th>Exams recalled</th>
     <td class="n">{{ scr.at_primary_operating_point.n_recalled }}</td></tr>
</table>
<div class="row">
 <div><img src="{{ plots.triage }}" alt="triage trade-off"></div>
 <div><img src="{{ plots.risk_bands }}" alt="risk bands"></div>
</div>
<p class="sub">{{ scr.cohort_caveat }}</p>
{% else %}
<div class="err verdict">Screening section failed: {{ scr.error }}</div>
{% endif %}

{# ---------- subgroups ---------- #}
<h2>7 &nbsp; Subgroup analysis (effect modifiers)</h2>
<p>Pooled AUC can mask a subgroup where the device underperforms &mdash; most
 critically <b>dense breasts</b>. Cochran's Q tests whether the subgroups
 genuinely differ.</p>
{% if plots.subgroups %}<img src="{{ plots.subgroups }}" alt="subgroup forest plot">{% endif %}
{% for col, block in subgroups.items() %}
 {% if block and not block.error and block.groups %}
 <h3>By {{ col }}</h3>
 <table>
  <tr><th>Subgroup</th><th>n</th><th>malignant</th><th>AUC</th><th>95% CI</th></tr>
 {% for g in block.groups %}
  <tr><td>{{ g.group }}</td><td class="n">{{ g.n }}</td>
      <td class="n">{{ g.n_pos }}</td><td class="n">{{ g.auc|num }}</td>
      <td class="n">{{ [g.ci_low, g.ci_high]|ci }}</td></tr>
 {% endfor %}
 </table>
 {% set het = block.heterogeneity %}
 {% if het and het.applicable %}
 <div class="verdict {{ 'warn' if het.effect_modifier else 'ok' }}">
  Heterogeneity: Q={{ het.Q|num(2) }}, df={{ het.df }},
  p={{ het.p_value|num(3) }}, I&sup2;={{ het.I_squared|pct(0) }} &mdash;
  {{ '"' ~ col ~ '" IS an effect modifier (performance differs across strata).'
     if het.effect_modifier
     else 'no significant performance difference across strata.' }}
 </div>
 {% endif %}
 {% endif %}
{% endfor %}
{% if not subgroups %}<p class="sub">No subgroup columns supplied.</p>{% endif %}

{# ---------- reader comparison ---------- #}
<h2>8 &nbsp; AI vs reference reader</h2>
{% if reader and not reader.skipped and not reader.error %}
<table>
 <tr><th>Reader reference</th><td><code>{{ reader.reader_col }}</code></td>
     <th>Cases with reader score</th>
     <td class="n">{{ reader.n_cases_with_reader }}</td></tr>
 <tr><th>AI AUC</th><td class="n">{{ reader.auc_ai|num }}</td>
     <th>Reader AUC</th><td class="n">{{ reader.auc_reader|num }}</td></tr>
 <tr><th>AUC difference (AI &minus; reader)</th>
     <td class="n">{{ reader.delong_test.auc_diff|num }}</td>
     <th>DeLong p-value</th>
     <td class="n">{{ reader.delong_test.p_value|num(4) }}</td></tr>
 <tr><th>Non-inferiority margin</th>
     <td class="n">{{ reader.noninferiority.margin|num(2) }}</td>
     <th>Lower bound of difference</th>
     <td class="n">{{ reader.noninferiority.lower_bound|num }}</td></tr>
</table>
<div class="verdict {{ 'ok' if reader.noninferiority.non_inferior else 'warn' }}">
 {{ reader.interpretation }}</div>
<p class="sub">Paired (correlated-ROC) design: AI and reader scored the same
 cases, so the DeLong covariance term is included. The reader reference is a
 proxy and its caveats are listed under Limitations.</p>
{% else %}
<div class="note verdict">Not run &mdash;
 {{ reader.skipped or reader.error or 'no reader column' }}.</div>
{% endif %}

{# ---------- localisation ---------- #}
{% if froc %}
<h2>9 &nbsp; Lesion localisation (FROC)</h2>
<p>Exam-level AUC ignores <i>where</i> a mark lands. FROC scores each mark: a
 mark is a true positive only if it hits the lesion under the geometric
 criterion. With operating points 1/2/3/4 FP per image the mean reproduces the
 official Duke BCS-DBT ranking metric.</p>
<div class="row">
 <div><img src="{{ plots.froc }}" alt="FROC curve"></div>
 <div>
 <table>
  <tr><th>False marks / image</th><th>Localisation sensitivity</th></tr>
 {% for label, sens in froc.sensitivity_at_fp.items() %}
  <tr><td class="n">{{ label.split('_')[1] }}</td>
      <td class="n">{{ sens|pct }}</td></tr>
 {% endfor %}
  <tr><th>Mean sensitivity</th>
      <th class="n">{{ froc.mean_sensitivity|num }}</th></tr>
  <tr><td>95% CI (case bootstrap)</td>
      <td class="n">{{ froc.mean_sensitivity_ci|ci }}</td></tr>
  <tr><td>Cases / lesions</td>
      <td class="n">{{ froc.meta.n_cases }} / {{ froc.meta.n_lesions }}</td></tr>
 </table>
 </div>
</div>
{% endif %}

{# ---------- limitations ---------- #}
<h2>{{ '10' if froc else '9' }} &nbsp; Limitations &amp; interpretation notes</h2>
<ul>
{% for item in limitations %}<li>{{ item }}</li>{% endfor %}
</ul>

<footer>
 mammoval v{{ version }} &middot; Confidence intervals: DeLong for AUC,
 Wilson for proportions, percentile bootstrap elsewhere &middot;
 This is a methodological / educational validation pipeline, not a regulatory
 submission and not a cleared medical device.
</footer>
</body></html>
"""


# Compact template for a localisation-only (FROC) report — used when the study
# has no exam-level classification arm (e.g. the 3D FROC pipeline run alone).
_LOCALIZATION_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   color:#1d2630;max-width:1080px;margin:0 auto;padding:32px 28px;line-height:1.5;}
 h1{font-size:25px;border-bottom:3px solid #1f5fa8;padding-bottom:8px;margin-bottom:4px;}
 h2{font-size:19px;color:#1f5fa8;margin-top:34px;border-bottom:1px solid #d8dee6;padding-bottom:4px;}
 .sub{color:#6b7785;font-size:13px;margin-top:0;}
 table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;}
 th,td{border:1px solid #d8dee6;padding:6px 9px;text-align:left;}
 th{background:#eef2f7;}
 td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;}
 .kpi{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0;}
 .card{flex:1 1 200px;background:#f6f8fb;border:1px solid #d8dee6;border-radius:8px;padding:12px 14px;}
 .card .v{font-size:23px;font-weight:600;color:#1f5fa8;}
 .card .l{font-size:11px;color:#6b7785;text-transform:uppercase;letter-spacing:.4px;}
 .card .s{font-size:11px;color:#6b7785;}
 img{max-width:100%;border:1px solid #e3e8ee;border-radius:6px;margin:8px 0;}
 .row{display:flex;gap:18px;flex-wrap:wrap;}
 .row>div{flex:1 1 340px;}
 .note{padding:10px 14px;border-radius:6px;font-size:13px;margin:10px 0;
   background:#eef2f7;border-left:4px solid #1f5fa8;}
 footer{margin-top:40px;border-top:1px solid #d8dee6;padding-top:10px;
   color:#6b7785;font-size:11px;}
 code{background:#eef2f7;padding:1px 4px;border-radius:3px;font-size:12px;}
</style></head><body>

<h1>{{ title }}</h1>
<p class="sub">Generated {{ generated }} &middot; mammoval clinical validation pipeline</p>

<div class="note">
 <b>Scope.</b> Standalone <b>lesion-localisation</b> validation of a breast-cancer
 detection AI. FROC scores every region <i>mark</i> the device emits — a mark
 counts only when it lands on a real lesion under the geometric hit criterion —
 so it measures detection quality that an exam-level AUC cannot see.
</div>

<h2>1 &nbsp; Localisation cohort</h2>
<table>
 <tr><th>Dataset</th><td>{{ froc.meta.dataset }}</td>
     <th>Volumes / images</th><td class="n">{{ froc.meta.n_cases }}</td></tr>
 <tr><th>Ground-truth lesions</th><td class="n">{{ froc.meta.n_lesions }}</td>
     <th>Operating points</th>
     <td class="n">{{ froc.meta.fp_points|join(', ') }} FP/volume</td></tr>
</table>

<h2>2 &nbsp; FROC &mdash; free-response ROC</h2>
<p>FROC plots lesion-localisation sensitivity against the mean number of false
 marks per volume. The mean sensitivity at 1/2/3/4 false marks per volume
 reproduces the official Duke BCS-DBT / DBTex challenge ranking metric.</p>
<div class="kpi">
 <div class="card"><div class="l">Mean FROC sensitivity</div>
  <div class="v">{{ froc.mean_sensitivity|num }}</div>
  <div class="s">95% CI {{ froc.mean_sensitivity_ci|ci }} (case bootstrap)</div></div>
 <div class="card"><div class="l">Ground-truth lesions</div>
  <div class="v">{{ froc.meta.n_lesions }}</div>
  <div class="s">over {{ froc.meta.n_cases }} volumes</div></div>
</div>
<div class="row">
 <div><img src="{{ plots.froc }}" alt="FROC curve"></div>
 <div>
 <table>
  <tr><th>False marks / volume</th><th>Localisation sensitivity</th></tr>
 {% for label, sens in froc.sensitivity_at_fp.items() %}
  <tr><td class="n">{{ label.split('_')[1] }}</td>
      <td class="n">{{ sens|pct }}</td></tr>
 {% endfor %}
  <tr><th>Mean</th><th class="n">{{ froc.mean_sensitivity|pct }}</th></tr>
 </table>
 </div>
</div>

<h2>3 &nbsp; Method</h2>
<p>True-positive criterion (the official Duke BCS-DBT rule): a predicted box
 centre is a hit when it lies within
 <code>max(&radic;(W&sup2;+H&sup2;)/2, 100)</code> pixels of a ground-truth box
 centre <b>and</b> within <code>VolumeSlices/4</code> slices of it. Within each
 volume, detections are matched to lesions greedily in descending score order;
 the confidence interval on mean sensitivity is a case-level bootstrap.</p>

<h2>4 &nbsp; Limitations &amp; interpretation notes</h2>
<ul>{% for item in limitations %}<li>{{ item }}</li>{% endfor %}</ul>

<footer>
 mammoval v{{ version }} &middot; FROC per the Duke BCS-DBT criterion &middot;
 This is a methodological / educational validation pipeline, not a regulatory
 submission and not a cleared medical device.
</footer>
</body></html>
"""


def build_report(classification_results=None, localization_results=None,
                 output_path="validation_report.html",
                 title="Mammography AI - Clinical Validation Report",
                 extra_limitations=None):
    """Render results to a standalone HTML file.

    Supply ``classification_results`` for an exam-level report (optionally with
    ``localization_results`` adding a FROC section), or ``localization_results``
    alone for a standalone FROC localisation report.

    Parameters
    ----------
    classification_results : dict, optional
        Output of :func:`mammoval.pipeline.run_classification_validation`.
    localization_results : dict, optional
        Output of :func:`mammoval.pipeline.run_localization_validation`.
    output_path : str
        Where to write the HTML. Returns the same path.
    extra_limitations : list of str, optional
        Dataset- or study-specific caveats appended to the standard list.

    Returns
    -------
    str : ``output_path``.
    """
    from . import __version__

    env = Environment(autoescape=False)
    env.filters.update(num=_num, pct=_pct, ci=_ci)
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- localisation-only report (no classification arm) -----------------
    if classification_results is None:
        if localization_results is None:
            raise ValueError("build_report needs classification_results, "
                             "localization_results, or both")
        loc_limits = [
            "Localisation is scored against ground-truth boxes for biopsied "
            "lesions only; other findings are not part of the lesion set.",
            "The true-positive criterion is geometric (a mark within tolerance "
            "of a lesion centre), not a radiologist's judgement of relevance.",
            "Confidence intervals are case-level bootstrap and cover sampling "
            "error only - not distribution shift across vendor, site or era.",
            "Standalone retrospective localisation: it does not establish how "
            "radiologists perform WITH the device (that needs an MRMC study).",
        ]
        if extra_limitations:
            loc_limits.extend(extra_limitations)
        html = env.from_string(_LOCALIZATION_TEMPLATE).render(
            title=title, generated=generated, version=__version__,
            froc=localization_results,
            plots={"froc": plotting.froc_plot(localization_results)},
            limitations=loc_limits,
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return output_path

    r = classification_results
    ops = r.get("operating_points", {})
    by_spec = ops.get("by_target_specificity", [])

    plots = {
        "roc": plotting.roc_plot(r),
        "calibration": plotting.calibration_plot(r),
        "decision": plotting.decision_curve_plot(r),
        "triage": plotting.triage_plot(r),
        "risk_bands": plotting.risk_band_plot(r),
        "subgroups": plotting.subgroup_forest_plot(r),
        "froc": plotting.froc_plot(localization_results),
    }

    limitations = [
        "Standalone retrospective validation: it measures the AI device in "
        "isolation. It does not establish how radiologists perform WITH the AI "
        "- that requires a prospective multi-reader multi-case (MRMC) study.",
        "The reference standard is biopsy/pathology where available; cases "
        "without tissue confirmation inherit the dataset's labelling rule.",
        "Confidence intervals quantify sampling error only. They do not cover "
        "distribution shift across scanner vendor, site, ethnicity or "
        "acquisition era - assess generalisation on an external cohort.",
        "Operating points are derived on this cohort; a deployed threshold "
        "must be fixed prospectively and monitored.",
    ]
    if extra_limitations:
        limitations.extend(extra_limitations)

    html = env.from_string(_TEMPLATE).render(
        title=title,
        generated=generated,
        version=__version__,
        meta=r["meta"],
        d=r["discrimination"],
        ops=ops,
        primary_op=by_spec[0] if by_spec else None,
        cal=r.get("calibration", {}),
        scr=r.get("screening", {}),
        subgroups=r.get("subgroups", {}),
        reader=r.get("reader_comparison", {}),
        froc=localization_results,
        plots=plots,
        limitations=limitations,
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path
