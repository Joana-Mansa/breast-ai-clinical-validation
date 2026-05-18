# Clinical Validation Methodology

This document explains *what* the pipeline measures and *why* — the reasoning a
clinical validation function is responsible for, independent of any particular
model or dataset. It is written against the use case of an AI device for breast
cancer detection in 2D digital mammography (DM) and 3D digital breast
tomosynthesis (DBT), such as a CADe/CADt decision-support product.

---

## 1. Validation is not model development

Model development asks *"can I make this network better?"*. Clinical validation
asks a different, narrower, higher-stakes question:

> *Given this device, exactly as it will ship — does the evidence support its
> intended use, and where does it fail?*

The discipline that follows from that question:

- **The model is a sealed black box.** The pipeline never trains, fine-tunes or
  threshold-tunes the model on the evaluation data. It runs inference and
  measures. In this codebase that stance is structural: a model enters only
  through the `ImageClassifier` / `VolumeClassifier` interface
  (`mammoval/models/base.py`), and the metrics never see anything but a
  predictions table.
- **The reference standard is fixed in advance.** Ground truth is biopsy /
  pathology where available, and the dataset's labelling rule otherwise.
- **Analyses are pre-specified.** Operating points, the non-inferiority margin,
  subgroups and the primary endpoint are chosen *before* seeing results, so the
  study cannot drift into post-hoc cherry-picking.
- **Every estimate carries an interval.** A point estimate without a confidence
  interval is not a validation result.

---

## 2. Regulatory frame

A breast AI device is *Software as a Medical Device* (SaMD). Two themes shape
the evidence it needs.

**Intended use determines the bar.** The same algorithm is a different device
depending on its claim:

| Claim | Regulatory character | What must be shown |
|---|---|---|
| Concurrent reader aid (CADe) | Marks shown alongside the radiologist | The radiologist performs **better with** the aid — needs a reader study |
| Triage / worklist (CADt) | Re-orders or flags exams | Time-to-decision and that cancers are not down-triaged |
| Standalone / autonomous reader | AI replaces a read | AI is **non-inferior** (or superior) to a radiologist |

**Standalone performance is necessary but not sufficient.** Retrospective
standalone metrics (this pipeline's Sections 3–8) characterise the device in
isolation. A *clinical benefit* claim additionally needs to show what happens
when a radiologist uses it — Section 9 below. Regulators (FDA SaMD guidance and
the predetermined change control plan / PCCP for model updates; EU MDR with
CE-marking under a notified body) expect both, plus evidence the device
generalises beyond its development data.

This pipeline produces the **standalone retrospective** layer of that evidence
and is explicit about what it does *not* establish.

---

## 3. Standalone discrimination — ROC / AUC

The first question is purely about ranking: does the AI score order cancers
above non-cancers? The **ROC AUC** answers it threshold-free and
prevalence-free.

- **Confidence interval — DeLong.** The AUC is an estimate; its uncertainty is
  reported with **DeLong's method** (`metrics/delong.py`), the standard
  closed-form, non-parametric variance estimator. It is the right tool because
  validation data is *paired*: every case has an AI score and a reference
  standard, so a single CI and any model-vs-model comparison must use the
  correlated-ROC variance.
- **Partial AUC.** Screening runs at high specificity — a radiologist recalls
  only a few percent of women. Discrimination in the low-specificity part of
  the ROC is clinically irrelevant and can inflate the global AUC. The pipeline
  also reports the **standardised partial AUC** over specificity 0.80–1.00.
- **Patient-level confidence interval.** A woman contributes up to four views
  (L/R × CC/MLO); views from one woman are correlated. Treating views as
  independent *understates* uncertainty. The pipeline adds a **cluster
  bootstrap** that resamples whole patients (`metrics/bootstrap.py`); its
  interval is wider, and that extra width is real.

A useful external benchmark: Rodríguez-Ruiz et al. (2019) found a commercial
breast AI reached a standalone AUC of **0.840** against a mean radiologist AUC
of **0.814** across nine datasets — i.e. a credible standalone device sits in
radiologist territory, and a validation should be powered to resolve
differences of that size.

---

## 4. Operating points

An ROC curve is a menu; a deployed device eats one item from it. The pipeline
reports performance at **fixed, clinically meaningful specificities** (default
0.90 and 0.96), because in screening the false-positive rate is the binding
constraint — every false positive is a recalled, anxious, healthy woman and a
downstream cost.

At each operating point it gives sensitivity (with a bootstrap CI), the
achieved specificity, the full confusion matrix and PPV. The Youden point is
reported too, but only as a reference: a real screening threshold is set to a
target recall rate, not to a statistical optimum.

**The threshold-transfer problem.** A threshold chosen on this cohort will not
deliver the same specificity on a population with a different prevalence or
case mix. Operating points are characterised here; a deployed threshold must be
fixed prospectively and then monitored.

---

## 5. Lesion localisation — FROC

Exam-level AUC has a blind spot: a model can call the exam correctly *for the
wrong reason* — high score, mark in the wrong place. For a CADe device that
draws regions, **where the mark lands is the product**.

**Free-response ROC (FROC)** scores each mark. A mark is a true positive only
if it satisfies a geometric hit criterion against a ground-truth lesion;
everything else is a false mark. FROC plots lesion-localisation sensitivity
against **mean false marks per image**. Unlike ROC its x-axis is unbounded, so
performance is read off at clinically tolerable mark rates.

The pipeline implements the **official Duke BCS-DBT criterion** exactly
(`metrics/localization.py: duke_dbt_hit`): a hit requires the predicted box
centre within `max(√(W²+H²)/2, 100)` pixels of the ground-truth centre **and**
within `VolumeSlices/4` slices of it. The reported mean sensitivity at 1/2/3/4
false marks per volume reproduces the official DBTex challenge ranking metric,
so numbers are comparable to published baselines.

---

## 6. Calibration and clinical utility

Discrimination and calibration are independent. A model can rank perfectly yet
be badly calibrated — and any decision that treats the score as a *probability*
(a rule-out threshold, a risk band, an exchange-rate calculation) is corrupted
by miscalibration.

- **Reliability diagram + Brier score + ECE** — does "score 0.1" mean roughly a
  10% cancer rate?
- **Calibration slope and intercept.** Slope < 1 means the model is
  over-confident. A non-zero intercept is *calibration-in-the-large* drift — the
  signature of deploying a model where prevalence differs from training, which
  is the normal situation when an enriched-data model meets a screening
  population.
- **Decision-curve analysis (net benefit).** Translates the score into the unit
  a clinician values — net true positives per patient after charging for false
  positives at the rate a decision threshold implies. The device is useful over
  the threshold range where its curve beats both *treat-all* and *treat-none*.

---

## 7. Screening behaviour and AI triage

Discrimination is abstract; a radiology department reports concrete numbers.
The pipeline computes, at the primary operating point, the metrics a screening
programme and a regulator actually quote:

- **Cancer detection rate (CDR)** — screen-detected cancers per 1,000 exams;
- **Recall / abnormal-interpretation rate** — fraction of women called back;
- **PPV of recall (PPV1)**, sensitivity, specificity.

It then simulates the two clinical uses a Transpara-style device is positioned
for:

- **Rule-out / triage.** Exams below a low score are deprioritised or
  auto-classified normal. The pipeline sweeps the rule-out threshold and reports
  the trade-off curve: workload reduction (efficiency) against sensitivity
  retained and **missed cancers** (safety). The defensible operating point is
  the largest workload cut that keeps sensitivity above a pre-agreed floor —
  the safety question, *how many cancers are in the ruled-out band*, is never
  hidden.
- **Risk banding.** The continuous score is mapped to ordinal bands (the spirit
  of a 1–10 exam score) and the observed cancer rate per band is checked for
  monotonic increase — evidence the score is a genuine risk stratifier.

**Cohort caveat.** On an enriched, non-consecutive dataset (e.g. CBIS-DDSM)
prevalence is a curation artefact, so CDR and recall rate are *illustrative of
the metric*, not programme estimates. The pipeline states this in the report
rather than letting the numbers be misread.

---

## 8. Subgroup analysis — effect modifiers

A pooled AUC can conceal a subgroup where the device is unsafe. The pre-eminent
example in mammography is **breast density**: dense tissue masks tumours and
depresses sensitivity for radiologists and AI alike, and dense-breast women are
exactly the group with the greatest unmet need. Density is a candidate
**effect modifier**, not a nuisance variable.

The pipeline estimates AUC within each subgroup with a DeLong CI and applies
**Cochran's Q** heterogeneity test across strata (`metrics/subgroups.py`). A
significant Q is evidence the variable modifies performance: the device should
then be reported — and possibly operating-point-tuned — per stratum, not
pooled. Standard strata: breast density, lesion type (mass vs calcification),
view, and where available age, scanner vendor and site.

This is also where **generalisation** is probed. The confidence intervals in
Sections 3–7 quantify *sampling error only*. They say nothing about
distribution shift across vendor, site, ethnicity or acquisition era. That
requires an *external* validation cohort — the single most important extension
of this pipeline, and a known strength claim for multi-vendor-validated
devices.

---

## 9. From standalone to clinical benefit — the reader study

Standalone metrics do not tell you whether *patients* are better off, because
the device is used by a radiologist, not instead of one. The definitive design
is the **Multi-Reader Multi-Case (MRMC) study**: many radiologists read the
same cases with and without the AI, and the change in their performance —
analysed with the Dorfman-Berbaum-Metz / Obuchowski-Rockette framework that
accounts for reader *and* case variability — is the endpoint.

This pipeline does **not** run an MRMC study. What it does provide is the
honest adjacent analysis: an **AI-vs-reference-reader comparison**. Using a
proxy reader signal — BI-RADS *assessment* in CBIS-DDSM — it runs:

- a **paired DeLong test** of AI AUC vs reader AUC (paired because both scored
  the same cases), and
- a **non-inferiority test**: AI is declared non-inferior if the lower
  confidence bound on `AUC_AI − AUC_reader` exceeds a pre-specified `−margin`.

Non-inferiority, not superiority, is usually the right standalone question — the
claim is rarely "AI beats the radiologist" but "AI is *not meaningfully worse*",
which is what supports a workload-reduction or autonomous-read use case. The
margin is a clinical judgement and must be fixed before analysis.

**Caveat, stated plainly.** BI-RADS assessment is an ordinal proxy, not an
independent prospective radiologist read, and category 0 ("incomplete") is not
strictly ordered between 1 and 2. The comparison demonstrates the *method*; a
real claim needs real reader data.

---

## 10. Statistical principles

- **Pairing.** AI, reference standard and (where present) reader are measured
  on the same cases. Comparisons use paired methods — DeLong's correlated-ROC
  variance, or a paired bootstrap that resamples shared indices.
- **Confidence intervals.** DeLong for AUC; **Wilson** score intervals for
  proportions (sensitivity, specificity, PPV — well behaved for the small
  counts of a thin subgroup); **percentile bootstrap** for everything else.
- **Clustering.** The unit of analysis is the image/view; the unit of
  independence is the woman. Patient-level cluster bootstrap respects that.
- **Non-inferiority.** A one-sided test against a pre-specified margin — the
  correct frame for a "not worse than the radiologist" claim.
- **Multiplicity.** Subgroups and multiple operating points inflate the
  family-wise error rate. The pipeline reports them as *exploratory*; a
  confirmatory study pre-specifies a single primary endpoint and adjusts.
- **Power.** A validation must be large enough to resolve a clinically
  meaningful AUC difference (≈0.02–0.05); cancers, not exams, are the scarce
  resource that drives sample size.

---

## 11. Limitations of this pipeline

- It is a **retrospective standalone** pipeline. It does not run a prospective
  trial or an MRMC reader study.
- Public datasets diverge from a live screening population: CBIS-DDSM is
  **digitised film**, lesion-enriched and non-consecutive; the Duke DBT subset
  used here is small. Absolute numbers are not programme estimates.
- The default models are **baselines** chosen for runnable reproducibility, not
  state-of-the-art devices; the pipeline quantifies their shortfall rather than
  hiding it.
- Confidence intervals cover **sampling error only** — not distribution shift,
  not label noise, not verification bias in the reference standard.
- It is an **educational / methodological** tool. It is not a regulatory
  submission and not a cleared medical device.

These limitations are surfaced in every generated report, by design — a
validation tool that oversells is worse than useless.

---

## References

- DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Comparing areas under
  correlated ROC curves. *Biometrics*.
- Sun X, Xu W (2014). Fast implementation of DeLong's algorithm. *IEEE SPL*.
- Rodríguez-Ruiz A et al. (2019). Stand-alone AI for breast cancer detection
  vs 101 radiologists. *JNCI*.
- Bunch PC et al.; Chakraborty DP. Free-response ROC (FROC) methodology.
- Buda M et al. (2021). A data set and deep-learning algorithm for mass
  detection in DBT (Duke BCS-DBT). *JAMA Network Open*.
- Vickers AJ, Elkin EB (2006). Decision-curve analysis. *Med Decis Making*.
- Gallas BD et al. MRMC ROC analysis for imaging-device assessment.
- FDA — *Clinical Performance Assessment for CADe Devices*; SaMD guidance and
  the Predetermined Change Control Plan (PCCP).
