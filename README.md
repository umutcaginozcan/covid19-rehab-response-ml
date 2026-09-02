# Post-COVID Rehabilitation Response Prediction using ML algorithms

Machine-learning pipelines for predicting response to an 8-week post-COVID
pulmonary rehabilitation program (IMT / HIIT / control), for two endpoints:
6-minute walk test response (Δ ≥ 25 m) and VO₂peak response (Δ ≥ 1.2 mL·kg⁻¹·min⁻¹).
This is a classification task; 1: responder, 0: non-responder.

## Try the calculators

- [6-MWT calculator](https://colab.research.google.com/drive/1vi-hdYhZyMiF3VQXAQrh6BCCzg3aMe9L)
- [VO₂peak calculator](https://colab.research.google.com/drive/1PcojcJT0PXdfimnPXBesGnAWxwmL9E-D)

Both notebooks use model coefficients only; no patient data is included.

## The scores

| Endpoint | Winning model | F1 | AUC |
|---|---|---|---|
| 6-MWT (Δ ≥ 25 m) | Logistic regression (compact-8) | 0.80 ± 0.09 | 0.87 ± 0.08 |
| VO₂peak (Δ ≥ 1.2 mL·kg⁻¹·min⁻¹) | Elastic-net logistic (compact-5) | 0.86 ± 0.09 | 0.92 ± 0.07 |

Mean ± SD over person-grouped, stratified, repeated 5-fold cross-validation
(13 algorithms from 8 families were compared; F1 was the primary metric).

## Repository layout

- `ml_6mwt/code/` and `ml_vo2peak/code/` — analysis pipelines (Python 3,
  scikit-learn, XGBoost, LightGBM, SHAP). Person-grouped, stratified, repeated
  5-fold cross-validation; imputation, standardization and feature selection
  are fitted inside each training fold.
- Run order: `benchmark.py` → `validate.py` / `blocks.py` → `figures.py` → `tables.js`.

Scripts read the study dataset from the `COVID_SAV` environment variable;
the dataset is not distributed.
