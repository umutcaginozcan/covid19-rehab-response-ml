"""All journal figures (500 dpi) and the OOF metric summary.

Run after benchmark.py, validate.py and treatment.py:
  Figure6  calibration + decision-curve analysis (from validation results)
  Figure7  counterfactual treatment selection (from what-if results)
  roc_repeats / calibration_curve / confusion_matrix / shap_importance
  pdp_arms / ice_fss  (per-arm partial dependence, individual curves)
"""
import warnings
warnings.filterwarnings('ignore')
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score, precision_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix)
from core import load_data, grouped_folds, winner, WINNERS, METRICS, FIGURES, SEED

DPI = 500
BAR_GREEN = '#66C2A5'
C_IMT, C_HIIT, C_CTRL = '#27A883', '#F97B4C', '#7C8CE4'
ARMS = [('IMT', (1, 0), C_IMT), ('HIIT', (0, 1), C_HIIT), ('Control', (0, 0), C_CTRL)]
CLF_LABEL = WINNERS['clf'][0]
DISPLAY = {'grp_IMT': 'Training arm: IMT', 'grp_HIIT': 'Training arm: HIIT',
           'SixMWT_meter_BT': 'Baseline 6-MWT distance', 'MIP_BT': 'Baseline MIP',
           'VO2KG_max_BT': 'Baseline VO₂peak', 'FSS_BT': 'Baseline FSS',
           'SGQR_total_score_BT': 'Baseline SGRQ total', 'Age': 'Age', 'BKI': 'BMI'}
PDP_PANELS = [('FSS_BT', 'Baseline FSS (score)'), ('SixMWT_meter_BT', 'Baseline 6-MWT distance (m)'),
              ('BKI', 'BMI (kg/m²)'), ('MIP_BT', 'Baseline MIP (cmH₂O)'),
              ('VO2KG_max_BT', 'Baseline VO₂peak (mL·kg⁻¹·min⁻¹)'), ('Age', 'Age (years)')]

plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                     'font.size': 11, 'axes.linewidth': 1.0})


def _save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'written: {name}')


def _oof_repeat(factory, X, y, groups, rep):
    p = np.full(len(y), np.nan)
    for tr, te in grouped_folds(y, groups, 5, 1, 'clf', seed=SEED + rep):
        m = factory()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def figure6_validation():
    V = json.load(open(os.path.join(METRICS, 'validation_results.json')))
    c = V['clf']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=DPI)
    ax1.plot([0, 1], [0, 1], ls='--', c='#999999', lw=1)
    ax1.plot(c['cal']['mean_pred'], c['cal']['frac_pos'], 'o-', c='#222222', lw=1.4, ms=6)
    ax1.set_xlabel('Predicted probability'); ax1.set_ylabel('Observed responder fraction')
    ax1.set_title(f"Calibration (Brier = {c['brier']:.2f})", fontweight='bold', fontsize=13)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    d = c['dca']
    ax2.plot(d['thr'], d['model'], c='#222222', lw=1.6, label=f'Model ({CLF_LABEL})')
    ax2.plot(d['thr'], d['all'], c='#888888', ls='--', lw=1.4, label='Treat all')
    ax2.axhline(0, c='#bbbbbb', ls=':', lw=1.2, label='Treat none')
    ax2.set_xlabel('Threshold probability'); ax2.set_ylabel('Net benefit')
    ax2.set_title('Decision curve analysis', fontweight='bold', fontsize=13)
    ax2.set_ylim(-0.05, 0.65)
    ax2.legend(frameon=False, fontsize=10)
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f"Responder model validation — OOF AUC {c['auc_oof']:.2f} "
                 f"(95% CI {c['ci'][0]:.2f}–{c['ci'][1]:.2f}), permutation p = {c['p_perm']:.3f}",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'Figure6_model_validation.png')


def figure7_treatment():
    w = pd.read_csv(os.path.join(METRICS, 'whatif_recommendations.csv'))
    w = w.sort_values('p_IMT').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=DPI)
    x = np.arange(len(w))
    ax.plot(x, w.p_IMT, 'o', c='#222222', ms=4.5, label='P(responder | IMT)')
    ax.plot(x, w.p_HIIT, 's', c='#888888', ms=4, label='P(responder | HIIT)')
    ax.plot(x, w.p_Control, '^', c='#c4c4c4', ms=4, label='P(responder | Control)')
    ax.axhline(0.5, c='#bbbbbb', ls=':', lw=1)
    ax.set_xlabel('Patients (sorted by predicted probability under IMT)')
    ax.set_ylabel('Predicted probability of response')
    ax.set_title('Counterfactual treatment-selection: predicted 6-MWT response under each program',
                 fontweight='bold', fontsize=12.5)
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    _save(fig, 'Figure7_treatment_selection.png')


def interpretation_set():
    """ROC (per repeat), calibration, confusion matrix, SHAP + metric summary."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    factory, feats, mname, sname = winner('clf')
    df = load_data()
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    X, y, per = d[feats].values, d.responder.values, d.person.values
    reps = [_oof_repeat(factory, X, y, per, r) for r in range(5)]
    prob = np.mean(reps, axis=0)

    yh = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, yh)
    metrics = dict(
        model=f'{mname} ({sname})', n=int(len(y)),
        AUC=roc_auc_score(y, prob), BRIER=brier_score_loss(y, prob),
        ACC=accuracy_score(y, yh), BAL=balanced_accuracy_score(y, yh),
        F1=f1_score(y, yh), SENS=recall_score(y, yh),
        SPEC=recall_score(y, yh, pos_label=0), PPV=precision_score(y, yh),
        NPV=precision_score(y, yh, pos_label=0), MCC=matthews_corrcoef(y, yh),
        confusion=dict(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1])))
    os.makedirs(METRICS, exist_ok=True)
    json.dump(metrics, open(os.path.join(METRICS, 'interpret_metrics.json'), 'w'), indent=1)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    # ROC per repeat with mean ± SD band
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=DPI)
    grid = np.linspace(0, 1, 101)
    tprs, aucs = [], []
    for r, p in enumerate(reps):
        fpr, tpr, _ = roc_curve(y, p)
        aucs.append(roc_auc_score(y, p))
        tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=1, alpha=0.45, label=f'ROC repeat {r+1} (AUC = {aucs[-1]:.2f})')
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    sd = np.std(tprs, axis=0)
    ax.fill_between(grid, np.maximum(mean_tpr - sd, 0), np.minimum(mean_tpr + sd, 1), color='grey', alpha=0.25)
    ax.plot(grid, mean_tpr, c='black', lw=2.6, label=f'Mean ROC (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f})')
    ax.plot([0, 1], [0, 1], ls='--', c='red', lw=2, label='Chance')
    ax.set_xlabel('1 - Specificity', fontsize=17); ax.set_ylabel('Sensitivity', fontsize=17)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.03)
    ax.tick_params(labelsize=12); ax.legend(loc='lower right', fontsize=10.5)
    fig.tight_layout()
    _save(fig, 'roc_repeats.png')

    # calibration
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=5, strategy='quantile')
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=DPI)
    ax.plot([0, 1], [0, 1], ls='--', c='firebrick', lw=2, label='Perfectly calibrated')
    ax.plot(mean_pred, frac_pos, 'o-', c='#1f77b4', lw=2, ms=7, label='Responder (Δ ≥ 25 m)')
    ax.set_title('Calibration Curve', fontsize=20)
    ax.set_xlabel('Mean predicted probability', fontsize=16)
    ax.set_ylabel('Fraction of positives', fontsize=16)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.4); ax.tick_params(labelsize=12); ax.legend(loc='upper left', fontsize=12)
    fig.tight_layout()
    _save(fig, 'calibration_curve.png')

    # confusion matrix (row %)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=DPI)
    sns.heatmap(pct, annot=False, cmap='YlGnBu', vmin=0, vmax=100, square=True,
                xticklabels=['Non-responder', 'Responder'],
                yticklabels=['Non-responder', 'Responder'], cbar_kws={'shrink': 0.9}, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, f'{pct[i, j]:.2f}', ha='center', va='center',
                    fontsize=22, color='white' if pct[i, j] > 50 else 'black')
    ax.set_xlabel('Predicted Labels', fontsize=17); ax.set_ylabel('True Labels', fontsize=17)
    ax.tick_params(labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    _save(fig, 'confusion_matrix.png')

    # SHAP importance (fit on all labeled records; arm dummies combined)
    np.random.seed(SEED)
    m = factory(); m.fit(X, y)
    expl = shap.KernelExplainer(lambda Z: m.predict_proba(Z)[:, 1], X)
    sv = expl.shap_values(X, nsamples=200, silent=True)
    arm = np.abs(sv[:, 0] + sv[:, 1]).mean()
    imp = np.concatenate([[arm], np.abs(sv[:, 2:]).mean(axis=0)])
    order = np.argsort(imp)
    names = ['Training arm (IMT/HIIT/Control)'] + [DISPLAY.get(c, c) for c in feats[2:]]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)
    ax.barh(np.arange(len(imp)), imp[order], color=BAR_GREEN, height=0.78)
    for i, v in enumerate(imp[order]):
        ax.text(v + imp.max() * 0.012, i, f'{v:.3f}', va='center', fontsize=11)
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels([names[i] for i in order], fontsize=14)
    ax.set_xlabel('Feature Importance (mean |SHAP value|)', fontsize=17)
    ax.set_xlim(0, imp.max() * 1.12)
    ax.tick_params(axis='x', labelsize=12)
    fig.tight_layout()
    _save(fig, 'shap_importance.png')


def pdp_set():
    """Per-arm partial dependence panels and individual curves for FSS."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    factory, feats, _, _ = winner('clf')
    df = load_data()
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    X, y = d[feats].values, d.responder.values
    m = factory(); m.fit(X, y)

    def pdp(col, grid, arm):
        Xc = X.copy(); Xc[:, 0], Xc[:, 1] = arm
        out = []
        for v in grid:
            Xg = Xc.copy(); Xg[:, col] = v
            out.append(m.predict_proba(Xg)[:, 1].mean())
        return np.array(out)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), dpi=DPI, sharey=True)
    for ax, (colname, label) in zip(axes.ravel(), PDP_PANELS):
        ci = feats.index(colname)
        vals = d[colname].dropna()
        grid = np.linspace(vals.quantile(0.05), vals.quantile(0.95), 40)
        for arm_name, arm, color in ARMS:
            ax.plot(grid, pdp(ci, grid, arm), c=color, lw=2.4)
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylim(0, 1); ax.tick_params(labelsize=11); ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('Predicted probability of response', fontsize=12.5)
    handles = [plt.Line2D([0], [0], c=c, lw=3) for _, _, c in ARMS]
    fig.legend(handles, [a for a, _, _ in ARMS], loc='upper center', ncol=3,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'pdp_arms.png')

    ci = feats.index('FSS_BT')
    vals = d['FSS_BT'].dropna()
    grid = np.linspace(vals.quantile(0.05), vals.quantile(0.95), 40)
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=DPI)
    curves = []
    for i in range(len(d)):
        Xg = np.repeat(X[i:i + 1], len(grid), axis=0)
        Xg[:, ci] = grid
        p = m.predict_proba(Xg)[:, 1]
        curves.append(p)
        ax.plot(grid, p, c='#B8B8B8', lw=0.7, alpha=0.6)
    ax.plot(grid, np.mean(curves, axis=0), c='#22876B', lw=3, label='Mean (PDP)')
    ax.set_xlabel('Baseline FSS (score)', fontsize=16)
    ax.set_ylabel('Predicted probability of response', fontsize=15)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12); ax.legend(fontsize=12)
    fig.tight_layout()
    _save(fig, 'ice_fss.png')


if __name__ == '__main__':
    figure6_validation()
    figure7_treatment()
    interpretation_set()
    pdp_set()


# ---- VO2-family additions (paper-format figures, ported from ml_vo2peak) ----

SHAP_TAGS = {'SixMWT_meter_BT': '6-MWT distance (N)', 'MIP_BT': 'MIP (N)',
             'VO2KG_max_BT': 'VO2peak (N)', 'FSS_BT': 'FSS (N)',
             'SGQR_total_score_BT': 'SGRQ total (N)', 'Age': 'Age (N)', 'BKI': 'BMI (N)'}
SHORT = {'SixMWT_meter_BT': '6mwt', 'MIP_BT': 'mip', 'VO2KG_max_BT': 'vo2',
         'FSS_BT': 'fss', 'SGQR_total_score_BT': 'sgrq', 'Age': 'age', 'BKI': 'bmi'}


def response_rate_by_arm():
    """Observed responder share per arm (threshold: delta 6-MWT >= 25 m)."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    df = load_data()
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=DPI)
    for i, (name, code, color) in enumerate([('IMT', 1.0, C_IMT), ('HIIT', 3.0, C_HIIT),
                                             ('Control', 2.0, C_CTRL)]):
        sub = d[d.Group == code]
        rate = sub.responder.mean() * 100
        ax.bar(i, rate, color=color, width=0.62)
        ax.text(i, rate + 2, f'{rate:.0f}%\n({int(sub.responder.sum())}/{len(sub)})',
                ha='center', fontsize=13)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['IMT', 'HIIT', 'Control'], fontsize=14)
    ax.set_ylabel('6-MWT responders (Δ ≥ 25 m), %', fontsize=13.5)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    _save(fig, 'response_rate_by_arm.png')


def _shap_compact8():
    """Shared SHAP inputs for the beeswarm and dependence scatters (final model, all 55)."""
    factory, feats, _, _ = winner('clf')
    df = load_data()
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    X, y = d[feats].values.astype(float), d.responder.values
    m = factory()
    m.fit(X, y)
    np.random.seed(SEED)
    expl = shap.KernelExplainer(lambda Z: m.predict_proba(Z)[:, 1], X)
    sv = expl.shap_values(X, nsamples=200, silent=True)
    return d, X, sv, feats


def beeswarm():
    """SHAP beeswarm in the visual format of Oliver-Roig et al. 2022, Fig. 4.

    Arm rows first (incl. a derived Arm: Control row — colour: 1 when IMT=0 and
    HIIT=0; position: summed contribution of the two arm indicators), then the
    clinical variables by mean |SHAP|.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    _, X, sv, feats = _shap_compact8()
    ctrl = ((X[:, 0] == 0) & (X[:, 1] == 0)).astype(float)
    clin = list(range(2, len(feats)))
    clin.sort(key=lambda i: -np.nanmean(np.abs(sv[:, i])))
    Xd = np.column_stack([X[:, 0], X[:, 1], ctrl] + [X[:, i] for i in clin])
    svd = np.column_stack([sv[:, 0], sv[:, 1], sv[:, 0] + sv[:, 1]] + [sv[:, i] for i in clin])
    names = ['Arm: IMT (D)', 'Arm: HIIT (D)', 'Arm: Control (D)'] + \
            [SHAP_TAGS[feats[i]] for i in clin]
    plt.figure(figsize=(9.6, 6.6), dpi=DPI)
    shap.summary_plot(svd, Xd, feature_names=names, sort=False, show=False, plot_size=None)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_xlabel('Impact on model output', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    for t in ax.get_yticklabels():
        t.set_fontsize(13.5)
    if len(fig.axes) > 1:
        cax = fig.axes[-1]
        cax.tick_params(labelsize=12)
        cax.set_ylabel('Feature value', fontsize=13)
    _save(fig, 'shap_beeswarm.png')


def shap_scatters():
    """Dependence scatters (Oliver-Roig Figs 5-7 format) for the two clinical
    variables with the largest mean |SHAP|; the second panel is coloured by the
    first variable."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    from shap.plots import colors as shap_colors
    _, X, sv, feats = _shap_compact8()
    clin = list(range(2, len(feats)))
    clin.sort(key=lambda i: -np.nanmean(np.abs(sv[:, i])))
    top1, top2 = clin[0], clin[1]

    def panel(ci, fname, color_ci=None):
        ok = np.isfinite(X[:, ci]) if color_ci is None else \
            np.isfinite(X[:, ci]) & np.isfinite(X[:, color_ci])
        x, s = X[ok, ci], sv[ok, ci]
        fig, ax = plt.subplots(figsize=(7.8 if color_ci is None else 8.8, 5.8), dpi=DPI)
        lo, hi = float(s.min()), float(s.max())
        band = 0.15 * (hi - lo)
        y0 = lo - 1.5 * band
        cnt, edges = np.histogram(x, bins=16)
        ax.bar((edges[:-1] + edges[1:]) / 2, cnt / cnt.max() * band, bottom=y0,
               width=(edges[1] - edges[0]) * 0.97, color='#ababab', zorder=1)
        ax.axhline(0, ls='--', c='#333333', lw=1.0, zorder=2)
        if color_ci is None:
            ax.scatter(x, s, s=48, c='#1f77b4', alpha=0.88, edgecolors='none', zorder=3)
        else:
            sc = ax.scatter(x, s, s=52, c=X[ok, color_ci], cmap=shap_colors.red_blue,
                            alpha=0.92, edgecolors='none', zorder=3)
            cb = fig.colorbar(sc, ax=ax, aspect=32, pad=0.02)
            cb.outline.set_visible(False)
            cb.set_ticks([])
            cb.ax.text(0.5, 1.005, 'High', ha='center', va='bottom',
                       transform=cb.ax.transAxes, fontsize=12)
            cb.ax.text(0.5, -0.005, 'Low', ha='center', va='top',
                       transform=cb.ax.transAxes, fontsize=12)
            cb.set_label(SHAP_TAGS[feats[color_ci]], fontsize=13)
        ax.set_xlabel(SHAP_TAGS[feats[ci]], fontsize=15)
        ax.set_ylabel('Impact on the model output', fontsize=15)
        ax.set_ylim(y0, hi + band * 0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        _save(fig, fname)

    panel(top1, f'shap_scatter_{SHORT[feats[top1]]}.png')
    panel(top2, f'shap_scatter_{SHORT[feats[top2]]}.png', color_ci=top1)


def decision_plot():
    """Same-arm patient pairs in the visual format of Oliver-Roig et al. 2022,
    Fig. 8 — one figure per arm (IMT, HIIT, Control); an observed responder
    (red) and an observed non-responder (blue) start from the arm-typical
    probability and diverge on the seven clinical variables. Patients A-B: IMT,
    C-D: HIIT, E-F: Control; duplicate-enrolled persons are used at most once;
    only complete-data patients are eligible. Exact Shapley with same-arm
    conditioning; per-figure row order keeps cumulative positions in 0-100%.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    from math import factorial
    from itertools import permutations
    from shap.plots import colors as shap_colors
    factory, feats, _, _ = winner('clf')
    df = load_data()
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    X, y, per = d[feats].values.astype(float), d.responder.values, d.person.values
    reps = [_oof_repeat(factory, X, y, per, r) for r in range(5)]
    prob = np.mean(reps, axis=0)
    m = factory()
    m.fit(X, y)
    pfull = m.predict_proba(X)[:, 1]

    n = len(feats) - 2
    REF = [float(np.nanmedian(X[:, 2 + i])) for i in range(n)]
    LABELS = [SHAP_TAGS[c] for c in feats[2:]]
    masks = np.array([[(mm >> i) & 1 for i in range(n)] for mm in range(2 ** n)])

    def explain(idx):
        Xm = np.tile([X[idx, 0], X[idx, 1]] + REF, (2 ** n, 1))
        for i in range(n):
            Xm[masks[:, i] == 1, 2 + i] = X[idx, 2 + i]
        fv = m.predict_proba(Xm)[:, 1]
        phi = np.zeros(n)
        for mask in range(2 ** n):
            s = int(masks[mask].sum())
            for i in range(n):
                if not masks[mask][i]:
                    w = factorial(s) * factorial(n - s - 1) / factorial(n)
                    phi[i] += w * (fv[mask | (1 << i)] - fv[mask])
        return phi * 100, fv[0] * 100, fv[-1] * 100

    complete = np.isfinite(X[:, 2:]).all(axis=1)
    arm_masks = [('IMT', X[:, 0] == 1), ('HIIT', X[:, 1] == 1),
                 ('Control', (X[:, 0] == 0) & (X[:, 1] == 0))]
    pairs, expl = [], {}
    used = set()
    for name, am in arm_masks:
        if not (am & (y == 1)).any() or not (am & (y == 0)).any():
            print(f'skip {name}: no responder-non-responder pair exists in this arm '
                  f'({int((am & (y == 1)).sum())}/{int(am.sum())} responders)')
            continue
        free = np.array([p not in used for p in per]) & complete
        okp = am & free & (y == 1) & (prob >= 0.5) & (pfull >= 0.5)
        cand = okp if okp.any() else (am & free & (y == 1))
        r = int(np.argmax(np.where(cand, prob, -1)))
        used.add(per[r])
        free = np.array([p not in used for p in per]) & complete
        okn = am & free & (y == 0) & (prob < 0.5) & (pfull < 0.5)
        cand_n = okn if okn.any() else (am & free & (y == 0))
        nr = int(np.argmin(np.where(cand_n, prob, 2)))
        used.add(per[nr])
        pairs.append((name, r, nr))
        expl[r], expl[nr] = explain(r), explain(nr)

    imp = np.mean([np.abs(e[0]) for e in expl.values()], axis=0)
    want = list(np.argsort(imp))

    def n_bad(perm, idx_pair):
        bad = 0
        for phi, base, _ in (expl[j] for j in idx_pair):
            cum = base
            for i in perm:
                cum += phi[i]
                bad += not (0.0 <= cum <= 100.0)
        return bad

    RED, BLUE = '#C8385A', '#3A6FB0'
    XLIM = (-1.5, 101.5)
    letters = iter('ABCDEF')

    for name, r, nr in pairs:
        order = want if n_bad(want, (r, nr)) == 0 else min(
            (list(p) for p in permutations(range(n))), key=lambda p: n_bad(p, (r, nr)))
        rows = list(reversed(order))
        ytop = {i: rows.index(i) for i in order}
        fig, (strip, ax) = plt.subplots(
            2, 1, figsize=(10.5, 7.6), dpi=DPI,
            gridspec_kw=dict(height_ratios=[1, 15], hspace=0.06))
        base_p = expl[r][1]

        strip.imshow(np.linspace(0, 1, 512)[None, :], aspect='auto',
                     cmap=shap_colors.red_blue, extent=[0, 100, 0, 1])
        strip.set_xlim(XLIM)
        strip.set_yticks([])
        strip.xaxis.set_ticks_position('top')
        strip.set_xticks([0, 20, 40, 60, 80, 100])
        strip.tick_params(labelsize=10, length=2)
        for sp in strip.spines.values():
            sp.set_visible(False)
        strip.set_title(f'{name} arm', fontsize=13.5, pad=26)
        shown = []
        for v in (base_p, expl[r][2], expl[nr][2]):
            strip.plot(v, 1.45, marker='v', ms=6, c='black', clip_on=False)
            if all(abs(v - s) >= 3.5 for s in shown):  # avoid overlapping numbers
                strip.text(min(max(v, 2.2), 97.8), 0.5, f'{v:.0f}', ha='center',
                           va='center', fontsize=10, fontweight='bold', color='white')
                shown.append(v)

        for rr in range(n):
            ax.axhline(rr, ls=(0, (1, 4)), c='#bbbbbb', lw=0.9, zorder=0)
        ax.axvline(base_p, c='#888888', lw=1.3, zorder=1)
        placed = []  # (ly, lx) of drawn value labels, to dodge overlaps
        for idx, colr, sign in [(r, RED, +1), (nr, BLUE, -1)]:
            phi, _, fin = expl[idx]
            cum, xs = base_p, [base_p]
            for i in order:
                cum += phi[i]
                xs.append(cum)
            xs[-1] = fin
            ys = [n - 0.35] + [ytop[i] for i in order]
            ax.plot(xs, ys, c=colr, lw=1.6, zorder=3)
            ax.plot([fin, fin], [ys[-1], -0.55], c=colr, lw=1.6, zorder=3,
                    clip_on=False)
            vals = [f'{v:.0f}' if abs(v) >= 100 or float(v).is_integer()
                    else f'{v:.1f}' for v in X[idx, 2:]]
            for k, i in enumerate(order):
                lx = xs[k + 1] + sign * 2.0
                lha, ly = 'left' if sign > 0 else 'right', ytop[i] - 0.16
                if ytop[i] == 0:
                    ly = 0.34
                if sign > 0 and lx > 92:
                    lx, lha, ly = xs[k + 1] - 2.0, 'right', ytop[i] + 0.34
                if sign < 0 and lx < 8:
                    lx, lha, ly = xs[k + 1] + 2.0, 'left', ytop[i] + 0.34
                for py, px in placed:  # dodge a label already sitting there
                    if abs(py - ly) < 0.3 and abs(px - lx) < 11:
                        ly += 0.5
                        break
                placed.append((ly, lx))
                ax.text(lx, ly, f'({vals[i]})', ha=lha, fontsize=10,
                        color='#555555')
            tx = min(max(expl[idx][2], 6.5), 93.5)
            ax.text(tx, -0.28, f'Patient {next(letters)}', ha='center', va='center',
                    fontsize=11.5, color=colr,
                    bbox=dict(edgecolor=colr, facecolor='white', lw=1.2))

        ax.set_yticks(range(n))
        ax.set_yticklabels([LABELS[i] for i in rows], fontsize=12)
        ax.set_ylim(n - 0.4, -0.55)
        ax.set_xlim(XLIM)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel('Probability of 6-MWT response (%)', fontsize=13)
        ax.tick_params(labelsize=10.5)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        _save(fig, f'decision_plot_{name.lower()}.png')
