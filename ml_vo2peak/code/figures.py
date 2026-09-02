"""Interpretation figures for the VO₂peak response task (500 dpi).

Run after benchmark.py (reads its scoreboard to pick the winner: best mean F1
among the named, non-selection feature sets). SHAP importance, calibration,
confusion matrix, per-repeat ROC, response rate by arm.
"""
import warnings
warnings.filterwarnings('ignore')
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score, precision_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix)
from core import load_data, grouped_folds, zoo_factory, FEATURE_SETS, METRICS, FIGURES, SEED
from benchmark import VO2_THRESHOLD

DPI = 500
BAR_GREEN = '#66C2A5'
DISPLAY = {'grp_IMT': 'Training arm: IMT', 'grp_HIIT': 'Training arm: HIIT',
           'SixMWT_meter_BT': 'Baseline 6-MWT distance', 'MIP_BT': 'Baseline MIP',
           'VO2KG_max_BT': 'Baseline VO₂peak', 'FSS_BT': 'Baseline FSS',
           'SGQR_total_score_BT': 'Baseline SGRQ total', 'Age': 'Age', 'BKI': 'BMI',
           'female': 'Female sex', 'COVID_19_time': 'Time since COVID-19',
           'ct_gt50': 'CT involvement >50%', 'hospitalized': 'Hospitalized',
           'charlson': 'Charlson index', 'ever_smoker': 'Smoking history',
           'exercise_habit': 'Exercise habit', 'MEP_BT': 'Baseline MEP',
           'Quadriceps_right_BT': 'Baseline quadriceps strength',
           'Step_count_BT': 'Daily step count', 'Physical_activity_duration_BT': 'Physical-activity duration'}


def _save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'written: {name}')


def pick_winner():
    rows = json.load(open(os.path.join(METRICS, 'zoo_results.json')))['vo2']
    named = [r for r in rows if r['feats'] in FEATURE_SETS]
    best = max(named, key=lambda r: r['mean']['F1'])
    return best['model'], best['feats'], best['mean']


def oof_repeat_probs(factory, X, y, groups, rep):
    p = np.full(len(y), np.nan)
    for tr, te in grouped_folds(y, groups, 5, 1, 'clf', seed=SEED + rep):
        m = factory()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main():
    mname, sname, zmean = pick_winner()
    print(f'winner (best F1, named sets): {mname} | {sname} | F1={zmean["F1"]:.3f} AUC={zmean["AUC"]:.3f}')
    factory, feats = zoo_factory('clf', mname), FEATURE_SETS[sname]

    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['responder_vo2'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'responder_vo2'] = np.nan
    d = df.dropna(subset=['responder_vo2']).reset_index(drop=True)
    X, y, per = d[feats].values, d.responder_vo2.values, d.person.values

    reps = [oof_repeat_probs(factory, X, y, per, r) for r in range(5)]
    prob = np.mean(reps, axis=0)

    yh = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, yh)
    metrics = dict(
        model=f'{mname} ({sname})', threshold=VO2_THRESHOLD, n=int(len(y)),
        AUC=roc_auc_score(y, prob), BRIER=brier_score_loss(y, prob),
        ACC=accuracy_score(y, yh), BAL=balanced_accuracy_score(y, yh),
        F1=f1_score(y, yh), SENS=recall_score(y, yh),
        SPEC=recall_score(y, yh, pos_label=0), PPV=precision_score(y, yh),
        NPV=precision_score(y, yh, pos_label=0), MCC=matthews_corrcoef(y, yh),
        confusion=dict(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1])))
    json.dump(metrics, open(os.path.join(METRICS, 'interpret_metrics.json'), 'w'), indent=1)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    # per-repeat ROC + mean band
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=DPI)
    grid = np.linspace(0, 1, 101)
    tprs, aucs = [], []
    for r, p in enumerate(reps):
        fpr, tpr, _ = roc_curve(y, p)
        a = roc_auc_score(y, p)
        aucs.append(a)
        tprs.append(np.interp(grid, fpr, tpr))
        tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=1, alpha=0.45, label=f'ROC repeat {r+1} (AUC = {a:.2f})')
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    sd_tpr = np.std(tprs, axis=0)
    ax.fill_between(grid, np.maximum(mean_tpr - sd_tpr, 0), np.minimum(mean_tpr + sd_tpr, 1),
                    color='grey', alpha=0.25)
    ax.plot(grid, mean_tpr, c='black', lw=2.6,
            label=f'Mean ROC (AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f})')
    ax.plot([0, 1], [0, 1], ls='--', c='red', lw=2, label='Chance')
    ax.set_xlabel('1 - Specificity', fontsize=17)
    ax.set_ylabel('Sensitivity', fontsize=17)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.03)
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=10.5)
    fig.tight_layout()
    _save(fig, 'roc_repeats.png')

    # calibration
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=5, strategy='quantile')
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=DPI)
    ax.plot([0, 1], [0, 1], ls='--', c='firebrick', lw=2, label='Perfectly calibrated')
    ax.plot(mean_pred, frac_pos, 'o-', c='#1f77b4', lw=2, ms=7,
            label=f'Responder (ΔVO₂peak ≥ {VO2_THRESHOLD})')
    ax.set_title('Calibration Curve', fontsize=20)
    ax.set_xlabel('Mean predicted probability', fontsize=16)
    ax.set_ylabel('Fraction of positives', fontsize=16)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.legend(loc='upper left', fontsize=12)
    fig.tight_layout()
    _save(fig, 'calibration_curve.png')

    # confusion matrix (row %)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=DPI)
    sns.heatmap(pct, annot=False, cmap='YlGnBu', vmin=0, vmax=100, square=True,
                xticklabels=['Non-responder', 'Responder'],
                yticklabels=['Non-responder', 'Responder'],
                cbar_kws={'shrink': 0.9}, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, f'{pct[i, j]:.2f}', ha='center', va='center',
                    fontsize=22, color='white' if pct[i, j] > 50 else 'black')
    ax.set_xlabel('Predicted Labels', fontsize=17)
    ax.set_ylabel('True Labels', fontsize=17)
    ax.tick_params(labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    _save(fig, 'confusion_matrix.png')

    # SHAP importance (model fit on all labeled records; arm dummies combined)
    np.random.seed(SEED)
    m = factory()
    m.fit(X, y)
    f = lambda Z: m.predict_proba(Z)[:, 1]
    expl = shap.KernelExplainer(f, X)
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

    # response rate by arm — the clinically honest "why" figure for a group-only winner
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=DPI)
    arms = [('IMT', 1.0, '#27A883'), ('HIIT', 3.0, '#F97B4C'), ('Control', 2.0, '#7C8CE4')]
    for i, (name, code, color) in enumerate(arms):
        sub = d[d.Group == code]
        rate = sub.responder_vo2.mean() * 100
        ax.bar(i, rate, color=color, width=0.62)
        ax.text(i, rate + 2, f'{rate:.0f}%\n({int(sub.responder_vo2.sum())}/{len(sub)})',
                ha='center', fontsize=13)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[0] for a in arms], fontsize=14)
    ax.set_ylabel(f'VO₂peak responders (Δ ≥ {VO2_THRESHOLD} mL·kg⁻¹·min⁻¹), %', fontsize=13.5)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    _save(fig, 'response_rate_by_arm.png')


if __name__ == '__main__':
    main()



def waterfalls():
    """Per-patient decision explanations (exact Shapley, cohort-median reference).

    Three anonymized examples from the pooled OOF classification: a correctly
    identified responder (A), a correctly identified non-responder (B) and a
    missed responder (C). Explained with the final model under each patient's
    actual training arm.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    from math import factorial
    from matplotlib.patches import Polygon
    from core import COMPACT5
    factory, feats = zoo_factory('clf', 'Elastic-net logistic'), COMPACT5
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['responder_vo2'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'responder_vo2'] = np.nan
    d = df.dropna(subset=['responder_vo2']).reset_index(drop=True)
    X, y, per = d[feats].values, d.responder_vo2.values, d.person.values

    reps = [oof_repeat_probs(factory, X, y, per, r) for r in range(5)]
    prob = np.mean(reps, axis=0)
    m = factory()
    m.fit(X, y)
    pfull = m.predict_proba(X)[:, 1]
    imt, hii = X[:, 0] == 1, X[:, 1] == 1
    ctl = ~imt & ~hii
    ok_pos, ok_neg = (y == 1) & (prob >= 0.5) & (pfull >= 0.5), (y == 0) & (prob < 0.5) & (pfull < 0.5)
    # narrative picks: arm-covering, most confident example of each story
    a_imt = int(np.argmax(np.where(imt & ok_pos, prob, -1)))
    b_hii = int(np.argmax(np.where(hii & ok_pos, prob, -1)))
    c_hii = int(np.argmin(np.where(hii & ok_neg, prob - X[:, 3] * 1e-6, 2)))
    d_fn = int(np.argmin(np.where((y == 1) & (prob < 0.5) & (pfull < 0.5), prob, 2)))

    REF = [5324.0, 47.0, 510.0, 0.0]
    LABELS = ['Daily step count', 'Activity duration (min)', '6-MWT distance (m)', 'Exercise habit']
    POS, NEG = '#FF0D57', '#1E88E5'
    n = 4
    masks = np.array([[(mm >> i) & 1 for i in range(n)] for mm in range(2 ** n)])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), dpi=DPI)
    for ax, idx, title in zip(axes.ravel(), [a_imt, b_hii, c_hii, d_fn],
                              ['A  IMT responder — correctly identified',
                               'B  HIIT responder — correctly identified',
                               'C  HIIT non-responder — correctly identified',
                               'D  Missed responder']):
        gi, gh = X[idx, 0], X[idx, 1]
        entered = list(X[idx, 2:])
        Xm = np.tile([gi, gh] + REF, (2 ** n, 1))
        for i in range(n):
            Xm[masks[:, i] == 1, 2 + i] = entered[i]
        fv = m.predict_proba(Xm)[:, 1]
        phi = np.zeros(n)
        for mask in range(2 ** n):
            s = int(masks[mask].sum())
            for i in range(n):
                if not masks[mask][i]:
                    w = factorial(s) * factorial(n - s - 1) / factorial(n)
                    phi[i] += w * (fv[mask | (1 << i)] - fv[mask])
        base_p, final_p = fv[0] * 100, fv[-1] * 100
        # order bars largest-first but keep every partial sum inside [0, 100]
        remaining, order, cum = list(range(n)), [], base_p
        while remaining:
            bymag = sorted(remaining, key=lambda i: -abs(phi[i]))
            pick = next((i for i in bymag if 0.0 <= cum + phi[i] * 100 <= 100.0), bymag[0])
            order.append(pick)
            remaining.remove(pick)
            cum += phi[pick] * 100

        cum, points = base_p, [base_p]
        for i in order:
            cum += phi[i] * 100
            points.append(cum)
        lo, hi = min(points), max(points)
        pad = max((hi - lo) * 0.28, 4)
        x0, x1 = max(-0.5, lo - pad), min(100.5, hi + pad)
        rng = x1 - x0

        cum = base_p
        for row, i in enumerate(order):
            v = phi[i] * 100
            nxt = cum + v
            left, right = min(cum, nxt), max(cum, nxt)
            color = POS if v >= 0 else NEG
            head = min(abs(v) * 0.45, rng * 0.03)
            yb, h2 = row, 0.30
            if v >= 0:
                pts = [(left, yb - h2), (right - head, yb - h2), (right, yb),
                       (right - head, yb + h2), (left, yb + h2)]
            else:
                pts = [(right, yb - h2), (left + head, yb - h2), (left, yb),
                       (left + head, yb + h2), (right, yb + h2)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor='none'))
            txt = f'{v:+.0f}'
            if abs(v) > rng * 0.12:
                ax.text((left + right) / 2, yb, txt, ha='center', va='center',
                        color='white', fontsize=11, fontweight='bold')
            else:
                off = rng * 0.015
                if v >= 0:
                    ax.text(right + off, yb, txt, ha='left', va='center', color=color,
                            fontsize=10.5, fontweight='bold')
                else:
                    ax.text(left - off, yb, txt, ha='right', va='center', color=color,
                            fontsize=10.5, fontweight='bold')
            # dotted connector down to the next row
            if row < n - 1:
                ax.plot([nxt, nxt], [yb, yb + 1], ls=':', c='#bbbbbb', lw=1)
            cum = nxt

        val = lambda i: (('Yes' if entered[i] else 'No') if i == 3 else f'{entered[i]:g}')
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'{val(i)} = {LABELS[i]}' for i in order], fontsize=10.5)
        ax.set_ylim(n - 0.4, -1.1)
        ax.axvline(base_p, c='#999999', ls='--', lw=1.1, zorder=0)
        ax.text(base_p, n - 0.55, f'typical: {base_p:.0f}%', ha='center', fontsize=9.5,
                color='#777777')
        ax.text(points[-1], -0.8, f'this patient: {final_p:.0f}%', ha='center',
                fontsize=10.5, fontweight='bold')
        arm = 'IMT' if gi else ('HIIT' if gh else 'Control')
        ax.set_title(f'{title}  (arm: {arm})', fontsize=12)
        ax.set_xlim(x0, x1)
        ax.set_xlabel('P(responder), %', fontsize=11)
        ax.tick_params(labelsize=10)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, 'waterfall_examples.png')


def _shap_compact5():
    """Shared SHAP inputs for the beeswarm and dependence scatters (final model, all 53)."""
    from core import COMPACT5
    factory = zoo_factory('clf', 'Elastic-net logistic')
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['y'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'y'] = np.nan
    d = df.dropna(subset=['y']).reset_index(drop=True)
    X, y = d[COMPACT5].values, d.y.values
    m = factory()
    m.fit(X, y)
    np.random.seed(SEED)
    expl = shap.KernelExplainer(lambda Z: m.predict_proba(Z)[:, 1], X)
    sv = expl.shap_values(X, nsamples=200, silent=True)
    return d, X, sv


def beeswarm():
    """SHAP beeswarm in the visual format of Oliver-Roig et al. 2022, Fig. 4.

    Besides the two fitted arm indicators, a derived Arm: Control row is shown
    (colour: 1 when IMT=0 and HIIT=0; position: summed contribution of the two
    arm indicators), so the reference arm's negative pull is visible. Rows are
    kept in a fixed order: arm indicators first, then the remaining variables
    by mean |SHAP|.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    _, X, sv = _shap_compact5()
    ctrl = ((X[:, 0] == 0) & (X[:, 1] == 0)).astype(float)
    Xd = np.column_stack([X[:, 0], X[:, 1], ctrl, X[:, 2], X[:, 3], X[:, 5], X[:, 4]])
    svd = np.column_stack([sv[:, 0], sv[:, 1], sv[:, 0] + sv[:, 1],
                           sv[:, 2], sv[:, 3], sv[:, 5], sv[:, 4]])
    names = ['Arm: IMT (D)', 'Arm: HIIT (D)', 'Arm: Control (D)',
             'Daily step count (N)', 'Activity duration (N)',
             'Exercise habit (D)', '6-MWT distance (N)']
    plt.figure(figsize=(9.6, 5.6), dpi=DPI)
    shap.summary_plot(svd, Xd, feature_names=names, sort=False, show=False, plot_size=None)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_xlabel('Impact on model output', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    for t in ax.get_yticklabels():
        t.set_fontsize(14)
    if len(fig.axes) > 1:  # shap's colorbar (High / Low, "Feature value")
        cax = fig.axes[-1]
        cax.tick_params(labelsize=12)
        cax.set_ylabel('Feature value', fontsize=13)
    _save(fig, 'shap_beeswarm.png')


def shap_scatters():
    """Per-variable SHAP dependence scatters in the visual format of
    Oliver-Roig et al. 2022, Figs. 5-7: raw value (x) against impact on the
    model output (y), grey value histogram along the bottom. The
    activity-duration panel is coloured by daily step count (their Fig. 7
    interaction device)."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    from shap.plots import colors as shap_colors
    _, X, sv = _shap_compact5()

    def panel(ci, xlabel, fname, color_ci=None, cbar_label=None):
        x, s = X[:, ci].astype(float), sv[:, ci]
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
            sc = ax.scatter(x, s, s=52, c=X[:, color_ci].astype(float),
                            cmap=shap_colors.red_blue, alpha=0.92, edgecolors='none', zorder=3)
            cb = fig.colorbar(sc, ax=ax, aspect=32, pad=0.02)
            cb.outline.set_visible(False)
            cb.set_ticks([])
            cb.ax.text(0.5, 1.005, 'High', ha='center', va='bottom',
                       transform=cb.ax.transAxes, fontsize=12)
            cb.ax.text(0.5, -0.005, 'Low', ha='center', va='top',
                       transform=cb.ax.transAxes, fontsize=12)
            cb.set_label(cbar_label, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel('Impact on the model output', fontsize=15)
        ax.set_ylim(y0, hi + band * 0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        _save(fig, fname)

    panel(2, 'Daily step count (N)', 'shap_scatter_steps.png')
    panel(3, 'Activity duration (N)', 'shap_scatter_duration.png',
          color_ci=2, cbar_label='Daily step count (N)')


def decision_plot():
    """Same-arm patient pairs in the visual format of Oliver-Roig et al. 2022,
    Fig. 8 — one separate figure per arm (IMT, HIIT, Control). In each, an
    observed responder (red) and an observed non-responder (blue) start from
    the arm-typical probability and diverge on the four clinical variables
    alone. Patients A-B: IMT, C-D: HIIT, E-F: Control (the control responder
    is the cohort's single one — the model's missed responder). Exact Shapley
    with same-arm conditioning; one shared row order across all three figures
    keeps every cumulative position inside 0-100%.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    from math import factorial
    from itertools import permutations
    from shap.plots import colors as shap_colors
    from core import COMPACT5
    factory, feats = zoo_factory('clf', 'Elastic-net logistic'), COMPACT5
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['responder_vo2'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'responder_vo2'] = np.nan
    d = df.dropna(subset=['responder_vo2']).reset_index(drop=True)
    X, y, per = d[feats].values, d.responder_vo2.values, d.person.values
    reps = [oof_repeat_probs(factory, X, y, per, r) for r in range(5)]
    prob = np.mean(reps, axis=0)
    m = factory()
    m.fit(X, y)
    pfull = m.predict_proba(X)[:, 1]

    REF = [5324.0, 47.0, 510.0, 0.0]
    LABELS = ['Daily step count (N)', 'Activity duration (N)',
              '6-MWT distance (N)', 'Exercise habit (D)']
    n = 4
    masks = np.array([[(mm >> i) & 1 for i in range(n)] for mm in range(2 ** n)])

    def explain(idx):  # exact Shapley, same-arm conditioning (as in waterfalls)
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

    arm_masks = [('IMT', X[:, 0] == 1), ('HIIT', X[:, 1] == 1),
                 ('Control', (X[:, 0] == 0) & (X[:, 1] == 0))]
    pairs, expl = [], {}
    used = set()  # duplicate-enrolled persons appear at most once across figures
    for name, am in arm_masks:
        free = np.array([p not in used for p in per])
        okp = am & free & (y == 1) & (prob >= 0.5) & (pfull >= 0.5)
        cand = okp if okp.any() else (am & free & (y == 1))
        r = int(np.argmax(np.where(cand, prob, -1)))
        used.add(per[r])
        free = np.array([p not in used for p in per])
        okn = am & free & (y == 0) & (prob < 0.5) & (pfull < 0.5)
        nr = int(np.argmin(np.where(okn, prob - X[:, 3] * 1e-6, 2)))
        used.add(per[nr])
        pairs.append((name, r, nr))
        expl[r], expl[nr] = explain(r), explain(nr)

    # preferred row order: ascending mean importance over the six patients
    imp = np.mean([np.abs(e[0]) for e in expl.values()], axis=0)
    want = list(np.argsort(imp))

    def n_bad(perm, idx_pair):  # cumulative positions outside [0, 100]
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
        # this figure's row order: the preferred one if its two paths stay in
        # bounds, otherwise the best bound-preserving permutation
        order = want if n_bad(want, (r, nr)) == 0 else min(
            (list(p) for p in permutations(range(n))), key=lambda p: n_bad(p, (r, nr)))
        rows = list(reversed(order))
        ytop = {i: rows.index(i) for i in order}
        fig, (strip, ax) = plt.subplots(
            2, 1, figsize=(10.5, 6.2), dpi=DPI,
            gridspec_kw=dict(height_ratios=[1, 13], hspace=0.06))
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
        for v in (base_p, expl[r][2], expl[nr][2]):
            strip.plot(v, 1.45, marker='v', ms=6, c='black', clip_on=False)
            strip.text(min(max(v, 2.2), 97.8), 0.5, f'{v:.0f}', ha='center',
                       va='center', fontsize=10, fontweight='bold', color='white')

        for rr in range(n):
            ax.axhline(rr, ls=(0, (1, 4)), c='#bbbbbb', lw=0.9, zorder=0)
        ax.axvline(base_p, c='#888888', lw=1.3, zorder=1)
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
            e = X[idx, 2:]
            vals = ['{:g}'.format(e[0]), '{:g}'.format(e[1]),
                    '{:g}'.format(e[2]), 'Yes' if e[3] else 'No']
            for k, i in enumerate(order):
                lx = xs[k + 1] + sign * 2.0
                lha, ly = 'left' if sign > 0 else 'right', ytop[i] - 0.16
                if ytop[i] == 0:  # top row: keep clear of the name labels
                    ly = 0.34
                if sign > 0 and lx > 92:  # keep labels inside: mirror below row
                    lx, lha, ly = xs[k + 1] - 2.0, 'right', ytop[i] + 0.34
                if sign < 0 and lx < 8:
                    lx, lha, ly = xs[k + 1] + 2.0, 'left', ytop[i] + 0.34
                ax.text(lx, ly, f'({vals[i]})', ha=lha, fontsize=10.5,
                        color='#555555')
            tx = min(max(fin, 6.5), 93.5)  # name label caps its own line
            ax.text(tx, -0.28, f'Patient {next(letters)}', ha='center', va='center',
                    fontsize=11.5, color=colr,
                    bbox=dict(edgecolor=colr, facecolor='white', lw=1.2))

        ax.set_yticks(range(n))
        ax.set_yticklabels([LABELS[i] for i in rows], fontsize=12.5)
        ax.set_ylim(n - 0.4, -0.55)
        ax.set_xlim(XLIM)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel('Probability of VO₂peak response (%)', fontsize=13)
        ax.tick_params(labelsize=10.5)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        _save(fig, f'decision_plot_{name.lower()}.png')


def pdp_arms_vo2():
    """Per-arm partial dependence for the three continuous compact-5 variables."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    from core import COMPACT5
    factory = zoo_factory('clf', 'Elastic-net logistic')
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['y'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'y'] = np.nan
    d = df.dropna(subset=['y']).reset_index(drop=True)
    X, y = d[COMPACT5].values, d.y.values
    m = factory()
    m.fit(X, y)
    ARMS = [('IMT', (1, 0), '#27A883'), ('HIIT', (0, 1), '#F97B4C'), ('Control', (0, 0), '#7C8CE4')]
    panels = [(2, 'Daily step count (steps/day)'), (3, 'Activity duration (min/day)'),
              (4, 'Baseline 6-MWT distance (m)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=DPI, sharey=True)
    for ax, (ci, label) in zip(axes, panels):
        vals = d[COMPACT5[ci]]
        grid = np.linspace(vals.quantile(0.05), vals.quantile(0.95), 40)
        for arm_name, arm, color in ARMS:
            Xc = X.copy()
            Xc[:, 0], Xc[:, 1] = arm
            out = []
            for v in grid:
                Xg = Xc.copy()
                Xg[:, ci] = v
                out.append(m.predict_proba(Xg)[:, 1].mean())
            ax.plot(grid, out, c=color, lw=2.4, label=arm_name)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10.5)
    axes[0].set_ylabel('Predicted P(responder)', fontsize=12)
    axes[0].legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    _save(fig, 'pdp_arms_vo2.png')


def activity_scatter():
    """Holistic view: step count vs activity duration, trained arms, by outcome."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['y'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'y'] = np.nan
    d = df[df.y.notna() & df.Group.isin([1.0, 3.0])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=DPI)
    for val, marker, color, label in [(1.0, 'o', '#FF0D57', 'Responder'),
                                      (0.0, 'X', '#1E88E5', 'Non-responder')]:
        s = d[d.y == val]
        ax.scatter(s.Step_count_BT, s.Physical_activity_duration_BT, marker=marker,
                   s=95, c=color, alpha=0.85, edgecolors='white', linewidths=0.8, label=label)
    hab = d[d.exercise_habit == 1]
    ax.scatter(hab.Step_count_BT, hab.Physical_activity_duration_BT, facecolors='none',
               edgecolors='#222222', s=230, linewidths=1.4, label='Regular exercise habit')
    ax.set_xlabel('Daily step count (steps/day)', fontsize=13)
    ax.set_ylabel('Physical-activity duration (min/day)', fontsize=13)
    ax.set_title('Trained arms (IMT + HIIT): who responds?', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10.5, loc='upper left')
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, 'activity_scatter.png')
