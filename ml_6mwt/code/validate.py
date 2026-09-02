"""Winner validation and supporting analyses.

Sections: (1) classification winner — permutation test, person-clustered
bootstrap CI, calibration, decision-curve analysis; (2) regression winner —
same permutation/bootstrap design; (3) arm ablation (feature value beyond treatment assignment);
(4) person-clustered bootstrap of the counterfactual arm contrast.
All results land in results/metrics/validation_results.json.
"""
import warnings
warnings.filterwarnings('ignore')
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, brier_score_loss, f1_score
from sklearn.calibration import calibration_curve
from core import load_data, grouped_folds, winner, METRICS, SEED

B_BOOT, B_PERM = 2000, 200
rng = np.random.RandomState(SEED)


def oof_preds(factory, X, y, groups, task, n_repeats=5, seed=SEED):
    # per-patient out-of-fold average
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in grouped_folds(y, groups, 5, n_repeats, task, seed):
        m = factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1] if task == 'clf' else m.predict(X[te])
        acc[te] += p; cnt[te] += 1
    return acc / cnt


def cluster_boot_ci(stat_fn, y, p, persons, B=B_BOOT):
    # person-clustered bootstrap
    uniq = np.unique(persons)
    idx_of = {u: np.flatnonzero(persons == u) for u in uniq}
    stats = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[u] for u in pick])
        try:
            stats.append(stat_fn(y[rows], p[rows]))
        except ValueError:
            continue
    return np.percentile(stats, [2.5, 97.5])


def perm_test(factory, X, y, groups, task, stat_obs, B=B_PERM):
    # label permutation under the identical CV scheme
    hits = 0
    for b in range(B):
        yp = rng.permutation(y)
        pv = oof_preds(factory, X, yp, groups, task, n_repeats=1, seed=b)
        if task == 'clf':
            hits += roc_auc_score(yp, pv) >= stat_obs
        else:
            hits += mean_absolute_error(yp, pv) <= stat_obs
    return (1 + hits) / (B + 1)


def validate_clf(df, out):
    factory, feats, name, sname = winner('clf')
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    X, y, per = d[feats].values, d.responder.values, d.person.values
    prob = oof_preds(factory, X, y, per, 'clf')
    auc = roc_auc_score(y, prob)
    ci = cluster_boot_ci(roc_auc_score, y, prob, per)
    brier = brier_score_loss(y, prob)
    p_perm = perm_test(factory, X, y, per, 'clf',
                       roc_auc_score(y, oof_preds(factory, X, y, per, 'clf', 1)))
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=5, strategy='quantile')
    thr = np.arange(0.05, 0.61, 0.01)
    prev, n = y.mean(), len(y)
    nb_model = [(((prob >= t) & (y == 1)).sum() / n) - (((prob >= t) & (y == 0)).sum() / n) * t / (1 - t) for t in thr]
    nb_all = [prev - (1 - prev) * t / (1 - t) for t in thr]
    out['clf'] = dict(auc_oof=auc, ci=list(ci), brier=brier, p_perm=p_perm,
                      cal=dict(frac_pos=list(frac_pos), mean_pred=list(mean_pred)),
                      dca=dict(thr=list(thr), model=nb_model, all=nb_all), n=int(n), prev=float(prev))
    print(f'CLF ({name} {sname}): OOF AUC={auc:.3f}  95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]  '
          f'Brier={brier:.3f}  permutation p={p_perm:.4f}')


def validate_reg(df, out):
    factory, feats, name, sname = winner('reg')
    d = df.dropna(subset=['post_6mwt']).reset_index(drop=True)
    X, y, per = d[feats].values, d.post_6mwt.values, d.person.values
    pred = oof_preds(factory, X, y, per, 'reg')
    mae = mean_absolute_error(y, pred)
    ci = cluster_boot_ci(mean_absolute_error, y, pred, per)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p_perm = perm_test(factory, X, y, per, 'reg',
                       mean_absolute_error(y, oof_preds(factory, X, y, per, 'reg', 1)), B=500)
    out['reg'] = dict(mae_oof=mae, ci=list(ci), r2_oof=r2, p_perm=p_perm)
    print(f'REG ({name} {sname}): OOF MAE={mae:.1f} m  95% CI=[{ci[0]:.1f}, {ci[1]:.1f}]  '
          f'OOF R2={r2:.2f}  permutation p={p_perm:.4f}')


def ablation(df, out):
    # how much signal survives without the treatment-arm indicators
    factory, feats, _, _ = winner('clf')
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    y, per = d.responder.values, d.person.values

    def cv_scores(cols):
        X = d[cols].values
        f1s, aucs = [], []
        for tr, te in grouped_folds(y, per, 5, 5, 'clf', SEED):
            m = factory()
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan)
            f1s.append(f1_score(y[te], (p >= 0.5).astype(int), zero_division=0))
        return dict(AUC=float(np.nanmean(aucs)), AUC_sd=float(np.nanstd(aucs)),
                    F1=float(np.nanmean(f1s)), F1_sd=float(np.nanstd(f1s)))

    out['ablation'] = {'with_arm': cv_scores(feats), 'without_arm': cv_scores(feats[2:]),
                       'arm_only': cv_scores(feats[:2])}
    for k, v in out['ablation'].items():
        print(f"ablation {k}: AUC={v['AUC']:.3f}±{v['AUC_sd']:.2f} F1={v['F1']:.3f}±{v['F1_sd']:.2f}")


def counterfactual_boot(df, out, B=500):
    # person-clustered bootstrap of the mean arm contrast (refit each draw)
    factory, feats, _, _ = winner('clf')
    d = df.dropna(subset=['responder']).reset_index(drop=True)
    X, y, per = d[feats].values, d.responder.values, d.person.values
    Xall = df[feats].values
    boot_rng = np.random.RandomState(SEED)
    uniq = d.person.unique()
    rows_of = {u: np.flatnonzero(per == u) for u in uniq}
    dIH, dIC, all_imt = [], [], 0
    for _ in range(B):
        pick = boot_rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_of[u] for u in pick])
        m = factory()
        m.fit(X[rows], y[rows])
        probs = {}
        for arm, (gi, gh) in {'IMT': (1, 0), 'HIIT': (0, 1), 'Control': (0, 0)}.items():
            Xc = Xall.copy()
            Xc[:, 0], Xc[:, 1] = gi, gh
            probs[arm] = m.predict_proba(Xc)[:, 1]
        dIH.append(float(np.mean(probs['IMT'] - probs['HIIT'])))
        dIC.append(float(np.mean(probs['IMT'] - probs['Control'])))
        all_imt += bool(np.all((probs['IMT'] >= probs['HIIT']) & (probs['IMT'] >= probs['Control'])))
    out['counterfactual'] = dict(
        dP_IMT_HIIT=dict(mean=float(np.mean(dIH)), ci=[float(x) for x in np.percentile(dIH, [2.5, 97.5])]),
        dP_IMT_Control=dict(mean=float(np.mean(dIC)), ci=[float(x) for x in np.percentile(dIC, [2.5, 97.5])]),
        NNT_vs_Control=dict(mean=float(1 / np.mean(dIC)),
                            ci=[float(1 / np.percentile(dIC, 97.5)), float(1 / np.percentile(dIC, 2.5))]),
        pct_boot_IMT_best_for_all=float(all_imt / B * 100), B=B)
    c = out['counterfactual']
    print(f"counterfactual: dP(IMT-HIIT)={c['dP_IMT_HIIT']['mean']:.2f} "
          f"dP(IMT-Control)={c['dP_IMT_Control']['mean']:.2f} NNT={c['NNT_vs_Control']['mean']:.1f} "
          f"IMT best for all in {c['pct_boot_IMT_best_for_all']:.0f}% of bootstraps")


def run():
    df = load_data()
    out = {}
    validate_clf(df, out)
    validate_reg(df, out)
    ablation(df, out)
    counterfactual_boot(df, out)
    os.makedirs(METRICS, exist_ok=True)
    path = os.path.join(METRICS, 'validation_results.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'saved: {path}')
    return out


if __name__ == '__main__':
    run()
