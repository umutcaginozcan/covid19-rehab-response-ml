"""Shared engine: data loading, feature sets, person-grouped CV, model pool.

Adapted copy of ml_6mwt/code/core.py (adds COMPACT5); WINNERS and the regression half are
unused in this project (the VO2 scoreboard picks its winner from the results).
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score, precision_score, matthews_corrcoef,
                             brier_score_loss, r2_score, mean_absolute_error, mean_squared_error)

SEED = 0
N_SELECT = 6
_HERE = os.path.dirname(os.path.abspath(__file__))
SAV = os.environ.get('COVID_SAV', os.path.join(
    _HERE, '..', '..', 'data', 'COVID19_Alldata_set_Gazi University (1).sav'))
_RES = os.path.join(_HERE, '..', 'results')
METRICS = os.path.join(_RES, 'metrics')
FIGURES = os.path.join(_RES, 'figures')

CLF_METRICS = ['AUC', 'ACC', 'BAL', 'F1', 'SENS', 'SPEC', 'PPV', 'NPV', 'MCC', 'BRIER']
REG_METRICS = ['R2', 'MAE', 'RMSE', 'PEARSON']

FULL16 = ['grp_IMT', 'grp_HIIT', 'Age', 'female', 'BKI', 'COVID_19_time',
          'ct_gt50', 'hospitalized', 'charlson', 'ever_smoker', 'exercise_habit',
          'MIP_BT', 'MEP_BT', 'SixMWT_meter_BT', 'VO2KG_max_BT',
          'Quadriceps_right_BT', 'FSS_BT', 'SGQR_total_score_BT']
COMPACT8 = ['grp_IMT', 'grp_HIIT', 'SixMWT_meter_BT', 'MIP_BT', 'VO2KG_max_BT',
            'FSS_BT', 'SGQR_total_score_BT', 'Age', 'BKI']
GROUP_ONLY = ['grp_IMT', 'grp_HIIT']
PRE_GROUP = ['grp_IMT', 'grp_HIIT', 'SixMWT_meter_BT']
# activity-based set for VO2 response; selection evidence archived in
# vo2_explore_archive_2026-08-16.zip (project root, kept out of the repo)
COMPACT5 = ['grp_IMT', 'grp_HIIT', 'Step_count_BT', 'Physical_activity_duration_BT',
            'SixMWT_meter_BT', 'exercise_habit']
FEATURE_SETS = {'group-only': GROUP_ONLY, 'compact-5': COMPACT5, 'compact-8': COMPACT8,
                'full-16': FULL16, 'base+group': PRE_GROUP}

# Selected winners (model name, feature set). clf primary metric: F1; reg: R2.
# clf switched SVM->LR 2026-08-16: SVM collapses outside the cohort range (deployment).
WINNERS = {'clf': ('Logistic regression', 'compact-8'), 'reg': ('Linear regression', 'base+group')}


def _person_key(name):
    # same person may appear twice (enrolled in two arms)
    s = str(name).lower()
    s = re.sub(r'_(imt|hiit|kontrol|control)$', '', s)
    for a, b in zip('ığşçöü', 'igscou'):
        s = s.replace(a, b)
    return re.sub(r'[^a-z]', '', s)


def load_data():
    df, _ = pyreadstat.read_sav(SAV)
    df.columns = [unicodedata.normalize('NFKD', c).replace('‎', '') for c in df.columns]
    idcol = df.columns[0]
    df['person'] = df[idcol].map(_person_key)
    df['grp_IMT'] = (df.Group == 1.0).astype(int)
    df['grp_HIIT'] = (df.Group == 3.0).astype(int)
    df['female'] = (df.Gender == 1.0).astype(int)
    df['ct_gt50'] = (df.Pulmonary_involvement_percentage == 2.0).astype(int)
    df['hospitalized'] = (df.Hospitialization == 1.0).astype(int)
    df['ever_smoker'] = df.Historyofsmoking.isin([1.0, 2.0]).astype(int)
    df['exercise_habit'] = (df.Exercise_habit == 1.0).astype(int)
    df['charlson'] = df.Charlsoncomorbiditeindex_point
    # targets
    df['delta_6mwt'] = df.SixMWT_meter_AT - df.SixMWT_meter_BT
    df['responder'] = (df.delta_6mwt >= 25).astype(float)
    df.loc[df.delta_6mwt.isna(), 'responder'] = np.nan
    df['post_6mwt'] = df.SixMWT_meter_AT
    return df


def full_bt_features(df):
    # all clean baseline columns (<=30% missing)
    feats = ['grp_IMT', 'grp_HIIT', 'Age', 'female', 'BKI', 'COVID_19_time',
             'ct_gt50', 'hospitalized', 'charlson', 'ever_smoker', 'exercise_habit',
             'Shoulder_abductor_right_AT']  # baseline measure mislabeled _AT in the SAV (true post columns are Shoulderabductor_*_AT)
    for c in df.columns:
        if not c.endswith('_BT'):
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].isna().mean() > 0.30:
            continue
        feats.append(c)
    return feats


def grouped_folds(y, groups, n_splits, n_repeats, task, seed=0):
    # person-grouped folds: both records of an individual stay in the same fold
    idx_all = np.arange(len(y))
    for rep in range(n_repeats):
        if task == 'clf':
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
            yield from sgkf.split(idx_all.reshape(-1, 1), y, groups)
        else:
            rng = np.random.RandomState(seed + rep)
            ug = rng.permutation(pd.unique(groups))
            fold_of = {g: i % n_splits for i, g in enumerate(ug)}
            fmap = np.array([fold_of[g] for g in groups])
            for f in range(n_splits):
                yield idx_all[fmap != f], idx_all[fmap == f]


def pipe(model, scale=True):
    steps = [('imp', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('sc', StandardScaler()))
    return Pipeline(steps + [('m', model)])


def ttr(model):
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


CLF_ZOO = [
    ('Linear',   'Logistic regression', lambda: pipe(LogisticRegression(max_iter=5000))),
    ('Linear',   'Elastic-net logistic', lambda: pipe(GridSearchCV(
        LogisticRegression(penalty='elasticnet', solver='saga', max_iter=8000, random_state=SEED),
        {'C': [0.03, 0.3, 3.0], 'l1_ratio': [0.2, 0.5, 0.8]}, cv=3, scoring='roc_auc', n_jobs=-1))),
    ('Kernel',   'SVM (RBF)', lambda: pipe(SVC(probability=True, random_state=SEED))),
    ('Instance', 'k-NN (k=7)', lambda: pipe(KNeighborsClassifier(7))),
    ('Bayes',    'Gaussian Naive Bayes', lambda: pipe(GaussianNB())),
    ('Tree',     'Decision tree', lambda: pipe(DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=3, random_state=SEED))),
    ('Bagging',  'Random Forest', lambda: pipe(RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=SEED, n_jobs=-1))),
    ('Bagging',  'Extra Trees', lambda: pipe(ExtraTreesClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=SEED, n_jobs=-1))),
    ('Boosting', 'XGBoost', lambda: pipe(XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, eval_metric='logloss',
        verbosity=0, random_state=SEED))),
    ('Boosting', 'LightGBM', lambda: pipe(LGBMClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=5, verbosity=-1, random_state=SEED))),
    ('Boosting', 'Hist. gradient boosting', lambda: pipe(HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED))),
    ('Boosting', 'AdaBoost', lambda: pipe(AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=SEED))),
    ('Neural',   'MLP (16)', lambda: pipe(MLPClassifier(
        hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000, random_state=SEED))),
]

REG_ZOO = [
    ('Linear',   'Linear regression', lambda: pipe(LinearRegression())),
    ('Linear',   'Ridge', lambda: pipe(GridSearchCV(
        Ridge(), {'alpha': [0.1, 1.0, 10.0, 100.0]}, cv=3, scoring='r2', n_jobs=-1))),
    ('Kernel',   'SVR (RBF)', lambda: pipe(ttr(SVR(C=1.0)))),
    ('Instance', 'k-NN (k=7)', lambda: pipe(KNeighborsRegressor(7))),
    ('Tree',     'Decision tree', lambda: pipe(DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=3, random_state=SEED))),
    ('Bagging',  'Random Forest', lambda: pipe(RandomForestRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=SEED, n_jobs=-1))),
    ('Bagging',  'Extra Trees', lambda: pipe(ExtraTreesRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=SEED, n_jobs=-1))),
    ('Boosting', 'XGBoost', lambda: pipe(XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, verbosity=0, random_state=SEED))),
    ('Boosting', 'LightGBM', lambda: pipe(LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=5, verbosity=-1, random_state=SEED))),
    ('Boosting', 'Hist. gradient boosting', lambda: pipe(HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED))),
    ('Boosting', 'AdaBoost', lambda: pipe(AdaBoostRegressor(
        n_estimators=200, learning_rate=0.5, random_state=SEED))),
    ('Neural',   'MLP (16)', lambda: pipe(ttr(MLPRegressor(
        hidden_layer_sizes=(16,), alpha=1.0, max_iter=5000, random_state=SEED)))),
]


def zoo_factory(task, model_name):
    zoo = CLF_ZOO if task == 'clf' else REG_ZOO
    for _, name, factory in zoo:
        if name == model_name:
            return factory
    raise ValueError(model_name)


def winner(task):
    model_name, set_name = WINNERS[task]
    return zoo_factory(task, model_name), FEATURE_SETS[set_name], model_name, set_name


def clf_scores(yt, p):
    yh = (p >= 0.5).astype(int)
    return dict(AUC=roc_auc_score(yt, p) if len(set(yt)) > 1 else np.nan,
                ACC=accuracy_score(yt, yh), BAL=balanced_accuracy_score(yt, yh),
                F1=f1_score(yt, yh, zero_division=0),
                SENS=recall_score(yt, yh, zero_division=0),
                SPEC=recall_score(yt, yh, pos_label=0, zero_division=0),
                PPV=precision_score(yt, yh, zero_division=0),
                NPV=precision_score(yt, yh, pos_label=0, zero_division=0),
                MCC=matthews_corrcoef(yt, yh), BRIER=brier_score_loss(yt, p))


def reg_scores(yt, p):
    r = np.corrcoef(yt, p)[0, 1] if np.std(p) > 0 else np.nan
    return dict(R2=r2_score(yt, p), MAE=mean_absolute_error(yt, p),
                RMSE=float(np.sqrt(mean_squared_error(yt, p))), PEARSON=r)


def fold_selection(Xfull, y, tr, task):
    # fold-internal selection from the full baseline pool
    prep = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    Xtr = prep.fit_transform(Xfull[tr])
    lin = LogisticRegression(max_iter=5000) if task == 'clf' else Ridge(alpha=1.0)
    sfs = SequentialFeatureSelector(lin, n_features_to_select=N_SELECT,
                                    direction='forward', cv=3, n_jobs=-1)
    sfs.fit(Xtr, y[tr])
    fwd = np.flatnonzero(sfs.get_support())
    rfe = RFE(lin, n_features_to_select=N_SELECT, step=10)
    rfe.fit(Xtr, y[tr])
    bwd = np.flatnonzero(rfe.support_)
    return fwd, bwd


def benchmark(d, task, target, zoo, base_sets, metrics, header, fullbt):
    # factorial comparison: every model x 5 feature sets
    y, groups = d[target].values, d.person.values
    Xfull = d[fullbt].values
    print(f'\n===== {header} (n={len(d)}) =====')
    folds = list(grouped_folds(y, groups, 5, 5, task, SEED))
    # selections computed once per fold
    sels, freq = [], {'fwd': {}, 'bwd': {}}
    for tr, te in folds:
        fwd, bwd = fold_selection(Xfull, y, tr, task)
        sels.append((fwd, bwd))
        for i in fwd:
            freq['fwd'][fullbt[i]] = freq['fwd'].get(fullbt[i], 0) + 1
        for i in bwd:
            freq['bwd'][fullbt[i]] = freq['bwd'].get(fullbt[i], 0) + 1
    set_names = [n for n, _ in base_sets] + [f'full-211 → forward-{N_SELECT}',
                                             f'full-211 → backward-{N_SELECT} (RFE)']
    print(f'{"family":9} {"model":24} {"feats":28} ' + ' '.join(f'{m:>11}' for m in metrics))
    out = []
    for family, mname, factory in zoo:
        for si, sname in enumerate(set_names):
            scores = []
            for fi, (tr, te) in enumerate(folds):
                if si < len(base_sets):
                    X = d[base_sets[si][1]].values
                else:
                    idx = sels[fi][si - len(base_sets)]
                    X = Xfull[:, idx]
                m = factory()
                m.fit(X[tr], y[tr])
                if task == 'clf':
                    scores.append(clf_scores(y[te], m.predict_proba(X[te])[:, 1]))
                else:
                    scores.append(reg_scores(y[te], m.predict(X[te])))
            keys = list(scores[0].keys())
            arr = np.array([[s[k] for k in keys] for s in scores], dtype=float)
            mean = dict(zip(keys, np.nanmean(arr, axis=0)))
            sd = dict(zip(keys, np.nanstd(arr, axis=0)))
            cells = ' '.join(f'{mean[k]:5.2f}±{sd[k]:4.2f}' for k in metrics)
            print(f'{family:9} {mname:24} {sname:28} {cells}')
            out.append(dict(family=family, model=mname, feats=sname, mean=mean, sd=sd))
    for k in ('fwd', 'bwd'):
        top = sorted(freq[k].items(), key=lambda kv: -kv[1])[:8]
        print(f'  selected[{k}]: ' + ', '.join(f'{a}({b})' for a, b in top))
    return out, freq


def save_zoo(key, out, freq):
    # merge this task's scoreboard into the shared metrics file
    import json
    os.makedirs(METRICS, exist_ok=True)
    path = os.path.join(METRICS, 'zoo_results.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[key], res[key + '_sel'] = out, freq
    with open(path, 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f'saved: {path}')
