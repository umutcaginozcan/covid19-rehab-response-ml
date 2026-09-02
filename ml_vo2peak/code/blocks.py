"""Block-wise signal scan for VO₂peak response (Table S4).

Physiologically coherent variable blocks are defined MECHANICALLY from column
names (blind to outcome), capped at 12 variables (lowest missingness first).
Every block — plus arm-only and best-single-variable references — is evaluated
with the identical person-grouped repeated 5-fold CV on the trained subset
(IMT+HIIT, where the personalization question lives).
Output: results/metrics/block_results.json
"""
import warnings
warnings.filterwarnings('ignore')
import json
import os
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score
from core import load_data, full_bt_features, grouped_folds, zoo_factory, COMPACT5, METRICS, SEED

VO2_THRESHOLD = 1.2

# (domain, block label, name patterns) — membership is by pattern, not by outcome
BLOCKS = [
    ('Demographics',   'Demographics / anthropometrics', r'^(Age|female|BKI)$'),
    ('History',        'COVID-19 history & comorbidity', r'^(COVID_19_time|ct_gt50|hospitalized|charlson|ever_smoker)$'),
    ('Pulmonary',      'Spirometry & lung volumes',      r'^(FEV1|FVC|PEF|FEF|TLC|RV|VC|DLCO)'),
    ('Muscle',         'Respiratory muscle strength',    r'^(MIP|MEP)'),
    ('Muscle',         'Peripheral muscle strength',     r'^(Quadriceps|Shoulder|Handgrip)'),
    ('CPET',           'Gas exchange (PETO2/PETCO2/RER)', r'^(PETO2|PETCO2|RER|VE_)'),
    ('CPET',           'Cardiovascular response (CPET)', r'^(HR\w*_CPET|O2HR|SBP\d_CPET|DBP\d_CPET)'),
    ('CPET',           'Muscle oxygenation (SmO2)',      r'^SmO2'),
    ('CPET',           'Submaximal / anaerobic profile', r'(anaerobic|resting)'),
    ('Walk test',      '6-MWT physiological response',   r'(_6MWT_BT$)'),
    ('Walk test',      'Walking & endurance capacity',   r'^(SixMWT_meter|SixMWT_percentage|Endurance_time)'),
    ('Activity',       'Daily physical activity',        r'^(Step_count|Physical_activity|Lying_down|exercise_habit)'),
    ('Symptoms/QoL',   'Fatigue & quality of life',      r'^(FSS|SGQR|LCDAL|mMRC|Borg)'),
    ('Fitness',        'Baseline aerobic fitness',       r'^(VO2KG_max_BT|VO2_max)'),
]
CLF_METRICS = ['AUC', 'BAL', 'F1', 'SENS', 'SPEC']


def fold_scores(d, feats, y, per, factory):
    X = d[['grp_IMT', 'grp_HIIT'] + feats].values
    scores = []
    for tr, te in grouped_folds(y, per, 5, 5, 'clf', SEED):
        m = factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yh = (p >= 0.5).astype(int)
        scores.append(dict(
            AUC=roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan,
            BAL=balanced_accuracy_score(y[te], yh), F1=f1_score(y[te], yh, zero_division=0),
            SENS=recall_score(y[te], yh, zero_division=0),
            SPEC=recall_score(y[te], yh, pos_label=0, zero_division=0)))
    mean = {k: float(np.nanmean([s[k] for s in scores])) for k in CLF_METRICS}
    sd = {k: float(np.nanstd([s[k] for s in scores])) for k in CLF_METRICS}
    return mean, sd


def run():
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['y'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'y'] = np.nan
    d = df[df.y.notna() & df.Group.isin([1.0, 3.0])].reset_index(drop=True)
    y, per = d.y.values, d.person.values
    pool = [c for c in full_bt_features(df) if c not in ('grp_IMT', 'grp_HIIT')]
    print(f'trained subset: n={len(d)}, responders={int(y.sum())}, pool={len(pool)} variables')

    out = []
    used = set()
    for domain, label, pat in BLOCKS:
        members = [c for c in pool if re.search(pat, c)]
        if not members:
            print(f'!! empty block: {label}')
            continue
        if len(members) > 12:  # EPV cap: keep the 12 with least missingness, then alphabetical
            members = sorted(members, key=lambda c: (df[c].isna().mean(), c))[:12]
        used.update(members)
        best = None
        for mname, factory in [('Logistic regression', zoo_factory('clf', 'Logistic regression')),
                               ('Elastic-net logistic', zoo_factory('clf', 'Elastic-net logistic'))]:
            mean, sd = fold_scores(d, members, y, per, factory)
            if best is None or mean['AUC'] > best['mean']['AUC']:
                best = dict(family=domain, model=label, feats=f'{len(members)} variables',
                            best_model=mname, members=members, mean=mean, sd=sd)
        out.append(best)
        print(f"{label:34} n={len(members):2} best={best['best_model']:20} AUC={best['mean']['AUC']:.3f}±{best['sd']['AUC']:.2f}")

    # winner set (compact-5) with the published elastic-net
    mean, sd = fold_scores(d, [c for c in COMPACT5 if c not in ('grp_IMT', 'grp_HIIT')], y, per,
                           zoo_factory('clf', 'Elastic-net logistic'))
    out.append(dict(family='Activity', model='compact-5 (selected set)', feats='4 variables',
                    best_model='Elastic-net logistic', members=COMPACT5[2:], mean=mean, sd=sd))
    print(f"{'compact-5 (selected set)':34} n= 4 best=Elastic-net logistic  AUC={mean['AUC']:.3f}±{sd['AUC']:.2f}")

    # reference: arm only
    mean, sd = fold_scores(d, [], y, per, zoo_factory('clf', 'Logistic regression'))
    out.append(dict(family='Reference', model='Treatment arm only', feats='0 variables',
                    best_model='Logistic regression', members=[], mean=mean, sd=sd))
    print(f"{'Treatment arm only':34}       AUC={mean['AUC']:.3f}")

    # reference: best single variable (best of the whole pool — optimistic by construction)
    best_single = None
    for c in pool:
        if d[c].nunique() < 2:
            continue
        mean, sd = fold_scores(d, [c], y, per, zoo_factory('clf', 'Logistic regression'))
        if best_single is None or mean['AUC'] > best_single['mean']['AUC']:
            best_single = dict(family='Reference', model='Best single variable (of 209)', feats=c,
                               best_model='Logistic regression', members=[c], mean=mean, sd=sd)
    out.append(best_single)
    print(f"en iyi tekil: {best_single['feats']} AUC={best_single['mean']['AUC']:.3f}")

    os.makedirs(METRICS, exist_ok=True)
    path = os.path.join(METRICS, 'block_results.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'saved: {path}')
    return out


if __name__ == '__main__':
    run()
