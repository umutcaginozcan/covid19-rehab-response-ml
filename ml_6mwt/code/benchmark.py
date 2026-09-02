"""Scoreboard: every model × 5 feature sets, person-grouped repeated 5-fold CV.

Usage: python benchmark.py [clf|reg|all]   (default: all)
Feature sets per task: group-only (clf) or base+group (reg) / compact-8 /
full-16 / full-211 → forward-6 / full-211 → backward-6 (RFE), with selection
performed inside each training fold.
"""
import sys
import warnings
warnings.filterwarnings('ignore')
from core import (load_data, full_bt_features, benchmark, save_zoo,
                  CLF_ZOO, REG_ZOO, CLF_METRICS, REG_METRICS,
                  GROUP_ONLY, COMPACT8, FULL16, PRE_GROUP)


def run(task='all'):
    df = load_data()
    fullbt = full_bt_features(df)
    if task in ('clf', 'all'):
        d = df.dropna(subset=['responder']).reset_index(drop=True)
        out, freq = benchmark(d, 'clf', 'responder', CLF_ZOO,
                              [('group-only', GROUP_ONLY), ('compact-8', COMPACT8), ('full-16', FULL16)],
                              CLF_METRICS, 'RESPONDER (Δ 6-MWT ≥ 25 m)', fullbt)
        save_zoo('clf', out, freq)
    if task in ('reg', 'all'):
        d = df.dropna(subset=['post_6mwt']).reset_index(drop=True)
        out, freq = benchmark(d, 'reg', 'post_6mwt', REG_ZOO,
                              [('base+group', PRE_GROUP), ('compact-8', COMPACT8), ('full-16', FULL16)],
                              REG_METRICS, 'POST-TRAINING 6-MWT DISTANCE (m)', fullbt)
        save_zoo('reg', out, freq)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'all')
