"""Synthetic-data tests; no patient data required."""
import numpy as np
import pytest
from core import (grouped_folds, clf_scores, reg_scores, zoo_factory, winner,
                  FULL16, COMPACT8, GROUP_ONLY, PRE_GROUP, FEATURE_SETS, WINNERS)


def _groups(n):
    # persons g0..g14, some enrolled twice
    return np.array([f'g{i % 15}' for i in range(n)])


@pytest.mark.parametrize('task', ['clf', 'reg'])
def test_grouped_folds_keep_persons_together(task):
    n = 40
    y = np.tile([0.0, 1.0], n // 2) if task == 'clf' else np.random.RandomState(0).randn(n)
    g = _groups(n)
    for tr, te in grouped_folds(y, g, 5, 2, task, seed=0):
        assert set(g[tr]).isdisjoint(set(g[te]))
        assert len(tr) + len(te) == n


def test_grouped_folds_deterministic():
    y = np.tile([0.0, 1.0], 20)
    g = _groups(40)
    a = [(list(tr), list(te)) for tr, te in grouped_folds(y, g, 5, 2, 'clf', seed=0)]
    b = [(list(tr), list(te)) for tr, te in grouped_folds(y, g, 5, 2, 'clf', seed=0)]
    assert a == b


def test_clf_scores_perfect():
    y = np.array([0, 1, 0, 1, 1])
    s = clf_scores(y, y.astype(float))
    assert s['AUC'] == 1.0 and s['ACC'] == 1.0 and s['BRIER'] == 0.0


def test_reg_scores_perfect():
    y = np.array([400.0, 450.0, 500.0, 380.0])
    s = reg_scores(y, y.copy())
    assert s['R2'] == 1.0 and s['MAE'] == 0.0


def test_feature_sets_consistent():
    assert set(GROUP_ONLY) <= set(COMPACT8) <= set(FULL16)
    assert set(PRE_GROUP) <= set(FULL16)
    assert list(FEATURE_SETS) == ['group-only', 'compact-8', 'full-16', 'base+group']


def test_zoo_factory_and_winner():
    for task in ('clf', 'reg'):
        factory, feats, mname, sname = winner(task)
        assert (mname, sname) == WINNERS[task]
        assert feats == FEATURE_SETS[sname]
        assert hasattr(factory(), 'fit')
    with pytest.raises(ValueError):
        zoo_factory('clf', 'no-such-model')
