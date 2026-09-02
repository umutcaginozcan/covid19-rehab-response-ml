"""Task 3 — counterfactual treatment selection (S-learner what-if)."""
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from core import load_data, winner, METRICS


def run():
    df = load_data()
    factory, feats, _, _ = winner('clf')
    train = df.dropna(subset=['responder'])
    m = factory()
    m.fit(train[feats].values, train.responder.values)

    # each patient under all three scenarios
    rows = []
    for i, r in df.iterrows():
        probs = {}
        for arm, (gi, gh) in {'IMT': (1, 0), 'HIIT': (0, 1), 'Control': (0, 0)}.items():
            x = r[feats].copy()
            x['grp_IMT'], x['grp_HIIT'] = gi, gh
            probs[arm] = float(m.predict_proba(x.values.reshape(1, -1))[0, 1])
        rec = max(('IMT', 'HIIT'), key=lambda a: probs[a])  # control never recommended
        rows.append(dict(patient=f'#{i+1:02d}', actual={1.0: 'IMT', 3.0: 'HIIT', 2.0: 'Control'}[r.Group],
                         p_IMT=round(probs['IMT'], 3), p_HIIT=round(probs['HIIT'], 3),
                         p_Control=round(probs['Control'], 3), recommended=rec,
                         margin=round(probs['IMT'] - probs['HIIT'], 3)))
    w = pd.DataFrame(rows)
    os.makedirs(METRICS, exist_ok=True)
    path = os.path.join(METRICS, 'whatif_recommendations.csv')
    w.to_csv(path, index=False)
    print('recommendations:', w.recommended.value_counts().to_dict())
    print(f'mean P(responder): IMT={w.p_IMT.mean():.2f}  HIIT={w.p_HIIT.mean():.2f}  Control={w.p_Control.mean():.2f}')
    big = w[w.margin.abs() >= 0.15]
    print(f'patients with |ΔP| >= 0.15: {len(big)} (favoring IMT {int((big.margin>0).sum())}, HIIT {int((big.margin<0).sum())})')
    print(f'saved: {path}')
    return w


if __name__ == '__main__':
    run()
