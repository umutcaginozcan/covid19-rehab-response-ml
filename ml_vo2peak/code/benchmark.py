"""Secondary task — VO₂peak response classification: Δ ≥ 1.2 mL·kg⁻¹·min⁻¹.

Threshold anchored to the mean rehabilitation gain reported for ILD (ERS);
update VO2_THRESHOLD if the definitive MCID source says otherwise.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from core import (load_data, full_bt_features, benchmark, save_zoo,
                  CLF_ZOO, CLF_METRICS, GROUP_ONLY, COMPACT5)

VO2_THRESHOLD = 1.2


def run():
    df = load_data()
    dvo2 = df.VO2KG_max_AT - df.VO2KG_max_BT
    df['responder_vo2'] = (dvo2 >= VO2_THRESHOLD).astype(float)
    df.loc[dvo2.isna(), 'responder_vo2'] = np.nan
    fullbt = full_bt_features(df)
    d = df.dropna(subset=['responder_vo2']).reset_index(drop=True)
    out, freq = benchmark(d, 'clf', 'responder_vo2', CLF_ZOO,
                          [('group-only', GROUP_ONLY), ('compact-5', COMPACT5)],
                          CLF_METRICS, f'SECONDARY — VO2peak responder (Δ ≥ {VO2_THRESHOLD})', fullbt)
    save_zoo('vo2', out, freq)
    return out


if __name__ == '__main__':
    run()
