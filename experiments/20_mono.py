"""
Transformer non-monotonicity diagnostic, single-run: PGD recall for the six DL
models at eps in {0.05,0.10,0.20}, WITH and WITHOUT the GNSS enforcer, at the
recall=0.95 operating point. Confirms the non-monotonicity is not an
enforcer-clipping artifact (rules out H1). Output: results/tables/monotonicity.csv
Run: PYTHONPATH=. python experiments/20_mono.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from sklearn.preprocessing import MinMaxScaler                # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.metrics import recall_score                        # noqa: E402
from data.loader import load_track_splits                       # noqa: E402
from config.paths import DL_MODELS, TABLES_DIR                  # noqa: E402
from config.model_configs import get_config                     # noqa: E402
from models.deep_learning import (                               # noqa: E402
    CNN1DModel, LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel, TCNModel)
from attacks.pgd import PGDAttack                                # noqa: E402
from utils.gnss_constraints import GNSSConstraintEnforcer         # noqa: E402

DL = {'CNN-1D': (CNN1DModel, 'cnn_1d'), 'LSTM': (LSTMModel, 'lstm'),
      'BiLSTM': (BiLSTMModel, 'bilstm'), 'CNN-LSTM': (CNNLSTMModel, 'cnn_lstm'),
      'Transformer': (TransformerModel, 'transformer'), 'TCN': (TCNModel, 'tcn')}
EPS = [0.05, 0.10, 0.20]


def flat(p):
    p = np.asarray(p)
    return (p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()).astype(np.float64)


def tau_for(yv, pv):
    best, fb, br = None, 0.5, -1
    for t in np.linspace(0.01, 0.99, 197):
        r = recall_score(yv, (pv >= t).astype(int), zero_division=0)
        if r >= 0.95:
            best = t
        if r > br:
            br, fb = r, t
    return float(best) if best is not None else float(fb)


def main():
    (Xtr_e, Xv_e, Xte_e, _ytr, yv, yte, fn, sc) = load_track_splits(verbose=False)
    Xte_e = Xte_e.astype(np.float64)
    Xtr_s = sc.transform(Xtr_e).astype(np.float32)
    Xv_s = sc.transform(Xv_e).astype(np.float32)
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(yte == c)[0], int(round(4000 * (yte == c).mean())), replace=False) for c in (0, 1)])
    Xs = sc.transform(Xte_e[idx]).astype(np.float32); ys = yte[idx]
    enf = GNSSConstraintEnforcer(fn); enf.fit(Xtr_s, fn)
    enf_phys = GNSSConstraintEnforcer(fn); enf_phys.fit(Xtr_e.astype(np.float32), fn)
    enf_phys.fit_coupling(Xtr_e.astype(np.float64), fn)

    rows = []
    for name, (cls, cfg) in DL.items():
        c = get_config(cfg); c['input_dim'] = Xs.shape[1]
        m = cls(input_dim=Xs.shape[1], config=c); m.build_model()
        m.model.load_state_dict(torch.load(str(DL_MODELS / f'{cfg}.pt'), map_location='cpu'))
        m.model.eval(); m.is_trained = True
        tau = tau_for(yv, flat(m.predict_proba(Xv_s)))
        for mode, e_ in [('With enf.', enf), ('Without enf.', None)]:
            rec = {}
            for eps in EPS:
                adv = PGDAttack(m, epsilon=eps, num_iter=40, random_start=True, gnss_enforcer=e_).generate(Xs, ys)
                if mode == 'With enf.':
                    adv_phys = sc.inverse_transform(adv).astype(np.float64)
                    adv_phys = enf_phys.enforce_coupling_phys(adv_phys)
                    adv = sc.transform(adv_phys).astype(np.float32)
                pa = flat(m.predict_proba(adv.astype(np.float32)))
                rec[eps] = recall_score(ys, (pa >= tau).astype(int), zero_division=0)
            mono = (rec[0.05] >= rec[0.10] >= rec[0.20])
            rows.append(dict(model=name, mode=mode, r05=round(rec[0.05], 3),
                             r10=round(rec[0.10], 3), r20=round(rec[0.20], 3), monotonic=mono))
            print(f"{name:12s} {mode:12s} {rec[0.05]:.3f} {rec[0.10]:.3f} {rec[0.20]:.3f} mono={mono}")
    pd.DataFrame(rows).to_csv(TABLES_DIR / 'monotonicity.csv', index=False)
    print("wrote monotonicity.csv")


if __name__ == '__main__':
    main()
