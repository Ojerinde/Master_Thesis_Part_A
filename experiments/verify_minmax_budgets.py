"""
Verification harness for the min-max unification.

Confirms, WITHOUT needing trained models, that after the StandardScaler ->
MinMaxScaler change every fixed-budget attack (FGSM, PGD, DLSA, SNA, TPA)
realizes an L-inf perturbation <= epsilon IN THE MIN-MAX [0,1] SPACE, so the
budgets are directly comparable to the decision-based boundary attack and to the
epsilon values the paper defines. Also asserts the loader now returns a
MinMaxScaler mapping the train block into [0,1].

The gradient attacks use a randomly-initialised model: the epsilon projection is
model-independent (delta is clipped to +/-eps regardless of the gradient), so a
random net is sufficient to verify the budget geometry.

Run: PYTHONPATH=. python experiments/verify_minmax_budgets.py
"""
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.preprocessing import MinMaxScaler                     # noqa: E402
from data.loader import load_track_splits                          # noqa: E402
from config.model_configs import get_config                        # noqa: E402
from models.deep_learning import CNN1DModel                        # noqa: E402
from attacks.fgsm import FGSMAttack                                # noqa: E402
from attacks.pgd import PGDAttack                                  # noqa: E402
from attacks.gnss_attacks import (                                 # noqa: E402
    DataLocationShiftAttack, SimilarityNoiseAttack, TemporalPatternAttack)
from utils.gnss_constraints import GNSSConstraintEnforcer          # noqa: E402

EPS = [0.05, 0.10, 0.20]
TOL = 1e-6
N = 512


def main():
    (Xtr, Xv, Xte, ytr, yv, yte, fn, sc) = load_track_splits(verbose=False)

    # 1) loader now returns a MinMaxScaler that maps the train block to [0,1].
    assert isinstance(sc, MinMaxScaler), f"loader scaler is {type(sc).__name__}, expected MinMaxScaler"
    Xtr_s = sc.transform(Xtr)
    assert Xtr_s.min() >= -TOL and Xtr_s.max() <= 1 + TOL, \
        f"train not in [0,1]: min={Xtr_s.min():.4f} max={Xtr_s.max():.4f}"
    print(f"[OK] loader scaler = MinMaxScaler; train range = "
          f"[{Xtr_s.min():.3f}, {Xtr_s.max():.3f}]")

    # sample a scaled batch
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xte), size=min(N, len(Xte)), replace=False)
    Xs = sc.transform(Xte[idx]).astype(np.float32)
    ys = yte[idx].astype(np.int64)

    # fitted enforcer in the min-max space (as in 18_full_eval), plus physical coupling
    enf = GNSSConstraintEnforcer(fn); enf.fit(Xtr_s.astype(np.float32), fn)
    enf.fit_coupling(Xtr, fn)

    # random DL model for the gradient-attack geometry check
    cfg = get_config('cnn_1d'); cfg['input_dim'] = Xs.shape[1]
    m = CNN1DModel(input_dim=Xs.shape[1], config=cfg); m.build_model(); m.model.eval()

    print(f"\n{'attack':16s} " + " ".join(f"eps={e:<6.2f}" for e in EPS) + "  (realized min-max L-inf)")
    worst = 0.0
    for label, mk in [
        ('FGSM',  lambda e: FGSMAttack(m, epsilon=e, gnss_enforcer=enf).generate(Xs, ys)),
        ('PGD',   lambda e: PGDAttack(m, epsilon=e, num_iter=40, random_start=True,
                                      gnss_enforcer=enf).generate(Xs, ys)),
        ('DLSA',  lambda e: DataLocationShiftAttack(shift_scale=e, feature_names=fn,
                                                    gnss_enforcer=enf).generate(Xs, ys)),
        ('SNA',   lambda e: SimilarityNoiseAttack(epsilon=e, gnss_enforcer=enf).fit(Xs).generate(Xs, ys)),
        ('TPA',   lambda e: TemporalPatternAttack(doppler_amp=e, cn0_amp=e * 0.6,
                                                  feature_names=fn, gnss_enforcer=enf).generate(Xs, ys)),
    ]:
        realized = []
        for e in EPS:
            Xa = np.asarray(mk(e), dtype=np.float64)
            linf = float(np.max(np.abs(Xa - Xs)))
            realized.append(linf)
            worst = max(worst, linf - e)
        flag = "OK " if all(r <= e + 1e-4 for r, e in zip(realized, EPS)) else "BAD"
        print(f"[{flag}] {label:11s} " + " ".join(f"{r:<10.4f}" for r in realized))

    assert worst <= 1e-4, f"an attack exceeded its min-max budget by {worst:.6f}"
    print(f"\nALL fixed-budget attacks respect epsilon in min-max space "
          f"(max overshoot {worst:.2e}). Budgets are now unified with the "
          f"decision-based boundary attack (13).")


if __name__ == '__main__':
    main()
