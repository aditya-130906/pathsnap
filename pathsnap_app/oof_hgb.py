"""Out-of-fold HGB probabilities for TRAIN trips (5-fold GroupKFold by trip, original HGB configuration).

Usage: python -m pathsnap_app.oof_hgb --train work/full_train.csv --output work/train_oof_probs.csv
Each train trip's probabilities come from a model that never saw that trip, so HMM parameters tuned on them
reflect how the decoder behaves on unseen trips (in-sample probabilities are over-confident).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parents[1]
HGB = ROOT / "models" / "histogram_gradient_boosting"
ap = argparse.ArgumentParser(); ap.add_argument("--train", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
feats = json.loads((HGB / "feature_schema.json").read_text())["model_input_features"]
cfg = json.loads((HGB / "config.json").read_text())["config"]
df = pd.read_csv(a.train, dtype={"trip_id": str})
fit = df[df.label_valid == 1]
oof = np.full(len(df), np.nan)
for tr, va in GroupKFold(5).split(df, groups=df.trip_id):
    va_trips = set(df.trip_id.iloc[va])
    f = fit[~fit.trip_id.isin(va_trips)]
    m = HistGradientBoostingClassifier(**cfg).fit(f[feats].to_numpy(np.float32), f.label.to_numpy())
    oof[va] = m.predict_proba(df.iloc[va][feats].to_numpy(np.float32))[:, 1]
df[["trip_id", "observation_index", "candidate_arc_id"]].assign(prob=oof).to_csv(a.output, index=False)
print("written", a.output, len(df))
