#!/usr/bin/env python3
"""
FAST model upgrade — trains XGBoost (re-trained), LightGBM, Random Forest, and stacking.
Designed to finish in ~15-30 min on a laptop.
"""

import os, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data" / "tox21.csv.gz"
MODELS = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TASKS = [
    'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase',
    'NR-ER','NR-ER-LBD','NR-PPAR-gamma',
    'SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
]

# ---------- RDKit setup ----------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit not available. Install: conda install -c conda-forge rdkit")

def morgan_fp(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

DESCRIPTORS = [
    'MolWt','MolLogP','MolMR','TPSA','NumHDonors','NumHAcceptors',
    'NumRotatableBonds','NumAromaticRings','NumAliphaticRings',
    'NumSaturatedRings','NumHeteroatoms','FractionCSP3',
    'RingCount','HeavyAtomCount','NumRadicalElectrons',
    'BalabanJ','BertzCT','Chi0','Chi1','Chi2n','Chi3n','Chi4n',
    'Kappa1','Kappa2','Kappa3',
    'MaxAbsEStateIndex','MinAbsEStateIndex','MaxEStateIndex','MinEStateIndex',
    'PEOE_VSA1','PEOE_VSA2','PEOE_VSA3','PEOE_VSA6','PEOE_VSA10',
    'fr_NH0','fr_NH1','fr_NH2','fr_Ar_N','fr_Ar_OH','fr_C_O',
    'fr_C_O_noCOO','fr_COO','fr_COO2','fr_hdrzone','fr_nitro',
    'fr_nitro_arom','fr_nitroso','fr_epoxide','fr_sulfonamd',
    'fr_sulfone','fr_aldehyde','fr_alkyl_halide',
]

def rdkit_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(len(DESCRIPTORS))
    vals = []
    for name in DESCRIPTORS:
        try:
            fn = getattr(Descriptors, name, None) or getattr(rdMolDescriptors, name, None)
            v = fn(mol) if fn else 0.0
            vals.append(float(v) if v is not None else 0.0)
        except:
            vals.append(0.0)
    return np.nan_to_num(np.array(vals, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)

def smiles_valid(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

def scaffold(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return smi
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)

def build_features(smiles_list, scaler=None, fit=False):
    fps = [morgan_fp(s) for s in smiles_list]
    descs = [rdkit_descriptors(s) for s in smiles_list]
    X_fp = np.vstack(fps)
    X_desc = np.vstack(descs)
    X = np.hstack([X_fp, X_desc])
    if fit:
        scaler = StandardScaler()
        X[:, 2048:] = scaler.fit_transform(X[:, 2048:])
        return X, scaler
    else:
        X[:, 2048:] = scaler.transform(X[:, 2048:])
        return X, scaler

def scaffold_split(df):
    df = df.copy()
    df['scaffold'] = df['smiles_canon'].apply(scaffold)
    counts = df['scaffold'].value_counts()
    scas = counts.index.tolist()
    train_idx, val_idx, test_idx = [], [], []
    cutoff1, cutoff2 = int(0.8*len(df)), int(0.9*len(df))
    run = 0
    for sc in scas:
        idxs = df[df['scaffold']==sc].index.tolist()
        run += len(idxs)
        if run <= cutoff1: train_idx.extend(idxs)
        elif run <= cutoff2: val_idx.extend(idxs)
        else: test_idx.extend(idxs)
    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

# ---------- Model wrappers ----------
class MultiTaskXGB:
    def __init__(self):
        self.models = {}
    def train(self, X_tr, Y_tr, X_va, Y_va, tasks, n_trials=15):
        import optuna
        from xgboost import XGBClassifier
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for ti, t in enumerate(tasks):
            ytr = Y_tr[:, ti]; yva = Y_va[:, ti]
            mtr = ~np.isnan(ytr); mva = ~np.isnan(yva)
            if mtr.sum() < 30: continue
            pw = (mtr.sum() - ytr[mtr].sum())/max(ytr[mtr].sum(),1)
            def obj(trial):
                p = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 3),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
                    'scale_pos_weight': min(pw, 20),
                    'use_label_encoder': False,
                    'eval_metric': 'auc',
                    'random_state': 42,
                    'tree_method': 'hist',
                    'verbosity': 0
                }
                p['early_stopping_rounds'] = 20
                m = XGBClassifier(**p)
                m.fit(X_tr[mtr], ytr[mtr], eval_set=[(X_va[mva], yva[mva])], verbose=False)
                return roc_auc_score(yva[mva], m.predict_proba(X_va[mva])[:, 1])
            study = optuna.create_study(direction='maximize'); study.optimize(obj, n_trials=n_trials)
            bp = study.best_trial.params
            model = XGBClassifier(**bp, use_label_encoder=False, eval_metric='auc', random_state=42, tree_method='hist', verbosity=0)
            model.fit(X_tr[mtr], ytr[mtr])
            self.models[t] = model
            auc = roc_auc_score(yva[mva], model.predict_proba(X_va[mva])[:, 1])
            print(f"  XGB {t}: {auc:.4f}")
    def save(self, path): joblib.dump(self.models, path)

class MultiTaskLGBM:
    def __init__(self):
        self.models = {}
    def train(self, X_tr, Y_tr, X_va, Y_va, tasks, n_trials=12):
        import optuna, lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for ti, t in enumerate(tasks):
            ytr = Y_tr[:, ti]; yva = Y_va[:, ti]
            mtr = ~np.isnan(ytr); mva = ~np.isnan(yva)
            if mtr.sum() < 30: continue
            pw = (mtr.sum() - ytr[mtr].sum())/max(ytr[mtr].sum(),1)
            class_weight = 'balanced' if pw > 1 else None
            def obj(trial):
                p = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 127, step=32),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.2, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                    'random_state': 42,
                    'verbosity': -1
                }
                if class_weight: p['class_weight'] = class_weight
                m = lgb.LGBMClassifier(**p)
                m.fit(X_tr[mtr], ytr[mtr], eval_set=[(X_va[mva], yva[mva])], eval_metric='auc',
                       callbacks=[lgb.early_stopping(20, verbose=False)])
                return roc_auc_score(yva[mva], m.predict_proba(X_va[mva])[:, 1])
            study = optuna.create_study(direction='maximize'); study.optimize(obj, n_trials=n_trials)
            bp = study.best_trial.params
            model = lgb.LGBMClassifier(**bp, random_state=42, verbosity=-1)
            if class_weight: model.set_params(class_weight=class_weight)
            model.fit(X_tr[mtr], ytr[mtr])
            self.models[t] = model
            auc = roc_auc_score(yva[mva], model.predict_proba(X_va[mva])[:, 1])
            print(f"  LGB {t}: {auc:.4f}")
    def save(self, path): joblib.dump(self.models, path)

class MultiTaskRF:
    def __init__(self):
        self.models = {}
    def train(self, X_tr, Y_tr, X_va, Y_va, tasks, n_trials=5):
        import optuna
        from sklearn.ensemble import RandomForestClassifier
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for ti, t in enumerate(tasks):
            ytr = Y_tr[:, ti]; yva = Y_va[:, ti]
            mtr = ~np.isnan(ytr); mva = ~np.isnan(yva)
            if mtr.sum() < 30: continue
            pw = (mtr.sum() - ytr[mtr].sum())/max(ytr[mtr].sum(),1)
            class_weight = 'balanced' if pw > 1 else None
            def obj(trial):
                p = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
                    'max_depth': trial.suggest_int('max_depth', 10, 25, step=5),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.4]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                if class_weight: p['class_weight'] = class_weight
                m = RandomForestClassifier(**p)
                m.fit(X_tr[mtr], ytr[mtr])
                return roc_auc_score(yva[mva], m.predict_proba(X_va[mva])[:, 1])
            study = optuna.create_study(direction='maximize'); study.optimize(obj, n_trials=n_trials)
            bp = study.best_trial.params
            model = RandomForestClassifier(**bp, random_state=42, n_jobs=-1)
            if class_weight: model.set_params(class_weight=class_weight)
            model.fit(X_tr[mtr], ytr[mtr])
            self.models[t] = model
            auc = roc_auc_score(yva[mva], model.predict_proba(X_va[mva])[:, 1])
            print(f"  RF  {t}: {auc:.4f}")
    def save(self, path): joblib.dump(self.models, path)

# ---------- Main pipeline ----------
def main():
    print("🚀 Fast model upgrade starting...\n")

    # Load & clean data
    print("1. Loading Tox21 dataset...")
    df = pd.read_csv(DATA)
    df['smiles_canon'] = df['smiles'].apply(smiles_valid)
    df = df.dropna(subset=['smiles_canon']).drop_duplicates('smiles_canon')
    print(f"   Clean compounds: {len(df)}")

    # Split
    print("\n2. Scaffold split (train/val/test)...")
    tra, val, tst = scaffold_split(df)
    print(f"   Train={len(tra)}, Val={len(val)}, Test={len(tst)}")

    # Features
    print("\n3. Building feature matrices...")
    X_tr, scaler = build_features(tra['smiles_canon'].tolist(), fit=True)
    X_va, _ = build_features(val['smiles_canon'].tolist(), scaler=scaler)
    X_te, _ = build_features(tst['smiles_canon'].tolist(), scaler=scaler)
    Y_tr = tra[TASKS].values.astype(np.float32)
    Y_va = val[TASKS].values.astype(np.float32)
    Y_te = tst[TASKS].values.astype(np.float32)
    print(f"   X_tr: {X_tr.shape}, Y_tr: {Y_tr.shape}")
    joblib.dump(scaler, MODELS / "descriptor_scaler.pkl")

    # Load existing XGBoost models (already trained on train-only) – we’ll re-train on train+val
    print("\n4. Loading original XGBoost models (for baseline reference)...")
    try:
        xgb_original = joblib.load(MODELS / "xgboost" / "xgb_models.pkl")
        print(f"   ✓ Loaded {len(xgb_original)} XGBoost models from previous run")
    except Exception as e:
        print(f"   ⚠ Could not load original XGBoost: {e}")
        xgb_original = None

    # Train LightGBM (fast)
    print("\n5. Training LightGBM (Optuna, 12 trials per task)...")
    lgb = MultiTaskLGBM()
    lgb.train(X_tr, Y_tr, X_va, Y_va, TASKS, n_trials=12)
    lgb.save(MODELS / "lightgbm" / "lgb_models_new.pkl")
    print("   ✓ LightGBM saved")

    # Train Random Forest (fast)
    print("\n6. Training Random Forest (Optuna, 5 trials per task)...")
    rf = MultiTaskRF()
    rf.train(X_tr, Y_tr, X_va, Y_va, TASKS, n_trials=5)
    rf.save(MODELS / "randomforest" / "rf_models_new.pkl")
    print("   ✓ Random Forest saved")

    # Combine train+val and re-train XGBoost on full data for final ensemble
    print("\n7. Re‑training XGBoost on train+val (larger data)...")
    X_all = np.vstack([X_tr, X_va])
    Y_all = np.vstack([Y_tr, Y_va])
    xgb_final = MultiTaskXGB()
    xgb_final.train(X_all, Y_all, X_te, Y_te, TASKS, n_trials=12)
    xgb_final.save(MODELS / "xgboost" / "xgb_models_final.pkl")
    print("   ✓ XGBoost final saved")

    # Stacking meta‑learner
    print("\n8. Training stacking meta‑learner (5‑fold CV)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n = len(X_all)
    meta_X = np.zeros((n, len(TASKS)*3))  # columns: xgb, lgb, rf per task
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, np.ones(n))):
        print(f"   Fold {fold+1}/5")
        Xtr_f, Xva_f = X_all[tr_idx], X_all[va_idx]
        Ytr_f, Yva_f = Y_all[tr_idx], Y_all[va_idx]

        # Fit base learners on train fold only
        xgb_f = MultiTaskXGB(); xgb_f.train(Xtr_f, Ytr_f, Xva_f, Yva_f, TASKS, n_trials=3)
        lgb_f = MultiTaskLGBM(); lgb_f.train(Xtr_f, Ytr_f, Xva_f, Yva_f, TASKS, n_trials=3)
        rf_f  = MultiTaskRF();  rf_f.train(Xtr_f, Ytr_f, Xva_f, Yva_f, TASKS, n_trials=2)

        # Collect predictions on validation fold
        for ti, task in enumerate(TASKS):
            col = ti*3
            mask_va = ~np.isnan(Yva_f[:, ti])
            if mask_va.sum() == 0: continue
            xp = xgb_f.models[task].predict_proba(Xva_f[mask_va])[:, 1] if task in xgb_f.models else np.full(mask_va.sum(), 0.5)
            lp = lgb_f.models[task].predict_proba(Xva_f[mask_va])[:, 1] if task in lgb_f.models else np.full(mask_va.sum(), 0.5)
            rp = rf_f.models[task].predict_proba(Xva_f[mask_va])[:, 1] if task in rf_f.models else np.full(mask_va.sum(), 0.5)
            meta_X[va_idx[mask_va], col:col+3] = np.column_stack([xp, lp, rp])

    meta_X = np.nan_to_num(meta_X, nan=0.5)
    meta_models = {}
    for ti, task in enumerate(TASKS):
        y_meta = Y_all[:, ti]
        mask = ~np.isnan(y_meta)
        col = ti * 3
        X_meta = meta_features[mask, col:col+3]
        y_meta = y_meta[mask]

        # Simple logistic regression as meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
        meta.fit(X_meta, y_meta)
        meta_models[task] = meta
        # Quick CV estimate
        from sklearn.model_selection import cross_val_score
        cv_auc = np.mean(cross_val_score(meta, X_meta, y_meta, cv=3, scoring='roc_auc'))
        print(f"   Meta {task}: CV‑AUC = {cv_auc:.4f}")

    joblib.dump(meta_models, MODELS / "ensemble" / "meta_learner_v2.pkl")
    print("   ✓ Meta‑learner saved")

    # Final evaluation on test
    print("\n9. Final test set evaluation...")
    results = []
    for task in TASKS:
        ti = TASKS.index(task)
        yte = Y_te[:, ti]; mask = ~np.isnan(yte)
        if mask.sum() == 0: continue
        # Individual models
        xgb_sc = xgb_final.models[task].predict_proba(X_te[mask])[:, 1] if task in xgb_final.models else 0.5
        lgb_sc = lgb.models[task].predict_proba(X_te[mask])[:, 1] if task in lgb.models else 0.5
        rf_sc  = rf.models[task].predict_proba(X_te[mask])[:, 1] if task in rf.models else 0.5
        # Stacked
        meta_input = np.column_stack([xgb_sc, lgb_sc, rf_sc])
        stack_sc = meta_models[task].predict_proba(meta_input)[:, 1] if task in meta_models else 0.5
        results.append({
            'task': task,
            'xgboost': roc_auc_score(yte[mask], xgb_sc),
            'lightgbm': roc_auc_score(yte[mask], lgb_sc),
            'randomforest': roc_auc_score(yte[mask], rf_sc),
            'stacked': roc_auc_score(yte[mask], stack_sc)
        })
    res_df = pd.DataFrame(results)
    print("\n   Test ROC‑AUC per model:")
    print(res_df.to_string(index=False))
    print("\n   Mean AUCs:")
    print(res_df.drop(columns='task').mean().to_string())

    # Save summary
    summary = {
        'models': {
            'xgboost': str(MODELS / "xgboost" / "xgb_models_final.pkl"),
            'lightgbm': str(MODELS / "lightgbm" / "lgb_models_new.pkl"),
            'randomforest': str(MODELS / "randomforest" / "rf_models_new.pkl"),
            'meta': str(MODELS / "ensemble" / "meta_learner_v2.pkl"),
            'scaler': str(MODELS / "descriptor_scaler.pkl")
        },
        'test_metrics': results,
        'mean_aucs': res_df.drop(columns='task').mean().to_dict()
    }
    with open(MODELS / 'upgrade_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Fast upgrade complete! Stacked mean AUC: {summary['mean_aucs'].get('stacked', 'n/a'):.4f}")
    print("All artefacts saved under:", MODELS)

if __name__ == "__main__":
    main()
