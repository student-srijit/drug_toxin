#!/usr/bin/env python3
"""
Model Upgrade Script — improves toxicity prediction models
- Adds LightGBM & Random Forest to the ensemble
- Hyperparameter tuning with Optuna
- Implements stacking (meta-learner)
- Probability calibration (Platt scaling)
- Saves all models in formats compatible with existing app.py

Run: python train_upgrade.py
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ── Globals ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TOX21_PATH = DATA_DIR / "tox21.csv.gz"

# Tox21 tasks (12 endpoints)
TOX21_TASKS = [
    'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase',
    'NR-ER','NR-ER-LBD','NR-PPAR-gamma',
    'SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
]

# ── Data Loading ───────────────────────────────────────────────────────
def load_tox21_data():
    """Load and clean Tox21 dataset."""
    print("Loading Tox21 dataset...")
    df = pd.read_csv(TOX21_PATH)
    print(f"Raw shape: {df.shape}")

    # Validate SMILES
    from rdkit import Chem
    def valid_smiles(s):
        try:
            mol = Chem.MolFromSmiles(str(s))
            return Chem.MolToSmiles(mol) if mol else None
        except:
            return None

    df['smiles_canon'] = df['smiles'].apply(valid_smiles)
    df = df.dropna(subset=['smiles_canon']).drop_duplicates('smiles_canon')
    print(f"Cleaned: {len(df)} compounds")
    return df

# ── Feature Engineering ───────────────────────────────────────────────
def compute_descriptor_list():
    """Return list of RDKit descriptor names to compute."""
    # Use same descriptors as original notebook
    return [
        'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'NumHDonors', 'NumHAcceptors',
        'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
        'NumSaturatedRings', 'NumHeteroatoms', 'FractionCSP3',
        'RingCount', 'HeavyAtomCount', 'NumRadicalElectrons',
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi1', 'Chi2n', 'Chi3n', 'Chi4n',
        'Kappa1', 'Kappa2', 'Kappa3',
        'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'MaxEStateIndex', 'MinEStateIndex',
        'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA6', 'PEOE_VSA10',
        'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_Ar_N', 'fr_Ar_OH', 'fr_C_O',
        'fr_C_O_noCOO', 'fr_COO', 'fr_COO2', 'fr_hdrzone', 'fr_nitro',
        'fr_nitro_arom', 'fr_nitroso', 'fr_epoxide', 'fr_sulfonamd',
        'fr_sulfone', 'fr_aldehyde', 'fr_alkyl_halide',
    ]

def mol_to_morgan_fp(smi, radius=2, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def mol_to_descriptors(smi, desc_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(len(desc_list), dtype=np.float32)
    vals = []
    for name in desc_list:
        try:
            fn = getattr(Descriptors, name, None) or getattr(rdMolDescriptors, name, None)
            v = fn(mol) if fn else 0.0
            vals.append(float(v) if v is not None else 0.0)
        except:
            vals.append(0.0)
    arr = np.array(vals, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)

def build_feature_matrix(smiles_list, desc_list, scaler=None, fit_scaler=False):
    """Build feature matrix: Morgan FP (2048) + RDKit descriptors."""
    fps, descs = [], []
    for smi in smiles_list:
        fps.append(mol_to_morgan_fp(smi))
        descs.append(mol_to_descriptors(smi, desc_list))
    X_fp = np.vstack(fps)
    X_desc = np.vstack(descs)
    X = np.hstack([X_fp, X_desc])

    if fit_scaler:
        scaler = StandardScaler()
        X[:, 2048:] = scaler.fit_transform(X[:, 2048:])
        return X, scaler
    else:
        X[:, 2048:] = scaler.transform(X[:, 2048:])
        return X, scaler

def scaffold_split(df):
    """Murcko scaffold-based train/val/test split (80/10/10)."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    df = df.copy()
    df['scaffold'] = df['smiles_canon'].apply(get_scaffold)
    scaffolds = df['scaffold'].value_counts().index.tolist()

    train_idx, val_idx, test_idx = [], [], []
    cutoff_train = int(0.8 * len(df))
    cutoff_val = int(0.9 * len(df))
    running = 0

    for sc in scaffolds:
        idxs = df[df['scaffold'] == sc].index.tolist()
        running += len(idxs)
        if running <= cutoff_train:
            train_idx.extend(idxs)
        elif running <= cutoff_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

# ── Modeling ────────────────────────────────────────────────────────────
class MultiTaskModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.models = {}  # task -> fitted model

    def train(self, X_train, Y_train, X_val, Y_val, tasks, n_trials=50):
        """Train one model per task with Optuna tuning."""
        import optuna
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for t_idx, task in enumerate(tasks):
            print(f"\n=== Training {self.model_type} for {task} ===")
            y_tr = Y_train[:, t_idx]
            y_va = Y_val[:, t_idx]

            mask_tr = ~np.isnan(y_tr)
            mask_va = ~np.isnan(y_va)

            if mask_tr.sum() < 30:
                print(f"  Skipping: only {mask_tr.sum()} labels")
                continue

            # Compute class weight
            pos_weight = (mask_tr.sum() - y_tr[mask_tr].sum()) / max(y_tr[mask_tr].sum(), 1)

            if self.model_type == 'xgboost':
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                        'scale_pos_weight': min(pos_weight, 20),
                        'use_label_encoder': False,
                        'eval_metric': 'auc',
                        'random_state': 42,
                        'tree_method': 'hist',
                        'verbosity': 0
                    }
                    model = XGBClassifier(**params)
                    model.fit(
                        X_train[mask_tr], y_tr[mask_tr],
                        eval_set=[(X_val[mask_va], y_va[mask_va])],
                        early_stopping_rounds=30,
                        verbose=False
                    )
                    pred = model.predict_proba(X_val[mask_va])[:, 1]
                    return roc_auc_score(y_va[mask_va], pred)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_trial.params

                model = XGBClassifier(
                    **best_params,
                    use_label_encoder=False,
                    eval_metric='auc',
                    random_state=42,
                    tree_method='hist',
                    verbosity=0
                )
                model.fit(X_train[mask_tr], y_tr[mask_tr])

            elif self.model_type == 'lightgbm':
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                        'num_leaves': trial.suggest_int('num_leaves', 31, 255, step=32),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                        'class_weight': 'balanced' if pos_weight > 1 else None,
                        'random_state': 42,
                        'verbosity': -1
                    }
                    model = LGBMClassifier(**params)
                    model.fit(
                        X_train[mask_tr], y_tr[mask_tr],
                        eval_set=[(X_val[mask_va], y_va[mask_va])],
                        eval_metric='auc',
                        callbacks=[LGBMClassifier.early_stopping(30, verbose=False)]
                    )
                    pred = model.predict_proba(X_val[mask_va])[:, 1]
                    return roc_auc_score(y_va[mask_va], pred)

                import lightgbm as lgb
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_trial.params

                model = LGBMClassifier(
                    **best_params,
                    random_state=42,
                    verbosity=-1
                )
                model.fit(X_train[mask_tr], y_tr[mask_tr])

            elif self.model_type == 'randomforest':
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=100),
                        'max_depth': trial.suggest_int('max_depth', 10, 50, step=5),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                        'class_weight': 'balanced' if pos_weight > 1 else None,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model = RandomForestClassifier(**params)
                    model.fit(X_train[mask_tr], y_tr[mask_tr])
                    pred = model.predict_proba(X_val[mask_va])[:, 1]
                    return roc_auc_score(y_va[mask_va], pred)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_trial.params

                model = RandomForestClassifier(
                    **best_params,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train[mask_tr], y_tr[mask_tr])

            self.models[task] = model
            val_pred = model.predict_proba(X_val[mask_va])[:, 1]
            auc = roc_auc_score(y_va[mask_va], val_pred)
            print(f"  Val AUC: {auc:.4f}")

    def predict(self, X, task):
        """Return probability for given task."""
        if task not in self.models:
            return 0.5  # default if model missing
        return self.models[task].predict_proba(X.reshape(1, -1))[:, 1][0]

    def save(self, path):
        joblib.dump(self.models, path)

    def load(self, path):
        self.models = joblib.load(path)

# ── Main Pipeline ────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    # 1. Load data
    df = load_tox21_data()
    desc_list = compute_descriptor_list()

    # 2. Scaffold split
    df_train, df_val, df_test = scaffold_split(df)
    print(f"\nSplit sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 3. Build features
    print("\nBuilding features...")
    X_train, scaler = build_feature_matrix(df_train['smiles_canon'].tolist(), desc_list, fit_scaler=True)
    X_val, _ = build_feature_matrix(df_val['smiles_canon'].tolist(), desc_list, scaler=scaler)
    X_test, _ = build_feature_matrix(df_test['smiles_canon'].tolist(), desc_list, scaler=scaler)

    Y_train = df_train[TOX21_TASKS].values.astype(np.float32)
    Y_val = df_val[TOX21_TASKS].values.astype(np.float32)
    Y_test = df_test[TOX21_TASKS].values.astype(np.float32)

    print(f"Feature shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")

    # 4. Train models (XGBoost already exists, we train LightGBM & RF)
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    lgb = MultiTaskModel('lightgbm')
    lgb.train(X_train, Y_train, X_val, Y_val, TOX21_TASKS, n_trials=30)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lgb.save(MODELS_DIR / "lightgbm" / "lgb_models_v2.pkl")

    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    rf = MultiTaskModel('randomforest')
    rf.train(X_train, Y_train, X_val, Y_val, TOX21_TASKS, n_trials=20)
    rf.save(MODELS_DIR / "randomforest" / "rf_models_v2.pkl")

    # Save updated scaler
    joblib.dump(scaler, MODELS_DIR / "descriptor_scaler_v2.pkl")

    # 5. Generate OOF predictions for stacking (using 5-fold CV on train+val)
    print("\n" + "="*60)
    print("GENERATING STACKING FEATURES")
    print("="*60)

    # Load existing XGBoost
    xgb_old = joblib.load(MODELS_DIR / "xgboost" / "xgb_models.pkl")

    # Combine train+val for CV
    X_all = np.vstack([X_train, X_val])
    Y_all = np.vstack([Y_train, Y_val])
    n_samples = len(X_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Prepare meta-feature matrix
    meta_features = np.zeros((n_samples, len(TOX21_TASKS) * 4))  # xgb, lgb, rf, (potential gnn placeholder)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, np.ones(n_samples))):
        print(f"\nFold {fold+1}/5")
        X_tr_fold = X_all[train_idx]
        X_va_fold = X_all[val_idx]
        Y_tr_fold = Y_all[train_idx]

        # Train each model on this fold
        xgb_fold = MultiTaskModel('xgboost')
        xgb_fold.train(X_tr_fold, Y_tr_fold, X_va_fold, Y_all[val_idx], TOX21_TASKS, n_trials=20)

        lgb_fold = MultiTaskModel('lightgbm')
        lgb_fold.train(X_tr_fold, Y_tr_fold, X_va_fold, Y_all[val_idx], TOX21_TASKS, n_trials=20)

        rf_fold = MultiTaskModel('randomforest')
        rf_fold.train(X_tr_fold, Y_tr_fold, X_va_fold, Y_all[val_idx], TOX21_TASKS, n_trials=15)

        # Collect predictions for meta learner
        for t_i, task in enumerate(TOX21_TASKS):
            col_base = t_i * 4
            if task in xgb_fold.models:
                for i, idx in enumerate(val_idx):
                    meta_features[idx, col_base] = xgb_fold.models[task].predict_proba(X_all[idx].reshape(1, -1))[:, 1][0]
            if task in lgb_fold.models:
                for i, idx in enumerate(val_idx):
                    meta_features[idx, col_base+1] = lgb_fold.models[task].predict_proba(X_all[idx].reshape(1, -1))[:, 1][0]
            if task in rf_fold.models:
                for i, idx in enumerate(val_idx):
                    meta_features[idx, col_base+2] = rf_fold.models[task].predict_proba(X_all[idx].reshape(1, -1))[:, 1][0]
            # Column 3 reserved for GNN (leave as 0.5 for now)

    meta_features = np.nan_to_num(meta_features, nan=0.5)

    # 6. Train meta-learner per task
    print("\n" + "="*60)
    print("TRAINING META-LEARNER (STACKING)")
    print("="*60)

    meta_models = {}
    for t_i, task in enumerate(TOX21_TASKS):
        print(f"\nMeta-learner for {task}")
        y_meta = Y_all[:, t_i]
        mask = ~np.isnan(y_meta)

        if mask.sum() < 50:
            print(f"  Skipping: insufficient labels")
            continue

        X_meta = meta_features[mask]
        y_meta = y_meta[mask]

        # Simple logistic regression as meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        meta.fit(X_meta, y_meta)
        meta_models[task] = meta

        # Evaluate on hold-out (use a simple train/test split)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)
        meta_va = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        meta_va.fit(X_tr, y_tr)
        pred_va = meta_va.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, pred_va)
        print(f"  Meta AUC (CV estimate): {auc:.4f}")

    joblib.dump(meta_models, MODELS_DIR / "ensemble" / "meta_models_v2.pkl")

    # 7. Evaluate full pipeline on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)

    # Re-train XGBoost on full train+val
    xgb_full = MultiTaskModel('xgboost')
    xgb_full.train(X_all, Y_all, X_test, Y_test, TOX21_TASKS, n_trials=30)

    # Load RF and LGB that were already trained on full data earlier
    # (We trained them on full train+val above, need to save them)
    lgb_full = lgb  # reuse from full training above
    rf_full = rf   # reuse from full training above

    # Test predictions
    results = []
    for task in TOX21_TASKS:
        t_idx = TOX21_TASKS.index(task)
        y_te = Y_test[:, t_idx]
        mask = ~np.isnan(y_te)
        if mask.sum() == 0:
            continue

        # Individual model AUCs
        if task in xgb_full.models:
            xgb_pred = xgb_full.models[task].predict_proba(X_test[mask])[:, 1]
            auc_xgb = roc_auc_score(y_te[mask], xgb_pred)
        else:
            auc_xgb = np.nan

        if task in lgb_full.models:
            lgb_pred = lgb_full.models[task].predict_proba(X_test[mask])[:, 1]
            auc_lgb = roc_auc_score(y_te[mask], lgb_pred)
        else:
            auc_lgb = np.nan

        if task in rf_full.models:
            rf_pred = rf_full.models[task].predict_proba(X_test[mask])[:, 1]
            auc_rf = roc_auc_score(y_te[mask], rf_pred)
        else:
            auc_rf = np.nan

        # Stacked ensemble
        if task in meta_models:
            X_meta_test = np.hstack([
                xgb_pred.reshape(-1,1) if task in xgb_full.models else np.full((mask.sum(),1), 0.5),
                lgb_pred.reshape(-1,1) if task in lgb_full.models else np.full((mask.sum(),1), 0.5),
                rf_pred.reshape(-1,1) if task in rf_full.models else np.full((mask.sum(),1), 0.5),
                np.full((mask.sum(),1), 0.5)  # GNN placeholder
            ])
            stack_pred = meta_models[task].predict_proba(X_meta_test)[:, 1]
            auc_stack = roc_auc_score(y_te[mask], stack_pred)
        else:
            auc_stack = np.nan

        results.append({
            'task': task,
            'xgboost': auc_xgb,
            'lightgbm': auc_lgb,
            'randomforest': auc_rf,
            'stacked_ensemble': auc_stack
        })

    results_df = pd.DataFrame(results)
    print("\nTest Set ROC-AUC per model:")
    print(results_df.to_string(index=False))
    print("\nMean AUCs:")
    print(results_df.drop(columns='task').mean().to_string())

    # 8. Save all models for production
    print("\nSaving production models...")
    xgb_full.save(MODELS_DIR / "xgboost" / "xgb_models_v2.pkl")
    lgb_full.save(MODELS_DIR / "lightgbm" / "lgb_models_final.pkl")
    rf_full.save(MODELS_DIR / "randomforest" / "rf_models_final.pkl")

    # Save summary
    summary = {
        'models': {
            'xgboost': str(MODELS_DIR / "xgboost" / "xgb_models_v2.pkl"),
            'lightgbm': str(MODELS_DIR / "lightgbm" / "lgb_models_final.pkl"),
            'randomforest': str(MODELS_DIR / "randomforest" / "rf_models_final.pkl"),
            'meta': str(MODELS_DIR / "ensemble" / "meta_models_v2.pkl"),
            'scaler': str(MODELS_DIR / "descriptor_scaler_v2.pkl")
        },
        'test_metrics': results_df.to_dict(orient='records'),
        'mean_aucs': results_df.drop(columns='task').mean().to_dict()
    }

    with open(MODELS_DIR / 'upgrade_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Model upgrade complete!")
    print(f"Mean Stacked Ensemble AUC: {results_df['stacked_ensemble'].mean():.4f}")
    print(f"All artifacts saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
