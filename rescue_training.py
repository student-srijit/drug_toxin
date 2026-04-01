#!/usr/bin/env python3
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, DataStructs

# Setup paths
BASE = Path(__file__).parent.resolve()
DATA_PATH = BASE / "data" / "tox21.csv.gz"
MODELS_DIR = BASE / "models"
TASKS = [
    'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
    'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
]

# Feature helpers
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

def build_features(smiles_list, scaler):
    feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append(np.zeros(2100))
            continue
        # FP
        fp = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), fp)
        # Descriptors
        desc_vals = []
        for name in DESCRIPTORS:
            try:
                fn = getattr(Descriptors, name, None) or getattr(rdMolDescriptors, name, None)
                v = fn(mol) if fn else 0.0
                desc_vals.append(float(v) if v is not None else 0.0)
            except: desc_vals.append(0.0)
        desc = np.nan_to_num(np.array(desc_vals, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
        # Combine
        combined = np.hstack([fp, desc])
        combined[2048:] = scaler.transform(combined[2048:].reshape(1, -1))[0]
        feats.append(combined)
    return np.vstack(feats)

def main():
    print("🚀 Repair Script: Rescuing your 40-minute training work...")

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    df['smiles_canon'] = df['smiles'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None)
    df = df.dropna(subset=['smiles_canon']).drop_duplicates('smiles_canon')
    
    # 2. Re-create splits (same seed as train_fast.py)
    np.random.seed(42)
    # Re-implement scaffold split
    from rdkit.Chem.Scaffolds import MurckoScaffold
    df['scaffold'] = df['smiles_canon'].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False))
    scas = df['scaffold'].value_counts().index.tolist()
    train_idx, val_idx, test_idx = [], [], []
    cutoff1, cutoff2 = int(0.8*len(df)), int(0.9*len(df))
    run = 0
    for sc in scas:
        idxs = df[df['scaffold']==sc].index.tolist()
        run += len(idxs); 
        if run <= cutoff1: train_idx.extend(idxs)
        elif run <= cutoff2: val_idx.extend(idxs)
        else: test_idx.extend(idxs)
    
    tra_val = df.loc[train_idx + val_idx]
    tst = df.loc[test_idx]

    # 3. Load Base Models
    print("Loading pre-trained base models...")
    scaler = joblib.load(MODELS_DIR / "descriptor_scaler.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost" / "xgb_models_final.pkl")
    lgb = joblib.load(MODELS_DIR / "lightgbm" / "lgb_models_new.pkl")
    rf = joblib.load(MODELS_DIR / "randomforest" / "rf_models_new.pkl")

    # 4. Generate Meta-features on Train+Val (Fast estimation)
    print("Generating features for stacking...")
    X_stack_data = build_features(tra_val['smiles_canon'].tolist(), scaler)
    Y_stack_data = tra_val[TASKS].values.astype(np.float32)
    X_test_data = build_features(tst['smiles_canon'].tolist(), scaler)
    Y_test_data = tst[TASKS].values.astype(np.float32)

    meta_models = {}
    test_results = []

    print("Building final Meta-Learners...")
    for ti, task in enumerate(TASKS):
        # Base predictions for this task
        y_tr = Y_stack_data[:, ti]; m_tr = ~np.isnan(y_tr)
        y_te = Y_test_data[:, ti]; m_te = ~np.isnan(y_te)
        
        # Train Meta-Learner on the available base models
        xp_tr = xgb[task].predict_proba(X_stack_data[m_tr])[:, 1] if task in xgb else np.full(m_tr.sum(), 0.5)
        lp_tr = lgb[task].predict_proba(X_stack_data[m_tr])[:, 1] if task in lgb else np.full(m_tr.sum(), 0.5)
        rp_tr = rf[task].predict_proba(X_stack_data[m_tr])[:, 1] if task in rf else np.full(m_tr.sum(), 0.5)
        
        X_meta_tr = np.column_stack([xp_tr, lp_tr, rp_tr])
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(X_meta_tr, y_tr[m_tr])
        meta_models[task] = meta
        
        # Final Evaluation
        xp_te = xgb[task].predict_proba(X_test_data[m_te])[:, 1] if task in xgb else 0.5
        lp_te = lgb[task].predict_proba(X_test_data[m_te])[:, 1] if task in lgb else 0.5
        rp_te = rf[task].predict_proba(X_test_data[m_te])[:, 1] if task in rf else 0.5
        
        X_meta_te = np.column_stack([xp_te, lp_te, rp_te])
        stack_sc = meta.predict_proba(X_meta_te)[:, 1]
        
        auc_stack = roc_auc_score(y_te[m_te], stack_sc)
        test_results.append({
            'task': task,
            'xgboost': roc_auc_score(y_te[m_te], xp_te),
            'lightgbm': roc_auc_score(y_te[m_te], lp_te),
            'randomforest': roc_auc_score(y_te[m_te], rp_te),
            'stacked': auc_stack
        })
        print(f"  ✓ {task}: Stacked AUC = {auc_stack:.4f}")

    # 5. Save Final Artifacts
    joblib.dump(meta_models, MODELS_DIR / "ensemble" / "meta_learner_v2.pkl")
    
    res_df = pd.DataFrame(test_results)
    summary = {
        'models': {
            'xgboost': str(MODELS_DIR / "xgboost" / "xgb_models_final.pkl"),
            'lightgbm': str(MODELS_DIR / "lightgbm" / "lgb_models_new.pkl"),
            'randomforest': str(MODELS_DIR / "randomforest" / "rf_models_new.pkl"),
            'meta': str(MODELS_DIR / "ensemble" / "meta_learner_v2.pkl"),
            'scaler': str(MODELS_DIR / "descriptor_scaler.pkl")
        },
        'test_metrics': test_results,
        'mean_aucs': res_df.drop(columns='task').mean().to_dict()
    }
    with open(MODELS_DIR / 'upgrade_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ WORK RESCUED! All models saved and evaluated.")
    print(f"Final mean Stacked AUC: {res_df['stacked'].mean():.4f}")

if __name__ == "__main__":
    main()
