
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib, numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import QED, rdMolDescriptors
import py3Dmol
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

BASE_DIR = Path(__file__).resolve().parent.parent   # project root
MODELS_DIR = BASE_DIR / "models"

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER',
         'NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE',
         'SR-MMP','SR-p53']

EXAMPLE_MOLECULES = {
    "Custom": "",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Bisphenol A (endocrine disruptor)": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    "Nicotine": "CN1CCCC1c1cccnc1",
    "Tamoxifen (SERM)": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Sarin (nerve agent)": "CC(OP(C)(F)=O)C",
    "DFP (nerve agent class)": "FP(=O)(OC(C)C)OC(C)C",
}

# ══════════════════════════════════════════════════════════════
# Structural Alerts — catch toxicity Tox21 endpoints DON'T cover
# ══════════════════════════════════════════════════════════════
STRUCTURAL_ALERTS = [
    ("Organophosphate (P=O + halide)",
     "[P](=O)([F,Cl,Br,I])", "CRITICAL",
     "Acetylcholinesterase inhibitor — nerve agent / pesticide class. "
     "Causes neurotoxicity. NOT detected by Tox21 in-vitro endpoints."),
    ("Organophosphate (P=S + halide)",
     "[P](=S)([F,Cl,Br,I])", "CRITICAL",
     "Thio-organophosphate insecticide class. AChE inhibition risk."),
    ("Phosphoramide",
     "[P](=O)([N])([Cl,F])", "CRITICAL",
     "Phosphoramide mustard / alkylating agent class."),
    ("Organophosphate (P-F bond)",
     "[P]~[F]", "CRITICAL",
     "P-F bond detected — characteristic of nerve agents (Sarin, Soman, DFP). "
     "Lethal AChE inhibitor. NOT detected by any Tox21 assay."),
    ("Mustard agent (S-mustard)",
     "ClCCS", "HIGH",
     "Sulfur mustard / vesicant — alkylating agent. NOT detected by Tox21."),
    ("Mustard agent (N-mustard)",
     "ClCCN", "HIGH",
     "Nitrogen mustard — alkylating agent / chemotherapy class."),
    ("Nitrosamine",
     "[N;R0](=O)[N;R0]", "HIGH",
     "Potent carcinogen — may not trigger Tox21 in-vitro assays."),
    ("Acyl halide",
     "[C](=O)[F,Cl,Br,I]", "HIGH",
     "Highly reactive — causes chemical burns, respiratory damage."),
    ("Isocyanate",
     "[N]=[C]=[O]", "HIGH",
     "Severe respiratory sensitizer (Bhopal-class). NOT detected by Tox21."),
    ("Organic azide",
     "[N-]=[N+]=[N]", "HIGH",
     "Explosive / cytotoxic — shock-sensitive."),
    ("Acyl cyanide",
     "[C](=O)C#N", "HIGH",
     "Releases HCN upon hydrolysis — acute lethal toxicity."),
    ("Heavy metal",
     "[Tl,Hg,Pb,Cd,As]", "CRITICAL",
     "Heavy metal toxicity — NOT captured by Tox21 bioassays."),
    ("Organic peroxide",
     "OO", "MEDIUM",
     "Oxidizer — explosive, causes tissue damage."),
    ("Hydrazine",
     "[NH2][NH2]", "MEDIUM",
     "Hepatotoxic carcinogen."),
    ("Epoxide",
     "[C]1[O][C]1", "MEDIUM",
     "Electrophilic — DNA-alkylating potential."),
    ("Dioxin / Dibenzodioxin",
     "c1ccc2c(c1)Oc3ccccc3O2", "CRITICAL",
     "Polychlorinated dibenzo-p-dioxin class — potent AhR agonist. "
     "Causes cancer, immunotoxicity, endocrine disruption. "
     "TCDD is the most toxic congener."),
    ("Dibenzofuran",
     "c1ccc2c(c1)Oc3ccccc32", "HIGH",
     "Polyhalogenated dibenzofuran — persistent organic pollutant. "
     "Similar toxicological profile to dioxins."),
    ("Polyhalogenated biphenyl (PCB-like)",
     "c1cc([F,Cl,Br])c([F,Cl,Br])cc1-c2cc([F,Cl,Br])c([F,Cl,Br])cc2", "HIGH",
     "Polychlorinated biphenyl pattern — endocrine disruptor, "
     "bioaccumulative persistent organic pollutant."),
]


def get_pharmaco_profile(mol):
    """Calculates pharmacological drug-likeness parameters."""
    profile = {}
    try:
        profile["MW"] = Descriptors.MolWt(mol)
        profile["LogP"] = Descriptors.MolLogP(mol)
        profile["HBD"] = Descriptors.NumHDonors(mol)
        profile["HBA"] = Descriptors.NumHAcceptors(mol)
        profile["TPSA"] = Descriptors.TPSA(mol)
        profile["QED"] = QED.qed(mol)
        
        # SAS Score approximation (synthetic accessibility)
        profile["RotBonds"] = Descriptors.NumRotatableBonds(mol)
        
        # Rule of 5 check
        ro5_violations = 0
        if profile["MW"] > 500: ro5_violations += 1
        if profile["LogP"] > 5: ro5_violations += 1
        if profile["HBD"] > 5: ro5_violations += 1
        if profile["HBA"] > 10: ro5_violations += 1
        profile["RO5"] = "PASS" if ro5_violations <= 1 else f"FAIL ({ro5_violations} violations)"
    except Exception:
        return None
    return profile


def get_shap_explainer(task, models):
    """Returns a SHAP explainer for a specific task's XGBoost model."""
    xgb_model = models["xgb"].get(task)
    if xgb_model:
        return shap.TreeExplainer(xgb_model)
    return None


def display_shap_analysis(smiles, task, models):
    """Calculates and plots SHAP values for a specific task and molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return
    
    feats = build_features(mol, models["scaler"])
    explainer = get_shap_explainer(task, models)
    
    if explainer:
        shap_values = explainer.shap_values(feats)
        
        # Combine feature names
        feature_names = [f"FP_{i}" for i in range(2048)] + DESC_NAMES + ["ZINC_1", "ZINC_2", "ZINC_3"]
        
        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        
        # Draw a custom bar plot for top features to be clean
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        top_inds = np.argsort(np.abs(vals))[-15:]
        
        colors = ['#ef4444' if vals[i] > 0 else '#3b82f6' for i in top_inds]
        ax.barh([feature_names[i] for i in top_inds], [vals[i] for i in top_inds], color=colors)
        ax.set_title(f"Molecular Drivers: {task}", color="white", fontsize=14, fontweight="bold")
        ax.tick_params(axis='both', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
            
        st.pyplot(fig)
        st.caption("🔴 Red bars increase toxicity risk | 🔵 Blue bars decrease toxicity risk")


def show_3d_mol(smiles):
    """Renders a py3Dmol viewer for the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    mblock = Chem.MolToMolBlock(mol)
    
    view = py3Dmol.view(width=400, height=400)
    view.addModel(mblock, "mol")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.zoomTo()
    
    # Use raw HTML/JS to render the viewer in Streamlit
    html = view._make_html()
    components.html(html, height=400)


def generate_technical_report(smiles, results_df, alerts, profile):
    """Generates a formatted technical report of the findings."""
    mol = Chem.MolFromSmiles(smiles)
    report = f"DRUG TOXICITY ANALYSIS REPORT\n"
    report += "="*30 + "\n\n"
    report += f"Compound SMILES: {smiles}\n"
    report += f"Molecular Weight: {profile.get('MW', 'N/A'):.2f}\n"
    report += f"LogP: {profile.get('LogP', 'N/A'):.2f}\n"
    report += f"Drug-likeness (QED): {profile.get('QED', 'N/A'):.3f}\n"
    report += f"Rule of 5 Status: {profile.get('RO5', 'N/A')}\n\n"
    
    report += "TOXICITY RISK ASSESSMENTS\n"
    report += "-"*30 + "\n"
    for _, row in results_df.iterrows():
        report += f"{row['Task']}: {row['Ensemble']:.1%} ({row['Risk']})\n"
    
    report += "\nSTRUCTURAL SAFETY ALERTS\n"
    report += "-"*30 + "\n"
    if alerts:
        for aname, asev, amech in alerts:
            report += f"[{asev}] {aname}: {amech}\n"
    else:
        report += "No structural alerts detected.\n"
    
    report += "\n" + "="*30 + "\n"
    report += "Generated by Precision Drug Toxicity Engine v2.0\n"
    return report


def check_structural_alerts(mol):
    alerts = []
    for name, smarts, severity, mechanism in STRUCTURAL_ALERTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            alerts.append((name, severity, mechanism))
    return alerts


# ══════════════════════════════════════════════════════════════
# GNN Architecture
# ══════════════════════════════════════════════════════════════
if HAS_PYG:
    ATOM_TYPES = [
        "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca",
        "Fe","As","Al","I","B","V","K","Tl","Yb","Sb","Sn",
        "Ag","Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au",
        "Ni","Cd","In","Mn","Zr","Cr","Pt","Hg","Pb","Unknown",
    ]
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ]

    def _one_hot(v, choices):
        return [int(v == c) for c in choices]

    def _atom_features(atom):
        return (
            _one_hot(atom.GetSymbol(), ATOM_TYPES)
            + _one_hot(atom.GetDegree(), list(range(11)))
            + [int(atom.GetFormalCharge()), int(atom.GetNumImplicitHs()),
               int(atom.GetIsAromatic())]
            + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            + [int(atom.IsInRing()), atom.GetMass() / 100.0]
        )

    def smiles_to_pyg(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
        edges = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.zeros((2, 0), dtype=torch.long)
        )
        return Data(x=x, edge_index=edge_index)

    class ToxGNN(nn.Module):
        def __init__(self, in_channels, hidden=256, out_tasks=12, heads=4,
                     dropout=0.5, n_layers=3, shared_dim=128):
            super().__init__()
            self.dropout = dropout
            self.n_layers = n_layers
            self.input_proj = nn.Linear(in_channels, hidden)
            self.input_norm = nn.LayerNorm(hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(n_layers):
                if i % 2 == 0:
                    self.convs.append(GINConv(nn.Sequential(
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden))))
                else:
                    self.convs.append(GATConv(
                        hidden, hidden // heads, heads=heads,
                        dropout=dropout, concat=True))
                self.norms.append(nn.LayerNorm(hidden))
            self.pool_proj = nn.Linear(hidden * 2, hidden)
            self.pool_norm = nn.LayerNorm(hidden)
            self.shared_head = nn.Sequential(
                nn.Linear(hidden, shared_dim), nn.LayerNorm(shared_dim),
                nn.ReLU(), nn.Dropout(dropout))
            self.task_heads = nn.ModuleList(
                [nn.Linear(shared_dim, 1) for _ in range(out_tasks)])

        def forward(self, x, edge_index, batch):
            h = self.input_norm(F.relu(self.input_proj(x)))
            h = F.dropout(h, p=self.dropout * 0.5, training=self.training)
            for i in range(self.n_layers):
                r = h
                h = F.elu(self.norms[i](self.convs[i](h, edge_index)))
                h = F.dropout(h, p=self.dropout, training=self.training) + r
            hm = global_mean_pool(h, batch)
            hs = global_add_pool(h, batch)
            hg = F.dropout(self.pool_norm(F.relu(self.pool_proj(
                torch.cat([hm, hs], dim=1)))),
                p=self.dropout, training=self.training)
            s = self.shared_head(hg)
            return torch.cat([t(s) for t in self.task_heads], dim=1)


# ══════════════════════════════════════════════════════════════
# ChemBERTa Architecture
# ══════════════════════════════════════════════════════════════
if HAS_TRANSFORMERS:
    class ChemBERTaTox(nn.Module):
        def __init__(self, backbone_name, n_tasks=12, hidden_dim=256, dropout=0.3):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(backbone_name)
            bb_dim = self.backbone.config.hidden_size  # 768
            self.head = nn.Sequential(
                nn.LayerNorm(bb_dim),
                nn.Linear(bb_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_tasks),
            )

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.head(cls)


# ══════════════════════════════════════════════════════════════
# Feature computation
# ══════════════════════════════════════════════════════════════
DESC_NAMES = [
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

def build_features(mol, scaler):
    fp = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), fp)
    
    desc_vals = []
    for name in DESC_NAMES:
        try:
            fn = getattr(Descriptors, name, None) or getattr(rdMolDescriptors, name, None)
            v = fn(mol) if fn else 0.0
            desc_vals.append(float(v) if v is not None else 0.0)
        except:
            desc_vals.append(0.0)
            
    desc = np.array(desc_vals, dtype=np.float32)
    # The new scaler on disk expects exactly 52 descriptors
    desc_scaled = scaler.transform(desc.reshape(1, -1))[0]
    
    # We no longer need the 'zinc' padding as the new models match the 52-descriptor set (2100 total)
    feats = np.concatenate([fp, desc_scaled])[np.newaxis, :]
    return np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════
def load_models():
    device = torch.device("cpu")
    # UPDATED: Using the fresh 'Fast' models and the Meta-Learner rescuer
    xgb = joblib.load(MODELS_DIR / "xgboost" / "xgb_models_final.pkl")
    lgb = joblib.load(MODELS_DIR / "lightgbm" / "lgb_models_new.pkl")
    rf  = joblib.load(MODELS_DIR / "randomforest" / "rf_models_new.pkl")
    meta = joblib.load(MODELS_DIR / "ensemble" / "meta_learner_v2.pkl")
    scaler = joblib.load(MODELS_DIR / "descriptor_scaler.pkl")

    gnn = None
    if HAS_PYG:
        try:
            for p in [MODELS_DIR / "gnn" / "gnn_best.pt",
                      MODELS_DIR / "gnn" / "gnn_final.pt"]:
                if p.exists():
                    sd = torch.load(p, map_location=device, weights_only=False)
                    if isinstance(sd, dict) and "model_state_dict" in sd:
                        sd = sd["model_state_dict"]
                    gnn = ToxGNN(
                        in_channels=sd["input_proj.weight"].shape[1],
                        hidden=sd["input_proj.weight"].shape[0],
                        out_tasks=sum(1 for k in sd
                                      if k.startswith("task_heads.") and k.endswith(".weight")),
                        heads=(sd["convs.1.att_src"].shape[1]
                               if "convs.1.att_src" in sd else 4),
                        n_layers=(max(int(k.split(".")[1])
                                      for k in sd if k.startswith("convs.")) + 1),
                        shared_dim=sd["shared_head.0.weight"].shape[0],
                    )
                    gnn.load_state_dict(sd)
                    gnn.to(device).eval()
                    break
        except Exception:
            gnn = None

    # ChemBERTa
    cb_model = None
    cb_tokenizer = None
    cb_max_len = 128
    if HAS_TRANSFORMERS:
        cb_path = MODELS_DIR / "chemberta" / "chemberta_final.pt"
        if cb_path.exists():
            try:
                ckpt = torch.load(cb_path, map_location=device, weights_only=False)
                bb_name = ckpt.get("backbone_name", "seyonec/ChemBERTa-zinc-base-v1")
                cb_max_len = ckpt.get("max_length", 128)
                n_tasks = ckpt.get("n_tasks", 12)
                h_dim = ckpt.get("hidden_dim", 256)
                cb_model = ChemBERTaTox(bb_name, n_tasks=n_tasks, hidden_dim=h_dim)
                cb_model.load_state_dict(ckpt["model_state_dict"])
                cb_model.to(device).eval()
                cb_tokenizer = AutoTokenizer.from_pretrained(bb_name)
            except Exception:
                cb_model = None
                cb_tokenizer = None

    return {"xgb": xgb, "lgb": lgb, "rf": rf, "meta": meta, "gnn": gnn,
            "cb_model": cb_model, "cb_tokenizer": cb_tokenizer,
            "cb_max_len": cb_max_len, "scaler": scaler, "device": device}


# ══════════════════════════════════════════════════════════════
# Risk Thresholds — standard classification on simple-average
# ensemble of XGBoost, LightGBM, Random Forest, and GNN probs.
# The meta-learner compresses outputs too aggressively for display;
# simple averaging preserves interpretable probabilities.
# ══════════════════════════════════════════════════════════════
RISK_HIGH = 0.50   # average prob > 0.5 → HIGH
RISK_MED  = 0.35   # average prob > 0.35 → MEDIUM


# ══════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════
def predict_single(smiles, models):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = build_features(mol, models["scaler"])

    # GNN inference
    gnn_probs = {}
    gnn = models["gnn"]
    if gnn is not None and HAS_PYG:
        data = smiles_to_pyg(smiles)
        if data is not None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            with torch.no_grad():
                probs = torch.sigmoid(
                    gnn(data.x, data.edge_index, data.batch)
                ).cpu().numpy()[0]
            for i, t in enumerate(TASKS):
                gnn_probs[t] = float(probs[i])

    # ChemBERTa inference
    cb_probs = {}
    cb_model = models.get("cb_model")
    cb_tokenizer = models.get("cb_tokenizer")
    if cb_model is not None and cb_tokenizer is not None:
        try:
            enc = cb_tokenizer(smiles, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=models.get("cb_max_len", 128))
            with torch.no_grad():
                logits = cb_model(enc["input_ids"], enc["attention_mask"])
                probs_cb = torch.sigmoid(logits).cpu().numpy()[0]
            for i, t in enumerate(TASKS):
                cb_probs[t] = float(probs_cb[i])
        except Exception:
            pass

    has_cb = len(cb_probs) > 0
    n_models = 5 if has_cb else 4

    rows = []
    for t in TASKS:
        xp = float(models["xgb"][t].predict_proba(feats)[:, 1][0]) if t in models["xgb"] else 0.5
        lp = float(models["lgb"][t].predict_proba(feats)[:, 1][0]) if t in models["lgb"] else 0.5
        rp = float(models["rf"][t].predict_proba(feats)[:, 1][0])  if t in models["rf"]  else 0.5
        gp = gnn_probs.get(t, 0.5)
        cp = cb_probs.get(t, 0.5)

        # STACKED ENSEMBLE (Logistic Regression Meta-Learner)
        # We pass the 3 base model predictions (XGB, LGB, RF) to the meta-model
        if t in models["meta"]:
            meta_input = np.column_stack([xp, lp, rp])
            prob = float(models["meta"][t].predict_proba(meta_input)[:, 1][0])
        else:
            # Fallback if meta-model is missing for a task
            prob = (xp + lp + rp + gp) / 4.0
            
        risk = "HIGH" if prob > RISK_HIGH else "MEDIUM" if prob > RISK_MED else "LOW"
        row = {
            "Task": t, "Ensemble": round(prob, 4), "Risk": risk,
            "XGBoost": round(xp, 4), "LightGBM": round(lp, 4),
            "Random Forest": round(rp, 4), "GNN": round(gp, 4),
        }
        if has_cb:
            row["ChemBERTa"] = round(cp, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def mol_to_svg(mol, size=(400, 300)):
    d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = d.drawOptions()
    opts.clearBackground = False
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText()


# ══════════════════════════════════════════════════════════════
# FastAPI App Definition
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Precision Drug Toxicity Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models globally on startup
print("Loading models...")
MODELS = load_models()
print("Models loaded successfully.")


class PredictRequest(BaseModel):
    smiles: str

class ShapRequest(BaseModel):
    smiles: str
    task: str


@app.post("/predict")
def api_predict(req: PredictRequest):
    smiles = req.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
    # 1. Structural Alerts
    alerts_raw = check_structural_alerts(mol)
    alerts = [{"name": a[0], "severity": a[1], "mechanism": a[2]} for a in alerts_raw]
    
    # 2. Tox21 Predictions
    canonical = Chem.MolToSmiles(mol)
    df = predict_single(canonical, MODELS)
    if df is None:
        raise HTTPException(status_code=500, detail="Prediction failed")
        
    predictions = df.to_dict(orient="records")
    
    # 3. Pharmacology Profile
    profile = get_pharmaco_profile(mol)
    
    # 4. Overall Risk Calculation
    n_crit = sum(1 for a in alerts_raw if a[1] == "CRITICAL")
    n_ahigh = sum(1 for a in alerts_raw if a[1] == "HIGH")
    n_amed = sum(1 for a in alerts_raw if a[1] == "MEDIUM")
    avg_score = df["Ensemble"].mean()
    hi = len(df[df.Risk == "HIGH"])
    md = len(df[df.Risk == "MEDIUM"])
    
    overall = "LOW CONCERN"
    if n_crit > 0: overall = "CRITICAL"
    elif n_ahigh > 0 or hi >= 3: overall = "DANGEROUS"
    elif hi > 0 or (md >= 3 and avg_score > 0.30): overall = "HIGH CONCERN"
    elif md > 0 or n_amed > 0 or avg_score > 0.25: overall = "MODERATE CONCERN"
    
    return {
        "smiles": canonical,
        "overall_risk": overall,
        "average_score": float(avg_score),
        "alerts": alerts,
        "predictions": predictions,
        "pharmacology": profile
    }

@app.post("/shap")
def api_shap(req: ShapRequest):
    smiles = req.smiles
    task = req.task
    
    if task not in TASKS:
        raise HTTPException(status_code=400, detail="Invalid task endpoint")
        
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
        
    feats = build_features(mol, MODELS["scaler"])
    explainer = get_shap_explainer(task, MODELS)
    
    if not explainer:
        raise HTTPException(status_code=500, detail="SHAP explainer not available for this task")
        
    shap_values = explainer.shap_values(feats)
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    
    feature_names = [f"FP_{i}" for i in range(2048)] + DESC_NAMES + ["ZINC_1", "ZINC_2", "ZINC_3"]
    
    # Get top 15 features
    top_inds = np.argsort(np.abs(vals))[-15:]
    
    results = []
    for i in top_inds:
        results.append({
            "feature": feature_names[i],
            "value": float(vals[i])
        })
        
    return {"task": task, "shap_top_features": results}


@app.get("/3d")
def get_3d_viewer(smiles: str):
    from fastapi.responses import HTMLResponse
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    mblock = Chem.MolToMolBlock(mol)
    
    view = py3Dmol.view(width=400, height=400)
    view.addModel(mblock, "mol")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.zoomTo()
    html = view._make_html()
    return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
