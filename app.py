
import streamlit as st
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
import streamlit.components.v1 as components

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

BASE_DIR = Path(__file__).resolve().parent
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
DESC_NAMES = [n for n, _ in Descriptors.descList[:52]]


def build_features(mol, scaler):
    fp = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), fp)
    desc = np.array([
        float(v) if v is not None else 0.0
        for v in (Descriptors.CalcMolDescriptors(mol).get(n, 0) for n in DESC_NAMES)
    ], dtype=np.float32)
    zinc = np.zeros(3, dtype=np.float32)
    feats = np.concatenate([fp, desc, zinc])[np.newaxis, :]
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    # Scale descriptor+zinc columns (2048:) to match training pipeline
    feats[:, 2048:] = scaler.transform(feats[:, 2048:])
    return feats


# ══════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    xgb = joblib.load(MODELS_DIR / "xgboost" / "xgb_models.pkl")
    lgb = joblib.load(MODELS_DIR / "lightgbm" / "lgb_models.pkl")
    rf  = joblib.load(MODELS_DIR / "randomforest" / "rf_models.pkl")
    meta = joblib.load(MODELS_DIR / "ensemble" / "meta_models.pkl")
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

        # Simple average ensemble
        if has_cb:
            prob = (xp + lp + rp + gp + cp) / 5.0
        else:
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
# Page config + CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Drug Toxicity Predictor", layout="wide", page_icon="🧬")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

html, body, [class*="st-"] { 
    font-family: 'Inter', sans-serif; 
}

.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
}

.block-container { 
    padding-top: 2rem; 
    max-width: 1400px; 
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: var(--glass-bg);
    padding: 10px;
    border-radius: 16px;
    border: 1px solid var(--glass-border);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
}

.stTabs [aria-selected="true"] {
    background: var(--primary-gradient) !important;
    color: white !important;
}

.alert-critical { 
    background: rgba(220, 38, 38, 0.1); 
    border: 1px solid #ef4444;
    border-radius: 16px; 
    padding: 1.5rem; 
    margin-bottom: 1rem;
}

.alert-high { 
    background: rgba(234, 88, 12, 0.08); 
    border: 1px solid #f97316;
    border-radius: 16px; 
    padding: 1.5rem; 
    margin-bottom: 1rem; 
}

.alert-sev { font-weight: 800; font-size: 1.1rem; letter-spacing: 0.05em; }
.alert-name { font-weight: 600; color: #fca5a5; font-size: 1.1rem; }
.alert-mech { color: #cbd5e1; font-size: 0.95rem; margin-top: 8px; line-height: 1.5; }

/* Custom slider/input styling */
.stTextInput > div > div > input {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
}

h1, h2, h3 {
    background: -webkit-linear-gradient(#f8fafc, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
</style>""", unsafe_allow_html=True)

st.title("🧬 Precision Drug Toxicity Engine")
st.caption("Advanced Pharmacology Intelligence | 5-Model Ensemble | IIT BHU Hackathon Edition")

models = load_models()

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_predict, tab_3d, tab_accuracy, tab_analysis, tab_batch = st.tabs([
    "🔬 Analysis & Predict", "🧊 3D Molecule", "📊 Model Intelligence", "🔍 Insights", "📋 Batch Engine"
])

# ─────────────────────────────────────────────────────────────
# TAB: Predict
# ─────────────────────────────────────────────────────────────
with tab_predict:
    col_input, col_mol = st.columns([1.5, 1])
    with col_input:
        example = st.selectbox("Example molecules", list(EXAMPLE_MOLECULES.keys()))
        default_smi = EXAMPLE_MOLECULES[example]
        smiles = st.text_input("Enter SMILES", value=default_smi,
                               placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
        predict_btn = st.button("🚀 Predict Toxicity", use_container_width=True)

    with col_mol:
        if smiles:
            mol_vis = Chem.MolFromSmiles(smiles)
            if mol_vis:
                col_2d, col_pharm = st.columns([1, 1])
                with col_2d:
                    svg = mol_to_svg(mol_vis, size=(400, 280))
                    st.markdown(
                        f'<div style="background:#fff;border-radius:14px;padding:1rem;text-align:center">'
                        f'{svg}<br><span style="color:#666;font-size:.85rem">{Chem.MolToSmiles(mol_vis)}</span></div>',
                        unsafe_allow_html=True)
                
                with col_pharm:
                    st.markdown("### Pharmacology Profile")
                    profile = get_pharmaco_profile(mol_vis)
                    if profile:
                        c1, c2 = st.columns(2)
                        c1.metric("LogP", f"{profile['LogP']:.2f}")
                        c2.metric("MW", f"{profile['MW']:.1f}")
                        c1.metric("QED", f"{profile['QED']:.3f}")
                        c2.metric("RO5", profile["RO5"])
                        st.divider()
                        st.caption(f"HBD: {profile['HBD']} | HBA: {profile['HBA']} | TPSA: {profile['TPSA']:.1f}")

    if predict_btn and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string")
        else:
            # ═══ STRUCTURAL ALERTS (shown FIRST, before Tox21) ═══
            alerts = check_structural_alerts(mol)
            if alerts:
                severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
                alerts.sort(key=lambda a: severity_order.get(a[1], 3))
                st.markdown("---")
                st.markdown("## 🚨 STRUCTURAL SAFETY ALERTS")
                st.error(
                    f"**{len(alerts)} structural alert(s) detected!** "
                    "These flag toxic mechanisms that Tox21 in-vitro assays CANNOT detect "
                    "(e.g. nerve agents, alkylating agents, heavy metals)."
                )
                for aname, asev, amech in alerts:
                    css_class = f"alert-{asev.lower()}"
                    icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "🔶"}.get(asev, "ℹ️")
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'{icon} <span class="alert-sev">{asev}</span> — '
                        f'<span class="alert-name">{aname}</span>'
                        f'<div class="alert-mech">{amech}</div></div>',
                        unsafe_allow_html=True)
                st.markdown(
                    "> ⚡ **A molecule can score LOW on all 12 Tox21 endpoints yet still be lethal** "
                    "through mechanisms like acetylcholinesterase inhibition (nerve agents), "
                    "DNA alkylation (mustards), or heavy metal poisoning — **none of these are "
                    "measured by Tox21 bioassays.**"
                )

            # ═══ Tox21 ML predictions ═══
            st.markdown("---")
            st.subheader("Tox21 Endpoint Predictions")
            if alerts:
                st.warning(
                    "⚠️ The Tox21 scores below measure only 12 specific in-vitro pathways "
                    "(nuclear receptors + stress response). They do NOT reflect the structural "
                    "alerts above. Low Tox21 scores ≠ safe molecule."
                )

            canonical = Chem.MolToSmiles(mol)
            n_mdls = 5 if models.get("cb_model") is not None else 4
            with st.spinner(f"Running {n_mdls}-model ensemble..."):
                df = predict_single(canonical, models)

            if df is not None:
                hi = len(df[df.Risk == "HIGH"])
                md = len(df[df.Risk == "MEDIUM"])
                lo = len(df[df.Risk == "LOW"])

                # ═══ OVERALL RISK VERDICT (combines ALL signals) ═══
                n_crit = sum(1 for a in alerts if a[1] == "CRITICAL")
                n_ahigh = sum(1 for a in alerts if a[1] == "HIGH")
                n_amed = sum(1 for a in alerts if a[1] == "MEDIUM")
                avg_score = df["Ensemble"].mean()
                max_score = df["Ensemble"].max()
                elevated = hi + md  # endpoints above LOW

                # Determine overall risk level from all evidence
                if n_crit > 0:
                    overall = "CRITICAL"
                    overall_icon = "☠️"
                    overall_color = "#dc2626"
                    overall_reason = (
                        f"**{n_crit} CRITICAL structural alert(s)** detected. "
                        "This compound contains motifs associated with extreme toxicity "
                        "(nerve agents, heavy metals, dioxins, etc.) through mechanisms "
                        "that Tox21 in-vitro assays CANNOT measure."
                    )
                elif n_ahigh > 0 or hi >= 3:
                    overall = "DANGEROUS"
                    overall_icon = "🚨"
                    overall_color = "#ea580c"
                    reasons = []
                    if n_ahigh > 0:
                        reasons.append(f"{n_ahigh} HIGH structural alert(s)")
                    if hi > 0:
                        reasons.append(f"{hi} HIGH Tox21 endpoint(s)")
                    overall_reason = f"**{' + '.join(reasons)}** — multiple toxicity signals."
                elif hi > 0 or (md >= 3 and avg_score > 0.30):
                    overall = "HIGH CONCERN"
                    overall_icon = "⚠️"
                    overall_color = "#d97706"
                    reasons = []
                    if hi > 0:
                        reasons.append(f"{hi} HIGH Tox21 endpoint(s)")
                    if md > 0:
                        reasons.append(f"{md} MEDIUM endpoint(s)")
                    overall_reason = f"**{' + '.join(reasons)}** — significant toxicity signals across Tox21 pathways."
                elif md > 0 or n_amed > 0 or avg_score > 0.25:
                    overall = "MODERATE CONCERN"
                    overall_icon = "🔶"
                    overall_color = "#ca8a04"
                    reasons = []
                    if n_amed > 0:
                        reasons.append(f"{n_amed} structural alert(s)")
                    if md > 0:
                        reasons.append(f"{md} MEDIUM Tox21 endpoint(s)")
                    if not reasons:
                        reasons.append(f"elevated average score ({avg_score:.0%})")
                    overall_reason = f"**{' + '.join(reasons)}** — warrants further investigation."
                else:
                    overall = "LOW CONCERN"
                    overall_icon = "✅"
                    overall_color = "#22c55e"
                    overall_reason = "No structural alerts and all Tox21 endpoints are LOW risk."

                st.markdown(
                    f'<div style="background:rgba({int(overall_color[1:3],16)},{int(overall_color[3:5],16)},{int(overall_color[5:7],16)},.15);'
                    f'border:2px solid {overall_color};border-radius:14px;padding:1.2rem 1.5rem;margin:1rem 0;">'
                    f'<span style="font-size:1.8rem">{overall_icon}</span> '
                    f'<span style="font-size:1.3rem;font-weight:800;color:{overall_color}">Overall Risk: {overall}</span>'
                    f'<div style="color:#e2e8f0;margin-top:6px;font-size:.95rem">{overall_reason}</div></div>',
                    unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔴 High Risk", hi)
                c2.metric("🟡 Medium Risk", md)
                c3.metric("🟢 Low Risk", lo)
                c4.metric("📊 Avg Score", f"{avg_score:.1%}")

                if alerts and elevated == 0:
                    st.info(
                        "💡 **Why are the Tox21 scores LOW despite structural alerts?** "
                        "Tox21 only covers 12 in-vitro pathways (nuclear receptors + stress response). "
                        "Many lethal compounds (nerve agents, ion channel toxins, heavy metals) "
                        "kill through mechanisms that Tox21 does NOT test. "
                        "The structural alerts above detect these uncovered mechanisms."
                    )

                st.dataframe(
                    df.sort_values("Ensemble", ascending=False),
                    use_container_width=True, hide_index=True)
                
                # Report Download
                report_txt = generate_technical_report(smiles, df, alerts, profile)
                st.download_button(
                    label="📄 Download Technical Report",
                    data=report_txt,
                    file_name=f"toxicity_report_{smiles[:10]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ─────────────────────────────────────────────────────────────
# TAB: 3D Molecule
# ─────────────────────────────────────────────────────────────
with tab_3d:
    st.subheader("🧊 Interactive 3D Conformers")
    if smiles:
        with st.spinner("Generating 3D conformer..."):
            show_3d_mol(smiles)
    else:
        st.info("Enter a SMILES string in the Analysis tab to view its 3D structure.")

# ─────────────────────────────────────────────────────────────
# TAB: Model Intelligence
# ─────────────────────────────────────────────────────────────
with tab_accuracy:
    metrics_path = BASE_DIR / "data" / "predictions" / "test_task_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.subheader("Test Set Metrics (per task)")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    else:
        st.info("No test metrics file found at data/predictions/test_task_metrics.csv")

    summary_path = BASE_DIR / "data" / "predictions" / "experiment_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        st.subheader("Experiment Summary")
        st.json(summary)

# ─────────────────────────────────────────────────────────────
# TAB: Insights
# ─────────────────────────────────────────────────────────────
with tab_analysis:
    col_drivers, col_alerts = st.columns([1.2, 1])
    
    with col_drivers:
        st.subheader("🧪 Interpretability: Task Drivers")
        if smiles:
            target_task = st.selectbox("Select Endpoint for Analysis", TASKS)
            with st.spinner("Analyzing molecular drivers..."):
                display_shap_analysis(smiles, target_task, models)
        else:
            st.info("Predict a molecule to see its specific toxicity drivers.")

    with col_alerts:
        st.subheader("🚨 Structural Alerts")
        if smiles:
            mol_ins = Chem.MolFromSmiles(smiles)
            alerts = check_structural_alerts(mol_ins)
            if alerts:
                for aname, asev, amech in alerts:
                    css_class = f"alert-{asev.lower()}"
                    st.markdown(f'<div class="{css_class}"><span class="alert-sev">{asev}</span> — <span class="alert-name">{aname}</span><div class="alert-mech">{amech}</div></div>', unsafe_allow_html=True)
            else:
                st.success("No structural alerts detected for this molecule.")
        else:
            st.info("Enter SMILES to run sub-structural safety check.")

    st.divider()

    col_space, col_coverage = st.columns([1, 1])
    with col_space:
        st.subheader("🌌 Chemical Space Mapping")
        umap_path = BASE_DIR / "data" / "plots" / "umap_chemical_space.png"
        if umap_path.exists():
            st.image(str(umap_path), caption="Tox21 Chemical Space (UMAP Projection)")
        else:
            st.info("Chemical space map not found in data/plots.")

    with col_coverage:
        st.subheader("📈 Tox21 Endpoint Coverage")
        st.markdown("""
| Endpoint | Pathway | What it measures |
|----------|---------|-----------------|
| NR-AR | Androgen Receptor | Endocrine disruption — male hormones |
| NR-AR-LBD | AR Ligand-Binding Domain | AR binding activity |
| NR-AhR | Aryl Hydrocarbon Receptor | Dioxin-like toxicity |
| NR-Aromatase | Aromatase | Estrogen synthesis inhibition |
| NR-ER | Estrogen Receptor | Endocrine disruption — female hormones |
| NR-ER-LBD | ER Ligand-Binding Domain | ER binding activity |
| NR-PPAR-gamma | PPARγ | Metabolic / lipid regulation |
| SR-ARE | Antioxidant Response | Oxidative stress (Nrf2) |
| SR-ATAD5 | ATAD5 | Genotoxicity / DNA damage |
| SR-HSE | Heat Shock Response | Protein folding stress |
| SR-MMP | Mitochondrial Membrane | Mitochondrial toxicity |
| SR-p53 | p53 Tumor Suppressor | DNA damage (p53 pathway) |
""")

# ─────────────────────────────────────────────────────────────
# TAB: Batch Predict
# ─────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV with a `smiles` column.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        batch = pd.read_csv(up)
        if "smiles" not in batch.columns:
            st.error("CSV must have a 'smiles' column")
        else:
            prog = st.progress(0)
            rows = []
            alert_counts = []
            for idx, smi in enumerate(batch.smiles):
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    rows.append({t: None for t in TASKS})
                    alert_counts.append(0)
                    continue
                alerts = check_structural_alerts(mol)
                alert_counts.append(len(alerts))
                res = predict_single(str(smi), models)
                if res is not None:
                    row = {t: res.loc[res.Task == t, "Ensemble"].values[0]
                           for t in TASKS if t in res.Task.values}
                    row["structural_alerts"] = len(alerts)
                    row["alert_details"] = "; ".join(f"{a[1]}:{a[0]}" for a in alerts) if alerts else ""
                else:
                    row = {t: None for t in TASKS}
                rows.append(row)
                prog.progress((idx + 1) / len(batch))
            out = pd.concat([batch.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚠️ Model Scope Limitation")
    st.markdown(
        "Tox21 measures **12 specific in-vitro endpoints** (nuclear receptors + "
        "stress response). It does **NOT** detect:\n"
        "- 🧠 Neurotoxicity (AChE inhibition)\n"
        "- ☠️ Acute systemic toxicity\n"
        "- 🫁 Respiratory damage\n"
        "- 🧬 Carcinogenicity\n"
        "- 🏥 Organ-specific damage\n\n"
        "**Structural alerts** supplement the model by flagging known dangerous motifs."
    )
    st.markdown("---")
    st.markdown("### Risk Levels")
    st.markdown(
        "- 🔴 **HIGH** (>50%) — Likely active\n"
        "- 🟡 **MEDIUM** (35-50%) — Borderline activity\n"
        "- 🟢 **LOW** (<35%) — Likely inactive *for this endpoint*\n\n"
        "**Overall Risk** combines Tox21 scores + structural alerts "
        "for a complete picture."
    )
    st.markdown("---")
    st.caption("IIT BHU | ToxGNN + ChemBERTa + XGBoost Ensemble")
