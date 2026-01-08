# app.py — Pregnancy Drug Card (Menon Lab / UTMB) with ML (Pregnancy_Risk_Tier)
# ------------------------------------------------------------
# Requires files in same folder:
#   - Master table_260 drugs_ADME_Protox.csv
#   - pregnancy_risk_model.pkl
#   - feature_columns.json
#   - menon_logo.png
#   - utmb_logo.png
# ------------------------------------------------------------

import io
import json
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# =========================
# Page config
# =========================
st.set_page_config(page_title="Pregnancy Drug Card", layout="wide")

CSV_PATH = "Master table_260 drugs_ADME_Protox.csv"
MENON_LOGO_PATH = "menon_logo.png"
UTMB_LOGO_PATH = "utmb_logo.png"

MODEL_PATH = "pregnancy_risk_model.pkl"
FEATURES_PATH = "feature_columns.json"

SPLASH_SECONDS = 5


# =========================
# Style
# =========================
st.markdown(
    """
    <style>
      .splash-card {
        width: 100%;
        padding: 18px 16px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(0,0,0,0.96), rgba(18,18,18,0.92));
        border: 1px solid rgba(255,77,166,0.35);
        box-shadow: 0 14px 40px rgba(0,0,0,0.40);
        color: #ffffff;
        animation: splashFadeIn 650ms ease-out;
      }
      .splash-title {
        font-size: 28px;
        font-weight: 950;
        margin: 6px 0 0 0;
        letter-spacing: 0.2px;
      }
      .splash-subtitle {
        font-size: 14px;
        opacity: 0.88;
        margin: 6px 0 0 0;
        line-height: 1.35;
      }
      .badge {
        display: inline-block;
        margin-top: 10px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,77,166,0.14);
        border: 1px solid rgba(255,77,166,0.30);
        font-size: 12px;
        font-weight: 800;
        color: #ff4da6;
      }
      @keyframes splashFadeIn {
        from {opacity: 0; transform: translateY(12px);}
        to {opacity: 1; transform: translateY(0px);}
      }
      .section-title {
        font-size: 15px;
        font-weight: 900;
        margin: 0 0 8px 0;
      }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 12px;
        border: 1px solid rgba(255,77,166,0.30);
        background: rgba(255,77,166,0.10);
        color: #ff4da6;
        margin-right: 8px;
        margin-bottom: 6px;
      }
      .menon-watermark {
        position: fixed;
        bottom: 14px;
        right: 16px;
        z-index: 9999;
        font-size: 13px;
        font-weight: 900;
        padding: 8px 12px;
        border-radius: 12px;
        background: rgba(0,0,0,0.88);
        color: #ff4da6;
        border: 1px solid rgba(255,77,166,0.35);
        box-shadow: 0 6px 22px rgba(0,0,0,0.22);
      }
      @media (max-width: 700px) {
        .splash-title { font-size: 22px; }
        .splash-subtitle { font-size: 13px; }
        .menon-watermark { font-size: 12px; padding: 7px 10px; }
      }
    </style>
    <div class="menon-watermark">Developed by The Menon Laboratory, UTMB</div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Conditions (optional dropdown)
# =========================
PREGNANCY_CONDITIONS = {
    "Preterm Birth (PTB) – inflammation-driven": "Reduce inflammatory cytokines and limit NF-κB/TLR4 activation while maintaining maternal–fetal safety.",
    "Preterm PROM (pPROM) – membrane weakening/inflammation": "Reduce inflammatory signaling and secondary tissue injury risk; prioritize safety and exposure predictability.",
    "Preeclampsia (PE) – inflammatory/vascular stress subtype": "Support anti-inflammatory profile with minimal DDI risk and favorable safety flags.",
    "Chorioamnionitis / intrauterine infection inflammation": "Anti-inflammatory potential with careful safety flags; interpret alongside infection management context.",
    "Fetal inflammatory response (FIRS) – fetal exposure concern": "Balance anti-inflammatory potential with minimized fetal exposure risk.",
}


# =========================
# Utilities
# =========================
def normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def as_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def safe_mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def compute_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def compute_rdkit_desc(mol):
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
        "RotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
        "FracCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def build_pdf_bytes(title: str, body: str, clinical_note: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    x = 50
    y = height - 60

    def draw_wrapped(text, y, max_chars=98, line_h=13, font="Helvetica", size=10):
        c.setFont(font, size)
        s = (text or "").strip()
        lines = []
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            lines.append(s[:cut])
            s = s[cut:].strip()
        if s:
            lines.append(s)
        for ln in lines:
            c.drawString(x, y, ln)
            y -= line_h
        return y

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 24

    y = draw_wrapped(body, y)
    y -= 10

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Clinical note (prototype guidance)")
    y -= 16

    y = draw_wrapped(clinical_note, y, max_chars=105, font="Helvetica-Oblique", size=10, line_h=12)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def risk_to_clinical_note(risk_label: str, confidence: float | None):
    conf_txt = f" (model confidence ~{confidence:.2f})" if confidence is not None else ""
    if risk_label == "Low":
        return (
            "Low predicted pregnancy risk tier" + conf_txt +
            ". This suggests a comparatively favorable in-silico risk profile within this curated dataset. "
            "Still confirm with authoritative pregnancy guidance, dosing, and patient context."
        )
    if risk_label == "Moderate":
        return (
            "Moderate predicted pregnancy risk tier" + conf_txt +
            ". Consider benefit-risk, gestational age, dose and duration. "
            "Prefer careful monitoring and confirm with authoritative pregnancy guidance."
        )
    if risk_label == "High":
        return (
            "High predicted pregnancy risk tier" + conf_txt +
            ". This suggests higher concern signals in the current model/dataset. "
            "Prefer alternatives when available; if used, keep to the minimum effective dose/duration and consider specialist oversight."
        )
    return (
        "Risk tier could not be determined confidently. Treat as research-only output and confirm with authoritative guidance."
    )


def splash_screen(duration_sec: int = 5):
    if st.session_state.get("splash_done", False):
        return

    splash = st.empty()
    with splash.container():
        st.markdown('<div class="splash-card">', unsafe_allow_html=True)

        # medium logos side-by-side
        lcol, rcol, scol = st.columns([2.0, 2.0, 1.0], gap="small")
        with lcol:
            try:
                st.image(MENON_LOGO_PATH, width=190)
            except Exception:
                st.caption("Missing menon_logo.png")
        with rcol:
            try:
                st.image(UTMB_LOGO_PATH, width=190)
            except Exception:
                st.caption("Missing utmb_logo.png")
        with scol:
            if st.button("Skip", use_container_width=True, key="skip_splash"):
                st.session_state["splash_done"] = True
                splash.empty()
                st.rerun()

        st.markdown('<div class="splash-title">Pregnancy Drug Card</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="splash-subtitle">'
            'Developed by <b>The Menon Laboratory, UTMB</b><br>'
            'Prototype for research decision-support (not a clinical recommendation)'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="badge">v1 • Curated dataset + ML</div>', unsafe_allow_html=True)

        prog = st.progress(0)
        steps = max(20, duration_sec * 20)
        for i in range(steps):
            prog.progress(int((i + 1) / steps * 100))
            time.sleep(duration_sec / steps)

        st.markdown("</div>", unsafe_allow_html=True)

    splash.empty()
    st.session_state["splash_done"] = True


@st.cache_data(show_spinner=False)
def load_master():
    df = pd.read_csv(CSV_PATH)
    df.columns = [normalize_colname(c) for c in df.columns]

    name_col = find_col(df, ["Drug name", "Drug_name", "Name", "drug_name", "drug"])
    smiles_col = find_col(df, ["SMILES", "Smiles", "Canonical SMILES", "canonical_smiles"])

    if name_col is None:
        raise ValueError("No drug name column found. Add a 'Drug name' (or similar) column.")
    if smiles_col is None:
        # ok: we won't require smiles for clinician flow
        smiles_col = None

    # normalize name for matching
    df["_name_norm"] = df[name_col].fillna("").astype(str).str.strip().str.lower()

    return df, name_col, smiles_col


@st.cache_resource(show_spinner=False)
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feat_cols = json.load(f)
    return model, feat_cols


def prep_features_for_row(row: pd.Series, feat_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_cols}])

    # map common yes/no style to 0/1 if needed
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].map({"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0})

    # numeric fill
    for col in X.columns:
        if X[col].dtype != object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X


def predict_risk(model, X: pd.DataFrame):
    # labels used in training: Low=0, Moderate=1, High=2
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        proba = float(np.max(p))
    label = {0: "Low", 1: "Moderate", 2: "High"}.get(pred, "Unknown")
    return label, proba


# =========================
# App start
# =========================
splash_screen(SPLASH_SECONDS)

# Load data + model
try:
    df, NAME_COL, SMILES_COL = load_master()
except Exception as e:
    st.error(f"Failed to load master CSV: {e}")
    st.stop()

try:
    model, MODEL_FEATURES = load_model_assets()
except Exception as e:
    st.error(
        f"ML model files not found or failed to load. Ensure '{MODEL_PATH}' and '{FEATURES_PATH}' are in the repo. Error: {e}"
    )
    st.stop()

# Header (single clean page)
top = st.container(border=True)
with top:
    c1, c2, c3 = st.columns([6, 2.4, 1.0])
    with c1:
        st.markdown("## Pregnancy Drug Card")
        st.caption("Product of The Menon Laboratory, UTMB")
    with c2:
        condition = st.selectbox("Pregnancy condition", list(PREGNANCY_CONDITIONS.keys()), index=0)
    with c3:
        if st.button("Exit", use_container_width=True, key="exit_btn"):
            st.markdown("<script>window.open('','_self'); window.close();</script>", unsafe_allow_html=True)

# Search only (no examples)
left, right = st.columns([1.1, 2.3], gap="large")

with left:
    st.markdown("### Search")
    drug_name_in = st.text_input(
        "Drug name",
        placeholder="Type a drug name (e.g., dexamethasone)",
        label_visibility="visible",
    )

    # Optional research mode for your own use (clinicians can ignore)
    with st.expander("Research mode (optional): paste SMILES to view structure)", expanded=False):
        smiles_in = st.text_area("SMILES (optional)", height=90, placeholder="Paste SMILES here")

with right:
    if not drug_name_in.strip():
        st.info("Enter a drug name to generate the pregnancy drug card.")
        st.stop()

    q = drug_name_in.strip().lower()

    hits = df[df["_name_norm"] == q]
    if hits.empty:
        # partial contains fallback
        hits = df[df["_name_norm"].str.contains(q, na=False)]

    if hits.empty:
        st.warning(
            "This drug name is not in the current curated master list. "
            "This prototype is configured to return results only for compounds present in the dataset."
        )
        st.stop()

    # if multiple matches, let user choose
    if len(hits) > 1:
        options = hits[NAME_COL].astype(str).tolist()
        chosen = st.selectbox("Multiple matches found. Select one:", options, index=0, key="match_pick")
        row = hits[hits[NAME_COL].astype(str) == chosen].iloc[0]
    else:
        row = hits.iloc[0]

    # ML prediction
    X = prep_features_for_row(row, MODEL_FEATURES)
    risk_label, confidence = predict_risk(model, X)

    # Build summary from available columns (don’t force columns that aren’t present)
    # Common fields (if present in your CSV)
    def get_any(cands):
        col = find_col(df, cands)
        return as_float(row[col]) if col else None

    MW = get_any(["MW", "Molecular weight", "Molecular Weight"]) or get_any(["RDKit_MW"])
    LogP = get_any(["LogP", "cLogP", "XlogP"])
    BBB = get_any(["BBB", "BBB permeability", "Blood brain barrier"])
    PGP = get_any(["P-gp substrate", "Pgp substrate", "P-gp_Substrate"])
    GI = row.get(find_col(df, ["GI Absorption"]), None)
    BA = get_any(["Bioavailability Score", "Bioavailability"])
    ESOL = get_any(["ESOL Solubility", "ESOL"])

    # Toxicity quick flags (if present)
    tox_cols = {
        "Hepatotoxicity": find_col(df, ["Hepatotoxicity", "Hepatotox"]),
        "Neurotoxicity": find_col(df, ["Neurotoxicity", "Neurotox"]),
        "Immunotoxicity": find_col(df, ["Immunotoxicity", "Immunotox"]),
        "Carcinogenicity": find_col(df, ["Carcinogenicity", "Carcinogen"]),
        "Mutagenicity": find_col(df, ["Mutagenicity", "Mutagenic"]),
    }
    tox_flags = []
    for lab, col in tox_cols.items():
        if col and col in row.index:
            v = row[col]
            try:
                if isinstance(v, str):
                    if v.strip().lower() in ["yes", "y", "true", "positive", "toxic"]:
                        tox_flags.append(lab)
                else:
                    if pd.notna(v) and float(v) >= 0.5:
                        tox_flags.append(lab)
            except Exception:
                pass

    tox_summary = ", ".join(tox_flags) if tox_flags else "No strong toxicity flags detected (threshold-based)"

    # Optional structure (16:9) if smiles exists or user pasted
    structure_shown = False
    mol = None
    smiles_val = None

    if SMILES_COL and isinstance(row.get(SMILES_COL, None), str) and row.get(SMILES_COL).strip():
        smiles_val = row.get(SMILES_COL).strip()
        mol = safe_mol_from_smiles(smiles_val)

    if mol is None and "smiles_in" in locals() and smiles_in and smiles_in.strip():
        smiles_val = smiles_in.strip()
        mol = safe_mol_from_smiles(smiles_val)

    # Clinical note (requested)
    clinical_note = risk_to_clinical_note(risk_label, confidence)

    # Render card (single page result)
    st.markdown(f"### {row[NAME_COL]}")
    pill_html = (
        f'<span class="pill">Predicted risk tier: {risk_label}</span>'
        f'<span class="pill">Confidence: {("" if confidence is None else f"{confidence:.2f}") or "NA"}</span>'
        f'<span class="pill">Condition: {condition}</span>'
    )
    st.markdown(pill_html, unsafe_allow_html=True)

    if mol is not None:
        img = Draw.MolToImage(mol, size=(1200, 675))  # 16:9
        st.image(img, caption="Structure (optional)", use_container_width=True)
        structure_shown = True

    # Two-column professional layout
    cA, cB = st.columns([1.25, 1.0], gap="large")

    with cA:
        with st.container(border=True):
            st.markdown('<div class="section-title">ADME (from curated dataset)</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("MW", "NA" if MW is None else f"{MW:.1f}")
            m2.metric("LogP", "NA" if LogP is None else f"{LogP:.2f}")
            m3.metric("BBB", "NA" if BBB is None else f"{BBB:.2f}")
            st.write(f"**GI Absorption:** {GI if GI is not None else 'NA'}")
            st.write(f"**P-gp substrate (prob/flag):** {('NA' if PGP is None else (f'{PGP:.2f}' if isinstance(PGP,float) else str(PGP)))}")
            st.write(f"**Bioavailability score:** {'NA' if BA is None else f'{BA:.2f}'}")
            st.write(f"**ESOL solubility:** {'NA' if ESOL is None else f'{ESOL:.2f}'}")

        with st.container(border=True):
            st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
**{row[NAME_COL]}** was profiled for **{condition}**.

- **Condition goal:** {PREGNANCY_CONDITIONS[condition]}
- **Model output:** **{risk_label}** pregnancy risk tier (confidence ~{confidence:.2f}).
- **Toxicity summary:** {tox_summary}

This output is intended for **research prioritization** and **does not constitute clinical recommendation**.
                """.strip()
            )

    with cB:
        with st.container(border=True):
            st.markdown('<div class="section-title">Clinical note (prototype)</div>', unsafe_allow_html=True)
            st.write(clinical_note)
            st.info(
                "Clinical caution: This tool summarizes a curated dataset plus an internal ML model trained on that dataset. "
                "Use alongside authoritative pregnancy drug guidance, patient context, and clinical judgment."
            )

        with st.container(border=True):
            st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)

            pdf_body = (
                f"{row[NAME_COL]} — Condition: {condition}\n\n"
                f"Predicted pregnancy risk tier: {risk_label} (confidence ~{confidence:.2f})\n\n"
                f"ADME snapshot: MW={('NA' if MW is None else round(MW,1))}, LogP={('NA' if LogP is None else round(LogP,2))}, "
                f"BBB={('NA' if BBB is None else round(BBB,2))}, P-gp substrate={('NA' if PGP is None else PGP)}, "
                f"GI Absorption={('NA' if GI is None else GI)}, Bioavailability={('NA' if BA is None else round(BA,2))}, "
                f"ESOL={('NA' if ESOL is None else round(ESOL,2))}.\n\n"
                f"Toxicity summary: {tox_summary}\n\n"
                f"Disclaimer: research decision-support only; not a clinical recommendation."
            )

            pdf_bytes = build_pdf_bytes(
                title=f"Pregnancy Drug Card — {row[NAME_COL]}",
                body=pdf_body,
                clinical_note=clinical_note,
            )

            # IMPORTANT: unique key avoids StreamlitDuplicateElementId
            st.download_button(
                "Print / Download PDF",
                data=pdf_bytes,
                file_name=f"{str(row[NAME_COL]).replace(' ', '_')}_pregnancy_drug_card.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="pdf_download_main",
            )
