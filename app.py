# app.py — Pregnancy Drug Card (Menon Lab / UTMB) + ML Risk Tier
# Put these files in the SAME repo folder as this app.py:
# 1) Master table_260 drugs_ADME_Protox.csv
# 2) menon_logo.png
# 3) utmb_logo.png
# 4) pregnancy_risk_model.pkl
# 5) feature_columns.json
# 6) label_map.json  (optional; app has a safe fallback)

import io
import json
import re
import time
import numpy as np
import pandas as pd
import streamlit as st

# RDKit (Streamlit Cloud Linux supports conda-based rdkit in many setups;
# if your app already works with RDKit, keep it. If not, you can disable structure rendering below.)
from rdkit import Chem
from rdkit.Chem import Draw

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ML artifacts
import joblib


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Pregnancy Drug Card", layout="wide")


# -----------------------------
# Paths
# -----------------------------
CSV_PATH = "Master table_260 drugs_ADME_Protox.csv"
MENON_LOGO_PATH = "menon_logo.png"
UTMB_LOGO_PATH = "utmb_logo.png"

MODEL_PATH = "pregnancy_risk_model.pkl"
FEATURES_PATH = "feature_columns.json"
LABEL_MAP_PATH = "label_map.json"  # optional

SPLASH_SECONDS = 5


# -----------------------------
# Minimal styling (professional + compact)
# -----------------------------
st.markdown(
    """
    <style>
      .small-muted {opacity:0.80; font-size: 13px;}
      .pill {
        display:inline-block; padding: 6px 10px; border-radius:999px;
        font-weight:800; font-size:12px;
        border:1px solid rgba(255,77,166,0.30);
        background: rgba(255,77,166,0.10);
        color:#ff4da6; margin-right:8px; margin-bottom:6px;
      }
      .brandline { font-weight:800; }
      .section-title { font-size:15px; font-weight:900; margin: 0 0 8px 0; }
      .splash-card {
        width:100%;
        padding:18px 16px;
        border-radius:18px;
        background: linear-gradient(180deg, rgba(0,0,0,0.95), rgba(18,18,18,0.92));
        border: 1px solid rgba(255,77,166,0.38);
        box-shadow: 0 14px 40px rgba(0,0,0,0.35);
        color:#fff;
      }
      .splash-title { font-size: 26px; font-weight: 950; margin: 4px 0 0 0; }
      .splash-sub { font-size: 13px; opacity: 0.88; margin: 6px 0 0 0; line-height: 1.35; }
      @media (max-width:700px){
        .splash-title{font-size:22px;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Pregnancy conditions (kept, but compact)
# -----------------------------
PREGNANCY_CONDITIONS = {
    "Preterm Birth (PTB) – inflammation-driven": "Reduce inflammatory cytokines and limit NF-κB/TLR4 activation while maintaining maternal–fetal safety.",
    "Preterm PROM (pPROM) – membrane weakening/inflammation": "Reduce inflammatory signaling and secondary tissue injury risk; prioritize safety and exposure predictability.",
    "Preeclampsia (PE) – inflammatory/vascular stress subtype": "Support anti-inflammatory profile with minimal DDI risk and favorable safety flags.",
    "Chorioamnionitis / intrauterine infection inflammation": "Anti-inflammatory potential with careful safety flags; interpret alongside infection management context.",
    "Fetal inflammatory response (FIRS) – fetal exposure concern": "Balance anti-inflammatory potential with minimized fetal exposure risk.",
}


# -----------------------------
# Utility helpers
# -----------------------------
def normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df: pd.DataFrame, candidates) -> str | None:
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


def yn_to_num(v):
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ["yes", "y", "true", "positive", "toxic"]:
            return 1.0
        if t in ["no", "n", "false", "negative", "non-toxic", "nontoxic"]:
            return 0.0
    return v


def safe_mol_from_smiles(smiles: str):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def fmt(v, digits=2):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def build_pdf_bytes(title: str, body_text: str, caution: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x = 50
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 22

    def draw_wrapped(text, y, max_chars=95, line_height=13, font="Helvetica", size=10):
        c.setFont(font, size)
        s = (text or "").replace("\r", "").strip()
        # keep blank lines
        parts = s.split("\n")
        for part in parts:
            if part.strip() == "":
                y -= line_height
                continue
            line = part.strip()
            while len(line) > max_chars:
                cut = line.rfind(" ", 0, max_chars)
                if cut == -1:
                    cut = max_chars
                c.drawString(x, y, line[:cut])
                y -= line_height
                line = line[cut:].strip()
            c.drawString(x, y, line)
            y -= line_height
        return y

    y = draw_wrapped(body_text, y, max_chars=105)
    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    y = draw_wrapped(caution, y, max_chars=110, line_height=12, font="Helvetica-Oblique", size=9)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Splash screen (once/session)
# -----------------------------
def splash_screen(duration_sec: int = 5):
    if st.session_state.get("splash_done", False):
        return

    holder = st.empty()
    with holder.container():
        st.markdown('<div class="splash-card">', unsafe_allow_html=True)

        # Compact logo row
        lcol, ccol, rcol = st.columns([1, 1, 1.2], gap="small")

        with lcol:
            try:
                st.image(MENON_LOGO_PATH, width=130)
            except Exception:
                st.caption("menon_logo.png missing")

        with ccol:
            try:
                st.image(UTMB_LOGO_PATH, width=130)
            except Exception:
                st.caption("utmb_logo.png missing")

        with rcol:
            if st.button("Skip", use_container_width=True, key="splash_skip"):
                st.session_state["splash_done"] = True
                holder.empty()
                st.rerun()

        st.markdown('<div class="splash-title">Pregnancy Drug Card</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="splash-sub"><span class="brandline">Developed by The Menon Laboratory, UTMB</span><br>Prototype • v1</div>',
            unsafe_allow_html=True,
        )

        prog = st.progress(0)
        steps = max(25, duration_sec * 25)
        for i in range(steps):
            prog.progress(int((i + 1) / steps * 100))
            time.sleep(duration_sec / steps)

        st.markdown("</div>", unsafe_allow_html=True)

    holder.empty()
    st.session_state["splash_done"] = True


# -----------------------------
# Load data + ML artifacts
# -----------------------------
@st.cache_data(show_spinner=False)
def load_master_table():
    df = pd.read_csv(CSV_PATH)
    df.columns = [normalize_colname(c) for c in df.columns]
    name_col = find_col(df, ["Drug name", "Drug_name", "Name", "drug_name", "drug"])
    smiles_col = find_col(df, ["SMILES", "Smiles", "Canonical SMILES", "canonical_smiles"])
    return df, name_col, smiles_col


@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    # label map optional (fallback ok)
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    try:
        with open(LABEL_MAP_PATH, "r") as f:
            lm = json.load(f)
        # handle either {"0":"Low"...} or {0:"Low"...}
        label_map = {int(k): v for k, v in lm.items()}
    except Exception:
        pass

    return model, feature_cols, label_map


def predict_risk_from_row(row: pd.Series, df: pd.DataFrame, model, feature_cols, label_map):
    # Ensure features exist in df
    usable = [c for c in feature_cols if c in df.columns]
    if not usable:
        return None

    X = df[usable].copy()

    # normalize Y/N
    for col in usable:
        X[col] = X[col].apply(yn_to_num)

    # numeric coercion where possible
    for col in usable:
        X[col] = pd.to_numeric(X[col], errors="ignore")

    medians = X.median(numeric_only=True)

    x_one = row[usable].copy()
    x_one = x_one.apply(yn_to_num)
    x_one = pd.to_numeric(x_one, errors="ignore")

    # fill numeric missings with medians
    x_one_df = pd.DataFrame([x_one])
    for col in usable:
        if col in medians.index:
            x_one_df[col] = pd.to_numeric(x_one_df[col], errors="coerce")
            x_one_df[col] = x_one_df[col].fillna(medians[col])

    # any remaining NaNs -> 0 (safe fallback)
    x_one_df = x_one_df.fillna(0)

    pred_idx = int(model.predict(x_one_df)[0])
    proba = None
    conf = None
    try:
        proba_arr = model.predict_proba(x_one_df)[0]  # shape (n_classes,)
        proba_arr = np.array(proba_arr, dtype=float)
        conf = float(np.max(proba_arr))
        proba = proba_arr.tolist()
    except Exception:
        pass

    return {
        "label": label_map.get(pred_idx, str(pred_idx)),
        "index": pred_idx,
        "confidence": conf,
        "proba": proba,
        "features_used": usable,
    }


# -----------------------------
# Build clinician-style narrative (no bullets)
# -----------------------------
def build_clinician_report(drug_name: str, condition: str, condition_goal: str, risk_label: str | None, confidence: float | None,
                           adme: dict, tox: dict, ddi: dict):
    # Keep it natural + compact; not a clinical recommendation.
    risk_line = ""
    if risk_label:
        if confidence is not None:
            risk_line = f"Model prediction: **Pregnancy Risk Tier = {risk_label}** (confidence {confidence:.2f})."
        else:
            risk_line = f"Model prediction: **Pregnancy Risk Tier = {risk_label}**."

    # ADME sentence
    adme_parts = []
    for k in ["MW", "LogP", "BBB", "GI Absorption", "P-gp substrate", "Bioavailability Score", "ESOL Solubility"]:
        if k in adme and adme[k] is not None:
            adme_parts.append(f"{k}: {adme[k]}")
    adme_line = "ADME snapshot: " + (", ".join(adme_parts) if adme_parts else "Not available in the master table for this compound.")

    # Toxicity sentence
    tox_parts = []
    for k in ["Hepatotoxicity", "Neurotoxicity", "Nephrotoxicity", "Cardiotoxicity", "Immunotoxicity", "Mutagenicity", "Carcinogenicity"]:
        if k in tox and tox[k] is not None:
            tox_parts.append(f"{k}: {tox[k]}")
    tox_line = "Toxicity snapshot: " + (", ".join(tox_parts) if tox_parts else "Not available in the master table for this compound.")

    # DDI sentence
    ddi_parts = []
    for k in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2E1"]:
        if k in ddi and ddi[k] is not None:
            ddi_parts.append(f"{k}: {ddi[k]}")
    ddi_line = "Metabolism/DDI hints: " + (", ".join(ddi_parts) if ddi_parts else "Not available in the master table for this compound.")

    text = (
        f"### {drug_name}\n"
        f"**Condition focus:** {condition}\n\n"
        f"**Goal:** {condition_goal}\n\n"
        f"{risk_line}\n\n"
        f"{adme_line}\n\n"
        f"{ddi_line}\n\n"
        f"{tox_line}\n"
    )

    caution = (
        "Clinical caution: This tool summarizes a curated master table plus an internal ML classifier trained on those columns. "
        "It is for research/triage only and is not a clinical recommendation."
    )
    return text, caution


# -----------------------------
# App start
# -----------------------------
splash_screen(SPLASH_SECONDS)

# Top header (compact)
top = st.container(border=True)
with top:
    a, b, c = st.columns([5.8, 2.2, 1.0])
    with a:
        st.markdown("## Pregnancy Drug Card")
        st.markdown('<div class="small-muted">Developed by The Menon Laboratory, UTMB</div>', unsafe_allow_html=True)
    with b:
        selected_condition = st.selectbox(
            "Pregnancy condition",
            list(PREGNANCY_CONDITIONS.keys()),
            index=0,
            key="condition_select",
        )
    with c:
        if st.button("Exit", use_container_width=True, key="exit_btn"):
            st.markdown("<script>window.open('','_self'); window.close();</script>", unsafe_allow_html=True)

# Load master table
try:
    df, name_col, smiles_col = load_master_table()
except Exception as e:
    st.error(f"Failed to load master CSV: {e}")
    st.stop()

if not name_col:
    st.error("Could not find a drug name column in your CSV (expected something like 'Drug name' or 'drug_name').")
    st.stop()

# Load ML artifacts (optional but expected)
model = None
feature_cols = None
label_map = None
ml_ready = True
try:
    model, feature_cols, label_map = load_ml_artifacts()
except Exception as e:
    ml_ready = False

# Single-screen UI: one input
left, right = st.columns([1.05, 2.25], gap="large")

with left:
    st.markdown("### Search")
    drug_query = st.text_input(
        "Drug name",
        value="",
        placeholder="Type a drug name (e.g., dexamethasone)",
        key="drug_name_input",
        label_visibility="visible",
    )

    # keep screen clean: only show matches after typing
    if drug_query.strip():
        q = drug_query.strip().lower()
        names = df[name_col].fillna("").astype(str)
        mask = names.str.lower().str.contains(q, na=False)
        matches = df[mask].copy()

        st.caption(f"Matches: {len(matches)}")

        if len(matches) == 0:
            st.warning("Not found in master list. Please try a different spelling (generic name).")
            selected_row = None
        else:
            # If exact match exists, auto-pick it
            exact_mask = names.str.lower().str.strip() == q
            exact = df[exact_mask]
            if len(exact) >= 1:
                selected_row = exact.iloc[0]
            else:
                # Keep dropdown minimal
                options = matches[name_col].fillna("Unknown").astype(str).tolist()
                pick = st.selectbox(
                    "Select match",
                    options,
                    index=0,
                    key="match_select",
                )
                selected_row = matches[matches[name_col].astype(str) == pick].iloc[0]
    else:
        selected_row = None
        st.caption("Enter a drug name to view the card.")

with right:
    if selected_row is None:
        st.info("Waiting for a drug name…")
        st.stop()

    drug_name = str(selected_row[name_col])

    # ---- Structure (16:9) from SMILES in dataset (if available) ----
    if smiles_col and smiles_col in df.columns:
        smiles_val = str(selected_row[smiles_col]) if not pd.isna(selected_row[smiles_col]) else ""
    else:
        smiles_val = ""

    if smiles_val:
        mol = safe_mol_from_smiles(smiles_val)
        if mol:
            st.image(
                Draw.MolToImage(mol, size=(1200, 675)),  # 16:9
                caption="Structure",
                use_container_width=True,
            )

    # ---- Pull commonly used columns (only if present) ----
    def get_col(*cands):
        return find_col(df, list(cands))

    # ADME
    MW_col = get_col("MW", "Molecular Weight", "RDKit_MW")
    LogP_col = get_col("LogP", "cLogP", "XlogP")
    BBB_col = get_col("BBB", "BBB permeability", "Blood brain barrier", "BBB_prob")
    GI_col = get_col("GI Absorption", "GI_absorption", "GIA")
    PGP_sub_col = get_col("P-gp substrate", "Pgp substrate", "P-gp_Substrate", "Pgp_Substrate")
    Bioav_col = get_col("Bioavailability Score", "Bioavailability", "BA Score")
    ESOL_col = get_col("ESOL Solubility", "ESOL", "Solubility")

    # Toxicity
    Hep_col = get_col("Hepatotoxicity", "Hepatotox")
    Neuro_col = get_col("Neurotoxicity")
    Nephro_col = get_col("Nephrotoxicity")
    Cardio_col = get_col("Cardiotoxicity")
    Immuno_col = get_col("Immunotoxicity", "Immunotox")
    Mut_col = get_col("Mutagenicity")
    Carc_col = get_col("Carcinogenicity")

    # CYP/DDI (use whatever exists)
    cyp_cols = {}
    for enz in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2E1"]:
        cyp_cols[enz] = get_col(f"{enz} substrate", f"{enz}_Substrate", f"{enz} inhibitor", f"{enz}_Inhibitor")

    adme = {
        "MW": fmt(as_float(selected_row[MW_col]), 1) if MW_col else None,
        "LogP": fmt(as_float(selected_row[LogP_col]), 2) if LogP_col else None,
        "BBB": fmt(as_float(selected_row[BBB_col]), 2) if BBB_col else None,
        "GI Absorption": str(selected_row[GI_col]) if GI_col and not pd.isna(selected_row[GI_col]) else None,
        "P-gp substrate": fmt(as_float(selected_row[PGP_sub_col]), 2) if PGP_sub_col else None,
        "Bioavailability Score": fmt(as_float(selected_row[Bioav_col]), 2) if Bioav_col else None,
        "ESOL Solubility": fmt(as_float(selected_row[ESOL_col]), 2) if ESOL_col else None,
    }

    tox = {
        "Hepatotoxicity": fmt(as_float(selected_row[Hep_col]), 2) if Hep_col else None,
        "Neurotoxicity": fmt(as_float(selected_row[Neuro_col]), 2) if Neuro_col else None,
        "Nephrotoxicity": fmt(as_float(selected_row[Nephro_col]), 2) if Nephro_col else None,
        "Cardiotoxicity": fmt(as_float(selected_row[Cardio_col]), 2) if Cardio_col else None,
        "Immunotoxicity": fmt(as_float(selected_row[Immuno_col]), 2) if Immuno_col else None,
        "Mutagenicity": fmt(as_float(selected_row[Mut_col]), 2) if Mut_col else None,
        "Carcinogenicity": fmt(as_float(selected_row[Carc_col]), 2) if Carc_col else None,
    }

    ddi = {}
    for enz, col in cyp_cols.items():
        if col and col in df.columns:
            v = selected_row[col]
            ddi[enz] = fmt(as_float(v), 2) if not isinstance(v, str) else str(v)
        else:
            ddi[enz] = None

    # ---- ML prediction ----
    condition_goal = PREGNANCY_CONDITIONS[selected_condition]

    pred = None
    if ml_ready:
        pred = predict_risk_from_row(selected_row, df, model, feature_cols, label_map)

    # ---- Quick pills (compact) ----
    pill_line = st.container()
    with pill_line:
        pills = []
        if pred:
            conf = pred["confidence"]
            conf_txt = f"{conf:.2f}" if conf is not None else "NA"
            pills.append(f'<span class="pill">Risk Tier: {pred["label"]}</span>')
            pills.append(f'<span class="pill">Confidence: {conf_txt}</span>')
        else:
            pills.append('<span class="pill">Risk Tier: NA (model not loaded)</span>')

        pills.append(f'<span class="pill">Condition: {selected_condition}</span>')
        st.markdown("".join(pills), unsafe_allow_html=True)

    # ---- Professional single-window “card” layout ----
    st.markdown(f"### {drug_name}")

    # compact metrics grid
    g1, g2, g3 = st.columns(3)
    g1.metric("MW", adme["MW"] or "NA")
    g2.metric("LogP", adme["LogP"] or "NA")
    g3.metric("BBB", adme["BBB"] or "NA")

    # two compact cards
    cA, cB = st.columns([1.2, 1.0], gap="large")

    with cA:
        with st.container(border=True):
            st.markdown('<div class="section-title">ADME (from master table)</div>', unsafe_allow_html=True)
            st.write(f"**GI Absorption:** {adme['GI Absorption'] or 'NA'}")
            st.write(f"**P-gp substrate:** {adme['P-gp substrate'] or 'NA'}")
            st.write(f"**Bioavailability Score:** {adme['Bioavailability Score'] or 'NA'}")
            st.write(f"**ESOL Solubility:** {adme['ESOL Solubility'] or 'NA'}")

        with st.container(border=True):
            st.markdown('<div class="section-title">Toxicity (from master table)</div>', unsafe_allow_html=True)
            # Only show non-empty fields (keeps it clean)
            shown_any = False
            for k, v in tox.items():
                if v is not None:
                    st.write(f"**{k}:** {v}")
                    shown_any = True
            if not shown_any:
                st.write("Not available in the master table for this compound.")

    with cB:
        with st.container(border=True):
            st.markdown('<div class="section-title">Metabolism / DDI hints</div>', unsafe_allow_html=True)
            shown_any = False
            for k, v in ddi.items():
                if v is not None:
                    st.write(f"**{k}:** {v}")
                    shown_any = True
            if not shown_any:
                st.write("Not available in the master table for this compound.")

        with st.container(border=True):
            st.markdown('<div class="section-title">ML prediction</div>', unsafe_allow_html=True)
            if pred:
                st.write(f"**Pregnancy Risk Tier:** {pred['label']}")
                if pred["confidence"] is not None:
                    st.write(f"**Confidence:** {pred['confidence']:.2f}")
                st.caption("Note: This is a model trained on the master table columns; it does not replace clinical judgement.")
            else:
                st.warning("Model artifacts not found/loaded. Upload model files to the repo (pkl + json).")

    # ---- Natural clinician-style report (NO bullets) ----
    report_text, caution = build_clinician_report(
        drug_name=drug_name,
        condition=selected_condition,
        condition_goal=condition_goal,
        risk_label=(pred["label"] if pred else None),
        confidence=(pred["confidence"] if pred else None),
        adme=adme,
        tox=tox,
        ddi=ddi,
    )

    with st.container(border=True):
        st.markdown('<div class="section-title">Pregnancy interpretation</div>', unsafe_allow_html=True)
        st.markdown(report_text)
        st.info(caution)

    # ---- PDF export (unique key to avoid StreamlitDuplicateElementId) ----
    pdf_bytes = build_pdf_bytes(
        title=f"Pregnancy Drug Card — {drug_name}",
        body_text=re.sub(r"\*\*(.*?)\*\*", r"\\1", report_text).replace("### ", "").strip(),
        caution=caution,
    )

    st.download_button(
        "Print / Download PDF",
        data=pdf_bytes,
        file_name=f"{drug_name}_pregnancy_drug_card.pdf".replace(" ", "_"),
        mime="application/pdf",
        use_container_width=True,
        key=f"pdf_{drug_name}",  # IMPORTANT: unique key prevents duplicate element IDs
    )
