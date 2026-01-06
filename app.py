# app.py — Pregnancy Drug Card (Menon Lab / UTMB)
# Single-flow clinician UI:
#   1) Splash (5 sec) with Menon + UTMB logos side-by-side
#   2) Drug name only (no SMILES input)
#   3) PubChem fetch → Canonical SMILES → RDKit structure + descriptors
#   4) If drug exists in your CSV → show table-based ADME/Tox values
#      else → show “computed-only” (RDKit) + “not available” for table fields
#   5) Natural pregnancy-focused narrative + PDF download
#
# Place these files in the SAME repo folder as app.py:
#   - Master table_260 drugs_ADME_Protox.csv
#   - menon_logo.png   (optional but recommended)
#   - utmb_logo.png    (optional but recommended)

import io
import re
import time
import textwrap
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw, Crippen

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# =========================
# Config
# =========================
APP_TITLE = "Pregnancy Drug Card"
CSV_PATH = "Master table_260 drugs_ADME_Protox.csv"
MENON_LOGO_PATH = "menon_logo.png"
UTMB_LOGO_PATH = "utmb_logo.png"
SPLASH_SECONDS = 5

st.set_page_config(page_title=APP_TITLE, layout="wide")


# =========================
# Styles / Branding
# =========================
st.markdown(
    """
    <style>
      /* Hide Streamlit default chrome a bit */
      [data-testid="stToolbar"] {visibility: hidden;}
      header {visibility: hidden;}
      .block-container {padding-top: 1.2rem; padding-bottom: 3.2rem;}

      .topcard {
        border-radius: 18px;
        border: 1px solid rgba(255,77,166,0.25);
        padding: 14px 14px;
        background: rgba(255,77,166,0.03);
      }
      .section-title {
        font-size: 16px;
        font-weight: 900;
        margin: 0 0 8px 0;
      }
      .muted {
        opacity: 0.82;
        font-size: 13px;
      }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 12px;
        border: 1px solid rgba(255,77,166,0.35);
        background: rgba(255,77,166,0.10);
        color: #ff4da6;
        margin-right: 8px;
        margin-bottom: 8px;
      }

      /* Watermark: pinkish text + black background */
      .menon-watermark {
        position: fixed;
        bottom: 14px;
        right: 16px;
        z-index: 9999;
        font-size: 13px;
        font-weight: 900;
        padding: 8px 12px;
        border-radius: 12px;
        background: rgba(0,0,0,0.90);
        color: #ff4da6;
        border: 1px solid rgba(255,77,166,0.45);
        box-shadow: 0 6px 22px rgba(0,0,0,0.25);
        letter-spacing: 0.2px;
      }
      @media (max-width: 700px) {
        .menon-watermark {
          font-size: 12px;
          padding: 6px 10px;
          bottom: 10px;
          right: 10px;
        }
      }

      /* Splash */
      .splash-card {
        width: 100%;
        padding: 22px 18px;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(0,0,0,0.95), rgba(18,18,18,0.92));
        border: 1px solid rgba(255,77,166,0.40);
        box-shadow: 0 14px 40px rgba(0,0,0,0.45);
        color: #ffffff;
        animation: splashFadeIn 650ms ease-out;
      }
      .splash-title {
        font-size: 28px;
        font-weight: 950;
        margin: 8px 0 0 0;
        letter-spacing: 0.2px;
      }
      .splash-subtitle {
        font-size: 14px;
        opacity: 0.88;
        margin: 6px 0 0 0;
        line-height: 1.35;
      }
      .splash-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,77,166,0.15);
        border: 1px solid rgba(255,77,166,0.35);
        font-size: 12px;
        font-weight: 900;
        color: #ff4da6;
      }
      @keyframes splashFadeIn {
        from {opacity: 0; transform: translateY(14px);}
        to {opacity: 1; transform: translateY(0px);}
      }
      @media (max-width: 700px) {
        .splash-title { font-size: 22px; }
        .splash-subtitle { font-size: 13px; }
      }
    </style>

    <div class="menon-watermark">Developed by The Menon Laboratory, UTMB</div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Pregnancy conditions
# =========================
PREGNANCY_CONDITIONS: Dict[str, str] = {
    "Preterm Birth (PTB) – inflammation-driven":
        "Reduce inflammatory cytokines and limit NF-κB/TLR4 pathway activation while maintaining maternal–fetal safety.",
    "Preterm PROM (pPROM) – membrane weakening/inflammation":
        "Reduce inflammatory signaling and secondary tissue injury risk; prioritize safety and exposure predictability.",
    "Preeclampsia (PE) – inflammatory/vascular stress subtype":
        "Support anti-inflammatory profile with minimal DDI risk and favorable safety flags.",
    "Chorioamnionitis / intrauterine infection inflammation":
        "Anti-inflammatory potential with careful safety flags; interpret alongside infection management context.",
    "Fetal inflammatory response (FIRS) – fetal exposure concern":
        "Balance anti-inflammatory potential with minimized fetal exposure risk.",
}


# =========================
# Utilities
# =========================
def normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def as_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def fmt(v, digits=2) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)

def safe_mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())

def compute_rdkit_bundle(mol) -> Dict[str, float]:
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "RotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "FracCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "cLogP_RDKit": float(Crippen.MolLogP(mol)),
        "InChIKey": (Chem.MolToInchiKey(mol) if mol is not None else "NA"),
    }

def pregnancy_transfer_risk(logp_or_logd, tpsa, pgp_substrate_prob, ppb) -> Tuple[str, list]:
    reasons = []
    score = 0

    if logp_or_logd is not None:
        if logp_or_logd >= 3:
            score += 2; reasons.append(f"lipophilicity high ({logp_or_logd:.2f})")
        elif logp_or_logd >= 2:
            score += 1; reasons.append(f"lipophilicity moderate ({logp_or_logd:.2f})")

    if tpsa is not None:
        if tpsa <= 60:
            score += 2; reasons.append(f"TPSA low ({tpsa:.1f})")
        elif tpsa <= 90:
            score += 1; reasons.append(f"TPSA moderate ({tpsa:.1f})")

    if pgp_substrate_prob is not None:
        if pgp_substrate_prob >= 0.5:
            score -= 2; reasons.append(f"P-gp substrate likely ({pgp_substrate_prob:.2f})")
        else:
            score += 1; reasons.append(f"P-gp substrate unlikely ({pgp_substrate_prob:.2f})")

    if ppb is not None:
        # percent or fraction
        if ppb > 1.0 and ppb >= 95:
            score -= 1; reasons.append(f"PPB high ({ppb:.0f}%)")
        elif ppb <= 1.0 and ppb >= 0.95:
            score -= 1; reasons.append(f"PPB high ({ppb:.2f})")

    if score >= 4: return "High", reasons
    if score >= 2: return "Moderate", reasons
    return "Low", reasons

def build_pdf_bytes(title: str, body_text: str, caution: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x = 50
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 22

    c.setFont("Helvetica", 10)
    wrapped = textwrap.wrap(body_text.replace("**", ""), width=105)
    for line in wrapped:
        c.drawString(x, y, line)
        y -= 13
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    for line in textwrap.wrap(caution, width=115):
        c.drawString(x, y, line)
        y -= 12
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica-Oblique", 9)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def pubchem_lookup(drug_name: str) -> Optional[dict]:
    name = (drug_name or "").strip()
    if not name:
        return None

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/CanonicalSMILES,IUPACName,MolecularWeight/JSON"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        props = data["PropertyTable"]["Properties"][0]
        return {
            "query": name,
            "canonical_smiles": props.get("CanonicalSMILES"),
            "iupac": props.get("IUPACName"),
            "mw_pubchem": props.get("MolecularWeight"),
        }
    except Exception:
        return None


# =========================
# Data load
# =========================
@st.cache_data(show_spinner=False)
def load_table():
    df = pd.read_csv(CSV_PATH)
    df.columns = [normalize_colname(c) for c in df.columns]
    name_col = find_col(df, ["Drug name", "Drug_name", "Name", "drug_name", "drug"])
    smiles_col = find_col(df, ["SMILES", "Smiles", "Canonical SMILES", "canonical_smiles"])
    return df, name_col, smiles_col


# =========================
# Splash screen (once/session)
# =========================
def splash_screen(duration_sec: int = 5):
    if st.session_state.get("splash_done", False):
        return

    splash = st.empty()
    with splash.container():
        st.markdown('<div class="splash-card">', unsafe_allow_html=True)

        # logos row
        lcol, rcol = st.columns(2, gap="large")
        with lcol:
            try:
                st.image(MENON_LOGO_PATH, use_container_width=True)
            except Exception:
                st.caption("Upload: menon_logo.png")
        with rcol:
            try:
                st.image(UTMB_LOGO_PATH, use_container_width=True)
            except Exception:
                st.caption("Upload: utmb_logo.png")

        st.markdown(f'<div class="splash-title">{APP_TITLE}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="splash-subtitle">'
            '<b>Developed by The Menon Laboratory, UTMB</b><br>'
            'Clinician-first pregnancy pharmacology profiling prototype'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="splash-badge">Prototype • v1</div>', unsafe_allow_html=True)

        prog = st.progress(0)
        steps = max(20, duration_sec * 20)
        for i in range(steps):
            prog.progress(int((i + 1) / steps * 100))
            time.sleep(duration_sec / steps)

        st.markdown("</div>", unsafe_allow_html=True)

    splash.empty()
    st.session_state["splash_done"] = True


# =========================
# Report builder (natural narrative)
# =========================
def build_natural_report(
    drug_display_name: str,
    condition: str,
    condition_goal: str,
    table_found: bool,
    pubchem_meta: dict,
    rdkit_desc: dict,
    adme: dict,
    tox: dict,
    heuristics: dict,
) -> Tuple[str, str]:
    # Core interpreted fields
    mw = adme.get("MW", None)
    tpsa = adme.get("TPSA", None)
    bbb = adme.get("BBB Permeability", None)
    pgp_sub = adme.get("P-gp Substrate", None)
    pgp_inh = adme.get("P-gp Inhibitor", None)
    gi_abs = adme.get("GI Absorption", None)
    sol = adme.get("ESOL Solubility", None)
    sol_class = adme.get("ESOL Class", None)

    hep = tox.get("Hepatotoxicity", None)
    neuro = tox.get("Neurotoxicity", None)
    neph = tox.get("Nephrotoxicity", None)
    cardio = tox.get("Cardiotoxicity", None)
    mut = tox.get("Mutagenicity", None)
    carc = tox.get("Carcinogenicity", None)
    immuno = tox.get("Immunotoxicity", None)

    transfer = heuristics.get("Placental transfer likelihood", "NA")
    transfer_basis = heuristics.get("Placental transfer basis", "")

    # Build narrative
    iupac = pubchem_meta.get("iupac", None) if pubchem_meta else None
    inchi = rdkit_desc.get("InChIKey", "NA")
    cLogP = rdkit_desc.get("cLogP_RDKit", None)

    # Interpret tox “overall”
    tox_signals = []
    for label, v in [
        ("hepatotoxicity", hep),
        ("neurotoxicity", neuro),
        ("nephrotoxicity", neph),
        ("cardiotoxicity", cardio),
        ("immunotoxicity", immuno),
        ("mutagenicity", mut),
        ("carcinogenicity", carc),
    ]:
        if v is None:
            continue
        try:
            if float(v) >= 0.5:
                tox_signals.append(label)
        except Exception:
            pass

    tox_summary = "no strong toxicity signals flagged by threshold" if not tox_signals else ("signals flagged for: " + ", ".join(tox_signals))

    # ADME “overall”
    adme_bits = []
    if gi_abs is not None:
        adme_bits.append(f"GI absorption {fmt(gi_abs,2)}")
    if bbb is not None:
        adme_bits.append(f"BBB permeability {fmt(bbb,2)}")
    if pgp_sub is not None:
        adme_bits.append(f"P-gp substrate {fmt(pgp_sub,2)}")
    if pgp_inh is not None:
        adme_bits.append(f"P-gp inhibitor {fmt(pgp_inh,2)}")
    if sol is not None:
        if sol_class is not None:
            adme_bits.append(f"ESOL solubility {fmt(sol,2)} ({sol_class})")
        else:
            adme_bits.append(f"ESOL solubility {fmt(sol,2)}")

    adme_summary = "limited ADME fields available in the current table" if not adme_bits else "; ".join(adme_bits)

    evidence_line = "Table-based ADME/Toxicity predictions were found in your master dataset." if table_found else (
        "This drug was not found in the master dataset by name; the report uses PubChem + RDKit chemistry only, "
        "and table-based ADME/Toxicity fields are shown as NA."
    )

    report = (
        f"**{drug_display_name}** was profiled for **{condition}**. The clinical objective for this condition is: "
        f"*{condition_goal}*.\n\n"
        f"{evidence_line}\n\n"
        f"**Identity & chemistry:** InChIKey {inchi}. "
        f"{('IUPAC: ' + iupac + '. ') if iupac else ''}"
        f"Computed MW {fmt(mw,1)} and TPSA {fmt(tpsa,1)}; RDKit cLogP {fmt(cLogP,2)}.\n\n"
        f"**ADME considerations for pregnancy:** {adme_summary}. "
        f"Based on a transparent heuristic ({transfer_basis}), **placental transfer likelihood is {transfer}**.\n\n"
        f"**Toxicity considerations:** {tox_summary}. "
        f"If toxicity probabilities are high, prioritize short-term/acute use only when clinically justified and confirm with targeted assays.\n\n"
        f"**Recommendation (research-support):** Use this summary to prioritize experimental validation "
        f"(e.g., inflammatory cytokine readouts, transporter assays, exposure modeling)."
    )

    caution = (
        "Clinical caution: This tool provides research-support summarization using PubChem retrieval, in-silico predictions "
        "from your dataset (when available), and simple transparent heuristics. It is not a clinical recommendation."
    )
    return report, caution


# =========================
# Main UI
# =========================
splash_screen(SPLASH_SECONDS)

# Load table
try:
    df, name_col, smiles_col = load_table()
except Exception as e:
    st.error(f"Failed to load CSV ({CSV_PATH}): {e}")
    st.stop()

# Top header (single tab feel)
top = st.container()
with top:
    st.markdown('<div class="topcard">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([6, 2.4, 1.2], gap="small")
    with c1:
        st.markdown(f"## {APP_TITLE}")
        st.markdown('<div class="muted">Product of The Menon Laboratory, UTMB</div>', unsafe_allow_html=True)
    with c2:
        selected_condition = st.selectbox(
            "Pregnancy condition",
            list(PREGNANCY_CONDITIONS.keys()),
            index=0,
            key="cond_select",
        )
    with c3:
        if st.button("Exit", use_container_width=True, key="exit_btn"):
            st.markdown("<script>window.open('','_self'); window.close();</script>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Single input area (no examples)
left, right = st.columns([1.05, 2.25], gap="large")

with left:
    st.markdown("### Search")
    drug_name = st.text_input("Drug name", label_visibility="visible", key="drug_name_input")
    run = st.button("Generate Pregnancy Drug Card", use_container_width=True, key="run_btn")
    st.caption("Input: drug name only. The app fetches structure via PubChem automatically.")

with right:
    if not run:
        st.info("Enter a drug name on the left, then click **Generate Pregnancy Drug Card**.")
        st.stop()

    if not drug_name.strip():
        st.warning("Please enter a drug name.")
        st.stop()

    # 1) PubChem lookup
    with st.spinner("Fetching structure from PubChem..."):
        meta = pubchem_lookup(drug_name)

    if not meta or not meta.get("canonical_smiles"):
        st.error("PubChem could not resolve this drug name. Try a generic name (e.g., 'dexamethasone').")
        st.stop()

    smiles = meta["canonical_smiles"]
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        st.error("PubChem returned a structure, but RDKit could not parse it.")
        st.stop()

    rd = compute_rdkit_bundle(mol)

    # 2) Try to find drug in your table by name
    table_found = False
    row = None
    if name_col:
        # Prefer exact match first, then contains
        names = df[name_col].fillna("").astype(str)
        exact = df[names.str.lower() == drug_name.strip().lower()]
        if len(exact) > 0:
            row = exact.iloc[0]
            table_found = True
        else:
            contains = df[names.str.lower().str.contains(drug_name.strip().lower(), na=False)]
            if len(contains) > 0:
                row = contains.iloc[0]
                table_found = True

    # 3) Collect ADME / Tox fields (from table when available)
    #    These are “best-effort” column matches. Missing columns -> NA.
    def get_table_value(candidates):
        if (not table_found) or (row is None):
            return None
        col = find_col(df, candidates)
        if not col:
            return None
        return row.get(col)

    # ADME columns you mentioned (best-effort)
    adme = {
        "MW": rd.get("MW"),
        "TPSA": rd.get("TPSA"),

        "GI Absorption": as_float(get_table_value(["GI Absorption", "GI_absorption", "GI absorption"])),
        "BBB Permeability": as_float(get_table_value(["BBB Permeability", "BBB permeability", "Blood brain permeability", "BBB"])),
        "P-gp Substrate": as_float(get_table_value(["Pgp Substrate", "P-gp Substrate", "P-gp substrate", "Pgp substrate"])),
        "P-gp Inhibitor": as_float(get_table_value(["Pgp Inhibitor", "P-gp Inhibitor", "P-gp inhibitor", "Pgp inhibitor"])),

        "Log Kp (Skin Permeability)": as_float(get_table_value(["LogKp", "Log Kp", "Skin permeability", "Log Kp (Skin Permeability)"])),
        "Lipinski Violations": as_float(get_table_value(["Lipinski Violations", "Lipinski", "Lipinski violations"])),
        "Leadlikeness Violations": as_float(get_table_value(["Leadlikeness Violations", "Leadlikeness", "Lead likeness"])),
        "Bioavailability Score": as_float(get_table_value(["Bioavailability Score", "bioavailability score"])),
        "PAINS Alerts": as_float(get_table_value(["PAINS", "PAINS Alerts", "PAINS alert"])),
        "Brenk Alerts": as_float(get_table_value(["Brenk", "Brenk Alerts", "Brenk alert"])),
        "ESOL Solubility": as_float(get_table_value(["ESOL Solubility", "ESOL", "Solubility"])),
        "ESOL Class": get_table_value(["ESOL Class", "ESOL class", "Solubility Class"]),
    }

    # Toxicity columns you mentioned (best-effort)
    tox = {
        "Hepatotoxicity": as_float(get_table_value(["Hepatotoxicity", "Hepatotox"])),
        "Nephrotoxicity": as_float(get_table_value(["Nephrotoxicity", "Nephrotox"])),
        "Neurotoxicity": as_float(get_table_value(["Neurotoxicity", "Neurotox"])),
        "Respiratory toxicity": as_float(get_table_value(["Respiratory toxicity", "Respiratory Toxicity"])),
        "Cardiotoxicity": as_float(get_table_value(["Cardiotoxicity", "Cardiotox"])),
        "Carcinogenicity": as_float(get_table_value(["Carcinogenicity", "Carcinogen"])),
        "Mutagenicity": as_float(get_table_value(["Mutagenicity", "Mutagenic"])),
        "Immunotoxicity": as_float(get_table_value(["Immunotoxicity", "Immunotox"])),

        # Tox21 pathway-like fields (if present in your table)
        "nrf2/ARE": as_float(get_table_value(["nrf2/ARE", "NRF2", "nrf2"])),
        "PPAR-gamma": as_float(get_table_value(["PPAR-gamma", "PPAR Gamma", "PPARγ"])),
        "p53": as_float(get_table_value(["p53", "P53"])),
        "MMP": as_float(get_table_value(["MMP", "Mitochondrial Membrane Potential", "mitochondrial membrane potential"])),
    }

    # 4) Heuristics (placental transfer)
    # Use table logP/logD if present; else RDKit cLogP
    logp_tbl = as_float(get_table_value(["LogP", "logP", "cLogP", "XlogP"]))
    logd_tbl = as_float(get_table_value(["LogD", "logD"]))
    lipoph = logp_tbl if logp_tbl is not None else (logd_tbl if logd_tbl is not None else rd.get("cLogP_RDKit"))
    ppb_tbl = as_float(get_table_value(["PPB", "Plasma Protein Binding", "protein binding"]))
    pgp_sub_tbl = adme.get("P-gp Substrate", None)
    transfer_label, transfer_reasons = pregnancy_transfer_risk(lipoph, rd.get("TPSA"), pgp_sub_tbl, ppb_tbl)

    heur = {
        "Placental transfer likelihood": transfer_label,
        "Placental transfer basis": "lipophilicity + TPSA + PPB + P-gp (when available)",
    }

    # =========================
    # Display: single professional “drug card”
    # =========================
    st.markdown("### Pregnancy Drug Card")

    # Structure image 16:9
    st.image(Draw.MolToImage(mol, size=(1200, 675)), caption="Structure (16:9)", use_container_width=True)

    # Identity
    st.markdown(f"### {drug_name.strip()}")
    if meta.get("iupac"):
        st.caption(meta["iupac"])

    st.code(f"PubChem Canonical SMILES: {smiles}", language="text")

    # Quick pills
    st.markdown(
        f"""
        <span class="pill">Condition: {selected_condition}</span>
        <span class="pill">Placental transfer: {transfer_label}</span>
        <span class="pill">{'Dataset-matched' if table_found else 'Not in dataset (RDKit-only)'} </span>
        """,
        unsafe_allow_html=True,
    )

    # Metrics grid
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("MW", fmt(rd.get("MW"), 1))
        st.metric("TPSA", fmt(rd.get("TPSA"), 1))
    with colB:
        st.metric("RDKit cLogP", fmt(rd.get("cLogP_RDKit"), 2))
        st.metric("GI Absorption", fmt(adme.get("GI Absorption"), 2))
    with colC:
        st.metric("BBB Permeability", fmt(adme.get("BBB Permeability"), 2))
        st.metric("Bioavailability Score", fmt(adme.get("Bioavailability Score"), 2))

    # Sub-sections: ADME / Toxicity (compact, clinician-friendly)
    s1, s2 = st.columns([1.1, 1.0], gap="large")

    with s1:
        with st.container(border=True):
            st.markdown('<div class="section-title">ADME parameters (table-based where available)</div>', unsafe_allow_html=True)
            st.write(f"**P-gp substrate:** {fmt(adme.get('P-gp Substrate'), 2)}")
            st.write(f"**P-gp inhibitor:** {fmt(adme.get('P-gp Inhibitor'), 2)}")
            st.write(f"**PPB:** {fmt(ppb_tbl, 2)}")
            st.write(f"**Log Kp (skin permeability):** {fmt(adme.get('Log Kp (Skin Permeability)'), 2)}")
            st.write(f"**Lipinski violations:** {fmt(adme.get('Lipinski Violations'), 0)}")
            st.write(f"**Leadlikeness violations:** {fmt(adme.get('Leadlikeness Violations'), 0)}")
            st.write(f"**PAINS alerts:** {fmt(adme.get('PAINS Alerts'), 0)}")
            st.write(f"**Brenk alerts:** {fmt(adme.get('Brenk Alerts'), 0)}")
            solv = adme.get("ESOL Solubility")
            solv_class = adme.get("ESOL Class")
            st.write(f"**ESOL solubility:** {fmt(solv, 2)}" + (f" (**{solv_class}**)" if solv_class not in [None, "NA", np.nan] else ""))

    with s2:
        with st.container(border=True):
            st.markdown('<div class="section-title">Toxicity parameters (table-based where available)</div>', unsafe_allow_html=True)
            st.write(f"**Hepatotoxicity:** {fmt(tox.get('Hepatotoxicity'), 2)}")
            st.write(f"**Neurotoxicity:** {fmt(tox.get('Neurotoxicity'), 2)}")
            st.write(f"**Nephrotoxicity:** {fmt(tox.get('Nephrotoxicity'), 2)}")
            st.write(f"**Cardiotoxicity:** {fmt(tox.get('Cardiotoxicity'), 2)}")
            st.write(f"**Respiratory toxicity:** {fmt(tox.get('Respiratory toxicity'), 2)}")
            st.write(f"**Mutagenicity:** {fmt(tox.get('Mutagenicity'), 2)}")
            st.write(f"**Carcinogenicity:** {fmt(tox.get('Carcinogenicity'), 2)}")
            st.write(f"**Immunotoxicity:** {fmt(tox.get('Immunotoxicity'), 2)}")
            # Optional tox21
            st.markdown('<div class="muted">Tox21 pathway flags (if present)</div>', unsafe_allow_html=True)
            st.write(f"**nrf2/ARE:** {fmt(tox.get('nrf2/ARE'), 2)}")
            st.write(f"**PPAR-gamma:** {fmt(tox.get('PPAR-gamma'), 2)}")
            st.write(f"**p53:** {fmt(tox.get('p53'), 2)}")
            st.write(f"**Mito membrane potential:** {fmt(tox.get('MMP'), 2)}")

    # Natural pregnancy interpretation (no bullet list shown)
    condition_goal = PREGNANCY_CONDITIONS[selected_condition]
    report_text, caution = build_natural_report(
        drug_display_name=drug_name.strip(),
        condition=selected_condition,
        condition_goal=condition_goal,
        table_found=table_found,
        pubchem_meta=meta,
        rdkit_desc=rd,
        adme=adme,
        tox=tox,
        heuristics=heur,
    )

    with st.container(border=True):
        st.markdown('<div class="section-title">Pregnancy interpretation</div>', unsafe_allow_html=True)
        st.markdown(report_text)
        st.info(caution)

    # PDF
    pdf_bytes = build_pdf_bytes(
        title=f"{APP_TITLE} — {drug_name.strip()}",
        body_text=re.sub(r"\*\*(.*?)\*\*", r"\1", report_text).replace("\n\n", "\n"),
        caution=caution,
    )

    st.download_button(
        "Print / Download PDF",
        data=pdf_bytes,
        file_name=f"{drug_name.strip().replace(' ', '_')}_pregnancy_drug_card.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"download_pdf_{drug_name.strip().lower()}_{selected_condition}",
    )
