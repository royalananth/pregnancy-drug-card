# app.py — Pregnancy Drug Card (Menon Lab / UTMB)
# Flow:
# 1) Splash (5 sec) with Menon + UTMB logos
# 2) Clinician enters DRUG NAME only
# 3) Search MASTER TABLE first:
#    - If found -> show drug card (no PubChem)
#    - If not found -> PubChem fallback (optional), show RDKit-only card if resolved
# 4) Natural pregnancy narrative + PDF

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
      .muted { opacity: 0.82; font-size: 13px; }
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
        .menon-watermark { font-size: 12px; padding: 6px 10px; bottom: 10px; right: 10px; }
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
      .splash-title { font-size: 28px; font-weight: 950; margin: 8px 0 0 0; letter-spacing: 0.2px; }
      .splash-subtitle { font-size: 14px; opacity: 0.88; margin: 6px 0 0 0; line-height: 1.35; }
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
      @keyframes splashFadeIn { from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: translateY(0px);} }
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
# Splash screen
# =========================
def splash_screen(duration_sec: int = 5):
    if st.session_state.get("splash_done", False):
        return
    splash = st.empty()
    with splash.container():
        st.markdown('<div class="splash-card">', unsafe_allow_html=True)

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
            '<div class="splash-subtitle"><b>Developed by The Menon Laboratory, UTMB</b><br>'
            'Clinician-first pregnancy pharmacology profiling prototype</div>',
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
# Matching master table
# =========================
def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s\-\(\)\+\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_in_master(df: pd.DataFrame, name_col: Optional[str], query: str) -> Tuple[bool, Optional[pd.Series], str]:
    """
    Returns: (found, row, match_type)
    match_type: "exact" | "contains" | "none"
    """
    if not name_col:
        return False, None, "none"
    q = normalize_name(query)
    if not q:
        return False, None, "none"

    names = df[name_col].fillna("").astype(str).apply(normalize_name)

    # Exact
    exact_idx = np.where(names.values == q)[0]
    if len(exact_idx) > 0:
        return True, df.iloc[int(exact_idx[0])], "exact"

    # Contains (best first hit)
    contains_mask = names.str.contains(q, na=False)
    if contains_mask.any():
        return True, df[contains_mask].iloc[0], "contains"

    return False, None, "none"


# =========================
# Report builder (natural narrative)
# =========================
def build_report_text(
    drug_display_name: str,
    condition: str,
    condition_goal: str,
    table_found: bool,
    match_type: str,
    rdkit_desc: Optional[dict],
    adme: dict,
    tox: dict,
    transfer_label: str,
) -> Tuple[str, str]:
    mw = adme.get("MW", None)
    tpsa = adme.get("TPSA", None)

    gi_abs = adme.get("GI Absorption", None)
    bbb = adme.get("BBB Permeability", None)
    pgp_sub = adme.get("P-gp Substrate", None)
    pgp_inh = adme.get("P-gp Inhibitor", None)
    bioav = adme.get("Bioavailability Score", None)

    hep = tox.get("Hepatotoxicity", None)
    neuro = tox.get("Neurotoxicity", None)
    neph = tox.get("Nephrotoxicity", None)
    cardio = tox.get("Cardiotoxicity", None)
    mut = tox.get("Mutagenicity", None)
    carc = tox.get("Carcinogenicity", None)
    immuno = tox.get("Immunotoxicity", None)

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

    if table_found:
        match_note = f"Matched in master dataset ({match_type} name match)."
    else:
        match_note = "Not found in master dataset by name; PubChem fallback was used for chemistry only."

    clogp = None
    inchikey = None
    if rdkit_desc:
        clogp = rdkit_desc.get("cLogP_RDKit", None)
        inchikey = rdkit_desc.get("InChIKey", "NA")

    report = (
        f"**{drug_display_name}** was profiled for **{condition}**. The clinical objective for this condition is: "
        f"*{condition_goal}*.\n\n"
        f"{match_note}\n\n"
        f"**Chemistry & exposure context:** MW {fmt(mw,1)}, TPSA {fmt(tpsa,1)}"
        f"{'' if clogp is None else f', RDKit cLogP {fmt(clogp,2)}'}"
        f"{'' if inchikey is None else f', InChIKey {inchikey}'}.\n\n"
        f"**ADME considerations for pregnancy:** GI absorption {fmt(gi_abs,2)}, BBB permeability {fmt(bbb,2)}, "
        f"P-gp substrate {fmt(pgp_sub,2)}, P-gp inhibitor {fmt(pgp_inh,2)}, bioavailability score {fmt(bioav,2)}. "
        f"Using a transparent heuristic, **placental transfer likelihood is {transfer_label}**.\n\n"
        f"**Toxicity considerations:** {tox_summary}. "
        f"If probabilities are high, prioritize short-term/acute use only when clinically justified and confirm with targeted assays.\n\n"
        f"**Recommendation (research-support):** Use this summary to prioritize validation in your pregnancy-relevant models."
    )

    caution = (
        "Clinical caution: This tool provides research-support summarization using your master dataset (when matched) "
        "and simple transparent heuristics. It is not a clinical recommendation."
    )
    return report, caution


# =========================
# Main UI
# =========================
splash_screen(SPLASH_SECONDS)

try:
    df, name_col, smiles_col = load_table()
except Exception as e:
    st.error(f"Failed to load CSV ({CSV_PATH}): {e}")
    st.stop()

# Header
with st.container():
    st.markdown('<div class="topcard">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([6, 2.4, 1.2], gap="small")
    with c1:
        st.markdown(f"## {APP_TITLE}")
        st.markdown('<div class="muted">Product of The Menon Laboratory, UTMB</div>', unsafe_allow_html=True)
    with c2:
        selected_condition = st.selectbox("Pregnancy condition", list(PREGNANCY_CONDITIONS.keys()), index=0, key="cond_select")
    with c3:
        if st.button("Exit", use_container_width=True, key="exit_btn"):
            st.markdown("<script>window.open('','_self'); window.close();</script>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([1.05, 2.25], gap="large")

with left:
    st.markdown("### Search")
    drug_name = st.text_input("Drug name", key="drug_name_input")
    run = st.button("Generate Pregnancy Drug Card", use_container_width=True, key="run_btn")
    st.caption("Primary: master dataset. Fallback: PubChem only if not found in master.")

with right:
    if not run:
        st.info("Enter a drug name on the left, then click **Generate Pregnancy Drug Card**.")
        st.stop()

    if not drug_name.strip():
        st.warning("Please enter a drug name.")
        st.stop()

    condition_goal = PREGNANCY_CONDITIONS[selected_condition]

    # 1) MASTER TABLE FIRST
    found, row, match_type = find_in_master(df, name_col, drug_name)

    # Helper to read from row safely
    def get_row_value(candidates):
        if (not found) or (row is None):
            return None
        col = find_col(df, candidates)
        if not col:
            return None
        return row.get(col)

    # Base values (some from RDKit if we have SMILES)
    rd = None
    mol = None
    smiles = None
    structure_source = None

    if found:
        # Prefer dataset SMILES to render structure (if you have it)
        smiles_val = None
        if smiles_col:
            smiles_val = row.get(smiles_col, None)
        if isinstance(smiles_val, str) and smiles_val.strip():
            smiles = smiles_val.strip()
            mol = safe_mol_from_smiles(smiles)
            if mol is not None:
                rd = compute_rdkit_bundle(mol)
                structure_source = "Master dataset SMILES"
        # If no SMILES in dataset, still show numeric table fields and skip structure safely
    else:
        # 2) PUBCHEM FALLBACK ONLY IF NOT FOUND
        with st.spinner("Not in master dataset — trying PubChem fallback..."):
            meta = pubchem_lookup(drug_name)
        if not meta or not meta.get("canonical_smiles"):
            st.error("Drug not found in master dataset and PubChem could not resolve this name. Please verify spelling / generic name.")
            st.stop()
        smiles = meta["canonical_smiles"]
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            st.error("PubChem returned a structure, but RDKit could not parse it.")
            st.stop()
        rd = compute_rdkit_bundle(mol)
        structure_source = "PubChem Canonical SMILES"

    # ADME columns (best-effort)
    adme = {
        "MW": (rd.get("MW") if rd else as_float(get_row_value(["RDKit_MW", "MW", "Molecular Weight"]))),
        "TPSA": (rd.get("TPSA") if rd else as_float(get_row_value(["RDKit_TPSA", "TPSA"]))),

        "GI Absorption": as_float(get_row_value(["GI Absorption", "GI_absorption", "GI absorption"])),
        "BBB Permeability": as_float(get_row_value(["BBB Permeability", "BBB permeability", "Blood brain permeability", "BBB"])),
        "P-gp Substrate": as_float(get_row_value(["Pgp Substrate", "P-gp Substrate", "P-gp substrate", "Pgp substrate"])),
        "P-gp Inhibitor": as_float(get_row_value(["Pgp Inhibitor", "P-gp Inhibitor", "P-gp inhibitor", "Pgp inhibitor"])),
        "Bioavailability Score": as_float(get_row_value(["Bioavailability Score", "bioavailability score"])),
    }

    # Toxicity columns (best-effort)
    tox = {
        "Hepatotoxicity": as_float(get_row_value(["Hepatotoxicity", "Hepatotox"])),
        "Nephrotoxicity": as_float(get_row_value(["Nephrotoxicity", "Nephrotox"])),
        "Neurotoxicity": as_float(get_row_value(["Neurotoxicity", "Neurotox"])),
        "Cardiotoxicity": as_float(get_row_value(["Cardiotoxicity", "Cardiotox"])),
        "Mutagenicity": as_float(get_row_value(["Mutagenicity", "Mutagenic"])),
        "Carcinogenicity": as_float(get_row_value(["Carcinogenicity", "Carcinogen"])),
        "Immunotoxicity": as_float(get_row_value(["Immunotoxicity", "Immunotox"])),
    }

    # Heuristic placental transfer
    logp_tbl = as_float(get_row_value(["LogP", "logP", "cLogP", "XlogP"]))
    logd_tbl = as_float(get_row_value(["LogD", "logD"]))
    lipoph = logp_tbl if logp_tbl is not None else (logd_tbl if logd_tbl is not None else (rd.get("cLogP_RDKit") if rd else None))
    ppb_tbl = as_float(get_row_value(["PPB", "Plasma Protein Binding", "protein binding"]))
    pgp_sub_tbl = adme.get("P-gp Substrate", None)
    tpsa_val = (rd.get("TPSA") if rd else adme.get("TPSA"))
    transfer_label, _ = pregnancy_transfer_risk(lipoph, tpsa_val, pgp_sub_tbl, ppb_tbl)

    # =========================
    # Display card
    # =========================
    st.markdown("### Pregnancy Drug Card")

    # Structure 16:9 if available
    if mol is not None:
        st.image(Draw.MolToImage(mol, size=(1200, 675)), caption=f"Structure (16:9) • Source: {structure_source}", use_container_width=True)

    st.markdown(f"### {drug_name.strip()}")

    st.markdown(
        f"""
        <span class="pill">Condition: {selected_condition}</span>
        <span class="pill">Placental transfer: {transfer_label}</span>
        <span class="pill">{'Matched in master dataset' if found else 'Fallback: PubChem (not in master)'} </span>
        """,
        unsafe_allow_html=True,
    )

    # Metrics
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("MW", fmt(adme.get("MW"), 1))
        st.metric("TPSA", fmt(adme.get("TPSA"), 1))
    with colB:
        st.metric("GI Absorption", fmt(adme.get("GI Absorption"), 2))
        st.metric("BBB Permeability", fmt(adme.get("BBB Permeability"), 2))
    with colC:
        st.metric("P-gp Substrate", fmt(adme.get("P-gp Substrate"), 2))
        st.metric("Bioavailability Score", fmt(adme.get("Bioavailability Score"), 2))

    s1, s2 = st.columns([1.1, 1.0], gap="large")
    with s1:
        with st.container(border=True):
            st.markdown('<div class="section-title">ADME (from master when matched)</div>', unsafe_allow_html=True)
            st.write(f"**P-gp inhibitor:** {fmt(adme.get('P-gp Inhibitor'), 2)}")
            st.write(f"**PPB:** {fmt(ppb_tbl, 2)}")
            st.write(f"**logP/logD (if available):** {fmt(logp_tbl,2)} / {fmt(logd_tbl,2)}")
    with s2:
        with st.container(border=True):
            st.markdown('<div class="section-title">Toxicity (from master when matched)</div>', unsafe_allow_html=True)
            st.write(f"**Hepatotoxicity:** {fmt(tox.get('Hepatotoxicity'), 2)}")
            st.write(f"**Neurotoxicity:** {fmt(tox.get('Neurotoxicity'), 2)}")
            st.write(f"**Nephrotoxicity:** {fmt(tox.get('Nephrotoxicity'), 2)}")
            st.write(f"**Cardiotoxicity:** {fmt(tox.get('Cardiotoxicity'), 2)}")
            st.write(f"**Mutagenicity / Carcinogenicity:** {fmt(tox.get('Mutagenicity'), 2)} / {fmt(tox.get('Carcinogenicity'), 2)}")
            st.write(f"**Immunotoxicity:** {fmt(tox.get('Immunotoxicity'), 2)}")

    # Natural narrative
    report_text, caution = build_report_text(
        drug_display_name=drug_name.strip(),
        condition=selected_condition,
        condition_goal=condition_goal,
        table_found=found,
        match_type=match_type,
        rdkit_desc=rd,
        adme=adme,
        tox=tox,
        transfer_label=transfer_label,
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
        key=f"download_pdf_{normalize_name(drug_name)}_{selected_condition}",
    )
