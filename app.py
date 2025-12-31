# app.py — Pregnancy Drug Card (Menon Lab / UTMB)
# Put these files in the SAME folder (repo root) as this app.py:
#   - Master table_260 drugs_ADME_Protox.csv
#   - menon_logo.png   (optional)
#   - utmb_logo.png    (optional)
#
# Optional GenAI (novel SMILES):
#   Add OPENAI_API_KEY in Streamlit Secrets.
#   (Streamlit Cloud → App → Settings → Secrets)

import re
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw, Crippen

# PDF generation (Print/Download)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Optional GenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Config
# =========================
st.set_page_config(page_title="Pregnancy Drug Card", layout="wide")

CSV_PATH = "Master table_260 drugs_ADME_Protox.csv"
MENON_LOGO_PATH = "menon_logo.png"
UTMB_LOGO_PATH = "utmb_logo.png"
SPLASH_SECONDS = 5


# =========================
# CSS (Professional UI)
# =========================
st.markdown(
    """
    <style>
      .menon-watermark {
        position: fixed;
        bottom: 14px;
        right: 16px;
        z-index: 9999;
        font-size: 14px;
        font-weight: 900;
        padding: 8px 12px;
        border-radius: 12px;
        background: rgba(0,0,0,0.92);
        color: #ff4da6;
        border: 1px solid rgba(255,77,166,0.45);
        box-shadow: 0 6px 22px rgba(0,0,0,0.25);
        letter-spacing: 0.2px;
      }
      @media (max-width: 700px) {
        .menon-watermark { font-size: 12px; padding: 6px 10px; bottom: 10px; right: 10px; }
      }
      .section-title { font-size: 16px; font-weight: 900; margin: 0 0 8px 0; }
      .muted { opacity: 0.82; font-size: 13px; }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 900;
        font-size: 12px;
        border: 1px solid rgba(255,77,166,0.35);
        background: rgba(255,77,166,0.10);
        color: #ff4da6;
        margin-right: 8px;
        margin-bottom: 6px;
      }
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
        font-size: 30px;
        font-weight: 950;
        margin: 6px 0 0 0;
        letter-spacing: 0.2px;
      }
      .splash-subtitle {
        font-size: 15px;
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
        from {opacity: 0; transform: translateY(12px);}
        to {opacity: 1; transform: translateY(0px);}
      }
      @media (max-width: 700px) {
        .splash-title { font-size: 24px; }
        .splash-subtitle { font-size: 13px; }
      }
    </style>
    <div class="menon-watermark">Developed by The Menon Laboratory, UTMB</div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Pregnancy Conditions (v1)
# =========================
PREGNANCY_CONDITIONS = {
    "Preterm Birth (PTB) – inflammation-driven": {
        "goal": "Reduce inflammatory cytokines and limit NF-κB/TLR4 pathway activation while maintaining maternal–fetal safety."
    },
    "Preterm PROM (pPROM) – membrane weakening/inflammation": {
        "goal": "Reduce inflammatory signaling and secondary tissue injury risk; prioritize safety and exposure predictability."
    },
    "Preeclampsia (PE) – inflammatory/vascular stress subtype": {
        "goal": "Support anti-inflammatory profile with minimal DDI risk and favorable safety flags."
    },
    "Chorioamnionitis / intrauterine infection inflammation": {
        "goal": "Anti-inflammatory potential with careful safety flags; interpret alongside infection management context."
    },
    "Fetal inflammatory response (FIRS) – fetal exposure concern": {
        "goal": "Balance anti-inflammatory potential with minimized fetal exposure risk."
    },
}


# =========================
# Splash Screen (once/session)
# =========================
def splash_screen(duration_sec: int = 5):
    if st.session_state.get("splash_done", False):
        return

    splash = st.empty()
    with splash.container():
        st.markdown('<div class="splash-card">', unsafe_allow_html=True)

        lcol, ccol, rcol = st.columns([2.4, 2.4, 1.2], gap="small")

        with lcol:
            try:
                st.image(MENON_LOGO_PATH, use_container_width=True)
            except Exception:
                st.caption("menon_logo.png not found (optional)")

        with ccol:
            try:
                st.image(UTMB_LOGO_PATH, use_container_width=True)
            except Exception:
                st.caption("utmb_logo.png not found (optional)")

        with rcol:
            if st.button("Skip", use_container_width=True, key="splash_skip"):
                st.session_state["splash_done"] = True
                splash.empty()
                st.rerun()

        st.markdown('<div class="splash-title">Pregnancy Drug Card</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="splash-subtitle">'
            'A research-grade pregnancy pharmacology profiling prototype<br>'
            '<b>Developed by The Menon Laboratory, UTMB</b>'
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
# Helpers
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


def fmt(v, digits=2):
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


def compute_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def compute_rdkit_descriptors(mol):
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
        "RotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
        "FracCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "cLogP_RDKit": float(Crippen.MolLogP(mol)),
    }


def pregnancy_transfer_risk(logp_or_logd, tpsa, pgp_substrate_prob, ppb):
    reasons = []
    score = 0

    if logp_or_logd is not None:
        if logp_or_logd >= 3:
            score += 2
            reasons.append(f"logP/logD high ({logp_or_logd:.2f})")
        elif logp_or_logd >= 2:
            score += 1
            reasons.append(f"logP/logD moderate ({logp_or_logd:.2f})")

    if tpsa is not None:
        if tpsa <= 60:
            score += 2
            reasons.append(f"TPSA low ({tpsa:.1f})")
        elif tpsa <= 90:
            score += 1
            reasons.append(f"TPSA moderate ({tpsa:.1f})")

    if pgp_substrate_prob is not None:
        if pgp_substrate_prob >= 0.5:
            score -= 2
            reasons.append(f"P-gp substrate likely ({pgp_substrate_prob:.2f})")
        else:
            score += 1
            reasons.append(f"P-gp substrate unlikely ({pgp_substrate_prob:.2f})")

    if ppb is not None:
        if ppb > 1.0:  # likely percent
            if ppb >= 95:
                score -= 1
                reasons.append(f"High PPB ({ppb:.0f}%)")
        else:  # fraction bound
            if ppb >= 0.95:
                score -= 1
                reasons.append(f"High PPB ({ppb:.2f})")

    if score >= 4:
        return "High", reasons
    if score >= 2:
        return "Moderate", reasons
    return "Low", reasons


def metabolism_ddi_risk(cyp_sub_probs, cyp_inhib_probs):
    reasons = []
    score = 0

    for k, v in cyp_sub_probs.items():
        if v is None:
            continue
        if v >= 0.5:
            score += 2
            reasons.append(f"{k} substrate likely ({v:.2f})")
        elif v >= 0.3:
            score += 1
            reasons.append(f"{k} substrate possible ({v:.2f})")

    for k, v in cyp_inhib_probs.items():
        if v is None:
            continue
        if v >= 0.5:
            score += 2
            reasons.append(f"{k} inhibitor likely ({v:.2f})")
        elif v >= 0.3:
            score += 1
            reasons.append(f"{k} inhibitor possible ({v:.2f})")

    if score >= 6:
        return "High", reasons
    if score >= 3:
        return "Moderate", reasons
    return "Low", reasons


def tox_flag_summary(row, tox_cols):
    flags = []
    for label, col in tox_cols.items():
        if col and col in row.index:
            v = row[col]
            if isinstance(v, (int, float)) and not pd.isna(v):
                if float(v) >= 0.5:
                    flags.append(label)
            elif isinstance(v, str):
                if v.strip().lower() in ["yes", "y", "true", "positive", "toxic"]:
                    flags.append(label)
    return flags


@st.cache_data(show_spinner=False)
def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    df.columns = [normalize_colname(c) for c in df.columns]

    name_col = find_col(df, ["Drug name", "Drug_name", "Name", "drug_name", "drug"])
    smiles_col = find_col(df, ["SMILES", "Smiles", "Canonical SMILES", "canonical_smiles"])
    if smiles_col is None:
        raise ValueError("Could not find a SMILES column. Ensure your CSV has a column named 'SMILES'.")

    # Compute minimal RDKit descriptors for dataset rows
    inchikeys, mw, tpsa = [], [], []
    for s in df[smiles_col].tolist():
        mol = safe_mol_from_smiles(s)
        if mol is None:
            inchikeys.append(None)
            mw.append(np.nan)
            tpsa.append(np.nan)
        else:
            inchikeys.append(compute_inchikey(mol))
            mw.append(float(Descriptors.MolWt(mol)))
            tpsa.append(float(rdMolDescriptors.CalcTPSA(mol)))

    df["InChIKey_calc"] = inchikeys
    df["RDKit_MW"] = mw
    df["RDKit_TPSA"] = tpsa

    # Stable numeric id for keys
    if "compound_id" not in [c.lower() for c in df.columns]:
        df.insert(0, "compound_id", range(1, len(df) + 1))

    return df, name_col, smiles_col


def strip_markdown(s: str) -> str:
    # Minimal cleanup for PDF: remove bold markers and extra whitespace
    if not s:
        return ""
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


def build_pdf_bytes(title: str, paragraph: str, caution: str) -> bytes:
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
            y -= line_height
        return y

    y = draw_wrapped(paragraph, y)
    y -= 10
    y = draw_wrapped(caution, y, max_chars=105, font="Helvetica-Oblique", size=9, line_height=12)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def get_openai_client():
    if OpenAI is None:
        return None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        if not key:
            return None
        return OpenAI(api_key=key)
    except Exception:
        return None


def genai_report_for_novel(smiles: str, inchikey: str, condition: str, facts: dict, heuristics: dict):
    """
    Returns (paragraph, caution) grounded ONLY to provided facts/heuristics.
    """
    client = get_openai_client()
    if client is None:
        return None, None

    schema = {
        "type": "object",
        "properties": {
            "paragraph": {"type": "string"},
            "caution": {"type": "string"},
        },
        "required": ["paragraph", "caution"],
        "additionalProperties": False,
    }

    prompt = f"""
You are drafting a cautious pregnancy pharmacology summary for RESEARCH USE.
Use ONLY the supplied facts and heuristics. Do NOT invent ADMETlab/ProTox, CYP, P-gp, BBB, LD50, or toxicity flags.
Return:
- One short paragraph
- One caution line
Pregnancy condition: {condition}
Condition goal: {PREGNANCY_CONDITIONS[condition]['goal']}

Identity:
- SMILES: {smiles}
- InChIKey: {inchikey}

Facts:
{json.dumps(facts, indent=2)}

Heuristics:
{json.dumps(heuristics, indent=2)}

Return JSON only.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        text={"format": {"type": "json_schema", "name": "novel_drug_card", "schema": schema}},
    )
    data = json.loads(resp.output_text)
    return data["paragraph"], data["caution"]


def build_natural_report(
    drug_name: str,
    condition: str,
    condition_goal: str,
    mw, tpsa, logp, logd, logs, ppb, bbb,
    pgp_sub, pgp_inh,
    transfer_label, ddi_label,
    tox_summary,
    cyp_sub_probs, cyp_inh_probs
):
    lipoph = logp if logp is not None else logd

    subs = [k for k, v in cyp_sub_probs.items() if v is not None and v >= 0.5]
    inhibs = [k for k, v in cyp_inh_probs.items() if v is not None and v >= 0.5]
    sub_str = ", ".join(subs[:4]) if subs else "no strong substrate signals"
    inh_str = ", ".join(inhibs[:4]) if inhibs else "no strong inhibitor signals"

    text = (
        f"**{drug_name}** was profiled for **{condition}**. The condition objective is: *{condition_goal}*\n\n"
        f"Physicochemical/ADME context: MW **{fmt(mw,1)}** and TPSA **{fmt(tpsa,1)}**, lipophilicity (logP/logD) "
        f"**{fmt(lipoph,2)}**, solubility (logS) **{fmt(logs,2)}**, and protein binding **{fmt(ppb,2)}**. "
        f"BBB permeability is **{fmt(bbb,2)}** (if available).\n\n"
        f"Transport signals: **P-gp substrate {fmt(pgp_sub,2)}** and **P-gp inhibitor {fmt(pgp_inh,2)}**. "
        f"Using a transparent heuristic, **placental transfer likelihood is {transfer_label}**.\n\n"
        f"Metabolism/DDI signals suggest **{ddi_label} pregnancy PK-shift/DDI risk**, driven by CYP patterns "
        f"(substrates: {sub_str}; inhibitors: {inh_str}).\n\n"
        f"Toxicity flags from the current table indicate **{tox_summary}**. "
        f"This interpretation is intended to help prioritize experimental validation and is not a clinical recommendation."
    )

    caution = (
        "Clinical caution: This report summarizes in-silico predictions (ADMETlab/ProTox) plus rule-based heuristics. "
        "Use alongside clinical guidance, patient context, and experimental validation."
    )
    return text, caution


# =========================
# UI Start
# =========================
splash_screen(SPLASH_SECONDS)

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Replay splash", key="replay_splash"):
        st.session_state["splash_done"] = False
        st.rerun()
    st.markdown("---")
    st.caption("Logos appear on splash only if menon_logo.png and utmb_logo.png exist in the repo root.")

# Top bar
top = st.container(border=True)
with top:
    h1, h2, h3 = st.columns([6, 2.2, 1])
    with h1:
        st.markdown("## Pregnancy Drug Card")
        st.caption("Product of The Menon Laboratory, UTMB")
    with h2:
        selected_condition = st.selectbox(
            "Pregnancy condition",
            list(PREGNANCY_CONDITIONS.keys()),
            index=0,
            key="condition_select"
        )
    with h3:
        if st.button("Exit", use_container_width=True, key="exit_btn"):
            st.markdown("<script>window.open('','_self'); window.close();</script>", unsafe_allow_html=True)

# Load dataset
try:
    df, name_col, smiles_col = load_and_prepare()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Main layout
left, right = st.columns([1.15, 2.2], gap="large")

with left:
    st.subheader("Search")

    query_name = st.text_input("Drug name", placeholder="Type drug name (partial ok)", key="q_name")
    pasted_smiles = st.text_area(
        "Or paste SMILES (novel compound)",
        height=90,
        placeholder="Paste SMILES here (if novel)",
        key="q_smiles"
    )

    is_novel = bool((pasted_smiles or "").strip())

    ai_enabled = st.toggle(
        "Use Generative AI for novel compounds (beta)",
        value=False,
        disabled=(not is_novel),
        key="ai_toggle"
    )
    st.caption("GenAI requires OPENAI_API_KEY in Streamlit Secrets. Otherwise it falls back to rule-based text.")

    if not is_novel:
        display_name = df[name_col] if name_col else pd.Series(["Unknown"] * len(df))

        if query_name.strip():
            q = query_name.strip().lower()
            filtered = df[display_name.fillna("").astype(str).str.lower().str.contains(q)].copy()
        else:
            filtered = df.copy()

        st.caption(f"Matches: {len(filtered)}")
        if len(filtered) == 0:
            st.stop()

        filtered_display = (
            (filtered[name_col].fillna("Unknown").astype(str) if name_col else pd.Series(["Unknown"] * len(filtered)))
            + "  |  "
            + filtered[smiles_col].fillna("").astype(str)
        ).tolist()

        sel = st.selectbox("Select compound", filtered_display, index=0, key="compound_select")
        row = filtered.iloc[filtered_display.index(sel)]
    else:
        row = None

with right:
    st.subheader("Report")

    # =====================
    # A) Dataset compound
    # =====================
    if not is_novel:
        name_val = row[name_col] if name_col else "Unknown"
        smiles_val = str(row[smiles_col])
        inchi = row.get("InChIKey_calc", "NA")
        compound_id = int(row.get("compound_id", 0)) if "compound_id" in row.index else 0

        mol = safe_mol_from_smiles(smiles_val)
        if mol:
            st.image(Draw.MolToImage(mol, size=(1200, 675)), caption="Structure", use_container_width=True)

        # Column detection (best effort)
        logp_col = find_col(df, ["LogP", "logP", "cLogP", "XlogP"])
        logd_col = find_col(df, ["LogD", "logD"])
        logs_col = find_col(df, ["LogS", "logS", "Solubility"])
        ppb_col  = find_col(df, ["PPB", "Plasma Protein Binding", "protein binding"])
        bbb_col  = find_col(df, ["BBB", "Blood brain barrier", "Blood brain permeability", "BBB permeability", "BBB_prob"])

        pgp_sub_col = find_col(df, ["P-gp substrate", "Pgp substrate", "Pgp_Substrate", "P-gp_Substrate"])
        pgp_inh_col = find_col(df, ["P-gp inhibitor", "Pgp inhibitor", "Pgp_Inhibitor", "P-gp_Inhibitor"])

        cyp_sub_cols = {
            "CYP3A4": find_col(df, ["CYP3A4 substrate", "CYP3A4_Substrate"]),
            "CYP2D6": find_col(df, ["CYP2D6 substrate", "CYP2D6_Substrate"]),
            "CYP2C9": find_col(df, ["CYP2C9 substrate", "CYP2C9_Substrate"]),
            "CYP2E1": find_col(df, ["CYP2E1 substrate", "CYP2E1_Substrate"]),
        }
        cyp_inh_cols = {
            "CYP3A4": find_col(df, ["CYP3A4 inhibitor", "CYP3A4_Inhibitor"]),
            "CYP2D6": find_col(df, ["CYP2D6 inhibitor", "CYP2D6_Inhibitor"]),
            "CYP2C9": find_col(df, ["CYP2C9 inhibitor", "CYP2C9_Inhibitor"]),
            "CYP2E1": find_col(df, ["CYP2E1 inhibitor", "CYP2E1_Inhibitor"]),
        }

        tox_cols = {
            "Hepatotoxicity": find_col(df, ["Hepatotoxicity", "Hepatotox"]),
            "Immunotoxicity": find_col(df, ["Immunotoxicity", "Immunotox"]),
            "Mutagenicity": find_col(df, ["Mutagenicity", "Mutagenic"]),
            "Carcinogenicity": find_col(df, ["Carcinogenicity", "Carcinogen"]),
        }
        toxclass_col = find_col(df, ["Toxicity Class", "toxicity class", "ProTox class", "tox_class"])
        ld50_col = find_col(df, ["LD50", "ld50"])

        # Values
        mw = as_float(row.get("RDKit_MW"))
        tpsa = as_float(row.get("RDKit_TPSA"))
        logp = as_float(row[logp_col]) if logp_col else None
        logd = as_float(row[logd_col]) if logd_col else None
        logs = as_float(row[logs_col]) if logs_col else None
        ppb  = as_float(row[ppb_col])  if ppb_col else None
        bbb  = as_float(row[bbb_col])  if bbb_col else None

        pgp_sub = as_float(row[pgp_sub_col]) if pgp_sub_col else None
        pgp_inh = as_float(row[pgp_inh_col]) if pgp_inh_col else None

        cyp_sub_probs = {k: (as_float(row[v]) if v else None) for k, v in cyp_sub_cols.items()}
        cyp_inh_probs = {k: (as_float(row[v]) if v else None) for k, v in cyp_inh_cols.items()}

        lipoph = logp if logp is not None else logd
        transfer_label, _ = pregnancy_transfer_risk(lipoph, tpsa, pgp_sub, ppb)
        ddi_label, _ = metabolism_ddi_risk(cyp_sub_probs, cyp_inh_probs)

        tox_flags = tox_flag_summary(row, tox_cols)
        tox_summary = ", ".join(tox_flags) if tox_flags else "no strong toxicity flags detected"
        if toxclass_col:
            tox_summary = tox_summary + f" (class {row[toxclass_col]})"

        cond_goal = PREGNANCY_CONDITIONS[selected_condition]["goal"]

        # Title block
        st.markdown(f"### {name_val}")
        st.code(f"SMILES: {smiles_val}", language="text")
        st.write(f"**InChIKey:** {inchi}")

        # Quick pills
        st.markdown(
            f"""
            <span class="pill">Placental transfer: {transfer_label}</span>
            <span class="pill">PK/DDI risk: {ddi_label}</span>
            <span class="pill">Condition: {selected_condition}</span>
            """,
            unsafe_allow_html=True
        )

        # Cards
        cA, cB = st.columns([1.25, 1.0], gap="large")

        with cA:
            with st.container(border=True):
                st.markdown('<div class="section-title">ADME</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("MW", fmt(mw, 1))
                m1.metric("TPSA", fmt(tpsa, 1))
                m2.metric("logP", fmt(logp, 2))
                m2.metric("logD", fmt(logd, 2))
                m3.metric("logS", fmt(logs, 2))
                m3.metric("PPB", fmt(ppb, 2))
                st.write(f"**BBB permeability:** {fmt(bbb,2)}")

            with st.container(border=True):
                st.markdown('<div class="section-title">Transport (P-gp)</div>', unsafe_allow_html=True)
                st.write(f"**P-gp substrate:** {fmt(pgp_sub,2)}")
                st.write(f"**P-gp inhibitor:** {fmt(pgp_inh,2)}")

        with cB:
            with st.container(border=True):
                st.markdown('<div class="section-title">CYP / DDI signals</div>', unsafe_allow_html=True)
                st.caption("Likelihoods shown where columns exist in the table.")
                st.write("**Substrate likelihoods:**", {k: (None if v is None else round(v, 2)) for k, v in cyp_sub_probs.items()})
                st.write("**Inhibitor likelihoods:**", {k: (None if v is None else round(v, 2)) for k, v in cyp_inh_probs.items()})

            with st.container(border=True):
                st.markdown('<div class="section-title">Toxicity</div>', unsafe_allow_html=True)
                st.write(f"**Summary:** {tox_summary}")
                if ld50_col:
                    st.write(f"**LD50:** {row[ld50_col]}")

        # Natural report (NO bullets shown)
        natural_report, caution = build_natural_report(
            drug_name=name_val,
            condition=selected_condition,
            condition_goal=cond_goal,
            mw=mw, tpsa=tpsa, logp=logp, logd=logd, logs=logs, ppb=ppb, bbb=bbb,
            pgp_sub=pgp_sub, pgp_inh=pgp_inh,
            transfer_label=transfer_label, ddi_label=ddi_label,
            tox_summary=tox_summary,
            cyp_sub_probs=cyp_sub_probs, cyp_inh_probs=cyp_inh_probs
        )

        with st.container(border=True):
            st.markdown('<div class="section-title">Pregnancy interpretation</div>', unsafe_allow_html=True)
            st.markdown(natural_report)
            st.info(caution)

        # PDF (one button only) — UNIQUE KEY avoids StreamlitDuplicateElementId
        pdf_bytes = build_pdf_bytes(
            title=f"Pregnancy Drug Card — {name_val}",
            paragraph=strip_markdown(natural_report),
            caution=caution
        )
        st.download_button(
            "Print / Download PDF",
            data=pdf_bytes,
            file_name=f"{name_val}_pregnancy_drug_card.pdf".replace(" ", "_"),
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_dataset_{compound_id}_{selected_condition}"
        )

    # =====================
    # B) Novel SMILES
    # =====================
    else:
        smiles_val = (pasted_smiles or "").strip()
        mol = safe_mol_from_smiles(smiles_val)
        if mol is None:
            st.error("Invalid SMILES. Please correct and try again.")
            st.stop()

        st.image(Draw.MolToImage(mol, size=(1200, 675)), caption="Structure (Novel compound)", use_container_width=True)

        d = compute_rdkit_descriptors(mol)
        inchikey = compute_inchikey(mol) or "NA"
        cond_goal = PREGNANCY_CONDITIONS[selected_condition]["goal"]

        # Novel: only RDKit + heuristics
        transfer_label, _ = pregnancy_transfer_risk(d["cLogP_RDKit"], d["TPSA"], None, None)
        ddi_label = "Unknown"
        tox_summary = "Unknown (no ProTox provided)"

        facts = {
            "MW": round(d["MW"], 2),
            "TPSA": round(d["TPSA"], 2),
            "cLogP_RDKit": round(d["cLogP_RDKit"], 2),
            "HBD": d["HBD"],
            "HBA": d["HBA"],
            "RotB": d["RotB"],
            "RingCount": d["RingCount"],
            "FracCSP3": round(d["FracCSP3"], 3),
            "BBB": "Unknown (not provided)",
            "P-gp substrate/inhibitor": "Unknown (not provided)",
            "CYP / DDI": "Unknown (not provided)",
            "Toxicity profiles": "Unknown (not provided)",
        }
        heuristics = {
            "Placental transfer likelihood (heuristic)": transfer_label,
            "Pregnancy PK shift/DDI risk": ddi_label,
        }

        # Default narrative
        natural_report = (
            f"**Novel compound** was profiled for **{selected_condition}**. The condition objective is: *{cond_goal}*\n\n"
            f"This structure has computed MW **{fmt(d['MW'],1)}** and TPSA **{fmt(d['TPSA'],1)}**, with RDKit cLogP "
            f"**{fmt(d['cLogP_RDKit'],2)}**. Using a transparent heuristic, **placental transfer likelihood is {transfer_label}**.\n\n"
            f"Transporter/CYP/toxicity predictions from ADMETlab/ProTox are not available for this novel structure in the current version, "
            f"so DDI risk and toxicity remain **Unknown** until those models/measurements are added."
        )
        caution = (
            "Clinical caution: This novel-compound report is based only on RDKit descriptors + simple heuristics. "
            "It is not a clinical recommendation."
        )

        # Optional GenAI (still grounded)
        if ai_enabled:
            with st.spinner("Generating AI-assisted report (grounded to computed facts)..."):
                p2, c2 = genai_report_for_novel(
                    smiles=smiles_val,
                    inchikey=inchikey,
                    condition=selected_condition,
                    facts=facts,
                    heuristics=heuristics,
                )
            if p2 and c2:
                natural_report, caution = p2, c2
            else:
                st.warning("GenAI not configured. Add OPENAI_API_KEY in Streamlit Secrets to enable it.")

        # Header
        st.markdown("### Novel compound")
        st.code(f"SMILES: {smiles_val}", language="text")
        st.write(f"**InChIKey:** {inchikey}")

        st.markdown(
            f"""
            <span class="pill">Placental transfer: {transfer_label}</span>
            <span class="pill">PK/DDI risk: {ddi_label}</span>
            <span class="pill">Condition: {selected_condition}</span>
            """,
            unsafe_allow_html=True
        )

        # Compact metrics
        a, b, c = st.columns(3)
        a.metric("MW", fmt(d["MW"], 1))
        a.metric("TPSA", fmt(d["TPSA"], 1))
        b.metric("RDKit cLogP", fmt(d["cLogP_RDKit"], 2))
        b.metric("Placental transfer", transfer_label)
        c.metric("DDI risk", ddi_label)
        c.metric("Toxicity", "Unknown")

        with st.container(border=True):
            st.markdown('<div class="section-title">Pregnancy interpretation</div>', unsafe_allow_html=True)
            st.markdown(natural_report)
            st.info(caution)

        # PDF — UNIQUE KEY
        pdf_bytes = build_pdf_bytes(
            title="Pregnancy Drug Card — Novel compound",
            paragraph=strip_markdown(natural_report),
            caution=caution
        )
        st.download_button(
            "Print / Download PDF",
            data=pdf_bytes,
            file_name="novel_compound_pregnancy_drug_card.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_novel_{inchikey}_{selected_condition}"
        )
