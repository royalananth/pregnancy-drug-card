import re
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw

st.set_page_config(page_title="Pregnancy Drug Card (v1)", layout="wide")

# --- Branding / watermark (fixed corner): pink text, black background ---
st.markdown(
    """
    <style>
      .menon-watermark {
        position: fixed;
        bottom: 14px;
        right: 16px;
        z-index: 9999;
        font-size: 14px;
        font-weight: 800;
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
    </style>
    <div class="menon-watermark">Developed by The Menon Laboratory, UTMB</div>
    """,
    unsafe_allow_html=True
)

CSV_PATH = "Master table_260 drugs_ADME_Protox.csv"

# -----------------------------
# Pregnancy conditions (v1)
# -----------------------------
PREGNANCY_CONDITIONS = {
    "Preterm Birth (PTB) – inflammation-driven": {
        "goal": "Reduce inflammatory cytokines and limit NF-κB/TLR4 pathway activation while maintaining maternal-fetal safety.",
        "notes": [
            "Prefer lower toxicity signals and manageable CYP/DDI burden for pregnancy use.",
            "Later: incorporate docking (NF-κB, TLR4, JNK/MAPK) + cytokine inhibition assay results."
        ]
    },
    "Preterm PROM (pPROM) – membrane weakening/inflammation": {
        "goal": "Reduce inflammatory signaling and secondary tissue injury risk; prioritize safety and exposure predictability.",
        "notes": [
            "Avoid strong toxicity flags; watch high placental transfer if fetal exposure is a concern.",
            "Later: add ECM/MMP links and membrane-specific endpoints."
        ]
    },
    "Preeclampsia (PE) – inflammatory/vascular stress subtype": {
        "goal": "Support anti-inflammatory profile with minimal DDI risk and favorable safety flags.",
        "notes": [
            "CYP/DDI caution important due to polypharmacy in PE management.",
            "Later: PBPK pregnancy module will help exposure predictions."
        ]
    },
    "Chorioamnionitis / intrauterine infection inflammation": {
        "goal": "Anti-inflammatory potential with careful safety flags; interpret alongside infection management context.",
        "notes": [
            "High DDI risk compounds may complicate clinical regimens.",
            "Later: add pathogen/LPS response validation data."
        ]
    },
    "Fetal inflammatory response (FIRS) – fetal exposure concern": {
        "goal": "Balance anti-inflammatory potential with minimized fetal exposure risk.",
        "notes": [
            "Placental transfer likelihood becomes a key driver once PBPK/fetal exposure estimates are added."
        ]
    }
}

# -----------------------------
# Helpers
# -----------------------------
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
        "MW": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def as_prob(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def pregnancy_transfer_risk(logp_or_logd, tpsa, pgp_substrate_prob, ppb):
    reasons = []
    score = 0

    if logp_or_logd is not None:
        if logp_or_logd >= 3:
            score += 2
            reasons.append(f"logP/logD high ({logp_or_logd:.2f}) → higher passive partitioning")
        elif logp_or_logd >= 2:
            score += 1
            reasons.append(f"logP/logD moderate ({logp_or_logd:.2f})")

    if tpsa is not None:
        if tpsa <= 60:
            score += 2
            reasons.append(f"TPSA low ({tpsa:.1f}) → easier passive diffusion")
        elif tpsa <= 90:
            score += 1
            reasons.append(f"TPSA moderate ({tpsa:.1f})")

    if pgp_substrate_prob is not None:
        if pgp_substrate_prob >= 0.5:
            score -= 2
            reasons.append(f"P-gp substrate likely ({pgp_substrate_prob:.2f}) → efflux may reduce transfer")
        else:
            score += 1
            reasons.append(f"P-gp substrate unlikely ({pgp_substrate_prob:.2f})")

    if ppb is not None:
        if ppb > 1.0:
            if ppb >= 95:
                score -= 1
                reasons.append(f"High protein binding ({ppb:.0f}%) → lower free fraction")
        else:
            if ppb >= 0.95:
                score -= 1
                reasons.append(f"High protein binding ({ppb:.2f}) → lower free fraction")

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

    inchikeys, mw, tpsa, hbd, hba, rotb, ring, fcsp3 = [], [], [], [], [], [], [], []

    for s in df[smiles_col].tolist():
        mol = safe_mol_from_smiles(s)
        if mol is None:
            inchikeys.append(None)
            mw.append(np.nan); tpsa.append(np.nan); hbd.append(np.nan); hba.append(np.nan)
            rotb.append(np.nan); ring.append(np.nan); fcsp3.append(np.nan)
            continue
        inchikeys.append(compute_inchikey(mol))
        d = compute_rdkit_descriptors(mol)
        mw.append(d["MW"]); tpsa.append(d["TPSA"]); hbd.append(d["HBD"]); hba.append(d["HBA"])
        rotb.append(d["RotB"]); ring.append(d["RingCount"]); fcsp3.append(d["FracCSP3"])

    df["InChIKey_calc"] = inchikeys
    df["RDKit_MW"] = mw
    df["RDKit_TPSA"] = tpsa
    df["RDKit_HBD"] = hbd
    df["RDKit_HBA"] = hba
    df["RDKit_RotB"] = rotb
    df["RDKit_RingCount"] = ring
    df["RDKit_FracCSP3"] = fcsp3

    if "compound_id" not in [c.lower() for c in df.columns]:
        df.insert(0, "compound_id", range(1, len(df) + 1))

    return df, name_col, smiles_col


# -----------------------------
# UI
# -----------------------------
splash_screen(duration_sec=5)

# Top bar (product header + exit)
top = st.container(border=True)
with top:
    h1, h2, h3 = st.columns([6, 2, 1])
    with h1:
        st.markdown("## Pregnancy Drug Card")
        st.caption("Developed by The Menon Laboratory, UTMB")
    with h2:
        selected_condition = st.selectbox("Pregnancy condition", list(PREGNANCY_CONDITIONS.keys()), index=0)
    with h3:
        # "Exit" on web = close tab; this works in many browsers
        if st.button("Exit", use_container_width=True):
            st.markdown(
                "<script>window.open('','_self'); window.close();</script>",
                unsafe_allow_html=True
            )

try:
    df, name_col, smiles_col = load_and_prepare()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Single screen: left search + right content
left, right = st.columns([1.15, 2.2], gap="large")

with left:
    st.subheader("Search")
    query_name = st.text_input("Drug name", placeholder="Type name (partial ok)")
    pasted_smiles = st.text_area("Or paste SMILES", height=90, placeholder="Paste SMILES here")

    display_name = df[name_col] if name_col else pd.Series(["(no name column)"] * len(df))

    if pasted_smiles and pasted_smiles.strip():
        q = pasted_smiles.strip()
        exact = df[df[smiles_col].fillna("").astype(str).str.strip() == q]
        filtered = exact.copy() if len(exact) else df[df[smiles_col].fillna("").astype(str).str.contains(q, na=False)].copy()
    else:
        if query_name.strip():
            q = query_name.strip().lower()
            mask = display_name.fillna("").astype(str).str.lower().str.contains(q)
            filtered = df[mask].copy()
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

    sel = st.selectbox("Select compound", filtered_display, index=0)
    row = filtered.iloc[filtered_display.index(sel)]

with right:
    # --- Key identity / structure ---
    name_val = row[name_col] if name_col else "Unknown"
    smiles_val = str(row[smiles_col])

    st.markdown(f"### {name_val}")
    st.code(f"SMILES: {smiles_val}", language="text")
    st.write(f"**InChIKey:** {row.get('InChIKey_calc','NA')}")

    mol = safe_mol_from_smiles(smiles_val)
    if mol:
        img = Draw.MolToImage(mol, size=(1200, 675))  # 16:9 render
        st.image(img, caption="Structure", use_container_width=True)

    # --- Column detection (same as your current logic + BBB) ---
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
    logp = as_prob(row[logp_col]) if logp_col else None
    logd = as_prob(row[logd_col]) if logd_col else None
    logs = as_prob(row[logs_col]) if logs_col else None
    ppb  = as_prob(row[ppb_col])  if ppb_col  else None
    bbb  = as_prob(row[bbb_col])  if bbb_col  else None
    pgp_sub = as_prob(row[pgp_sub_col]) if pgp_sub_col else None
    pgp_inh = as_prob(row[pgp_inh_col]) if pgp_inh_col else None

    cyp_sub_probs = {k: (as_prob(row[v]) if v else None) for k, v in cyp_sub_cols.items()}
    cyp_inh_probs = {k: (as_prob(row[v]) if v else None) for k, v in cyp_inh_cols.items()}

    # Pregnancy heuristics
    lipoph = logp if logp is not None else logd
    transfer_label, transfer_reasons = pregnancy_transfer_risk(
        lipoph,
        row["RDKit_TPSA"] if pd.notna(row["RDKit_TPSA"]) else None,
        pgp_sub,
        ppb
    )
    ddi_label, ddi_reasons = metabolism_ddi_risk(cyp_sub_probs, cyp_inh_probs)

    # --- One single "dashboard" area ---
    grid = st.container(border=True)
    with grid:
        st.markdown("#### ADME + Toxicity overview")
        a, b, c = st.columns(3)
        a.metric("MW", f"{row['RDKit_MW']:.1f}" if pd.notna(row["RDKit_MW"]) else "NA")
        a.metric("BBB", f"{bbb:.2f}" if bbb is not None else "NA")
        b.metric("logP", f"{logp:.2f}" if logp is not None else "NA")
        b.metric("logD", f"{logd:.2f}" if logd is not None else "NA")
        c.metric("logS", f"{logs:.2f}" if logs is not None else "NA")
        c.metric("PPB", f"{ppb}" if ppb is not None else "NA")

        st.write(f"**P-gp substrate:** {pgp_sub if pgp_sub is not None else 'NA'}")
        st.write(f"**P-gp inhibitor:** {pgp_inh if pgp_inh is not None else 'NA'}")

        st.write("**CYP substrate likelihoods:**", {k: (None if v is None else round(v, 2)) for k, v in cyp_sub_probs.items()})
        st.write("**CYP inhibitor likelihoods:**", {k: (None if v is None else round(v, 2)) for k, v in cyp_inh_probs.items()})

        tox_flags = tox_flag_summary(row, tox_cols)
        st.write("**Toxicity flags:** " + (", ".join(tox_flags) if tox_flags else "No strong flags detected (threshold-based)"))
        if toxclass_col:
            st.write(f"**Toxicity class:** {row[toxclass_col]}")
        if ld50_col:
            st.write(f"**LD50:** {row[ld50_col]}")

        st.markdown("---")
        st.markdown("#### Pregnancy report (1 paragraph + 5 bullets)")

        # Build report text (same format you already like)
        inchi = row.get("InChIKey_calc", "NA")
        mw_val = f"{row['RDKit_MW']:.1f}" if pd.notna(row["RDKit_MW"]) else "NA"
        tpsa_val = f"{row['RDKit_TPSA']:.1f}" if pd.notna(row["RDKit_TPSA"]) else "NA"
        logp_val = f"{logp:.2f}" if logp is not None else "NA"
        logd_val = f"{logd:.2f}" if logd is not None else "NA"
        logs_val = f"{logs:.2f}" if logs is not None else "NA"
        ppb_val  = f"{ppb}" if ppb is not None else "NA"
        bbb_val  = f"{bbb:.2f}" if bbb is not None else "NA"
        pgp_sub_val = f"{pgp_sub:.2f}" if pgp_sub is not None else "NA"
        pgp_inh_val = f"{pgp_inh:.2f}" if pgp_inh is not None else "NA"

        drivers = []
        for enz, v in cyp_sub_probs.items():
            if v is not None and v >= 0.5:
                drivers.append(f"{enz} substrate")
        for enz, v in cyp_inh_probs.items():
            if v is not None and v >= 0.5:
                drivers.append(f"{enz} inhibitor")
        top_cyps = ", ".join(drivers[:4]) if drivers else "no strong CYP substrate/inhibitor signals (by threshold)"

        tox_summary = ", ".join(tox_flags) if tox_flags else "no strong toxicity flags detected"
        cond_goal = PREGNANCY_CONDITIONS[selected_condition]["goal"]

        paragraph = (
            f"{name_val} is predicted to have BBB permeability {bbb_val}, P-gp substrate {pgp_sub_val} and inhibitor {pgp_inh_val}, "
            f"with {ddi_label} pregnancy PK shift/DDI potential (key drivers: {top_cyps}). "
            f"Placental transfer likelihood is estimated as {transfer_label} based on lipophilicity (logP/logD), polarity (TPSA), PPB, and P-gp interaction. "
            f"Toxicity signals indicate {tox_summary}. Condition focus: {selected_condition}. {cond_goal}"
        )

        bullets = [
            f"Identity: {name_val} | InChIKey: {inchi}",
            f"ADME: MW {mw_val} | TPSA {tpsa_val} | logP {logp_val} / logD {logd_val} | logS {logs_val} | PPB {ppb_val} | BBB {bbb_val}",
            f"Transporter/CYP/DDI: P-gp sub {pgp_sub_val} | P-gp inh {pgp_inh_val} | PK shift/DDI risk: {ddi_label}",
            f"Placental transfer likelihood: {transfer_label}",
            f"Toxicity: {tox_summary}" + (f" | class {row[toxclass_col]}" if toxclass_col else "")
        ]

        caution = (
            "Clinical caution: This report is based on in-silico predictions (ADMETlab/ProTox) plus rule-based heuristics; "
            "it is not a clinical recommendation. Interpret alongside guidelines, patient context, and experimental validation."
        )

        st.write(paragraph)
        st.markdown("- " + "\n- ".join([f"**{b}**" if i == 0 else b for i, b in enumerate(bullets)]))
        st.info(caution)

        # PDF export
        pdf_bytes = build_pdf_bytes(
            title=f"Pregnancy Drug Card — {name_val}",
            paragraph=paragraph,
            bullets=bullets,
            caution=caution
        )
        st.download_button(
            "Print / Download PDF",
            data=pdf_bytes,
            file_name=f"{name_val}_pregnancy_drug_card.pdf".replace(" ", "_"),
            mime="application/pdf",
            use_container_width=True
        )


    # -----------------------------
    # ADME snapshot (requested fields)
    # -----------------------------
    st.markdown("#### ADME parameters")
    c1, c2, c3 = st.columns(3)
    c1.metric("MW", f"{row['RDKit_MW']:.1f}" if pd.notna(row["RDKit_MW"]) else "NA")
    c1.metric("BBB permeability", f"{bbb:.2f}" if bbb is not None else "NA")
    c2.metric("logP", f"{logp:.2f}" if logp is not None else "NA")
    c2.metric("logD", f"{logd:.2f}" if logd is not None else "NA")
    c3.metric("logS", f"{logs:.2f}" if logs is not None else "NA")
    c3.metric("PPB", f"{ppb}" if ppb is not None else "NA")

    st.write(f"**P-gp substrate:** {pgp_sub if pgp_sub is not None else 'NA'}")
    st.write(f"**P-gp inhibitor:** {pgp_inh if pgp_inh is not None else 'NA'}")

    st.markdown("#### CYP / DDI signals")
    st.write("**Substrate likelihoods:**", {k: (None if v is None else round(v, 2)) for k, v in cyp_sub_probs.items()})
    st.write("**Inhibitor likelihoods:**", {k: (None if v is None else round(v, 2)) for k, v in cyp_inh_probs.items()})

    st.markdown("#### Pregnancy-relevant flags (v1 heuristics)")
    st.write(f"**Placental transfer likelihood:** {transfer_label}")
    with st.expander("Why? (placental transfer)"):
        for r in transfer_reasons:
            st.write(f"- {r}")

    st.write(f"**Pregnancy PK shift / DDI risk:** {ddi_label}")
    with st.expander("Why? (DDI/PK shift)"):
        for r in ddi_reasons[:10]:
            st.write(f"- {r}")

    st.markdown("#### Toxicity (ProTox signals)")
    tox_flags = tox_flag_summary(row, tox_cols)
    st.write("**Flags:** " + (", ".join(tox_flags) if tox_flags else "No strong flags detected (by current thresholds)"))
    if toxclass_col:
        st.write(f"**Toxicity class:** {row[toxclass_col]}")
    if ld50_col:
        st.write(f"**LD50:** {row[ld50_col]}")

    # -----------------------------
    # Report output: 1 paragraph + 5 bullets + clinical caution
    # -----------------------------
    st.markdown("#### Report (1 paragraph + 5 bullets)")

    inchi = row.get("InChIKey_calc", "NA")
    mw_val = f"{row['RDKit_MW']:.1f}" if pd.notna(row["RDKit_MW"]) else "NA"
    tpsa_val = f"{row['RDKit_TPSA']:.1f}" if pd.notna(row["RDKit_TPSA"]) else "NA"
    logp_val = f"{logp:.2f}" if logp is not None else "NA"
    logd_val = f"{logd:.2f}" if logd is not None else "NA"
    logs_val = f"{logs:.2f}" if logs is not None else "NA"
    ppb_val = f"{ppb}" if ppb is not None else "NA"
    bbb_val = f"{bbb:.2f}" if bbb is not None else "NA"
    pgp_sub_val = f"{pgp_sub:.2f}" if pgp_sub is not None else "NA"
    pgp_inh_val = f"{pgp_inh:.2f}" if pgp_inh is not None else "NA"

    drivers = []
    for enz, v in cyp_sub_probs.items():
        if v is not None and v >= 0.5:
            drivers.append(f"{enz} substrate")
    for enz, v in cyp_inh_probs.items():
        if v is not None and v >= 0.5:
            drivers.append(f"{enz} inhibitor")
    top_cyps = ", ".join(drivers[:4]) if drivers else "no strong CYP substrate/inhibitor signals detected (by threshold)"

    tox_summary = ", ".join(tox_flags) if tox_flags else "no strong toxicity flags detected (by current thresholds)"
    caution_level = "Higher" if len(tox_flags) >= 2 else ("Moderate" if len(tox_flags) == 1 else "Lower")

    cond = selected_condition
    cond_goal = PREGNANCY_CONDITIONS.get(cond, {}).get("goal", "")

    paragraph = (
        f"{name_val} is predicted to have BBB permeability {bbb_val}, P-gp substrate {pgp_sub_val} and inhibitor {pgp_inh_val}, "
        f"with {ddi_label} pregnancy PK shift/DDI potential (key drivers: {top_cyps}). "
        f"Placental transfer likelihood is estimated as {transfer_label} based on lipophilicity (logP/logD), polarity (TPSA), PPB, and P-gp interaction. "
        f"Toxicity signals indicate {tox_summary}. "
        f"Condition focus: {cond}. {cond_goal}"
    )

    st.write(paragraph)
    st.info(
        "Clinical caution: This report is based on in-silico predictions (ADMETlab/ProTox) plus simple heuristics; "
        "it is not a clinical recommendation. Interpret alongside guidelines, patient context, and your experimental validation."
    )

    bullets = [
        f"**Identity:** {name_val} | InChIKey: {inchi}",
        f"**ADME:** MW {mw_val} | TPSA {tpsa_val} | logP {logp_val} / logD {logd_val} | logS {logs_val} | PPB {ppb_val} | BBB {bbb_val}",
        f"**Transporter/CYP/DDI:** P-gp substrate {pgp_sub_val} | P-gp inhibitor {pgp_inh_val} | PK shift/DDI risk: {ddi_label}",
        f"**Substrate likelihoods:** { {k: (None if v is None else round(v,2)) for k,v in cyp_sub_probs.items()} }",
        f"**Inhibitor likelihoods:** { {k: (None if v is None else round(v,2)) for k,v in cyp_inh_probs.items()} }",
    ]
    st.markdown("- " + "\n- ".join(bullets))

    st.markdown("#### Condition-specific: what this drug might do (v1)")
    what_might_do = []

    # Base condition goal
    if cond_goal:
        what_might_do.append(cond_goal)

    # Auto hints
    if ddi_label == "High":
        what_might_do.append("High DDI/PK-shift risk: consider interaction screening and pregnancy-adjusted exposure modeling (later PBPK).")
    if transfer_label == "High":
        what_might_do.append("Higher placental transfer likelihood: fetal exposure may be more likely (future PBPK will refine).")
    if tox_flags:
        what_might_do.append(f"Toxicity flags present ({tox_summary}): prioritize careful dosing/monitoring and confirm with in-vitro safety assays.")

    # Add condition notes
    cond_notes = PREGNANCY_CONDITIONS.get(cond, {}).get("notes", [])
    for n in cond_notes[:3]:
        what_might_do.append(n)

    if what_might_do:
        for item in what_might_do:
            if item:
                st.write(f"- {item}")
    else:
        st.write("- Add pregnancy condition rules once your condition definitions are finalized.")

    st.caption("Note: v1 is rule-based using ADMET/ProTox columns + RDKit descriptors. Docking/PBPK/cytokines will plug in as additional evidence later.")

