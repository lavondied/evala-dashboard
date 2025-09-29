# Streamlit – Marketingový dashboard (Tenants, MQL, MQL goal, MQL %)
# ---------------------------------------------------------------
# Funkce:
# - Ruční editace hodnot v interaktivní tabulce (Leden–Prosinec)
# - Import CSV/XLSX a export aktuálních dat
# - Automatický výpočet MQL % (MQL / MQL goal * 100)
# - KPI karty (YTD součty/plnění, nejlepší/nejhorší měsíc)
# - Grafy: MQL vs. MQL goal + MQL % (kombinovaný), Tenants vs. MQL, a 4 mini-grafy
# - Moderní vzhled s barvami brandu (nahraďte hex kódy dle manuálu)

import io
import math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Konfigurace aplikace
# =========================
st.set_page_config(
    page_title="Marketingový dashboard – Evala",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# BRAND BARVY – Evala (dle dodaných HEX)
# Pozn.: poslední kód byl zřejmě překlep "fffff" → používám #FFFFFF
# ---------------------------------------------------------------
PRIMARY = "#9F1D7C"      # hlavní akční barva (MQL, klíčové prvky)
SECONDARY = "#EEAECF"    # srovnávací/plán (MQL goal, sekundární sloupce)
ACCENT = "#99426D"       # akcent – křivka MQL %, zvýraznění
NEUTRAL_900 = "#1A0E16"  # tmavé pozadí (odvozeno z #67154A ztmavením)
NEUTRAL_800 = "#2B1521"  # povrch karet (tmavý panel)
NEUTRAL_300 = "#E3BCD9"  # sekundární text / linky (světle růžová)
NEUTRAL_100 = "#FFFFFF"  # světlý text na tmavém pozadí

PLOTLY_TEMPLATE_DARK = "plotly_dark"

MONTHS_CS = [
    "Leden", "Únor", "Březen", "Duben", "Květen", "Červen",
    "Červenec", "Srpen", "Září", "Říjen", "Listopad", "Prosinec"
]

METRICS = ["Tenants", "MQL", "MQL goal", "MQL % (auto)"]

# =========================
# Pomocné funkce
# =========================

def empty_df() -> pd.DataFrame:
    data = {m: [0, 0, 0, 0.0] for m in MONTHS_CS}
    df = pd.DataFrame(data, index=METRICS)
    return df


def compute_mql_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ochrana proti chybějícím řádkům
    if "MQL" not in df.index or "MQL goal" not in df.index:
        return df
    for m in MONTHS_CS:
        goal = pd.to_numeric(df.loc["MQL goal", m], errors="coerce")
        mql = pd.to_numeric(df.loc["MQL", m], errors="coerce")
        if pd.isna(goal) or pd.isna(mql) or goal == 0:
            df.loc["MQL % (auto)", m] = 0.0
        else:
            df.loc["MQL % (auto)", m] = float(mql) / float(goal) * 100.0
    return df


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m in MONTHS_CS:
        for r in ["Tenants", "MQL", "MQL goal"]:
            val = pd.to_numeric(df.loc[r, m], errors="coerce")
            if pd.isna(val) or val < 0:
                val = 0
            df.loc[r, m] = int(val)
    return df


def kpi_cards(df: pd.DataFrame) -> Tuple[int, int, int, float, str, str]:
    mql_sum = int(pd.to_numeric(df.loc["MQL", MONTHS_CS]).sum())
    goal_sum = int(pd.to_numeric(df.loc["MQL goal", MONTHS_CS]).sum())
    tenants_sum = int(pd.to_numeric(df.loc["Tenants", MONTHS_CS]).sum())
    ytd_pct = (mql_sum / goal_sum * 100.0) if goal_sum > 0 else 0.0

    # nejlepší/nejhorší měsíc podle MQL %
    mql_pct_vals = df.loc["MQL % (auto)", MONTHS_CS].astype(float).values
    best_idx = int(np.argmax(mql_pct_vals))
    worst_idx = int(np.argmin(mql_pct_vals))
    best_month = MONTHS_CS[best_idx]
    worst_month = MONTHS_CS[worst_idx]
    return tenants_sum, mql_sum, goal_sum, ytd_pct, best_month, worst_month


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Převod na long form pro plotly
    long_rows = []
    for m in MONTHS_CS:
        long_rows.append({"Měsíc": m, "Metrika": "Tenants", "Hodnota": int(df.loc["Tenants", m])})
        long_rows.append({"Měsíc": m, "Metrika": "MQL", "Hodnota": int(df.loc["MQL", m])})
        long_rows.append({"Měsíc": m, "Metrika": "MQL goal", "Hodnota": int(df.loc["MQL goal", m])})
        long_rows.append({"Měsíc": m, "Metrika": "MQL %", "Hodnota": float(df.loc["MQL % (auto)", m])})
    return pd.DataFrame(long_rows)


# =========================
# Styl (CSS) – tmavý moderní vzhled
# =========================
CSS = f"""
<style>
    .main, .stApp {{ background-color: {NEUTRAL_900}; }}
    .elev-card {{
        background: {NEUTRAL_800};
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1rem; border-radius: 16px; box-shadow: 0 2px 16px rgba(0,0,0,.25);
    }}
    .kpi-number {{ font-size: 2rem; font-weight: 700; color: {NEUTRAL_100}; }}
    .kpi-label {{ color: {NEUTRAL_300}; font-size: 0.9rem; margin-top: .25rem; }}
    .good {{ color: {PRIMARY}; }}
    .warn {{ color: {ACCENT}; }}
    .muted {{ color: {NEUTRAL_300}; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# =========================
# Sidebar: Import / Theme
# =========================
st.sidebar.header("⚙️ Nastavení")
mode = st.sidebar.radio("Motiv", ["Tmavý", "Světlý"], index=0)
plotly_template = PLOTLY_TEMPLATE_DARK if mode == "Tmavý" else "plotly_white"

uploaded_file = st.sidebar.file_uploader("Import dat (CSV nebo XLSX)", type=["csv", "xlsx"])

def read_uploaded(_file) -> pd.DataFrame:
    if _file is None:
        return None
    try:
        if _file.name.lower().endswith(".csv"):
            df = pd.read_csv(_file)
        else:
            df = pd.read_excel(_file)
        # očekáváme sloupec "Metrika" + 12 měsíců
        # případně transformujeme z long do wide
        if "Metrika" in df.columns:
            df = df.set_index("Metrika")
        # ověření, že máme všechny měsíce
        missing = [m for m in MONTHS_CS if m not in df.columns]
        for m in missing:
            df[m] = 0
        # seřadit sloupce podle měsíců a řádky podle METRICS
        df = df[MONTHS_CS]
        for r in METRICS:
            if r not in df.index:
                df.loc[r] = 0
        df = df.loc[METRICS]
        return df
    except Exception as e:
        st.sidebar.error(f"Nepodařilo se načíst soubor: {e}")
        return None


# =========================
# Data – inicializace stavu
# =========================
if "data" not in st.session_state:
    st.session_state.data = empty_df()

# Když je nahrán soubor, načteme ho
uploaded_df = read_uploaded(uploaded_file)
if uploaded_df is not None:
    st.session_state.data = uploaded_df

# Titulek
st.markdown("<h1 style='color:#E8F1EE;margin-bottom:0.25rem;'>Marketingový dashboard</h1>", unsafe_allow_html=True)
st.caption("Tenants, MQL, MQL goal a plnění v čase")

# =========================
# Interaktivní tabulka (ruční editace)
# =========================
st.markdown("### Vstupní tabulka (měsíce Leden–Prosinec)")
st.caption("Řádek **MQL % (auto)** se přepočítává automaticky z hodnot *MQL* a *MQL goal*.")

# Vypočítat % před vykreslením (aby se zobrazila aktuální hodnota)
st.session_state.data = sanitize_numeric(st.session_state.data)
st.session_state.data = compute_mql_pct(st.session_state.data)

# Konfigurace editoru – číselné sloupce, MQL % jen pro čtení
col_cfg = {}
for m in MONTHS_CS:
    col_cfg[m] = st.column_config.NumberColumn(
        label=m,
        help=f"Hodnoty pro měsíc {m}",
        min_value=0,
        step=1,
        format="%d"
    )

edited = st.data_editor(
    st.session_state.data,
    use_container_width=True,
    hide_index=False,
    column_config=col_cfg,
    disabled=[m for m in MONTHS_CS if False],  # sloupce zůstávají editovatelné
    num_rows="fixed",
)

# Zpět do session state (s tím, že MQL % přepočítáme)
st.session_state.data.loc["Tenants", MONTHS_CS] = edited.loc["Tenants", MONTHS_CS]
st.session_state.data.loc["MQL", MONTHS_CS] = edited.loc["MQL", MONTHS_CS]
st.session_state.data.loc["MQL goal", MONTHS_CS] = edited.loc["MQL goal", MONTHS_CS]

# přepočet
st.session_state.data = sanitize_numeric(st.session_state.data)
st.session_state.data = compute_mql_pct(st.session_state.data)

# Export tlačítka
c1, c2, c3 = st.columns([1,1,6])
with c1:
    csv_buf = io.StringIO()
    st.session_state.data.to_csv(csv_buf)
    st.download_button("⬇️ Export CSV", csv_buf.getvalue(), file_name="marketing_dashboard.csv", mime="text/csv")
with c2:
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        st.session_state.data.to_excel(writer, sheet_name="Data")
    st.download_button("⬇️ Export XLSX", xls_buf.getvalue(), file_name="marketing_dashboard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()

# =========================
# KPI karty
# =========================
T_sum, MQL_sum, GOAL_sum, YTD_pct, best_m, worst_m = kpi_cards(st.session_state.data)

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"<div class='elev-card'><div class='kpi-number'>{T_sum:,}</div><div class='kpi-label'>YTD Tenants</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='elev-card'><div class='kpi-number'>{MQL_sum:,}</div><div class='kpi-label'>YTD MQL</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='elev-card'><div class='kpi-number'>{GOAL_sum:,}</div><div class='kpi-label'>YTD MQL goal</div></div>", unsafe_allow_html=True)
with k4:
    color_class = "good" if YTD_pct >= 100 else ("warn" if YTD_pct >= 80 else "")
    st.markdown(f"<div class='elev-card'><div class='kpi-number {color_class}'>{YTD_pct:.0f}%</div><div class='kpi-label'>YTD plnění</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='elev-card'><div class='kpi-number'>{best_m}</div><div class='kpi-label'>Nejlepší měsíc</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='elev-card'><div class='kpi-number'>{worst_m}</div><div class='kpi-label'>Nejhorší měsíc</div></div>", unsafe_allow_html=True)

st.divider()

# =========================
# Graf 1: MQL vs. MQL goal + MQL % (kombinovaný)
# =========================
wide_df = st.session_state.data.copy()

fig_combo = go.Figure()
fig_combo.add_trace(go.Bar(
    x=MONTHS_CS,
    y=wide_df.loc["MQL", MONTHS_CS],
    name="MQL",
    marker_color=PRIMARY,
))
fig_combo.add_trace(go.Bar(
    x=MONTHS_CS,
    y=wide_df.loc["MQL goal", MONTHS_CS],
    name="MQL goal",
    marker_color=SECONDARY,
    opacity=0.85,
))
fig_combo.add_trace(go.Scatter(
    x=MONTHS_CS,
    y=wide_df.loc["MQL % (auto)", MONTHS_CS],
    name="MQL %",
    mode="lines+markers",
    yaxis="y2",
    line=dict(width=3, color=ACCENT),
))
fig_combo.update_layout(
    title="MQL vs. MQL goal + MQL plnění (%)",
    barmode="group",
    template=plotly_template,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=10),
    height=420,
    xaxis_title="Měsíc",
    yaxis=dict(title="Počet"),
    yaxis2=dict(title="Plnění %", overlaying="y", side="right", rangemode="tozero", range=[0, max(120, float(wide_df.loc["MQL % (auto)", MONTHS_CS].max()) + 20)]),
)
# referenční linka 100 %
fig_combo.add_hline(y=100, line_dash="dot", line_color=ACCENT, yref="y2")

cA, cB = st.columns([2, 1])
with cA:
    st.plotly_chart(fig_combo, use_container_width=True)

# =========================
# Graf 2: Tenants vs. MQL (srovnávací)
# =========================
long_df = to_long(wide_df)
comp_df = long_df[long_df["Metrika"].isin(["Tenants", "MQL"])]
fig_comp = px.bar(
    comp_df,
    x="Měsíc", y="Hodnota", color="Metrika",
    barmode="group",
    color_discrete_map={"Tenants": SECONDARY, "MQL": PRIMARY},
    template=plotly_template,
)
fig_comp.update_layout(
    title="Tenants vs. MQL",
    height=420,
    margin=dict(l=20, r=20, t=30, b=10),
    legend_title="",
    xaxis_title="Měsíc",
    yaxis_title="Počet",
)
with cB:
    st.plotly_chart(fig_comp, use_container_width=True)

st.divider()

# =========================
# Mini-grafy (4 karty)
# =========================
mg1, mg2, mg3, mg4 = st.columns(4)

# Tenants
with mg1:
    df_tmp = pd.DataFrame({"Měsíc": MONTHS_CS, "Hodnota": wide_df.loc["Tenants", MONTHS_CS].astype(int)})
    fig = px.bar(df_tmp, x="Měsíc", y="Hodnota", template=plotly_template, color_discrete_sequence=[SECONDARY])
    fig.update_layout(title="Tenants (měsíčně)", xaxis_title="Měsíc", yaxis_title="Počet", height=260, margin=dict(l=12, r=12, t=20, b=12), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# MQL
with mg2:
    df_tmp = pd.DataFrame({"Měsíc": MONTHS_CS, "Hodnota": wide_df.loc["MQL", MONTHS_CS].astype(int)})
    fig = px.bar(df_tmp, x="Měsíc", y="Hodnota", template=plotly_template, color_discrete_sequence=[PRIMARY])
    fig.update_layout(title="MQL (měsíčně)", xaxis_title="Měsíc", yaxis_title="Počet", height=260, margin=dict(l=12, r=12, t=20, b=12), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# MQL goal
with mg3:
    df_tmp = pd.DataFrame({"Měsíc": MONTHS_CS, "Hodnota": wide_df.loc["MQL goal", MONTHS_CS].astype(int)})
    fig = px.bar(df_tmp, x="Měsíc", y="Hodnota", template=plotly_template, color_discrete_sequence=[SECONDARY])
    fig.update_layout(title="MQL goal (měsíčně)", xaxis_title="Měsíc", yaxis_title="Počet", height=260, margin=dict(l=12, r=12, t=20, b=12), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# MQL %
with mg4:
    df_tmp = pd.DataFrame({"Měsíc": MONTHS_CS, "Hodnota": wide_df.loc["MQL % (auto)", MONTHS_CS].astype(float)})
    fig = px.line(df_tmp, x="Měsíc", y="Hodnota", markers=True, template=plotly_template)
    fig.update_traces(line=dict(width=3, color=ACCENT))
    fig.add_hline(y=100, line_dash="dot", line_color=ACCENT)
    fig.update_layout(title="MQL plnění (%)", xaxis_title="Měsíc", yaxis_title="%", height=260, margin=dict(l=12, r=12, t=20, b=12), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Tip: Importuješ-li CSV/XLSX, očekává se tabulka s řádky Tenants, MQL, MQL goal a MQL % (auto) a sloupci Leden–Prosinec. MQL % se přepočítává.")
