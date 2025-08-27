# streamlit_app_splitplot_pro_fixed.py
# Modified to handle dataset format with two header rows
# and auto-merge headers like d1/r1 into d1_r1

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path
from io import BytesIO, StringIO
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from textwrap import dedent
import warnings
warnings.filterwarnings("ignore")

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Split Plot Analyzer", layout="wide")

st.title("🧪 Split Plot / Two-Factor Analyzer")
st.caption("Handles data with two header rows like d1/d2/d3 and r1/r2, plus Genotype and Mean columns.")

# ---------------- Helpers -----------------

def read_two_header(uploaded):
    """Read Excel/CSV with two header rows and merge them to d1_r1 format."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded, header=[0, 1])
    else:
        df = pd.read_csv(uploaded, header=[0, 1])
    # Merge headers
    df.columns = [
        "_".join([str(x) for x in tup if str(x) != "nan"]).strip()
        for tup in df.columns.values
    ]
    return df

def transform_df(df):
    # Remove Mean column if present
    drop_cols = [c for c in df.columns if c.lower().startswith("mean")]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def wide_to_long(df):
    geno_col = df.columns[0]
    cols = df.columns[1:]
    records = []
    for col in cols:
        col_s = str(col).strip()
        if "_" not in col_s:
            continue
        factor, rep = col_s.split("_")
        tmp = df[[geno_col, col]].copy()
        tmp.columns = ["Genotype", "Response"]
        tmp["Treatment"] = factor
        tmp["Replication"] = rep
        records.append(tmp)
    long_df = pd.concat(records, ignore_index=True)
    long_df["Response"] = pd.to_numeric(long_df["Response"], errors="coerce")
    long_df.dropna(subset=["Response"], inplace=True)
    return long_df

def run_anova(df):
    model = ols("Response ~ C(Genotype) * C(Treatment)", data=df).fit()
    anova_res = anova_lm(model, typ=2)
    return model, anova_res

def tukey_table(df, group_col):
    try:
        tk = pairwise_tukeyhsd(endog=df["Response"], groups=df[group_col], alpha=0.05)
        d = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
        return d
    except Exception:
        return None

def significance_stars(p):
    if pd.isna(p): return ""
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

# ---------------- App Flow ----------------

uploaded = st.file_uploader("Upload CSV/XLSX with two header rows", type=["csv","xlsx","xls"])

if uploaded:
    df_raw = read_two_header(uploaded)
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())

    df_fixed = transform_df(df_raw)
    df_long = wide_to_long(df_fixed)
    st.subheader("Transformed Long Format Data")
    st.dataframe(df_long.head())

    st.write(f"Total observations: {len(df_long)}")

    # ANOVA
    st.subheader("ANOVA Table")
    model, anova_res = run_anova(df_long)
    anova_show = anova_res.copy()
    if "PR(>F)" in anova_show.columns:
        anova_show["Signif."] = anova_show["PR(>F)"].apply(significance_stars)
    st.dataframe(anova_show.round(6))

    # Post-hoc
    st.subheader("Tukey HSD - Genotype")
    tuk_g = tukey_table(df_long, "Genotype")
    if tuk_g is not None:
        st.dataframe(tuk_g)
    else:
        st.info("Not enough data for Tukey (Genotype)")

    st.subheader("Tukey HSD - Treatment")
    tuk_t = tukey_table(df_long, "Treatment")
    if tuk_t is not None:
        st.dataframe(tuk_t)
    else:
        st.info("Not enough data for Tukey (Treatment)")

    # Visualization
    st.subheader("Visualizations")
    inter = df_long.groupby(["Genotype","Treatment"])["Response"].mean().reset_index()
    fig = px.line(inter, x="Treatment", y="Response", color="Genotype", markers=True,
                  title="Interaction Plot: Genotype × Treatment")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload your dataset to start analysis.")
