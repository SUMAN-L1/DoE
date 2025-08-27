# split_plot_analyser_app.py
# -------------------------------------------------------------------
# Split plot analyser — Streamlit app
# Developed by Bhavya
#
# Features:
# - Accepts CSV / XLS / XLSX uploads where the file has TWO header rows:
#     Row 1: treatment (d1, d2, d3, ...)
#     Row 2: replication (r1, r2, ...)
#   and first column contains Genotype labels (g1..gN) and optionally a Mean column (ignored).
# - Merges the two header rows to form column names like "d1_r1", "d2_r2", etc.
# - Transforms to long format, performs ANOVA (Genotype, Treatment, Genotype×Treatment).
# - Computes residual MSE, then performs pairwise post-hoc tests (Tukey-like contrasts)
#   and returns a results table with: contrast, estimate, SE, df, t.ratio, p.value, Significance.
# - Publication-ready visualizations (Plotly): main effects, interaction lines, heatmap, box/violin,
#   and diagnostic plots (Residuals vs Fitted, Q-Q).
# - Export options for long data, ANOVA table, post-hoc table, and textual report.
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO, StringIO
import warnings
warnings.filterwarnings("ignore")

# Stats & plotting
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from scipy.stats import t as tdist
import plotly.express as px
import plotly.graph_objects as go
from textwrap import dedent

# ----------------- App config -----------------
st.set_page_config(page_title="Split plot analyser", page_icon="🧪", layout="wide")
st.markdown(f"<h1 style='text-align:center'>🧪 Split plot analyser</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align:center; color:gray'>Developed by Bhavya</h4>", unsafe_allow_html=True)
st.write("---")

# ----------------- Helpers --------------------

def read_with_two_headers(uploaded_file):
    """
    Reads uploaded CSV/Excel that uses two header rows:
      header row 0: treatment (d1,d2,d3,...)
      header row 1: replication (r1,r2,...)
    Returns a DataFrame with merged column names ("d1_r1", ...)
    """
    suf = Path(uploaded_file.name).suffix.lower()
    # Try reading with two header rows first
    try:
        if suf in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded_file, header=[0,1])
        else:
            # CSV may also contain two header lines; handle similarly
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=[0,1])
    except Exception as e:
        # Fallback: read with single header
        uploaded_file.seek(0)
        if suf in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded_file, header=0)
        else:
            df = pd.read_csv(uploaded_file, header=0)
    # Merge multiindex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            left = "" if pd.isna(tup[0]) else str(tup[0]).strip()
            right = "" if pd.isna(tup[1]) else str(tup[1]).strip()
            merged = f"{left}_{right}".strip("_")
            new_cols.append(merged)
        df.columns = new_cols
    else:
        # If single-header but second row actually contains rep labels (common in poorly saved CSV),
        # try inspecting second row to create merged names.
        # If first column name looks like 'Genotype' and second row has rep names like r1,r2,...
        # we attempt to reconstruct.
        # We'll not overwrite unless we detect likely pattern: presence of 'r' tokens in row 0 or 1.
        pass
    # drop any fully-empty columns
    df = df.loc[:, df.columns.notna()]
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df

def tidy_and_transform(df):
    """
    - Ensures first column is Genotype (rename if not).
    - Drops a 'Mean' column if present.
    - Keeps only columns that match pattern like 'd<number>_r<number>'.
    - Converts to long format with columns: Genotype, Treatment, Replication, Response
    """
    df = df.copy()
    # First column is genotype (if header name isn't informative, enforce 'Genotype')
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Genotype"})
    # Drop Mean-like columns (case-insensitive)
    mean_cols = [c for c in df.columns if str(c).strip().lower().startswith("mean")]
    if mean_cols:
        df = df.drop(columns=mean_cols)

    # Identify data columns: those containing '_' indicating treatment_rep or like 'd1_r1'
    data_cols = [c for c in df.columns if ("_" in str(c)) or (str(c).lower().startswith("d") and "r" in str(c).lower())]
    # If no data_cols found, try to interpret multi-level columns by splitting on whitespace
    if not data_cols:
        data_cols = [c for c in df.columns if c != "Genotype"]

    records = []
    for col in data_cols:
        col_s = str(col).strip()
        # try to parse "d1_r1" or "d1 r1" or "d1r1"
        if "_" in col_s:
            t, r = col_s.split("_", 1)
        elif " " in col_s:
            parts = col_s.split()
            t = parts[0]
            r = parts[1] if len(parts) > 1 else "r1"
        else:
            # attempt to split letters from digits and r tokens
            # fallback: treat as single treatment and default rep r1
            if "r" in col_s.lower() and any(ch.isdigit() for ch in col_s):
                # try to find 'r' marker
                # rudimentary parse
                idx = col_s.lower().find("r")
                t = col_s[:idx]
                r = col_s[idx:]
            else:
                t = col_s
                r = "r1"
        tmp = df[["Genotype", col]].copy()
        tmp = tmp.rename(columns={col: "Response"})
        tmp["Treatment"] = t
        tmp["Replication"] = r
        records.append(tmp)

    long_df = pd.concat(records, ignore_index=True)
    long_df["Response"] = pd.to_numeric(long_df["Response"], errors="coerce")
    long_df = long_df.dropna(subset=["Response"]).reset_index(drop=True)
    # Clean whitespace
    long_df["Genotype"] = long_df["Genotype"].astype(str).str.strip()
    long_df["Treatment"] = long_df["Treatment"].astype(str).str.strip()
    long_df["Replication"] = long_df["Replication"].astype(str).str.strip().str.lower()
    return long_df

def compute_anova(long_df, typ=2):
    """
    Fit model Response ~ C(Genotype)*C(Treatment)
    Return fitted model and ANOVA table (and ensure mean_sq/residual info present).
    """
    formula = "Response ~ C(Genotype) * C(Treatment)"
    model = ols(formula, data=long_df).fit()
    anova_res = anova_lm(model, typ=typ)
    # add mean_sq
    anova_res = anova_res.copy()
    anova_res["mean_sq"] = anova_res["sum_sq"] / anova_res["df"]
    return model, anova_res

def pairwise_contrasts(long_df, anova_res, group_col="Treatment"):
    """
    Compute pairwise contrasts between levels of group_col.
    Use residual MSE from anova_res to compute SE, t, p.
    Returns DataFrame with columns:
      contrast, estimate, SE, df, t.ratio, p.value, Significance
    """
    # group means and sizes
    gm = long_df.groupby(group_col)["Response"].agg(["mean", "count"]).rename(columns={"mean":"mean", "count":"n"})
    levels = list(gm.index)
    # Determine residual mean square and df
    # Residual row label might be 'Residual' or 'Residuals'. Find last row with index not factors.
    if "Residual" in anova_res.index:
        resid_row = anova_res.loc["Residual"]
    else:
        # Typically residual is last row
        resid_row = anova_res.iloc[-1]
    mse = float(resid_row["mean_sq"])
    df_resid = float(resid_row["df"])

    records = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            A = levels[i]
            B = levels[j]
            meanA = gm.loc[A, "mean"]
            meanB = gm.loc[B, "mean"]
            nA = gm.loc[A, "n"]
            nB = gm.loc[B, "n"]
            est = meanA - meanB
            se = np.sqrt(mse * (1.0/nA + 1.0/nB))
            # handle if se==0
            if se == 0 or np.isnan(se):
                t_ratio = np.nan
                pval = np.nan
            else:
                t_ratio = est / se
                # two-sided p-value using t-distribution with df_resid
                pval = 2.0 * (1 - tdist.cdf(abs(t_ratio), df_resid))
            # significance stars
            if pd.isna(pval):
                sig = ""
            else:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            records.append({
                "contrast": f"{A} - {B}",
                "estimate": round(est, 6),
                "SE": round(se, 8),
                "df": int(df_resid),
                "t.ratio": round(t_ratio, 6) if not pd.isna(t_ratio) else np.nan,
                "p.value": round(pval, 9) if not pd.isna(pval) else np.nan,
                "Significance": sig
            })
    posthoc_df = pd.DataFrame(records)
    # Sort by absolute t.ratio descending so most significant first (optional)
    posthoc_df = posthoc_df.sort_values(by="p.value").reset_index(drop=True)
    return posthoc_df

def format_anova_for_display(anova_res):
    df = anova_res.copy()
    # Round numbers for display
    display = df[["sum_sq", "df", "mean_sq", "F", "PR(>F)"]].round(6)
    display = display.rename(columns={"sum_sq":"Sum Sq", "mean_sq":"Mean Sq", "PR(>F)":"p-value"})
    return display

def generate_report_text(long_df, anova_res, posthoc_df):
    n_geno = long_df["Genotype"].nunique()
    n_treat = long_df["Treatment"].nunique()
    total_obs = len(long_df)
    best = long_df.groupby(["Genotype","Treatment"])["Response"].mean().idxmax()
    best_mean = long_df.groupby(["Genotype","Treatment"])["Response"].mean().max()
    text = dedent(f"""
    Split Plot Analysis Report
    --------------------------
    Developed by Bhavya

    Design summary:
    - Genotypes (main plots): {n_geno}
    - Treatments (subplots): {n_treat}
    - Total observations: {total_obs}

    ANOVA (key results):
    {anova_res.round(6).to_string()}

    Best performing combination: {best[0]} × {best[1]}  (Mean = {best_mean:.3f})

    Top significant post-hoc contrasts:
    {posthoc_df.head(10).to_string(index=False)}

    Notes:
    - If Genotype×Treatment interaction is significant, interpret main effects with caution.
    - Residual diagnostics and publication-ready plots exported separately.
    """)
    return text

def to_csv_bytes(df):
    buff = StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")

# ----------------- UI: Upload -------------------
st.sidebar.header("Upload & Options")
uploaded = st.sidebar.file_uploader("Upload CSV / XLS / XLSX (two header rows)", type=["csv","xls","xlsx"])
anova_type = st.sidebar.selectbox("ANOVA Type", options=["Type II", "Type III"], index=0)
show_mixed = st.sidebar.checkbox("Also fit Mixed Effects (random intercept for Genotype) — optional", value=False)
export_report = st.sidebar.checkbox("Include downloadable report", value=True)

if uploaded is None:
    st.info("Please upload your dataset (CSV/XLS/XLSX). The file should have two header rows: treatment row (d1,d2,...) and replication row (r1,r2,...), and first column should be Genotype names.")
    st.stop()

# ----------------- Read & Transform -------------------
try:
    raw_df = read_with_two_headers(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Raw data (first 8 rows)")
st.dataframe(raw_df.head(8), use_container_width=True)

# Tidy and convert
long_df = tidy_and_transform(raw_df)
if long_df.empty:
    st.error("After transformation, no numeric Response values were found. Please verify file format.")
    st.stop()

st.subheader("Transformed long data (first 10 rows)")
st.dataframe(long_df.head(10), use_container_width=True)
st.markdown(f"**Observations:** {len(long_df)} — **Genotypes:** {long_df['Genotype'].nunique()} — **Treatments:** {long_df['Treatment'].nunique()}")

# ----------------- ANOVA -------------------
st.write("---")
st.header("ANOVA — Split plot style (Genotype × Treatment)")

typ = 2 if anova_type == "Type II" else 3
model, anova_res = compute_anova(long_df, typ=typ)

# Display ANOVA
anova_display = format_anova_for_display(anova_res)
st.subheader("ANOVA Table")
st.dataframe(anova_display)

# Show message on interaction
if f"C(Genotype):C(Treatment)" in anova_res.index:
    if anova_res.loc[f"C(Genotype):C(Treatment)", "PR(>F)"] < 0.05:
        st.warning("Significant Genotype × Treatment interaction detected (p < 0.05). Interpret main effects cautiously.")
    else:
        st.success("No significant Genotype × Treatment interaction detected.")

# Optionally fit mixed-effects (simple random intercept)
if show_mixed:
    st.subheader("Mixed-effects model (random intercept for Genotype) — quick fit")
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm("Response ~ C(Treatment)", long_df, groups=long_df["Genotype"])
        mdf = md.fit(method="lbfgs")
        st.text(mdf.summary().as_text())
    except Exception as e:
        st.error(f"Mixed model failed: {e}")

# ----------------- Post-hoc contrasts -------------------
st.write("---")
st.header("Post-hoc pairwise contrasts (Tukey-like)")

posthoc_df = pairwise_contrasts(long_df, anova_res, group_col="Treatment")
# present with requested column names formatting
posthoc_df_display = posthoc_df.rename(columns={
    "contrast":"contrast", "estimate":"estimate", "SE":"SE", "df":"df", "t.ratio":"t.ratio", "p.value":"p.value", "Significance":"Significance"
})
st.dataframe(posthoc_df_display, use_container_width=True)

# Also show classic Tukey HSD results (p-adj) for user reference
try:
    mc = pairwise_tukeyhsd(endog=long_df["Response"], groups=long_df["Treatment"], alpha=0.05)
    tukey_df = pd.DataFrame(mc.summary().data[1:], columns=mc.summary().data[0])
    st.subheader("Tukey HSD summary (statsmodels implementation)")
    st.dataframe(tukey_df, use_container_width=True)
except Exception:
    st.info("Tukey HSD (statsmodels) could not be computed on this dataset (insufficient/irregular data).")

# ----------------- Visualizations -------------------
st.write("---")
st.header("Publication-quality Visualizations (Plotly)")

# Styling helper
PLOTLY_LAYOUT = dict(template="plotly_white", font=dict(family="Arial", size=14), legend=dict(orientation="h"))

# 1) Main effect: Treatment means with error bars (95% CI)
tstats = long_df.groupby("Treatment")["Response"].agg(["mean","std","count"]).reset_index()
tstats["se"] = tstats["std"] / np.sqrt(tstats["count"])
# 95% CI
tstats["ci_lower"] = tstats["mean"] - 1.96 * tstats["se"]
tstats["ci_upper"] = tstats["mean"] + 1.96 * tstats["se"]

fig_t = go.Figure()
fig_t.add_trace(go.Bar(x=tstats["Treatment"], y=tstats["mean"], error_y=dict(type="data", array=tstats["se"], visible=True),
                       name="Mean ± SE"))
fig_t.update_layout(title="Treatment means (Mean ± SE)", xaxis_title="Treatment", yaxis_title="Response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_t, use_container_width=True)

# 2) Main effect: Genotype means
gstats = long_df.groupby("Genotype")["Response"].agg(["mean","std","count"]).reset_index()
gstats["se"] = gstats["std"] / np.sqrt(gstats["count"])
fig_g = go.Figure()
fig_g.add_trace(go.Bar(x=gstats["Genotype"], y=gstats["mean"], error_y=dict(type="data", array=gstats["se"], visible=True)))
fig_g.update_layout(title="Genotype means (Mean ± SE)", xaxis_title="Genotype", yaxis_title="Response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_g, use_container_width=True)

# 3) Interaction plot (lines for genotypes across treatments)
inter_mean = long_df.groupby(["Genotype","Treatment"])["Response"].mean().reset_index()
fig_int = px.line(inter_mean, x="Treatment", y="Response", color="Genotype", markers=True)
fig_int.update_layout(title="Interaction plot: Genotype × Treatment", xaxis_title="Treatment", yaxis_title="Mean Response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_int, use_container_width=True)

# 4) Heatmap of means (Genotype x Treatment)
heat = inter_mean.pivot(index="Genotype", columns="Treatment", values="Response")
fig_heat = px.imshow(heat.values, x=heat.columns, y=heat.index, aspect="auto", color_continuous_scale="Viridis")
fig_heat.update_layout(title="Heatmap of Genotype × Treatment mean response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_heat, use_container_width=True)

# 5) Boxplot (Response distribution by Treatment and colored by Genotype)
fig_box = px.box(long_df, x="Treatment", y="Response", color="Genotype", points="all")
fig_box.update_layout(title="Response distribution by Treatment (colored by Genotype)", **PLOTLY_LAYOUT)
st.plotly_chart(fig_box, use_container_width=True)

# 6) Violin (Treatment)
fig_violin = px.violin(long_df, x="Treatment", y="Response", box=True, points="all")
fig_violin.update_layout(title="Response distribution shape by Treatment", **PLOTLY_LAYOUT)
st.plotly_chart(fig_violin, use_container_width=True)

# Diagnostics
st.write("---")
st.header("Model diagnostics")

resid = model.resid
fitted = model.fittedvalues

# Residuals vs Fitted
fig_r = go.Figure()
fig_r.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", marker=dict(opacity=0.7)))
fig_r.add_hline(y=0, line_dash="dash", line_color="red")
fig_r.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted values", yaxis_title="Residuals", **PLOTLY_LAYOUT)
st.plotly_chart(fig_r, use_container_width=True)

# Q-Q plot
qq = stats.probplot(resid, dist="norm")
fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Sample"))
# theoretical line
slope, intercept = qq[1][0], qq[1][1]
fig_qq.add_trace(go.Scatter(x=qq[0][0], y=intercept + slope*qq[0][0], mode="lines", name="Theoretical", line=dict(color="red", dash="dash")))
fig_qq.update_layout(title="Normal Q-Q plot (residuals)", xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", **PLOTLY_LAYOUT)
st.plotly_chart(fig_qq, use_container_width=True)

# Shapiro & Levene tests
st.subheader("Assumption tests")
try:
    sh_w, sh_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    st.write(f"Shapiro-Wilk normality test p-value: {sh_p:.6f}" if not np.isnan(sh_p) else "Shapiro not computed for >5000 obs")
except Exception as e:
    st.write("Shapiro test failed:", e)

try:
    grouped = [g["Response"].values for _, g in long_df.groupby(["Genotype","Treatment"])]
    if len(grouped) > 1:
        lev_w, lev_p = stats.levene(*grouped)
        st.write(f"Levene's test for equal variances p-value: {lev_p:.6f}")
except Exception as e:
    st.write("Levene's test failed:", e)

# ----------------- Export / Downloads -------------------
st.write("---")
st.header("Export results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.download_button("Download long data (CSV)", data=to_csv_bytes(long_df), file_name="long_data.csv", mime="text/csv")
with col2:
    # ANOVA as CSV
    anova_out = anova_res.reset_index().rename(columns={"index":"Term"})
    st.download_button("Download ANOVA (CSV)", data=to_csv_bytes(anova_out), file_name="anova_table.csv", mime="text/csv")
with col3:
    st.download_button("Download post-hoc contrasts (CSV)", data=to_csv_bytes(posthoc_df), file_name="posthoc_contrasts.csv", mime="text/csv")
with col4:
    if export_report:
        report_text = generate_report_text(long_df, anova_res, posthoc_df)
        bio = BytesIO(report_text.encode("utf-8"))
        st.download_button("Download report (TXT)", data=bio, file_name="splitplot_report.txt", mime="text/plain")

st.markdown("---")
st.caption("Notes: This app fits a two-way model Response ~ Genotype * Treatment and uses the residual MSE\n"
           "from that model to compute pairwise contrasts with SE = sqrt(MSE*(1/n_i + 1/n_j)).\n"
           "For full split-plot models with separate whole-plot and sub-plot error terms, a mixed-model\n"
           "or specialized split-plot routine is recommended. The app provides an optional mixed-effect\n"
           "model (random intercept Genotype) as a quick check.")
