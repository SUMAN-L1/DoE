# split_plot_analyser_app.py
# -------------------------------------------------------------------
# Split plot analyser — Streamlit app
# Developed by Bhavya
#
# This version is tuned to your exact data structure:
# - Rows: genotypes (g1 .. g19) — MAIN PLOT factor
# - Columns: two header rows: top = days (d1, d2, d3), second = replications (r1, r2)
#   These are merged to column names like "d1_r1", "d2_r2", etc.
# - The app:
#    1) Reads CSV/XLS/XLSX files with two header rows and merges headers
#    2) Drops any 'Mean' column
#    3) Transforms to long format: Genotype, Treatment (day), Replication, Response
#    4) Runs OLS ANOVA: Response ~ C(Genotype) * C(Treatment) (Type II or Type III selectable)
#    5) Optionally fits a Mixed Effects model (random intercept for Genotype) — recommended for split-plot
#    6) Computes post-hoc pairwise contrasts (contrast estimate, SE, df, t.ratio, p.value, Significance)
#       — contrasts use the residual MSE from the OLS ANOVA to compute SE (classic approach).
#    7) Also provides statsmodels Tukey HSD for Treatments (if applicable)
#    8) Produces publication-quality Plotly visuals and diagnostic plots
#    9) Exports: long data, ANOVA table, post-hoc table, report
#
# Note about split-plot inference:
# - For full split-plot inference with separate whole-plot and subplot error terms, a mixed-effects
#   model (or specialized split-plot routine) is more appropriate. The app fits a mixed model
#   optionally and displays its summary, but pairwise contrasts are computed from the OLS residual MSE
#   for transparency and because getting correct SE/df for contrasts from mixed models requires
#   more advanced marginal means code (not in base statsmodels).
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
st.markdown("<h1 style='text-align:center'>🧪 Split plot analyser</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray'>Developed by Bhavya</h4>", unsafe_allow_html=True)
st.write("---")

# ----------------- Helper functions -----------------

def read_with_two_headers(uploaded_file):
    """
    Reads uploaded CSV/Excel that uses two header rows:
      header row 0: day (d1,d2,d3,...)
      header row 1: replication (r1,r2,...)
    Returns a DataFrame with merged column names ("d1_r1", ...)
    """
    suf = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    try:
        if suf in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded_file, header=[0,1])
        else:
            # try read with two header rows
            df = pd.read_csv(uploaded_file, header=[0,1])
    except Exception:
        # fallback single header
        uploaded_file.seek(0)
        if suf in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded_file, header=0)
        else:
            df = pd.read_csv(uploaded_file, header=0)
    # Merge multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            left = "" if pd.isna(tup[0]) else str(tup[0]).strip()
            right = "" if pd.isna(tup[1]) else str(tup[1]).strip()
            merged = f"{left}_{right}".strip("_")
            new_cols.append(merged)
        df.columns = new_cols
    # Drop entirely empty rows/cols
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df

def tidy_and_transform(df):
    """
    - Ensures first column is Genotype (rename to 'Genotype')
    - Drops a 'Mean' column if present
    - Extracts data columns like 'd1_r1' and transforms to long format:
       Genotype | Treatment (d1/d2/d3) | Replication (r1/r2) | Response
    """
    df = df.copy()
    # Rename first column to Genotype
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Genotype"})
    # Drop Mean-like columns
    mean_cols = [c for c in df.columns if str(c).strip().lower().startswith("mean")]
    if mean_cols:
        df = df.drop(columns=mean_cols)
    # Identify data columns
    data_cols = [c for c in df.columns if c != "Genotype"]
    records = []
    for col in data_cols:
        col_s = str(col).strip()
        # Expecting patterns like 'd1_r1' after header merge
        if "_" in col_s:
            treatment, rep = col_s.split("_", 1)
        else:
            # fallback: if header is 'd1 r1' or 'd1r1', attempt to parse
            if " " in col_s:
                parts = col_s.split()
                treatment = parts[0]
                rep = parts[1] if len(parts) > 1 else "r1"
            else:
                # try to separate letters+digits and 'r' token
                lower = col_s.lower()
                if "r" in lower and any(ch.isdigit() for ch in lower):
                    idx = lower.rfind("r")
                    treatment = col_s[:idx]
                    rep = col_s[idx:]
                else:
                    treatment = col_s
                    rep = "r1"
        tmp = df[["Genotype", col]].copy()
        tmp.columns = ["Genotype", "Response"]
        tmp["Treatment"] = treatment
        tmp["Replication"] = rep
        records.append(tmp)
    long_df = pd.concat(records, ignore_index=True)
    long_df["Response"] = pd.to_numeric(long_df["Response"], errors="coerce")
    long_df = long_df.dropna(subset=["Response"]).reset_index(drop=True)
    # Clean columns
    long_df["Genotype"] = long_df["Genotype"].astype(str).str.strip()
    long_df["Treatment"] = long_df["Treatment"].astype(str).str.strip()
    long_df["Replication"] = long_df["Replication"].astype(str).str.strip().str.lower()
    return long_df

def compute_ols_anova(long_df, typ=2):
    """
    Fit OLS model Response ~ C(Genotype) * C(Treatment) and return model + anova table.
    """
    formula = "Response ~ C(Genotype) * C(Treatment)"
    model = ols(formula, data=long_df).fit()
    anova_res = anova_lm(model, typ=typ)
    # add mean_sq for convenience
    anova_res = anova_res.copy()
    anova_res["mean_sq"] = anova_res["sum_sq"] / anova_res["df"]
    return model, anova_res

def fit_mixed_model(long_df):
    """
    Fit a simple mixed-effects model: Response ~ C(Treatment) with random intercept for Genotype.
    This is a typical approach to capture whole-plot Genotype variance and use residuals for subplot errors.
    """
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm("Response ~ C(Treatment)", long_df, groups=long_df["Genotype"])
        mdf = md.fit(method="lbfgs")
        return mdf
    except Exception as e:
        return None

def pairwise_contrasts(long_df, anova_res, group_col="Treatment"):
    """
    Compute pairwise contrasts between levels of group_col using Residual MSE from OLS ANOVA.
    Returns DataFrame with: contrast, estimate, SE, df, t.ratio, p.value, Significance
    """
    gm = long_df.groupby(group_col)["Response"].agg(["mean", "count"]).rename(columns={"mean":"mean", "count":"n"})
    levels = list(gm.index)
    # Residual row (last index usually)
    # Prefer 'Residual' label, else take last row
    if "Residual" in anova_res.index:
        resid_row = anova_res.loc["Residual"]
    elif "Residuals" in anova_res.index:
        resid_row = anova_res.loc["Residuals"]
    else:
        resid_row = anova_res.iloc[-1]
    mse = float(resid_row["mean_sq"])
    df_resid = float(resid_row["df"])
    records = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            A = levels[i]; B = levels[j]
            meanA = gm.loc[A, "mean"]; meanB = gm.loc[B, "mean"]
            nA = gm.loc[A, "n"]; nB = gm.loc[B, "n"]
            est = meanA - meanB
            se = np.sqrt(mse * (1.0/nA + 1.0/nB))
            if se == 0 or np.isnan(se):
                t_ratio = np.nan; pval = np.nan
            else:
                t_ratio = est / se
                pval = 2.0 * (1 - tdist.cdf(abs(t_ratio), df_resid))
            sig = "***" if (not pd.isna(pval) and pval < 0.001) else "**" if (not pd.isna(pval) and pval < 0.01) else "*" if (not pd.isna(pval) and pval < 0.05) else "ns"
            records.append({
                "contrast": f"{A} - {B}",
                "estimate": round(est, 6),
                "SE": round(se, 8),
                "df": int(df_resid) if not pd.isna(df_resid) else None,
                "t.ratio": round(t_ratio, 6) if not pd.isna(t_ratio) else np.nan,
                "p.value": round(pval, 9) if not pd.isna(pval) else np.nan,
                "Significance": sig
            })
    posthoc_df = pd.DataFrame(records).sort_values("p.value").reset_index(drop=True)
    return posthoc_df

def format_anova_display(anova_res):
    df = anova_res.copy()
    disp = df[["sum_sq","df","mean_sq","F","PR(>F)"]].rename(columns={"sum_sq":"Sum Sq","mean_sq":"Mean Sq","PR(>F)":"p-value"})
    return disp.round(6)

def to_csv_bytes(df):
    buff = StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")

def generate_report_text(long_df, anova_res, posthoc_df):
    n_geno = long_df["Genotype"].nunique()
    n_treat = long_df["Treatment"].nunique()
    total_obs = len(long_df)
    best_pair = long_df.groupby(["Genotype","Treatment"])["Response"].mean().idxmax()
    best_mean = long_df.groupby(["Genotype","Treatment"])["Response"].mean().max()
    text = dedent(f"""
    Split plot analyser — Report
    Developed by Bhavya

    Design summary:
    - Genotypes (main plots): {n_geno}
    - Days (treatments, subplots): {n_treat}
    - Total observations: {total_obs}

    ANOVA (key table):
    {anova_res.round(6).to_string()}

    Best combination: {best_pair[0]} × {best_pair[1]} (Mean = {best_mean:.3f})

    Top post-hoc contrasts:
    {posthoc_df.head(10).to_string(index=False)}

    Notes:
    - For formal split-plot inference prefer mixed-effects estimation for correct error partitioning.
    - This app fits both OLS ANOVA and an optional mixed model (random intercept Genotype).
    """)
    return text

# ----------------- Sidebar: Upload & Options -----------------
st.sidebar.header("Upload & Options")
uploaded = st.sidebar.file_uploader("Upload CSV / XLS / XLSX (two header rows)", type=["csv","xls","xlsx"])
anova_type = st.sidebar.selectbox("ANOVA type", ["Type II","Type III"], index=0,
                                  help="Type II common for balanced; Type III for unbalanced or when interactions present.")
fit_mixed = st.sidebar.checkbox("Fit Mixed-Effects model (random intercept for Genotype)", value=True)
export_report = st.sidebar.checkbox("Enable downloadable report", value=True)

if uploaded is None:
    st.info("Upload your dataset (CSV/XLS/XLSX). The file should have two header rows: top row days (d1,d2,d3), second row replications (r1,r2). The left-most column should be genotype labels (g1...). A 'Mean' column (if present) will be ignored.")
    st.stop()

# ----------------- Read & Transform -----------------
try:
    raw_df = read_with_two_headers(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Raw data (preview)")
st.dataframe(raw_df.head(8), use_container_width=True)

# Transform
long_df = tidy_and_transform(raw_df)
if long_df.empty:
    st.error("No numeric Response values found after transformation. Check file format.")
    st.stop()

st.subheader("Long format data (preview)")
st.dataframe(long_df.head(12), use_container_width=True)
st.markdown(f"**Observations:** {len(long_df)}  •  **Genotypes:** {long_df['Genotype'].nunique()}  •  **Days (Treatments):** {long_df['Treatment'].nunique()}")

# ----------------- ANOVA -----------------
st.write("---")
st.header("ANOVA (Genotype × Day)")

typ = 2 if anova_type == "Type II" else 3
ols_model, anova_res = compute_ols_anova(long_df, typ=typ)

st.subheader("OLS ANOVA table")
st.dataframe(format_anova_display(anova_res), use_container_width=True)

# Interaction message
interaction_label = f"C(Genotype):C(Treatment)"
if interaction_label in anova_res.index:
    p_int = anova_res.loc[interaction_label, "PR(>F)"]
    if p_int < 0.05:
        st.warning(f"Significant Genotype × Treatment interaction (p = {p_int:.4g}). Interpret main effects with caution.")
    else:
        st.success(f"No significant Genotype × Treatment interaction (p = {p_int:.4g}).")

# ----------------- Mixed Model (Split-plot) -----------------
if fit_mixed:
    st.write("---")
    st.header("Mixed-effects model (recommended for split-plot inference)")
    mdf = fit_mixed_model(long_df)
    if mdf is None:
        st.error("Mixed model could not be fit. This can happen if data lack variability or model convergence fails.")
    else:
        st.text("Mixed model summary (random intercept for Genotype):")
        st.text(mdf.summary().as_text())

# ----------------- Post-hoc contrasts -----------------
st.write("---")
st.header("Post-hoc pairwise contrasts (Days) — contrasts computed using OLS residual MSE")

posthoc_df = pairwise_contrasts(long_df, anova_res, group_col="Treatment")
# display in requested format (rounded)
st.dataframe(posthoc_df, use_container_width=True)

# Additionally show Tukey HSD (statsmodels) for the Treatment factor (p-adj)
try:
    tuk = pairwise_tukeyhsd(endog=long_df["Response"], groups=long_df["Treatment"], alpha=0.05)
    tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
    st.subheader("Tukey HSD (adjusted p-values) — Treatments")
    st.dataframe(tuk_df, use_container_width=True)
except Exception:
    st.info("Tukey HSD (statsmodels) could not be computed for Treatments on this dataset (insufficient or unbalanced data).")

# ----------------- Visualizations -----------------
st.write("---")
st.header("Publication-quality visualizations")

PLOTLY_LAYOUT = dict(template="plotly_white", font=dict(family="Arial", size=13), legend=dict(orientation="h"))

# 1) Treatment (day) means with 95% CI (Mean ± 1.96*SE)
tstats = long_df.groupby("Treatment")["Response"].agg(["mean","std","count"]).reset_index()
tstats["se"] = tstats["std"] / np.sqrt(tstats["count"])
tstats["ci_lower"] = tstats["mean"] - 1.96*tstats["se"]
tstats["ci_upper"] = tstats["mean"] + 1.96*tstats["se"]
fig_t = go.Figure()
fig_t.add_trace(go.Bar(x=tstats["Treatment"], y=tstats["mean"], marker=dict(color="rgb(31,119,180)"),
                       error_y=dict(type="data", array=tstats["se"], visible=True), name="Mean ± SE"))
fig_t.update_layout(title="Day (Treatment) means — Mean ± SE", xaxis_title="Day", yaxis_title="Response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_t, use_container_width=True)

# 2) Genotype means (for reference)
gstats = long_df.groupby("Genotype")["Response"].agg(["mean","std","count"]).reset_index()
gstats["se"] = gstats["std"] / np.sqrt(gstats["count"])
fig_g = go.Figure()
fig_g.add_trace(go.Bar(x=gstats["Genotype"], y=gstats["mean"], marker=dict(color="rgb(44,160,44)"),
                       error_y=dict(type="data", array=gstats["se"], visible=True)))
fig_g.update_layout(title="Genotype means — Mean ± SE", xaxis_title="Genotype", yaxis_title="Response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_g, use_container_width=True)

# 3) Interaction lines: genotype lines across days
inter_mean = long_df.groupby(["Genotype","Treatment"])["Response"].mean().reset_index()
fig_int = px.line(inter_mean, x="Treatment", y="Response", color="Genotype", markers=True)
fig_int.update_layout(title="Interaction plot — Genotype × Day", xaxis_title="Day", yaxis_title="Mean Response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_int, use_container_width=True)

# 4) Heatmap of means (Genotype x Day)
heat = inter_mean.pivot(index="Genotype", columns="Treatment", values="Response")
fig_heat = px.imshow(heat.values, x=heat.columns, y=heat.index, aspect="auto", color_continuous_scale="Viridis")
fig_heat.update_layout(title="Heatmap of Genotype × Day mean response", **PLOTLY_LAYOUT)
st.plotly_chart(fig_heat, use_container_width=True)

# 5) Boxplots: Response by Day, points colored by Genotype
fig_box = px.box(long_df, x="Treatment", y="Response", color="Genotype", points="all")
fig_box.update_layout(title="Response distribution by Day (colored by Genotype)", **PLOTLY_LAYOUT)
st.plotly_chart(fig_box, use_container_width=True)

# Diagnostics
st.write("---")
st.header("Model diagnostics (OLS model)")

resid = ols_model.resid
fitted = ols_model.fittedvalues

# Residuals vs Fitted
fig_r = go.Figure()
fig_r.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", marker=dict(opacity=0.7)))
fig_r.add_hline(y=0, line_dash="dash", line_color="red")
fig_r.update_layout(title="Residuals vs Fitted (OLS)", xaxis_title="Fitted values", yaxis_title="Residuals", **PLOTLY_LAYOUT)
st.plotly_chart(fig_r, use_container_width=True)

# Q-Q plot
qq = stats.probplot(resid, dist="norm")
qq_x, qq_y = qq[0][0], qq[0][1]
slope, intercept = qq[1][0], qq[1][1]
fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers", name="Sample"))
fig_qq.add_trace(go.Scatter(x=qq_x, y=intercept + slope*qq_x, mode="lines", name="Theoretical", line=dict(color="red", dash="dash")))
fig_qq.update_layout(title="Normal Q-Q plot (OLS residuals)", xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", **PLOTLY_LAYOUT)
st.plotly_chart(fig_qq, use_container_width=True)

# Shapiro-Wilk & Levene
st.subheader("Assumption tests")
try:
    sh_w, sh_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    st.write(f"Shapiro-Wilk normality test p-value: {sh_p:.6f}" if not np.isnan(sh_p) else "Shapiro-Wilk not computed for >5000 obs")
except Exception as e:
    st.write("Shapiro test unavailable:", e)

try:
    grouped = [g["Response"].values for _, g in long_df.groupby(["Genotype","Treatment"])]
    if len(grouped) > 1:
        lev_w, lev_p = stats.levene(*grouped)
        st.write(f"Levene's test for equal variances p-value: {lev_p:.6f}")
except Exception as e:
    st.write("Levene test unavailable:", e)

# ----------------- Export / Downloads -------------------
st.write("---")
st.header("Export results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.download_button("Download long data (CSV)", data=to_csv_bytes(long_df), file_name="long_data.csv", mime="text/csv")
with col2:
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
st.caption("Notes:\n"
           "- Genotype is treated as the main-plot factor and Day (d1,d2,d3) as subplot (within-genotype) factor.\n"
           "- OLS ANOVA is displayed (easy-to-read table) and pairwise contrasts are computed using the OLS residual MSE\n"
           "  to produce contrast SE, t and p-values in the requested format.\n"
           "- For fully correct split-plot inference, prefer the mixed-effects model results (random intercept for Genotype).\n"
           "- If you want contrasts computed from mixed-model marginal means (with Satterthwaite df), I can extend the app\n"
           "  to use emmeans-style approximations (requires additional implementation).")
