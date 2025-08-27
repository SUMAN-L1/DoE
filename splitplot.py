# streamlit_app_splitplot_pro.py
# --------------------------------------------------------------------
# 🧪 Split Plot / Two-Factor Analyzer — "best-ever" flexible version
# --------------------------------------------------------------------
# Key upgrades vs your original file:
# ✔ Robust reader for CSV/XLS/XLSX (handles 1-row or 2-row headers)
# ✔ Smart wide→long transformer for columns like d1_r1, d2 r2, etc.
# ✔ UI to confirm/override detected columns & factor names
# ✔ Choice of Type-II or Type-III ANOVA; clear significance flags
# ✔ Optional Mixed-Effects model (random intercept for Genotype)
# ✔ Tukey HSD for main effects + simple-effects Tukey when interaction is sig.
# ✔ Rich Plotly visuals (main effects, interaction, heatmap, distributions)
# ✔ Diagnostics (Shapiro, Levene; QQ & Residuals vs Fitted)
# ✔ Download Summary/ANOVA/Tukey/Report; Export a clean long-format dataset
# ✔ Generate a template file that matches your typical wide format
# ✔ Helpful messages for unbalanced/missing cells
# --------------------------------------------------------------------

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

# ----------------------------- UI SETUP ------------------------------
st.set_page_config(
    page_title="Split Plot / Two-Factor Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main-header {font-size: 2.2rem; font-weight: 800; color: #1f77b4; text-align:center; margin: 0.25rem 0;}
      .sub-header {font-size: 1rem; color:#555; text-align:center; margin-bottom: 1rem;}
      .metricbox {background:linear-gradient(90deg,#f6f8fb,#fff); padding:10px 14px; border-left:4px solid #1f77b4; border-radius:10px;}
      .note {background:#f8f9fa;border-left:4px solid #28a745;padding:10px;border-radius:6px;margin:8px 0;}
      .warn {background:#fff3cd;border-left:4px solid #f0ad4e;padding:10px;border-radius:6px;margin:8px 0;}
      .danger {background:#f8d7da;border-left:4px solid #dc3545;padding:10px;border-radius:6px;margin:8px 0;}
      .muted {color:#666;}
      .code-like {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🧪 Split Plot / Two-Factor Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Flexible, wide-to-long aware, publication-ready analysis</div>', unsafe_allow_html=True)

# --------------------------- HELPERS --------------------------------

def _read_any(uploaded):
    """Robust file reader. Supports 1-row or 2-row headers in Excel."""
    try:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            # Try common encodings
            for enc in [None, "utf-8", "utf-8-sig", "latin1"]:
                try:
                    df = pd.read_csv(uploaded, encoding=enc)
                    break
                except Exception:
                    uploaded.seek(0)
            else:
                return None, "Could not read CSV with common encodings."
        elif suffix in [".xlsx", ".xls"]:
            # Detect if first two rows look like headers (MultiIndex)
            df_try = pd.read_excel(uploaded, header=None)
            uploaded.seek(0)
            # Heuristic: if second row contains header-ish strings and overall few NaNs, treat as two-row header
            two_header = (
                df_try.iloc[0].notna().sum() >= 3
                and df_try.iloc[1].notna().sum() >= 3
                and df_try.shape[1] <= 100
            )
            if two_header:
                df = pd.read_excel(uploaded, header=[0, 1])
                # Join the multiindex headers with underscore and strip whites
                df.columns = ["_".join([str(x) for x in col if str(x) != "nan"]).strip() for col in df.columns.values]
            else:
                uploaded.seek(0)
                df = pd.read_excel(uploaded)
        else:
            return None, "Unsupported file format. Please upload CSV, XLS, or XLSX."
        # Drop all-empty rows/cols
        df = df.dropna(how="all").copy()
        df = df.loc[:, df.columns.notna()]
        return df, None
    except Exception as e:
        return None, f"Read error: {e}"

GENO_CANDIDATES = ["genotype", "genotypes", "variety", "varieties", "line", "entry", "treatment_main", "mainplot"]
REPL_REGEXES = [
    r"(?P<Factor>[A-Za-z]+[0-9]*)[\s_\-\.]*?(?P<Rep>[rR][0-9]+)",   # d1_r1  or d1r1
    r"(?P<Factor>[A-Za-z]+[0-9]*)\s*\(\s*(?P<Rep>[rR][0-9]+)\s*\)" # d1 (r1)
]

def detect_genotype_col(df):
    # Return best guess and all textual first columns
    cols = list(df.columns)
    scores = []
    for c in cols:
        lc = str(c).strip().lower()
        score = 0
        for key in GENO_CANDIDATES:
            if key in lc:
                score += 3
        # if values look like g1, g2...
        try:
            vals = df[c].astype(str).str.lower()
            if (vals.str.match(r"g[0-9]+").mean() > 0.2) or (vals.nunique() >= max(4, int(len(df)*0.2))):
                score += 1
        except Exception:
            pass
        scores.append((score, c))
    scores.sort(reverse=True)
    return scores[0][1] if scores else cols[0]

def wide_to_long(df, geno_col, factor_name="Treatment", rep_name="Replication"):
    """Transform wide data shaped like d1_r1 into long format."""
    # Make safe copy & ensure geno col is first
    cols = [c for c in df.columns if c != geno_col]
    df = df[[geno_col] + cols].copy()

    melt_records = []
    # Build a parser pipeline
    for col in cols:
        col_s = str(col).strip()
        # Skip empty columns
        if col_s == "" or col_s.lower() == "nan":
            continue

        matched = None
        for rgx in REPL_REGEXES:
            m = pd.Series([col_s]).str.extract(rgx, expand=True)
            if pd.notna(m.loc[0, "Factor"]).any():
                matched = {"Factor": m.loc[0, "Factor"], "Rep": m.loc[0, "Rep"]}
                break

        if matched is None:
            # Try splitting by underscores if like 'd1_r1' already present
            if "_" in col_s:
                parts = col_s.split("_")
                # guess: last one rep, others join as factor
                rep_guess = next((p for p in parts if p.lower().startswith("r")), "r1")
                factor_guess = [p for p in parts if p != rep_guess]
                factor_guess = "_".join(factor_guess) if factor_guess else col_s
                matched = {"Factor": factor_guess, "Rep": rep_guess}
            else:
                # treat as factor with default rep r1
                matched = {"Factor": col_s, "Rep": "r1"}

        # Build rows
        ser = df[[geno_col, col]].copy()
        ser.columns = [geno_col, "Response"]
        ser[factor_name] = matched["Factor"]
        ser[rep_name] = matched["Rep"]
        melt_records.append(ser)

    if not melt_records:
        return pd.DataFrame(columns=[geno_col, factor_name, rep_name, "Response"])
    long_df = pd.concat(melt_records, axis=0, ignore_index=True)

    # Clear bad values and coerce numeric
    long_df["Response"] = pd.to_numeric(long_df["Response"], errors="coerce")
    long_df = long_df.dropna(subset=["Response"])

    # Tidy names
    long_df.rename(columns={geno_col: "Genotype"}, inplace=True)
    long_df["Genotype"] = long_df["Genotype"].astype(str).str.strip()
    long_df[factor_name] = long_df[factor_name].astype(str).str.strip()
    long_df[rep_name] = long_df[rep_name].astype(str).str.strip().str.lower()

    # Sort nicely
    long_df = long_df.sort_values(["Genotype", factor_name, rep_name]).reset_index(drop=True)
    return long_df

def run_anova(df, factor_col, type_choice="Type II"):
    # Model: Response ~ C(Genotype)*C(Factor)
    formula = f"Response ~ C(Genotype) * C({factor_col})"
    model = ols(formula, data=df).fit()
    typ = 2 if type_choice == "Type II" else 3
    anova_res = anova_lm(model, typ=typ)
    return model, anova_res

def run_mixedlm(df, factor_col):
    # Random intercept for Genotype; fixed for factor + interaction via random slopes not supported in simple call
    # We fit: Response ~ C(Factor) with group=Genotype, then add interaction by including Genotype as fixed * Factor in OLS as comparison
    # MixedLM with categorical needs one-hot; use patsy via smf.mixedlm
    import statsmodels.formula.api as smf
    try:
        md = smf.mixedlm(f"Response ~ C({factor_col})", df, groups=df["Genotype"])
        mdf = md.fit(method="lbfgs", reml=False)
        return mdf
    except Exception as e:
        return None

def tukey_table(df, group_col, response_col="Response"):
    try:
        tk = pairwise_tukeyhsd(endog=df[response_col], groups=df[group_col], alpha=0.05)
        d = pd.DataFrame(tk.summary(data=True)[1:], columns=tk.summary().data[0])
        return d
    except Exception:
        return None

def significance_stars(p):
    if pd.isna(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def design_check(df, geno_col="Genotype", factor_col="Treatment", rep_col="Replication"):
    # Balanced? each (Genotype, Factor) has same # of reps?
    cnt = df.groupby([geno_col, factor_col]).size()
    balanced = cnt.nunique() == 1
    missing_cells = cnt.unstack(factor_col).isna().sum().sum() if not cnt.empty else 0
    return balanced, int(missing_cells)

def summary_table(df, geno_col="Genotype", factor_col="Treatment"):
    tab = (
        df.groupby([geno_col, factor_col])["Response"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .rename(columns={"mean": "Mean", "std": "SD", "min": "Min", "max": "Max"})
        .reset_index()
    )
    tab["CV%"] = (tab["SD"] / tab["Mean"] * 100).round(2)
    tab["Rank_overall"] = tab["Mean"].rank(ascending=False, method="min").astype(int)
    return tab

def full_report(df_long, anova_df, type_choice, factor_name):
    g_levels = ", ".join(sorted(df_long["Genotype"].unique()))
    t_levels = ", ".join(sorted(df_long[factor_name].unique()))
    n_g = df_long["Genotype"].nunique()
    n_t = df_long[factor_name].nunique()
    best = df_long.groupby(["Genotype", factor_name])["Response"].mean()
    best_pair = best.idxmax() if not best.empty else ("-", "-")
    worst_pair = best.idxmin() if not best.empty else ("-", "-")
    rep_cnt = df_long.groupby(["Genotype", factor_name]).size()
    is_bal = rep_cnt.nunique() == 1 if not rep_cnt.empty else True

    text = f"""
    SPLIT PLOT / TWO-FACTOR ANALYSIS REPORT
    =======================================

    Design summary
    --------------
    • Main plot (assumed): Genotype  — levels: {n_g}
    • Subplot (assumed): {factor_name} — levels: {n_t}
    • Total observations: {len(df_long)}
    • Genotype levels: {g_levels}
    • {factor_name} levels: {t_levels}
    • Balanced (same # of reps per cell): {'Yes' if is_bal else 'No'}

    ANOVA ({type_choice})
    ---------------------
    {anova_df.round(6).to_string()}

    Performance snapshot
    --------------------
    • Best combo: {best_pair[0]} × {best_pair[1]} (mean={best.max():.3f})
    • Poorest combo: {worst_pair[0]} × {worst_pair[1]} (mean={best.min():.3f})
    • Range: {best.max() - best.min():.3f}

    Notes
    -----
    - If the interaction Genotype×{factor_name} is significant, interpret main effects with caution.
    - Mixed-effects (random intercept for Genotype) is available in the app as an auxiliary model.
    """
    return dedent(text)

def to_csv_download(df, name_prefix):
    buff = StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue(), f"{name_prefix}.csv"

def to_txt_download(text, name_prefix):
    bio = BytesIO(text.encode("utf-8"))
    return bio, f"{name_prefix}.txt"

def template_wide(n_geno=7, factor_levels=("d1","d2","d3"), reps=("r1","r2")):
    data = {"Genotypes": [f"g{i}" for i in range(1, n_geno+1)]}
    rng = np.random.default_rng(7)
    for d in factor_levels:
        for r in reps:
            data[f"{d}_{r}"] = np.round(rng.normal(loc=22, scale=2.5, size=n_geno), 2)
    return pd.DataFrame(data)

# --------------------------- SIDEBAR --------------------------------

with st.sidebar:
    st.header("📁 Data")
    up = st.file_uploader("Upload your file (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

    st.markdown("— or —")
    if st.button("📄 Download a template"):
        tmpl = template_wide()
        csv_str, fname = to_csv_download(tmpl, "splitplot_template")
        st.download_button("Download CSV Template", data=csv_str, file_name=fname, mime="text/csv")

    st.markdown("---")
    st.header("⚙️ Options")
    anova_type = st.radio("ANOVA type", ["Type II", "Type III"], index=0, help="Type II is common when the design is balanced; Type III is typical with interactions/unbalanced data.")
    show_mixed = st.checkbox("Also fit Mixed-Effects (random intercept for Genotype)", value=False)

# --------------------------- MAIN FLOW -------------------------------

if up is None:
    st.info("👆 Upload a .csv/.xlsx/.xls with genotypes as rows and columns like d1_r1, d1_r2, d2_r1, d2_r2, etc.")
    st.caption("Tip: If your Excel has two header rows (like d1/d2/d3 on top and r1/r2 below), this app will unify them automatically.")
else:
    df_raw, err = _read_any(up)
    if err:
        st.error(err)
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Shape: {df_raw.shape[0]} × {df_raw.shape[1]}")

    # Detect genotype col; let user confirm/override
    guess = detect_genotype_col(df_raw)
    with st.expander("🧩 Confirm Structure & Names", expanded=True):
        geno_col = st.selectbox("Column that contains Genotype names", options=list(df_raw.columns), index=list(df_raw.columns).index(guess))
        factor_label = st.text_input("Name for your subplot factor (e.g., Days, Dose, SowingDay)", value="Treatment")
        rep_label = st.text_input("Name for Replication column", value="Replication")
        apply_btn = st.button("🔄 Transform to Long Format", type="primary")

    if apply_btn:
        df_long = wide_to_long(df_raw, geno_col=geno_col, factor_name=factor_label, rep_name=rep_label)
        if df_long.empty:
            st.error("Could not transform the dataset. Please verify headers like d1_r1, d2_r2, etc.")
            st.stop()

        st.success("✅ Data transformed to long format")
        st.dataframe(df_long.head(12), use_container_width=True)
        st.caption(f"Long-format shape: {df_long.shape[0]} × {df_long.shape[1]}")

        # Persist
        st.session_state["df_long"] = df_long
        st.session_state["factor_name"] = factor_label
        st.session_state["rep_label"] = rep_label

# After transform
if "df_long" in st.session_state:
    df_long = st.session_state["df_long"].copy()
    factor_name = st.session_state["factor_name"]
    rep_label = st.session_state["rep_label"]

    # Quick design metrics
    nG, nT, nN = df_long["Genotype"].nunique(), df_long[factor_name].nunique(), len(df_long)
    bal, missing_cells = design_check(df_long, "Genotype", factor_name, rep_label)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metricbox">Genotypes<br><b>{nG}</b></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metricbox">{factor_name} levels<br><b>{nT}</b></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metricbox">Total obs.<br><b>{nN}</b></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metricbox">Balanced design?<br><b>{"Yes" if bal else "No"}</b></div>', unsafe_allow_html=True)
    if not bal:
        st.markdown('<div class="warn">Design is unbalanced (unequal replicates per cell). Type-III ANOVA is recommended.</div>', unsafe_allow_html=True)

    # -------------------- SUMMARY TABLE --------------------
    with st.expander("📊 Summary Statistics (per Genotype × Factor cell)", expanded=True):
        tab = summary_table(df_long, "Genotype", factor_name)
        st.dataframe(tab, use_container_width=True)
        best_row = tab.loc[tab["Mean"].idxmax()]
        st.markdown(
            f"""
            <div class="note">
              <b>Top cell:</b> {best_row['Genotype']} × {best_row[factor_name]} — Mean = {best_row['Mean']:.3f} ± {best_row['SD']:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------- ANOVA --------------------------
    st.header("🔬 ANOVA")
    model, anova_res = run_anova(df_long, factor_col=factor_name, type_choice=anova_type)

    # Decorate with significance column
    anova_show = anova_res.copy()
    if "PR(>F)" in anova_show.columns:
        anova_show["Signif."] = anova_show["PR(>F)"].apply(significance_stars)

    st.dataframe(anova_show.round(6), use_container_width=True)
    st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Mixed-effects (optional)
    if show_mixed:
        st.subheader("Mixed-Effects Model (Random Intercept for Genotype)")
        mdf = run_mixedlm(df_long, factor_col=factor_name)
        if mdf is None:
            st.markdown('<div class="danger">Mixed-Effects model failed to converge or could not be fit.</div>', unsafe_allow_html=True)
        else:
            st.text(mdf.summary())

    # ------------------- POST-HOC TESTS --------------------
    st.header("🎯 Post-hoc (Tukey HSD)")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Genotype main effect")
        tuk_g = tukey_table(df_long, "Genotype")
        if tuk_g is not None:
            st.dataframe(tuk_g, use_container_width=True)
        else:
            st.info("Not enough data for Tukey on Genotype.")

    with colB:
        st.subheader(f"{factor_name} main effect")
        tuk_t = tukey_table(df_long, factor_name)
        if tuk_t is not None:
            st.dataframe(tuk_t, use_container_width=True)
        else:
            st.info(f"Not enough data for Tukey on {factor_name}.")

    # Simple-effects Tukey if interaction significant
    if "PR(>F)" in anova_res.columns and f"C(Genotype):C({factor_name})" in anova_res.index:
        p_int = anova_res.loc[f"C(Genotype):C({factor_name})", "PR(>F)"]
        if p_int < 0.05:
            st.subheader("Simple-Effects Tukey (because interaction is significant)")
            with st.expander(f"Tukey for Genotype within each level of {factor_name}", expanded=False):
                tabs = st.tabs([f"{factor_name} = {lvl}" for lvl in df_long[factor_name].unique()])
                for tab_i, lvl in zip(tabs, df_long[factor_name].unique()):
                    with tab_i:
                        sub = df_long[df_long[factor_name] == lvl]
                        tk = tukey_table(sub, "Genotype")
                        if tk is not None:
                            st.dataframe(tk, use_container_width=True)
                        else:
                            st.info("Insufficient data.")
            with st.expander(f"Tukey for {factor_name} within each Genotype", expanded=False):
                tabs = st.tabs([f"Genotype = {g}" for g in df_long["Genotype"].unique()])
                for tab_i, g in zip(tabs, df_long["Genotype"].unique()):
                    with tab_i:
                        sub = df_long[df_long["Genotype"] == g]
                        tk = tukey_table(sub, factor_name)
                        if tk is not None:
                            st.dataframe(tk, use_container_width=True)
                        else:
                            st.info("Insufficient data.")

    # -------------------- VISUALIZATIONS -------------------
    st.header("📊 Visualizations")

    # Main effects: Genotype
    geno_means = df_long.groupby("Genotype")["Response"].agg(["mean", "std", "count"]).reset_index()
    geno_means["se"] = geno_means["std"] / np.sqrt(geno_means["count"])
    fig1 = px.bar(
        geno_means, x="Genotype", y="mean", error_y="se",
        title="Main Effect — Genotype", labels={"mean": "Mean Response"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Main effects: Factor
    fac_means = df_long.groupby(factor_name)["Response"].agg(["mean", "std", "count"]).reset_index()
    fac_means["se"] = fac_means["std"] / np.sqrt(fac_means["count"])
    fig2 = px.bar(
        fac_means, x=factor_name, y="mean", color=factor_name, error_y="se",
        title=f"Main Effect — {factor_name}", labels={"mean": "Mean Response"}
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Interaction means
    inter = df_long.groupby(["Genotype", factor_name])["Response"].mean().reset_index()
    fig3 = px.line(
        inter, x=factor_name, y="Response", color="Genotype", markers=True,
        title=f"Interaction: Genotype × {factor_name}"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    heat = inter.pivot(index="Genotype", columns=factor_name, values="Response")
    fig4 = px.imshow(
        heat.values, x=heat.columns, y=heat.index, aspect="auto",
        title=f"Heatmap of Mean Response (Genotype × {factor_name})", color_continuous_scale="viridis"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Boxplot
    fig5 = px.box(df_long, x="Genotype", y="Response", color=factor_name,
                  title=f"Response distribution by Genotype and {factor_name}")
    st.plotly_chart(fig5, use_container_width=True)

    # Violin
    fig6 = px.violin(df_long, x=factor_name, y="Response", color=factor_name, box=True,
                     title=f"Distribution shape by {factor_name}")
    fig6.update_layout(showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

    # ----------------------- DIAGNOSTICS -------------------
    with st.expander("🔍 Diagnostics", expanded=False):
        resid = model.resid
        fitted = model.fittedvalues

        # Residuals vs Fitted
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", name="Residuals", opacity=0.75))
        fig_r.add_hline(y=0, line_dash="dash", line_color="red")
        fig_r.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted", yaxis_title="Residuals", height=380)
        st.plotly_chart(fig_r, use_container_width=True)

        # QQ plot
        from scipy.stats import probplot
        qq = probplot(resid, dist="norm")
        qq_x, qq_y = qq[0][0], qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers", name="Sample", opacity=0.75))
        fig_q.add_trace(go.Scatter(x=qq_x, y=intercept + slope * qq_x, mode="lines", name="Theoretical", line=dict(dash="dash", color="red")))
        fig_q.update_layout(title="Normal Q-Q Plot", xaxis_title="Theoretical", yaxis_title="Sample", height=380)
        st.plotly_chart(fig_q, use_container_width=True)

        # Assumptions
        sh_w, sh_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
        if pd.notna(sh_p) and sh_p > 0.05:
            st.markdown('<div class="note">Normality OK (Shapiro p = {:.4f}).</div>'.format(sh_p), unsafe_allow_html=True)
        elif pd.notna(sh_p):
            st.markdown('<div class="warn">Residuals deviate from normality (Shapiro p = {:.4f}).</div>'.format(sh_p), unsafe_allow_html=True)

        groups = [g["Response"].values for _, g in df_long.groupby(["Genotype", factor_name])]
        if len(groups) > 1:
            lev_w, lev_p = stats.levene(*groups)
            if lev_p > 0.05:
                st.markdown('<div class="note">Homogeneity OK (Levene p = {:.4f}).</div>'.format(lev_p), unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn">Heteroscedasticity detected (Levene p = {:.4f}).</div>'.format(lev_p), unsafe_allow_html=True)

    # ------------------------ EXPORTS ----------------------
    st.header("💾 Export")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        csv_data, fname = to_csv_download(df_long, "long_format_data")
        st.download_button("⬇️ Long data (CSV)", data=csv_data, file_name=fname, mime="text/csv")

    with c2:
        csv_sum, fname2 = to_csv_download(tab, "summary_table")
        st.download_button("⬇️ Summary (CSV)", data=csv_sum, file_name=fname2, mime="text/csv")

    with c3:
        anov_csv, fname3 = to_csv_download(anova_show.reset_index().rename(columns={"index":"Term"}), "anova_table")
        st.download_button("⬇️ ANOVA (CSV)", data=anov_csv, file_name=fname3, mime="text/csv")

    with c4:
        if tuk_t is not None:
            tk_csv, fn4 = to_csv_download(tuk_t, "tukey_factor")
            st.download_button("⬇️ Tukey Factor (CSV)", data=tk_csv, file_name=fn4, mime="text/csv")
        else:
            st.write(" ")

    with c5:
        report_txt = full_report(df_long, anova_res, anova_type, factor_name)
        bio_report, rname = to_txt_download(report_txt, "splitplot_report")
        st.download_button("⬇️ Full Report (TXT)", data=bio_report, file_name=rname, mime="text/plain")

    st.caption("Made with ❤️ for robust cph experimentation.")
