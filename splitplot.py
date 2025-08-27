# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
import textwrap
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy import stats

# ---------- Helper functions ----------

def read_input_file(uploaded_file):
    """Read csv/xlsx/xls into pandas DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

def safe_drop_mean(df):
    """Remove 'Mean' column if present (case-insensitive)."""
    cols = df.columns.str.lower()
    if "mean" in cols:
        idx = int(np.where(cols == "mean")[0][0])
        df2 = df.drop(df.columns[idx], axis=1)
        return df2
    return df.copy()

def pivot_to_long(df):
    """Expect a 'Genotype' column and treatment columns like d1_r1, d1_r2, etc."""
    if "Genotype" not in df.columns:
        raise ValueError("Input must contain a 'Genotype' column")
    id_col = "Genotype"
    value_cols = [c for c in df.columns if c != id_col]
    long = df.melt(id_vars=[id_col], value_vars=value_cols, var_name="Treatment", value_name="Value")
    # split Treatment into Date and Rep
    if long['Treatment'].str.contains('_').all():
        long[['Date','Rep']] = long['Treatment'].str.split('_', expand=True)
    else:
        # try splitting by '.' or '-' then underscore fallback
        parts = long['Treatment'].str.split('[._-]', expand=True)
        if parts.shape[1] >= 2:
            long['Date'] = parts[0]
            long['Rep'] = parts[1]
        else:
            raise ValueError("Treatment columns must be named like 'd1_r1' (Date_Rep).")
    long = long.drop(columns=["Treatment"])
    long['Date'] = pd.Categorical(long['Date'], categories=sorted(long['Date'].unique(), key=lambda x: x))
    long['Rep'] = pd.Categorical(long['Rep'])
    long['Genotype'] = pd.Categorical(long['Genotype'])
    long = long.dropna(subset=['Value'])
    # coerce numeric
    long['Value'] = pd.to_numeric(long['Value'], errors='coerce')
    long = long.dropna(subset=['Value'])
    return long

def compute_lm_anova(df_long):
    """OLS model used for emmeans-like estimates and ANOVA tables."""
    formula = "Value ~ C(Date)*C(Genotype) + C(Rep)"
    model = ols(formula, data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA
    return model, anova_table

def fit_mixedlm(df_long):
    """
    Fit a Mixed Linear Model approximating split-plot:
    Random intercept for Rep (block) and variance component for Rep:Date (whole-plot).
    """
    try:
        vc = {"RepDate": "0 + C(Rep):C(Date)"}
        md = MixedLM.from_formula("Value ~ C(Date)*C(Genotype)", groups="Rep", vc_formula=vc, data=df_long)
        mdf = md.fit(reml=True)
        return mdf
    except Exception as e:
        try:
            md = MixedLM.from_formula("Value ~ C(Date)*C(Genotype)", groups="Rep", data=df_long)
            mdf = md.fit(reml=True)
            return mdf
        except Exception as e2:
            raise RuntimeError(f"MixedLM failed: {e} | fallback failed: {e2}")

def tukey_tests(df_long, factor):
    """Run Tukey HSD using statsmodels MultiComparison."""
    mc = MultiComparison(df_long["Value"], df_long[factor])
    res = mc.tukeyhsd()
    return res, mc

def greedy_cld_from_tukey(mc: MultiComparison, tukey_res):
    """
    Approximate CLD (compact letter display) from tukeyhsd results.
    """
    # Defensive check: if no groups, return an empty DataFrame
    if not hasattr(mc, 'groupsunique') or len(mc.groupsunique) == 0:
        return pd.DataFrame(columns=["level", "cld", "mean"])
        
    groups = list(mc.groupsunique)
    table = tukey_res._results_table.data[1:]
    not_diff = {g:{} for g in groups}
    for r in table:
        g1, g2, meandiff, p_adj, lower, upper, reject = r
        not_diff[g1][g2] = (not reject)
        not_diff[g2][g1] = (not reject)
    for g in groups:
        not_diff[g][g] = True
    means = df_long.groupby(mc.groups)["Value"].mean().reindex(groups)
    sorted_groups = list(means.sort_values(ascending=False).index)
    letters = {}
    current_letter = ord('a')
    for g in sorted_groups:
        if g in letters:
            continue
        letter = chr(current_letter)
        letters[g] = letters.get(g, "") + letter
        for h in sorted_groups:
            if h in letters:
                continue
            assigned = [gg for gg,lab in letters.items() if letter in lab]
            ok = True
            for gg in assigned:
                if h not in not_diff or gg not in not_diff[h] or not not_diff[h][gg]:
                    ok = False
                    break
            if ok:
                letters[h] = letters.get(h, "") + letter
        current_letter += 1
    cld_df = pd.DataFrame({"level": sorted_groups, "cld": [letters.get(g, "") for g in sorted_groups], "mean": means.loc[sorted_groups].values})
    return cld_df

def download_button_df(df, filename, button_label):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(button_label, data=csv, file_name=filename, mime='text/csv')

def fig_to_bytes(fig, fmt="png", dpi=300):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    return buf

# ---------- New Interpretations Function ----------
def add_interpretations(anova_table, geno_cld, date_cld):
    st.markdown("### 📊 Interpretations of Results")
    st.markdown("""
    Here's a simple guide to help you understand the statistical output and plots.
    """)
    st.markdown("#### ANOVA Table")
    if anova_table is not None:
        p_date = anova_table.loc['C(Date)', 'PR(>F)']
        p_geno = anova_table.loc['C(Genotype)', 'PR(>F)']
        p_interaction = anova_table.loc['C(Date):C(Genotype)', 'PR(>F)']
        st.markdown(f"""
        -   **Date effect (P={p_date:.4f})**: The p-value for Date tells us if there's a significant difference between the overall averages of the dates. **Interpretation**: {'There is a significant difference between dates.' if p_date < 0.05 else 'There is no significant difference between dates.'}
        -   **Genotype effect (P={p_geno:.4f})**: This p-value indicates if the genotypes, on average, are different from each other. **Interpretation**: {'There is a significant difference among genotypes.' if p_geno < 0.05 else 'There is no significant difference among genotypes.'}
        -   **Interaction effect (P={p_interaction:.4f})**: The interaction p-value tells us if the performance of genotypes changes from date to date. **Interpretation**: {'There is a significant interaction.' if p_interaction < 0.05 else 'There is no significant interaction.'} A significant interaction means you should look at the results per date to understand the full picture.
        """)
    else:
        st.info("ANOVA table not available for interpretation.")
    st.markdown("---")
    st.markdown("#### 🏆 Top Performers")
    if geno_cld is not None and not geno_cld.empty:
        best_geno_row = geno_cld.iloc[0]
        best_geno = best_geno_row['level']
        best_mean = best_geno_row['mean']
        best_cld = best_geno_row['cld']
        st.markdown(f"""
        <div style="background-color: #e6f7ff; padding: 15px; border-radius: 5px; border: 1px solid #91d5ff;">
            <h3 style="color: #0050b3; margin-top: 0;">Top Performing Genotype</h3>
            <p style="font-size: 1.2em; font-weight: bold; margin: 0;">{best_geno} with an average value of {best_mean:.2f}.</p>
            <p style="margin: 0; font-style: italic;">It's in the letter group '{best_cld}' meaning it is statistically similar to other genotypes with the same letter.</p>
        </div>
        """, unsafe_allow_html=True)
    if date_cld is not None and not date_cld.empty:
        best_date_row = date_cld.iloc[0]
        best_date = best_date_row['level']
        best_mean = best_date_row['mean']
        best_cld = best_date_row['cld']
        st.markdown(f"""
        <div style="background-color: #f6ffed; padding: 15px; border-radius: 5px; border: 1px solid #b7eb8f; margin-top: 10px;">
            <h3 style="color: #237804; margin-top: 0;">Top Performing Date</h3>
            <p style="font-size: 1.2em; font-weight: bold; margin: 0;">{best_date} with an average value of {best_mean:.2f}.</p>
            <p style="margin: 0; font-style: italic;">This date performed statistically similarly to others with the same letter '{best_cld}'.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🖼️ Understanding the Plots")
    st.markdown("""
    -   **Interaction Plot**: If lines are parallel, there's no interaction. If they cross or have different slopes, it means there's a significant interaction.
    -   **Faceted Boxplots**: These help visualize variability. A tall boxplot means high variability.
    -   **Genotype & Date Means Plots**: The letters (CLD) on top of the bars indicate statistical similarity. Groups with at least one letter in common are **not** statistically different.
    """)
    st.markdown("---")


# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Split-Plot Analysis App")
st.title("🌱 Split-Plot Analysis App")
st.markdown("""
This app performs split-plot analysis for data with columns named like `d1_r1`, `d1_r2`, etc.
- **Main-plot** = Date (d1, d2, d3)
- **Sub-plot** = Genotype (g1..g19)
- **Replications** = r1, r2
Upload your file (.csv, .xlsx, .xls) or use the example dataset.
""")

# Example dataset
example_csv = """Genotype,d1_r1,d1_r2,d2_r1,d2_r2,d3_r1,d3_r2,Mean
g1,36.86,37.97,38.47,37.47,40.87,42.40,39.01
g2,39.58,41.30,44.03,42.21,45.86,41.58,42.43
g3,38.02,34.57,33.96,36.35,37.69,37.06,36.28
g4,39.91,37.84,35.67,33.78,37.82,43.35,38.06
g5,35.94,36.35,39.12,38.03,41.23,42.22,38.82
g6,49.45,48.56,47.49,49.25,49.90,49.21,48.98
g7,38.76,44.30,45.74,44.69,48.21,48.10,44.97
g8,43.28,40.37,34.71,38.14,47.52,41.39,40.90
g9,41.08,41.79,44.44,38.67,47.38,48.45,43.64
g10,39.93,40.56,39.67,44.64,42.80,45.18,42.13
g11,37.13,34.82,36.48,36.81,38.36,35.67,36.55
g12,31.50,32.13,33.40,32.29,37.22,39.80,34.39
g13,35.53,35.00,34.19,34.36,41.98,36.33,36.23
g14,39.51,38.78,40.34,36.04,46.08,44.16,40.82
g15,35.08,36.98,40.17,39.93,47.16,46.95,41.04
g16,31.56,28.94,31.49,31.30,37.92,36.67,32.98
g17,42.14,43.49,44.14,43.74,47.47,48.33,44.88
g18,37.38,31.96,42.46,43.79,39.74,39.24,39.10
g19,36.62,36.98,36.18,36.86,40.56,38.92,37.69
"""

uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv","xlsx","xls"])
use_example = st.checkbox("Use example dataset (ignore upload)", value=True)

if uploaded_file is None and not use_example:
    st.info("Upload a file or check 'Use example dataset'.")
    st.stop()

if use_example:
    data = pd.read_csv(StringIO(example_csv))
else:
    try:
        data = read_input_file(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

st.subheader("Raw Data Preview")
st.dataframe(data.head())

try:
    data2 = safe_drop_mean(data)
    df_long = pivot_to_long(data2)
except Exception as e:
    st.error(f"Data transformation failed: {e}")
    st.stop()

st.subheader("Data in Long Format")
st.dataframe(df_long.head())
st.markdown(f"- **Observations**: **{len(df_long)}** | **Dates**: {list(df_long['Date'].cat.categories)} | **Genotypes**: {len(df_long['Genotype'].cat.categories)} | **Replications**: {list(df_long['Rep'].cat.categories)}")

# Fit models
st.markdown("---")
st.markdown("## 🔬 Statistical Analysis")
st.info("We'll run a Mixed Linear Model to approximate the split-plot design and an OLS model to get the ANOVA table and Tukey tests.")
col1, col2 = st.columns(2)

with col1:
    st.write("### Mixed Linear Model")
    run_mixed = st.button("Fit MixedLM")
    if run_mixed:
        try:
            mdf = fit_mixedlm(df_long)
            st.success("MixedLM fitted successfully!")
            st.text(str(mdf.summary()))
        except Exception as e:
            st.error(f"MixedLM failed: {e}")

with col2:
    st.write("### ANOVA Table (Type II)")
    run_ols = st.button("Fit OLS & Get ANOVA")
    if run_ols:
        try:
            lm_model, anova_table = compute_lm_anova(df_long)
            st.dataframe(anova_table.style.format("{:.4f}"))
        except Exception as e:
            st.error(f"OLS/ANOVA failed: {e}")
            lm_model = anova_table = None

try:
    lm_model, anova_table = compute_lm_anova(df_long)
except Exception as e:
    lm_model = anova_table = None

# Post-hoc Tukey
st.markdown("---")
st.markdown("## 📊 Post-hoc Tests & Visualizations")
try:
    tukey_date_res, mc_date = tukey_tests(df_long, "Date")
    tukey_geno_res, mc_geno = tukey_tests(df_long, "Genotype")

    st.write("### Tukey HSD — Date")
    st.dataframe(pd.DataFrame(tukey_date_res.summary().data[1:], columns=tukey_date_res.summary().data[0]))
    st.write("### Tukey HSD — Genotype (first 200 comparisons shown)")
    st.dataframe(pd.DataFrame(tukey_geno_res.summary().data[1:], columns=tukey_geno_res.summary().data[0]))
    
    date_cld = greedy_cld_from_tukey(mc_date, tukey_date_res)
    geno_cld = greedy_cld_from_tukey(mc_geno, tukey_geno_res)
    st.write("### Compact Letter Display (approx.) — Date")
    st.dataframe(date_cld)
    st.write("### Compact Letter Display (approx.) — Genotype (top 100 shown)")
    st.dataframe(geno_cld.head(100))
except Exception as e:
    st.error(f"Tukey/CLD failed: {e}")
    tukey_date_res = tukey_geno_res = None
    date_cld = geno_cld = None

add_interpretations(anova_table, geno_cld, date_cld)

# HSD within each Date
st.markdown("---")
st.markdown("## HSD within Each Date (Genotype comparisons per Date)")
try:
    hsd_results = []
    for d, sub in df_long.groupby("Date"):
        if sub['Genotype'].nunique() > 1:
            mc = MultiComparison(sub['Value'], sub['Genotype'])
            res = mc.tukeyhsd()
            rows = res._results_table.data[1:]
            table_df = pd.DataFrame(rows, columns=res._results_table.data[0])
            table_df['Date'] = d
            hsd_results.append(table_df)
    if hsd_results:
        hsd_df = pd.concat(hsd_results, ignore_index=True)
        st.dataframe(hsd_df.head(200))
        download_button_df(hsd_df, "hsd_by_date.csv", "Download HSD-by-Date CSV")
    else:
        st.write("No HSD results (not enough groups).")
except Exception as e:
    st.warning(f"Per-Date HSD failed: {e}")

# ... (rest of the code remains the same until the plotting section) ...

# ---------- Publication-quality plots ----------
st.markdown("---")
st.markdown("## 📈 Publication-Quality Plots")

# Interaction plot: one line per genotype
fig1, ax1 = plt.subplots(figsize=(12,6))
all_genos = df_long['Genotype'].cat.categories.tolist()
geno_means_overall = df_long.groupby("Genotype")['Value'].mean()
best_geno = geno_means_overall.idxmax()

for name, grp in df_long.groupby("Genotype"):
    x = [list(df_long['Date'].cat.categories).index(v) for v in grp['Date']]
    vals = grp.groupby('Date')['Value'].mean().reindex(df_long['Date'].cat.categories).values
    
    line_style = 'solid'
    line_width = 1.5
    label = name
    alpha = 0.6
    
    # Highlight the best genotype
    if name == best_geno:
        color = 'red'  # A distinctive color for the best genotype
        line_width = 3
        alpha = 1.0
        label = f"{name} (Best)"
    else:
        color = 'grey'
    
    ax1.plot(range(len(vals)), vals, marker='o', linewidth=line_width, alpha=alpha, label=label, color=color)

# Create a custom legend for all genotypes
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, title='Genotypes')

ax1.set_xticks(range(len(df_long['Date'].cat.categories)))
ax1.set_xticklabels(list(df_long['Date'].cat.categories))
ax1.set_xlabel("Date (main plot)")
ax1.set_ylabel("Value")
ax1.set_title("Interaction plot: Date × Genotype (line per genotype)")
ax1.grid(True, linestyle=':', linewidth=0.4)
st.pyplot(fig1)
buf = fig_to_bytes(fig1, fmt="png")
st.download_button("Download interaction plot (PNG)", data=buf, file_name="interaction_plot.png", mime="image/png")

# ... (rest of the code for other plots) ...

# Genotype means with SE and CLD
geno_means = df_long.groupby("Genotype")['Value'].agg(['mean', 'sem']).reset_index().rename(columns={'sem':'se'})

# This is the crucial fix
if geno_means.empty:
    st.warning("No data available to plot genotype means.")
else:
    if geno_cld is not None and not geno_cld.empty:
        geno_plot_df = geno_means.merge(geno_cld.rename(columns={"level":"Genotype"}), how='left', left_on="Genotype", right_on="Genotype")
    else:
        geno_plot_df = geno_means.copy()
        geno_plot_df['cld'] = ""

    # Sort by mean
    geno_plot_df = geno_plot_df.sort_values('mean', ascending=False).reset_index(drop=True)

    fig3, ax3 = plt.subplots(figsize=(14,6))
    ax3.bar(range(len(geno_plot_df)), geno_plot_df['mean'])
    ax3.errorbar(range(len(geno_plot_df)), geno_plot_df['mean'], yerr=geno_plot_df['se'].fillna(0), fmt='none', capsize=3)
    ax3.set_xticks(range(len(geno_plot_df)))
    ax3.set_xticklabels(geno_plot_df['Genotype'], rotation=90, fontsize=8)
    ax3.set_ylabel("Estimated mean (EMM-like)")
    ax3.set_title("Genotype estimated means with SE and CLD (approx.)")
    for i, r in geno_plot_df.iterrows():
        lbl = r.get('cld', '')
        ax3.text(i, r['mean'] + (r['se'] if not np.isnan(r['se']) else 0.1), str(lbl), ha='center', va='bottom', fontsize=7)
    ax3.grid(axis='y', linestyle=':', linewidth=0.4)
    st.pyplot(fig3)
    buf3 = fig_to_bytes(fig3, fmt="png")
    st.download_button("Download genotype means plot (PNG)", data=buf3, file_name="genotype_means.png", mime="image/png")
    download_button_df(geno_plot_df, "genotype_means_cld.csv", "Download genotype means + CLD (CSV)")

st.markdown("---")
st.markdown("### 📝 Notes & Limitations")
st.markdown(textwrap.dedent("""
-   The app attempts to reflect split-plot structure by fitting a Mixed Linear Model (random Rep and Rep:Date vc).
-   For pairwise comparisons and compact letters we use Tukey HSD on raw observations with a greedy CLD algorithm (approximation). R's `emmeans` + `multcompView` or `agricolae` will give results that are sometimes slightly different.
-   If you need exact equivalence with R's split-plot ANOVA + emmeans, you may need to use dedicated R packages.
-   Plots are saved/downloadable at high resolution (PNG). For journal submission you can export SVG from matplotlib by changing `fmt='svg'` in the download helper.
-   For highly unbalanced designs or singularities, MixedLM may fail; OLS is used as a robust complement for EMM-like estimation and Tukey.
"""))

st.success("✅ Analysis complete. Use download buttons above to save your results & plots!")
