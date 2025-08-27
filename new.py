import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config with enhanced styling
st.set_page_config(
    page_title="🌱 Split-Plot Analyser",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57 0%, #228B22 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🌱 Advanced Split-Plot Analyser</h1>
    <h3>Comprehensive Statistical Analysis for Agricultural Research</h3>
    <p>Developed with 💚 by Bhavya, M.S.</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
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

@st.cache_data
def pivot_to_long(df):
    """Expect a 'Genotype' column and treatment columns like d1_r1, d1_r2, etc."""
    if "Genotype" not in df.columns:
        raise ValueError("Input must contain a 'Genotype' column")
    
    id_col = "Genotype"
    value_cols = [c for c in df.columns if c != id_col]
    long = df.melt(id_vars=[id_col], value_vars=value_cols, var_name="Treatment", value_name="Value")
    
    if long['Treatment'].str.contains('_').all():
        long[['Date','Rep']] = long['Treatment'].str.split('_', expand=True)
    else:
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
    long['Value'] = pd.to_numeric(long['Value'], errors='coerce')
    long = long.dropna(subset=['Value'])
    return long

def compute_lm_anova(df_long):
    """OLS model used for emmeans-like estimates and ANOVA tables."""
    formula = "Value ~ C(Date)*C(Genotype) + C(Rep)"
    model = ols(formula, data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table

def fit_mixedlm(df_long):
    """Fit a Mixed Linear Model approximating split-plot."""
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
    """Approximate CLD from tukeyhsd results."""
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
    
    # Get the correct DataFrame for means calculation
    df_long = st.session_state.get('df_long', None)
    if df_long is not None:
        means = df_long.groupby(mc.groups)["Value"].mean().reindex(groups)
    else:
        # Fallback if session state doesn't exist
        means = pd.Series(index=groups, dtype=float).fillna(0)
    
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
    
    cld_df = pd.DataFrame({
        "level": sorted_groups, 
        "cld": [letters.get(g, "") for g in sorted_groups], 
        "mean": means.loc[sorted_groups].values
    })
    return cld_df

def create_interactive_plots(df_long):
    """Create interactive Plotly visualizations."""
    plots = {}
    
    # 1. Interactive Interaction Plot
    fig_int = go.Figure()
    
    # Calculate means for each genotype-date combination
    means_df = df_long.groupby(['Genotype', 'Date'])['Value'].mean().reset_index()
    overall_means = df_long.groupby('Genotype')['Value'].mean()
    best_geno = overall_means.idxmax()
    
    colors = px.colors.qualitative.Set3
    for i, genotype in enumerate(df_long['Genotype'].cat.categories):
        geno_data = means_df[means_df['Genotype'] == genotype]
        
        line_width = 4 if genotype == best_geno else 2
        line_color = 'red' if genotype == best_geno else colors[i % len(colors)]
        
        fig_int.add_trace(go.Scatter(
            x=geno_data['Date'],
            y=geno_data['Value'],
            mode='lines+markers',
            name=f"{genotype} {'(Best)' if genotype == best_geno else ''}",
            line=dict(width=line_width, color=line_color),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Mean Value: %{y:.2f}<extra></extra>'
        ))
    
    fig_int.update_layout(
        title="Interactive Genotype × Date Performance",
        xaxis_title="Date",
        yaxis_title="Mean Value",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    plots['interaction'] = fig_int
    
    # 2. Stability Analysis Scatter Plot
    try:
        lm_model, _ = compute_lm_anova(df_long)
        residuals = pd.DataFrame(lm_model.resid, index=lm_model.resid.index, columns=['Residual'])
        residuals = df_long.assign(Residual=residuals['Residual'].values)
        
        wricke_eco = residuals.groupby('Genotype')['Residual'].apply(lambda x: (x**2).sum())
        geno_means = df_long.groupby("Genotype")['Value'].mean()
        
        stability_df = pd.DataFrame({
            'Genotype': geno_means.index,
            'Mean': geno_means.values,
            'Ecovalence': wricke_eco.values
        })
        
        fig_stab = px.scatter(
            stability_df, 
            x='Ecovalence', 
            y='Mean',
            text='Genotype',
            title="Genotype Stability Analysis (Wricke's Ecovalence)",
            labels={
                'Ecovalence': "Wricke's Ecovalence (Lower = More Stable)",
                'Mean': 'Overall Mean Performance'
            },
            template='plotly_white',
            height=500
        )
        
        fig_stab.update_traces(textposition="top center")
        fig_stab.add_annotation(
            x=stability_df['Ecovalence'].min(),
            y=stability_df['Mean'].max(),
            text="Ideal Zone:<br>High Mean + Low Ecovalence",
            showarrow=True,
            arrowhead=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="green",
            borderwidth=1
        )
        plots['stability'] = fig_stab
    except Exception as e:
        st.warning(f"Could not create stability plot: {e}")
    
    # 3. Heatmap with annotations
    heatmap_df = df_long.pivot_table(index='Genotype', columns='Date', values='Value', aggfunc='mean')
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='Viridis',
        text=np.round(heatmap_df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>Genotype: %{y}</b><br>Date: %{x}<br>Mean Value: %{z:.2f}<extra></extra>'
    ))
    
    fig_heat.update_layout(
        title="Performance Heatmap: Genotype × Date",
        xaxis_title="Date",
        yaxis_title="Genotype",
        template='plotly_white',
        height=max(400, len(heatmap_df) * 25)
    )
    plots['heatmap'] = fig_heat
    
    return plots

def create_summary_metrics(df_long, anova_table):
    """Create summary metrics cards."""
    total_obs = len(df_long)
    n_genotypes = df_long['Genotype'].nunique()
    n_dates = df_long['Date'].nunique()
    n_reps = df_long['Rep'].nunique()
    
    overall_mean = df_long['Value'].mean()
    overall_std = df_long['Value'].std()
    cv = (overall_std / overall_mean) * 100 if overall_mean != 0 else 0
    
    # Best performing genotype
    geno_means = df_long.groupby('Genotype')['Value'].mean()
    best_genotype = geno_means.idxmax()
    best_performance = geno_means.max()
    
    # ANOVA significance
    anova_summary = {}
    if anova_table is not None:
        for factor in ['C(Date)', 'C(Genotype)', 'C(Date):C(Genotype)']:
            if factor in anova_table.index:
                p_val = anova_table.loc[factor, 'PR(>F)']
                anova_summary[factor] = "Significant" if p_val < 0.05 else "Not Significant"
    
    return {
        'total_obs': total_obs,
        'n_genotypes': n_genotypes,
        'n_dates': n_dates,
        'n_reps': n_reps,
        'overall_mean': overall_mean,
        'cv': cv,
        'best_genotype': best_genotype,
        'best_performance': best_performance,
        'anova_summary': anova_summary
    }

# Sidebar configuration
with st.sidebar:
    st.markdown("## 📊 Analysis Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=["csv","xlsx","xls"])
    use_example = st.checkbox("Use example dataset", value=False)
    
    # Analysis options
    st.markdown("### Analysis Options")
    show_mixed_model = st.checkbox("Run Mixed Linear Model", value=True)
    show_tukey = st.checkbox("Perform Tukey HSD Tests", value=True)
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)

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

# Load data
if uploaded_file is None and not use_example:
    st.info("👆 Please upload a file or use the example dataset from the sidebar.")
    st.stop()

try:
    if use_example:
        data = pd.read_csv(StringIO(example_csv))
        st.success("✅ Example dataset loaded successfully!")
    else:
        data = read_input_file(uploaded_file)
        st.success("✅ File uploaded and loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

# Data preview
with st.expander("📋 Raw Data Preview", expanded=False):
    st.dataframe(data, use_container_width=True)

# Data transformation
try:
    data2 = safe_drop_mean(data)
    df_long = pivot_to_long(data2)
    
    # Store in session state for global access
    st.session_state['df_long'] = df_long
    
except Exception as e:
    st.error(f"❌ Data transformation failed: {e}")
    st.stop()

# Display summary metrics
try:
    lm_model, anova_table = compute_lm_anova(df_long)
    metrics = create_summary_metrics(df_long, anova_table)
    
    st.markdown("## 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", metrics['total_obs'])
    with col2:
        st.metric("Genotypes", metrics['n_genotypes'])
    with col3:
        st.metric("Dates", metrics['n_dates'])
    with col4:
        st.metric("Replications", metrics['n_reps'])
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Overall Mean", f"{metrics['overall_mean']:.2f}")
    with col6:
        st.metric("CV (%)", f"{metrics['cv']:.1f}%")
    with col7:
        st.metric("Best Genotype", metrics['best_genotype'])
    with col8:
        st.metric("Best Performance", f"{metrics['best_performance']:.2f}")
    
except Exception as e:
    st.error(f"❌ Error calculating summary metrics: {e}")
    lm_model = anova_table = None

# Statistical Analysis
st.markdown("## 🔬 Statistical Analysis")

tabs = st.tabs(["ANOVA Results", "Mixed Model", "Post-hoc Tests"])

with tabs[0]:
    if anova_table is not None:
        st.markdown("### 📊 ANOVA Table (Type II)")
        
        # Format ANOVA table with better styling
        anova_styled = anova_table.style.format({
            'sum_sq': '{:.4f}',
            'df': '{:.0f}',
            'F': '{:.4f}',
            'PR(>F)': '{:.4f}'
        }).background_gradient(subset=['F'], cmap='RdYlGn_r')
        
        st.dataframe(anova_styled, use_container_width=True)
        
        # Interpretation
        st.markdown("#### 🎯 ANOVA Interpretation")
        
        p_date = anova_table.loc['C(Date)', 'PR(>F)']
        p_geno = anova_table.loc['C(Genotype)', 'PR(>F)']
        p_interaction = anova_table.loc['C(Date):C(Genotype)', 'PR(>F)']
        
        interpretation_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;">
            <h4>Statistical Significance Summary</h4>
            <ul>
                <li><b>Date Effect (p = {p_date:.4f}):</b> {'🟢 Significant - Dates differ significantly' if p_date < 0.05 else '🔴 Not Significant - No date differences'}</li>
                <li><b>Genotype Effect (p = {p_geno:.4f}):</b> {'🟢 Significant - Genotypes differ significantly' if p_geno < 0.05 else '🔴 Not Significant - No genotype differences'}</li>
                <li><b>Interaction Effect (p = {p_interaction:.4f}):</b> {'🟢 Significant - Genotype performance varies by date' if p_interaction < 0.05 else '🔴 Not Significant - Consistent genotype ranking'}</li>
            </ul>
        </div>
        """
        st.markdown(interpretation_html, unsafe_allow_html=True)
    else:
        st.error("ANOVA analysis failed")

with tabs[1]:
    if show_mixed_model:
        try:
            with st.spinner("Fitting Mixed Linear Model..."):
                mdf = fit_mixedlm(df_long)
            
            st.success("✅ Mixed Linear Model fitted successfully!")
            
            # Display model summary in a more readable format
            summary_str = str(mdf.summary())
            st.text_area("Model Summary", summary_str, height=400)
            
        except Exception as e:
            st.error(f"❌ Mixed Linear Model failed: {e}")

with tabs[2]:
    if show_tukey:
        try:
            with st.spinner("Performing Tukey HSD tests..."):
                tukey_date_res, mc_date = tukey_tests(df_long, "Date")
                tukey_geno_res, mc_geno = tukey_tests(df_long, "Genotype")
                
                date_cld = greedy_cld_from_tukey(mc_date, tukey_date_res)
                geno_cld = greedy_cld_from_tukey(mc_geno, tukey_geno_res)
            
            st.success("✅ Tukey HSD tests completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Date Comparisons")
                date_tukey_df = pd.DataFrame(tukey_date_res.summary().data[1:], 
                                           columns=tukey_date_res.summary().data[0])
                st.dataframe(date_tukey_df, use_container_width=True)
                
                if not date_cld.empty:
                    st.markdown("**Compact Letter Display - Dates**")
                    st.dataframe(date_cld, use_container_width=True)
            
            with col2:
                st.markdown("#### Genotype Comparisons (Sample)")
                geno_tukey_df = pd.DataFrame(tukey_geno_res.summary().data[1:], 
                                           columns=tukey_geno_res.summary().data[0])
                st.dataframe(geno_tukey_df.head(20), use_container_width=True)
                st.info(f"Showing first 20 of {len(geno_tukey_df)} comparisons")
                
                if not geno_cld.empty:
                    st.markdown("**Compact Letter Display - Genotypes (Top 15)**")
                    st.dataframe(geno_cld.head(15), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Tukey HSD tests failed: {e}")
            date_cld = geno_cld = pd.DataFrame()

# Interactive Visualizations
st.markdown("## 📊 Interactive Visualizations")

try:
    plots = create_interactive_plots(df_long)
    
    viz_tabs = st.tabs(["Interaction Plot", "Stability Analysis", "Performance Heatmap"])
    
    with viz_tabs[0]:
        if 'interaction' in plots:
            st.plotly_chart(plots['interaction'], use_container_width=True)
            st.markdown("""
            **📖 Interpretation:** This plot shows how each genotype performs across different dates. 
            Parallel lines indicate no interaction (consistent ranking), while crossing lines suggest 
            genotype-by-date interactions. The best overall genotype is highlighted in red.
            """)
    
    with viz_tabs[1]:
        if 'stability' in plots:
            st.plotly_chart(plots['stability'], use_container_width=True)
            st.markdown("""
            **📖 Interpretation:** This stability plot helps identify genotypes that are both high-yielding 
            and stable. Genotypes in the top-left quadrant (high mean, low ecovalence) are most desirable 
            as they combine good performance with consistency across dates.
            """)
    
    with viz_tabs[2]:
        if 'heatmap' in plots:
            st.plotly_chart(plots['heatmap'], use_container_width=True)
            st.markdown("""
            **📖 Interpretation:** The heatmap visualizes performance patterns across genotypes and dates. 
            Darker colors indicate higher values. Look for genotypes (rows) that consistently show 
            high values across dates (columns).
            """)

except Exception as e:
    st.error(f"❌ Error creating interactive plots: {e}")

# Enhanced Static Plots
st.markdown("## 📈 Publication-Ready Plots")

plot_tabs = st.tabs(["Means Comparison", "Distribution Analysis", "PCA Analysis"])

with plot_tabs[0]:
    if 'geno_cld' in locals() and not geno_cld.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Genotype Performance Rankings")
            
            # Fixed the column name issue
            geno_means = df_long.groupby("Genotype")['Value'].agg(['mean', 'sem']).reset_index()
            geno_means.columns = ['Genotype', 'mean_value', 'se']  # Rename columns explicitly
            
            if not geno_cld.empty:
                geno_plot_df = geno_means.merge(
                    geno_cld.rename(columns={"level":"Genotype", "mean": "cld_mean"}), 
                    how='left', 
                    on="Genotype"
                )
                # Use the calculated mean_value column for sorting
                geno_plot_df = geno_plot_df.sort_values('mean_value', ascending=False).reset_index(drop=True)
            else:
                geno_plot_df = geno_means.copy()
                geno_plot_df['cld'] = ""
                geno_plot_df = geno_plot_df.sort_values('mean_value', ascending=False).reset_index(drop=True)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(range(len(geno_plot_df)), geno_plot_df['mean_value'], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(geno_plot_df))))
            
            # Add error bars
            ax.errorbar(range(len(geno_plot_df)), geno_plot_df['mean_value'], 
                       yerr=geno_plot_df['se'].fillna(0), fmt='none', capsize=3, color='black')
            
            # Add CLD labels
            for i, r in geno_plot_df.iterrows():
                cld_label = r.get('cld', '')
                ax.text(i, r['mean_value'] + (r['se'] if not pd.isna(r['se']) else 0.1), 
                       str(cld_label), ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_xticks(range(len(geno_plot_df)))
            ax.set_xticklabels(geno_plot_df['Genotype'], rotation=45, ha='right')
            ax.set_ylabel("Mean Value ± SE")
            ax.set_title("Genotype Performance with Statistical Groups")
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download options
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button("📥 Download Genotype Plot", data=buf, 
                             file_name="genotype_performance.png", mime="image/png")
        
        with col2:
            if 'date_cld' in locals() and not date_cld.empty:
                st.markdown("### 📅 Date Performance Rankings")
                
                date_means = df_long.groupby("Date")['Value'].agg(['mean','sem']).reset_index()
                date_means.columns = ['Date', 'mean_value', 'se']
                
                if not date_cld.empty:
                    date_plot_df = date_means.merge(
                        date_cld.rename(columns={"level":"Date", "mean": "cld_mean"}), 
                        how='left', on="Date"
                    )
                else:
                    date_plot_df = date_means.copy()
                    date_plot_df['cld'] = ""
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                bars = ax2.bar(range(len(date_plot_df)), date_plot_df['mean_value'], 
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1'], width=0.6)
                
                ax2.errorbar(range(len(date_plot_df)), date_plot_df['mean_value'], 
                           yerr=date_plot_df['se'].fillna(0), fmt='none', capsize=5, color='black')
                
                for i, r in date_plot_df.iterrows():
                    cld_label = r.get('cld', '')
                    ax2.text(i, r['mean_value'] + (r['se'] if not pd.isna(r['se']) else 0.1), 
                           str(cld_label), ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                ax2.set_xticks(range(len(date_plot_df)))
                ax2.set_xticklabels(date_plot_df['Date'], fontsize=12)
                ax2.set_ylabel("Mean Value ± SE")
                ax2.set_title("Date Performance with Statistical Groups")
                ax2.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                
                buf2 = BytesIO()
                fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
                buf2.seek(0)
                st.download_button("📥 Download Date Plot", data=buf2, 
                                 file_name="date_performance.png", mime="image/png")

with plot_tabs[1]:
    st.markdown("### 📊 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot by genotype
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        # Select top performing genotypes for cleaner visualization
        top_genos = df_long.groupby('Genotype')['Value'].mean().nlargest(10).index
        df_subset = df_long[df_long['Genotype'].isin(top_genos)]
        
        sns.boxplot(data=df_subset, x='Genotype', y='Value', ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_title("Distribution of Values by Top 10 Genotypes")
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.markdown("""
        **📖 Interpretation:** Box plots show the distribution and variability of each genotype. 
        The box shows the interquartile range, the line inside is the median, and whiskers 
        extend to show the range. Outliers appear as individual points.
        """)
    
    with col2:
        # Violin plot by date
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df_long, x='Date', y='Value', ax=ax4)
        ax4.set_title("Distribution Density by Date")
        ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.markdown("""
        **📖 Interpretation:** Violin plots show the probability density of data at different values. 
        Wider sections indicate higher probability of values in that range. This helps visualize 
        how the distribution shape changes across dates.
        """)

with plot_tabs[2]:
    st.markdown("### 🔍 Principal Component Analysis")
    
    try:
        # PCA Analysis
        df_pivot = df_long.pivot_table(index='Genotype', columns='Date', values='Value', aggfunc='mean').fillna(0)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_pivot)
        
        pca = PCA(n_components=min(3, df_pivot.shape[1]))
        pca_result = pca.fit_transform(scaled_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # PCA Biplot
            fig5, ax5 = plt.subplots(figsize=(12, 10))
            
            # Plot genotypes as points
            scatter = ax5.scatter(pca_result[:, 0], pca_result[:, 1], 
                                s=100, alpha=0.7, c=range(len(df_pivot.index)), cmap='viridis')
            
            # Add genotype labels
            for i, genotype in enumerate(df_pivot.index):
                ax5.annotate(genotype, (pca_result[i, 0], pca_result[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add feature vectors (dates)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, feature in enumerate(df_pivot.columns):
                ax5.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
                         color='red', alpha=0.8, head_width=0.1, head_length=0.1)
                ax5.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature, 
                        color='red', ha='center', va='center', fontsize=12, fontweight='bold')
            
            ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
            ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
            ax5.set_title('PCA Biplot: Genotype-Date Interaction Analysis')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig5)
            
        with col2:
            # PCA Explained Variance
            st.markdown("#### Explained Variance")
            
            explained_var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
                'Cumulative (%)': np.cumsum(pca.explained_variance_ratio_) * 100
            })
            
            st.dataframe(explained_var_df, use_container_width=True)
            
            # Scree plot
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            ax6.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_ * 100, 'bo-')
            ax6.set_xlabel('Principal Component')
            ax6.set_ylabel('Variance Explained (%)')
            ax6.set_title('Scree Plot')
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig6)
        
        st.markdown("""
        **📖 Interpretation:** The PCA biplot reveals patterns in genotype-by-date interactions. 
        - **Points (genotypes)** close together have similar performance patterns
        - **Arrows (dates)** show the direction of maximum variation for each date
        - **Genotypes** positioned in the direction of a date arrow performed well in that condition
        - **Length of arrows** indicates the importance of that date in explaining variance
        """)
        
    except Exception as e:
        st.error(f"❌ PCA analysis failed: {e}")

# Enhanced Results Summary
st.markdown("## 📋 Results Summary & Recommendations")

if 'geno_cld' in locals() and not geno_cld.empty and 'date_cld' in locals() and not date_cld.empty:
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Performers")
        
        # Top 5 genotypes
        top_5_genos = geno_cld.head(5)
        
        results_html = "<div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;'>"
        results_html += "<h4>🌟 Top 5 Genotypes</h4><ul>"
        
        for _, row in top_5_genos.iterrows():
            results_html += f"<li><b>{row['level']}</b>: {row['mean']:.2f} (Group: {row['cld']})</li>"
        
        results_html += "</ul></div>"
        st.markdown(results_html, unsafe_allow_html=True)
        
        # Best date
        best_date = date_cld.iloc[0]
        date_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;'>
            <h4>📅 Optimal Date</h4>
            <p><b>{best_date['level']}</b> with mean performance of <b>{best_date['mean']:.2f}</b></p>
        </div>
        """
        st.markdown(date_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Key Insights")
        
        # Calculate CV for stability assessment
        geno_cv = df_long.groupby('Genotype').agg({
            'Value': lambda x: (x.std() / x.mean()) * 100
        }).rename(columns={'Value': 'CV'})
        
        most_stable = geno_cv.idxmin()[0]
        
        insights = [
            f"📈 **Best Overall Genotype**: {geno_cld.iloc[0]['level']} ({geno_cld.iloc[0]['mean']:.2f})",
            f"📊 **Most Stable Genotype**: {most_stable} (lowest CV)",
            f"🎯 **Optimal Date**: {best_date['level']} shows highest mean performance",
            f"🔄 **Interaction Effect**: {'Significant - adapt strategies by date' if anova_table is not None and anova_table.loc['C(Date):C(Genotype)', 'PR(>F)'] < 0.05 else 'Not significant - consistent ranking'}"
        ]
        
        insights_html = "<div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #007bff;'>"
        for insight in insights:
            insights_html += f"<p style='margin: 0.5rem 0;'>{insight}</p>"
        insights_html += "</div>"
        
        st.markdown(insights_html, unsafe_allow_html=True)

# Download Section
st.markdown("## 📥 Download Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Prepare comprehensive results CSV
    if 'geno_cld' in locals() and not geno_cld.empty:
        geno_results = df_long.groupby('Genotype').agg({
            'Value': ['mean', 'std', 'count']
        }).round(4)
        geno_results.columns = ['Mean', 'StdDev', 'Count']
        geno_results = geno_results.merge(geno_cld.set_index('level')[['cld']], 
                                        left_index=True, right_index=True, how='left')
        
        csv_buffer = StringIO()
        geno_results.to_csv(csv_buffer)
        
        st.download_button(
            "📊 Download Genotype Results",
            data=csv_buffer.getvalue(),
            file_name="genotype_analysis_results.csv",
            mime="text/csv"
        )

with col2:
    if anova_table is not None:
        anova_buffer = StringIO()
        anova_table.to_csv(anova_buffer)
        
        st.download_button(
            "📈 Download ANOVA Table",
            data=anova_buffer.getvalue(),
            file_name="anova_results.csv",
            mime="text/csv"
        )

with col3:
    # Download long format data
    long_buffer = StringIO()
    df_long.to_csv(long_buffer, index=False)
    
    st.download_button(
        "📋 Download Processed Data",
        data=long_buffer.getvalue(),
        file_name="processed_long_format_data.csv",
        mime="text/csv"
    )

# Footer with methodology
st.markdown("---")
st.markdown("## 📚 Methodology & Notes")

with st.expander("🔬 Statistical Methods Used", expanded=False):
    st.markdown("""
    ### Statistical Analysis Pipeline
    
    1. **Data Structure**: Split-plot design with Date as main plot and Genotype as sub-plot
    2. **ANOVA**: Type II Sum of Squares for unbalanced designs
    3. **Mixed Linear Model**: Random effects for Rep and Rep:Date to account for split-plot structure
    4. **Post-hoc Tests**: Tukey HSD for pairwise comparisons with family-wise error control
    5. **Stability Analysis**: Wricke's Ecovalence for genotype stability assessment
    6. **Multivariate Analysis**: PCA for pattern recognition in genotype-by-environment interactions
    
    ### Key Assumptions
    - Normality of residuals
    - Homogeneity of variance
    - Independence of observations within plots
    - Proper randomization in split-plot design
    
    ### Interpretation Guidelines
    - **P-values < 0.05**: Statistically significant effects
    - **Compact Letter Display**: Groups with common letters are not significantly different
    - **Stability**: Lower Wricke's Ecovalence indicates higher stability
    - **PCA**: First two components typically explain most variation in G×E interactions
    """)

with st.expander("⚠️ Limitations & Recommendations", expanded=False):
    st.markdown("""
    ### Current Limitations
    - CLD approximation may differ slightly from R's `emmeans` + `multcompView`
    - Mixed model implementation is simplified compared to specialized split-plot packages
    - Assumes balanced or near-balanced designs for optimal performance
    
    ### For Publication-Quality Analysis
    - Verify assumptions through residual analysis
    - Consider using R with `agricolae`, `emmeans`, or `nlme` packages for complex designs
    - Validate stability measures with additional methods (AMMI, GGE biplot)
    - Include environmental characterization for better interpretation
    
    ### Best Practices
    - Always examine interaction effects before interpreting main effects
    - Use multiple stability measures for robust conclusions
    - Consider practical significance alongside statistical significance
    - Validate top performers across multiple seasons/locations
    """)

# Success message
st.markdown("""
<div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1rem; border-radius: 10px; color: white; margin: 2rem 0; text-align: center;'>
    <h3>🎉 Analysis Complete!</h3>
    <p>Your split-plot analysis has been successfully completed. Use the download buttons above to save your results and visualizations.</p>
    <p><em>For questions or suggestions, contact the developer through the institution.</em></p>
</div>
""", unsafe_allow_html=True)
