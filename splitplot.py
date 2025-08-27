import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import matplotlib.pyplot as plt
import io
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Split Plot Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .interpretation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .significant {
        color: #28a745;
        font-weight: bold;
    }
    .not-significant {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<div class="main-header">🧪 Split Plot Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Developed by Bhavya</div>', unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data(uploaded_file):
    """Load data from various file formats"""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, XLS, or XLSX files.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_split_plot_structure(df):
    """Detect if data follows genotype-treatment structure"""
    try:
        # Check if first column contains genotypes
        first_col = df.columns[0].lower()
        if 'genotype' in first_col or 'variety' in first_col or df.iloc[:, 0].astype(str).str.startswith('g').any():
            return True
        return False
    except:
        return False

def auto_transform_wide_data(df):
    """Automatically transform wide format data based on column structure"""
    try:
        # Identify potential genotype/main plot column (first column with text data)
        main_plot_col = df.columns[0]
        
        # Get all other columns as potential treatments
        treatment_cols = df.columns[1:].tolist()
         
        # Create long format
        df_melted = df.melt(id_vars=[main_plot_col], 
                           value_vars=treatment_cols,
                           var_name='Treatment', 
                           value_name='Response')
        
        # Parse treatment structure to extract treatment and replication
        df_melted['Treatment_Clean'] = df_melted['Treatment'].str.extract(r'([a-zA-Z]+\d+)', expand=False)
        df_melted['Replication'] = df_melted['Treatment'].str.extract(r'([rR]\d+)', expand=False)
        
        # Fill missing replications
        df_melted['Replication'] = df_melted['Replication'].fillna('r1')
        df_melted['Treatment_Clean'] = df_melted['Treatment_Clean'].fillna(df_melted['Treatment'])
        
        # Rename columns
        df_melted = df_melted.rename(columns={
            main_plot_col: 'Genotype',
            'Treatment_Clean': 'Treatment'
        })
        
        # Convert response to numeric
        df_melted['Response'] = pd.to_numeric(df_melted['Response'], errors='coerce')
        df_melted = df_melted.dropna(subset=['Response'])
        
        return df_melted[['Genotype', 'Treatment', 'Replication', 'Response']]
        
    except Exception as e:
        st.error(f"Error in automatic transformation: {str(e)}")
        return None

def perform_split_plot_anova(df):
    """Perform Split Plot ANOVA Analysis"""
    try:
        # Create the model formula for split plot design
        # Genotype = Main plot factor, Treatment = Subplot factor
        formula = "Response ~ C(Genotype) * C(Treatment)"
        
        # Fit the model
        model = ols(formula, data=df).fit()
        
        # Perform ANOVA
        anova_results = anova_lm(model, typ=2)
        
        return model, anova_results
    except Exception as e:
        st.error(f"Error in ANOVA analysis: {str(e)}")
        return None, None

def perform_tukey_hsd(df, factor_col, response_col):
    """Perform Tukey HSD post-hoc test"""
    try:
        tukey_results = pairwise_tukeyhsd(endog=df[response_col], 
                                        groups=df[factor_col], 
                                        alpha=0.05)
        return tukey_results
    except Exception as e:
        st.error(f"Error in Tukey HSD test: {str(e)}")
        return None

def create_comprehensive_plots(df):
    """Create comprehensive visualization suite"""
    plots = {}
    
    try:
        # 1. Main effects plot - Genotype
        genotype_means = df.groupby('Genotype')['Response'].agg(['mean', 'std', 'count']).reset_index()
        genotype_means['se'] = genotype_means['std'] / np.sqrt(genotype_means['count'])
        
        fig1 = px.bar(genotype_means, x='Genotype', y='mean', 
                      error_y='se',
                      title='Main Effect: Genotype Performance',
                      labels={'mean': 'Mean Response', 'Genotype': 'Genotype'})
        fig1.update_layout(showlegend=False, height=400)
        plots['genotype_main'] = fig1
        
        # 2. Main effects plot - Treatment
        treatment_means = df.groupby('Treatment')['Response'].agg(['mean', 'std', 'count']).reset_index()
        treatment_means['se'] = treatment_means['std'] / np.sqrt(treatment_means['count'])
        
        fig2 = px.bar(treatment_means, x='Treatment', y='mean', 
                      error_y='se',
                      title='Main Effect: Treatment Performance',
                      labels={'mean': 'Mean Response', 'Treatment': 'Treatment'},
                      color='Treatment')
        fig2.update_layout(showlegend=False, height=400)
        plots['treatment_main'] = fig2
        
        # 3. Interaction plot
        interaction_means = df.groupby(['Genotype', 'Treatment'])['Response'].mean().reset_index()
        fig3 = px.line(interaction_means, x='Treatment', y='Response', color='Genotype',
                      title='Genotype × Treatment Interaction',
                      markers=True)
        fig3.update_layout(height=500)
        plots['interaction'] = fig3
        
        # 4. Heatmap of means
        pivot_data = df.groupby(['Genotype', 'Treatment'])['Response'].mean().unstack()
        fig4 = px.imshow(pivot_data.values, 
                        x=pivot_data.columns, 
                        y=pivot_data.index,
                        aspect='auto',
                        title='Response Heatmap: Genotype × Treatment',
                        color_continuous_scale='viridis')
        fig4.update_xaxes(title='Treatment')
        fig4.update_yaxes(title='Genotype')
        plots['heatmap'] = fig4
        
        # 5. Box plots for variability assessment
        fig5 = px.box(df, x='Genotype', y='Response', color='Treatment',
                     title='Response Distribution by Genotype and Treatment')
        fig5.update_layout(height=500)
        plots['boxplot'] = fig5
        
        # 6. Violin plots for distribution shape
        fig6 = px.violin(df, x='Treatment', y='Response', color='Treatment',
                        title='Response Distribution Shape by Treatment')
        fig6.update_layout(showlegend=False, height=400)
        plots['violin'] = fig6
        
        return plots
        
    except Exception as e:
        st.error(f"Error creating plots: {str(e)}")
        return {}

def interpret_results(anova_results, tukey_genotype, tukey_treatment, df):
    """Provide comprehensive interpretation of results"""
    interpretations = []
    
    try:
        # ANOVA interpretation
        alpha = 0.05
        
        if 'PR(>F)' in anova_results.columns:
            # Genotype effect
            genotype_p = anova_results.loc['C(Genotype)', 'PR(>F)']
            if genotype_p < alpha:
                interpretations.append(f"🔍 **Genotype Effect**: Highly significant (p = {genotype_p:.6f})")
                interpretations.append("   → Different genotypes show significantly different responses")
            else:
                interpretations.append(f"🔍 **Genotype Effect**: Not significant (p = {genotype_p:.6f})")
                interpretations.append("   → No significant differences between genotypes")
            
            # Treatment effect
            treatment_p = anova_results.loc['C(Treatment)', 'PR(>F)']
            if treatment_p < alpha:
                interpretations.append(f"🔍 **Treatment Effect**: Significant (p = {treatment_p:.6f})")
                interpretations.append("   → Different treatments show significantly different effects")
            else:
                interpretations.append(f"🔍 **Treatment Effect**: Not significant (p = {treatment_p:.6f})")
                interpretations.append("   → No significant differences between treatments")
            
            # Interaction effect
            if 'C(Genotype):C(Treatment)' in anova_results.index:
                interaction_p = anova_results.loc['C(Genotype):C(Treatment)', 'PR(>F)']
                if interaction_p < alpha:
                    interpretations.append(f"🔍 **Interaction Effect**: Significant (p = {interaction_p:.6f})")
                    interpretations.append("   → The effect of treatment depends on genotype (or vice versa)")
                    interpretations.append("   → Simple effects analysis recommended")
                else:
                    interpretations.append(f"🔍 **Interaction Effect**: Not significant (p = {interaction_p:.6f})")
                    interpretations.append("   → Treatment effects are consistent across genotypes")
        
        # Best performing combinations
        best_combo = df.groupby(['Genotype', 'Treatment'])['Response'].mean().idxmax()
        best_mean = df.groupby(['Genotype', 'Treatment'])['Response'].mean().max()
        interpretations.append(f"🏆 **Best Combination**: {best_combo[0]} with {best_combo[1]} (Mean: {best_mean:.2f})")
        
        # Worst performing combination
        worst_combo = df.groupby(['Genotype', 'Treatment'])['Response'].mean().idxmin()
        worst_mean = df.groupby(['Genotype', 'Treatment'])['Response'].mean().min()
        interpretations.append(f"⚠️ **Poorest Combination**: {worst_combo[0]} with {worst_combo[1]} (Mean: {worst_mean:.2f})")
        
        # Performance range
        overall_range = best_mean - worst_mean
        interpretations.append(f"📊 **Performance Range**: {overall_range:.2f} units difference between best and worst")
        
        return interpretations
        
    except Exception as e:
        st.error(f"Error in interpretation: {str(e)}")
        return ["Error generating interpretations"]

def create_summary_table(df):
    """Create comprehensive summary statistics"""
    try:
        # Overall summary
        overall_stats = df.groupby(['Genotype', 'Treatment'])['Response'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        # Add coefficient of variation
        overall_stats['CV%'] = ((overall_stats['std'] / overall_stats['mean']) * 100).round(2)
        
        # Add ranking
        overall_stats['Rank'] = overall_stats['mean'].rank(ascending=False, method='min')
        
        return overall_stats.reset_index()
        
    except Exception as e:
        st.error(f"Error creating summary table: {str(e)}")
        return None

# Main application
def main():
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV, XLS, or XLSX files with genotype-treatment structure"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Load data
            df_raw = load_data(uploaded_file)
            
            if df_raw is not None:
                st.markdown("### 📊 Raw Data Preview")
                st.dataframe(df_raw.head(10), use_container_width=True)
                st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
                
                # Auto-transform data
                if st.button("🔄 Transform to Long Format", type="primary"):
                    with st.spinner("Transforming data..."):
                        # Try genotype-specific transformation first
                        if detect_split_plot_structure(df_raw):
                            df_transformed = transform_genotype_data(df_raw)
                        else:
                            df_transformed = auto_transform_wide_data(df_raw)
                        
                        if df_transformed is not None and len(df_transformed) > 0:
                            st.success("✅ Data transformed successfully!")
                            st.session_state['df_analysis'] = df_transformed
                            
                            st.markdown("### 📈 Transformed Data Preview")
                            st.dataframe(df_transformed.head(), use_container_width=True)
                            st.markdown(f"**Transformed Shape:** {df_transformed.shape[0]} rows × {df_transformed.shape[1]} columns")
                            
                            # Data validation
                            n_genotypes = df_transformed['Genotype'].nunique()
                            n_treatments = df_transformed['Treatment'].nunique()
                            n_observations = len(df_transformed)
                            
                            st.markdown("### 📋 Design Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Genotypes", n_genotypes)
                            with col2:
                                st.metric("Treatments", n_treatments)
                            with col3:
                                st.metric("Total Obs.", n_observations)
                        else:
                            st.error("❌ Failed to transform data. Please check your data format.")

    # Main content area
    if uploaded_file is None:
        st.info("👆 Please upload a data file to begin analysis")
        
        # Show expected data format
        st.markdown("### 📋 Expected Data Format")
        
        st.markdown("**Your data should have genotypes as rows and treatments as columns:**")
        
        # Create example data matching the user's format
        example_data = pd.DataFrame({
            'Genotypes': [f'g{i}' for i in range(1, 8)],
            'd1_r1': [23.5, 21.2, 25.8, 22.1, 24.5, 20.9, 23.2],
            'd1_r2': [24.1, 22.0, 26.2, 22.8, 25.1, 21.5, 23.8],
            'd2_r1': [21.3, 19.8, 23.5, 20.2, 22.1, 18.9, 21.0],
            'd2_r2': [22.0, 20.5, 24.1, 21.0, 22.8, 19.6, 21.7],
            'd3_r1': [19.8, 18.2, 21.5, 18.5, 20.2, 17.1, 19.3],
            'd3_r2': [20.5, 19.0, 22.2, 19.2, 20.9, 17.8, 20.0]
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        st.markdown("""
        **Where:**
        - **Genotypes**: Your genotype/variety names (g1, g2, etc.)
        - **d1, d2, d3**: Different treatments/doses
        - **r1, r2**: Replications for each treatment
        """)
    
    elif 'df_analysis' in st.session_state:
        df_analysis = st.session_state['df_analysis']
        
        # Main analysis
        st.header("📈 Split Plot Analysis Results")
        
        # Summary statistics
        with st.expander("📊 Comprehensive Summary Statistics", expanded=True):
            summary_table = create_summary_table(df_analysis)
            if summary_table is not None:
                st.dataframe(summary_table, use_container_width=True)
                
                # Quick insights
                best_performer = summary_table.loc[summary_table['mean'].idxmax()]
                st.markdown(f"""
                <div class="interpretation-box">
                <h4>🏆 Top Performer</h4>
                <strong>{best_performer['Genotype']} with {best_performer['Treatment']}</strong><br>
                Mean Response: {best_performer['mean']:.3f} ± {best_performer['std']:.3f}<br>
                Rank: #{int(best_performer['Rank'])} out of {len(summary_table)}
                </div>
                """, unsafe_allow_html=True)
        
        # ANOVA Analysis
        with st.expander("🔬 Split Plot ANOVA", expanded=True):
            model, anova_results = perform_split_plot_anova(df_analysis)
            
            if anova_results is not None:
                st.subheader("ANOVA Table")
                
                # Format ANOVA results
                anova_formatted = anova_results.copy()
                anova_formatted['Significance'] = anova_formatted['PR(>F)'].apply(
                    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns' if pd.notna(x) else ''
                )
                
                # Round for display
                for col in ['sum_sq', 'mean_sq', 'F', 'PR(>F)']:
                    if col in anova_formatted.columns:
                        anova_formatted[col] = anova_formatted[col].round(6)
                
                st.dataframe(anova_formatted, use_container_width=True)
                
                # Legend
                st.markdown("**Significance levels:** *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # Post-hoc Analysis
        with st.expander("🎯 Post-hoc Analysis (Tukey HSD)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Genotype Comparisons")
                tukey_genotype = perform_tukey_hsd(df_analysis, 'Genotype', 'Response')
                if tukey_genotype is not None:
                    # Convert to DataFrame for better display
                    tukey_df = pd.DataFrame(tukey_genotype.summary().data[1:], 
                                          columns=tukey_genotype.summary().data[0])
                    st.dataframe(tukey_df, use_container_width=True)
            
            with col2:
                st.subheader("Treatment Comparisons")
                tukey_treatment = perform_tukey_hsd(df_analysis, 'Treatment', 'Response')
                if tukey_treatment is not None:
                    tukey_df2 = pd.DataFrame(tukey_treatment.summary().data[1:], 
                                           columns=tukey_treatment.summary().data[0])
                    st.dataframe(tukey_df2, use_container_width=True)
        
        # Comprehensive Visualizations
        st.header("📊 Comprehensive Visualizations")
        
        plots = create_comprehensive_plots(df_analysis)
        
        if plots:
            # Main effects
            col1, col2 = st.columns(2)
            with col1:
                if 'genotype_main' in plots:
                    st.plotly_chart(plots['genotype_main'], use_container_width=True)
            with col2:
                if 'treatment_main' in plots:
                    st.plotly_chart(plots['treatment_main'], use_container_width=True)
            
            # Interaction and heatmap
            col1, col2 = st.columns(2)
            with col1:
                if 'interaction' in plots:
                    st.plotly_chart(plots['interaction'], use_container_width=True)
            with col2:
                if 'heatmap' in plots:
                    st.plotly_chart(plots['heatmap'], use_container_width=True)
            
            # Distribution analysis
            if 'boxplot' in plots:
                st.plotly_chart(plots['boxplot'], use_container_width=True)
            
            if 'violin' in plots:
                st.plotly_chart(plots['violin'], use_container_width=True)
        
        # Interpretation
        with st.expander("🔍 Statistical Interpretation & Recommendations", expanded=True):
            if anova_results is not None:
                interpretations = interpret_results(
                    anova_results, 
                    tukey_genotype if 'tukey_genotype' in locals() else None,
                    tukey_treatment if 'tukey_treatment' in locals() else None,
                    df_analysis
                )
                
                for interpretation in interpretations:
                    st.markdown(interpretation)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                # Check for significant effects and provide recommendations
                if 'PR(>F)' in anova_results.columns:
                    genotype_p = anova_results.loc['C(Genotype)', 'PR(>F)']
                    treatment_p = anova_results.loc['C(Treatment)', 'PR(>F)']
                    
                    if genotype_p < 0.05:
                        best_genotype = df_analysis.groupby('Genotype')['Response'].mean().idxmax()
                        st.success(f"✅ **Genotype Selection**: Choose **{best_genotype}** for best performance")
                    
                    if treatment_p < 0.05:
                        best_treatment = df_analysis.groupby('Treatment')['Response'].mean().idxmax()
                        st.success(f"✅ **Treatment Selection**: Use **{best_treatment}** for optimal results")
                    
                    # Interaction recommendations
                    if 'C(Genotype):C(Treatment)' in anova_results.index:
                        interaction_p = anova_results.loc['C(Genotype):C(Treatment)', 'PR(>F)']
                        if interaction_p < 0.05:
                            st.warning("⚠️ **Important**: Significant interaction detected! Treatment effectiveness varies by genotype. Consider genotype-specific treatment recommendations.")
        
        # Model Diagnostics
        with st.expander("🔍 Model Diagnostics", expanded=False):
            if model is not None:
                residuals = model.resid
                fitted = model.fittedvalues
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residuals vs Fitted
                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers',
                                                 name='Residuals', opacity=0.7))
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_resid.update_layout(title="Residuals vs Fitted Values",
                                          xaxis_title="Fitted Values",
                                          yaxis_title="Residuals",
                                          height=400)
                    st.plotly_chart(fig_resid, use_container_width=True)
                
                with col2:
                    # Q-Q plot
                    from scipy.stats import probplot
                    qq = probplot(residuals, dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                              name='Sample Quantiles', opacity=0.7))
                    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                                              mode='lines', name='Theoretical Line',
                                              line=dict(color='red', dash='dash')))
                    fig_qq.update_layout(title="Normal Q-Q Plot",
                                       xaxis_title="Theoretical Quantiles",
                                       yaxis_title="Sample Quantiles",
                                       height=400)
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Model assumptions check
                st.markdown("### Model Assumptions Assessment")
                
                # Shapiro-Wilk test for normality
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                if shapiro_p > 0.05:
                    st.success(f"✅ **Normality**: Residuals appear normally distributed (Shapiro-Wilk p = {shapiro_p:.4f})")
                else:
                    st.warning(f"⚠️ **Normality**: Residuals may not be normally distributed (Shapiro-Wilk p = {shapiro_p:.4f})")
                
                # Levene's test for homogeneity of variance
                groups = [group['Response'].values for name, group in df_analysis.groupby(['Genotype', 'Treatment'])]
                if len(groups) > 1:
                    levene_stat, levene_p = stats.levene(*groups)
                    if levene_p > 0.05:
                        st.success(f"✅ **Homogeneity**: Equal variances assumption satisfied (Levene's p = {levene_p:.4f})")
                    else:
                        st.warning(f"⚠️ **Homogeneity**: Unequal variances detected (Levene's p = {levene_p:.4f})")
        
        # Export functionality
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Summary Data", type="primary"):
                summary_csv = create_summary_table(df_analysis).to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=summary_csv,
                    file_name=f"split_plot_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Generate Full Report", type="primary"):
                # Create comprehensive report
                report_content = generate_comprehensive_report(df_analysis, anova_results, summary_table)
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_content,
                    file_name=f"split_plot_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

def generate_comprehensive_report(df, anova_results, summary_table):
    """Generate a comprehensive analysis report"""
    try:
        # Calculate key statistics
        n_genotypes = df['Genotype'].nunique()
        n_treatments = df['Treatment'].nunique()
        n_observations = len(df)
        overall_mean = df['Response'].mean()
        overall_std = df['Response'].std()
        
        # Best and worst performers
        best_combo = df.groupby(['Genotype', 'Treatment'])['Response'].mean().idxmax()
        best_mean = df.groupby(['Genotype', 'Treatment'])['Response'].mean().max()
        worst_combo = df.groupby(['Genotype', 'Treatment'])['Response'].mean().idxmin()
        worst_mean = df.groupby(['Genotype', 'Treatment'])['Response'].mean().min()
        
        # Generate report
        report = f"""
# SPLIT PLOT ANALYSIS REPORT
## Developed by Bhavya
### Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

===============================================================================

## EXPERIMENTAL DESIGN SUMMARY
- Design Type: Split Plot Design
- Main Plot Factor: Genotype ({n_genotypes} levels)
- Subplot Factor: Treatment ({n_treatments} levels)
- Total Observations: {n_observations}
- Overall Mean Response: {overall_mean:.3f} ± {overall_std:.3f}

## GENOTYPE LEVELS
{', '.join(sorted(df['Genotype'].unique()))}

## TREATMENT LEVELS
{', '.join(sorted(df['Treatment'].unique()))}

===============================================================================

## ANALYSIS OF VARIANCE (ANOVA) RESULTS

{anova_results.round(6).to_string()}

## SIGNIFICANCE INTERPRETATION
"""
        
        # Add significance interpretation
        if 'PR(>F)' in anova_results.columns:
            alpha = 0.05
            
            # Genotype effect
            genotype_p = anova_results.loc['C(Genotype)', 'PR(>F)']
            genotype_status = "SIGNIFICANT" if genotype_p < alpha else "NOT SIGNIFICANT"
            report += f"- Genotype Effect: {genotype_status} (p = {genotype_p:.6f})\n"
            
            # Treatment effect
            treatment_p = anova_results.loc['C(Treatment)', 'PR(>F)']
            treatment_status = "SIGNIFICANT" if treatment_p < alpha else "NOT SIGNIFICANT"
            report += f"- Treatment Effect: {treatment_status} (p = {treatment_p:.6f})\n"
            
            # Interaction effect
            if 'C(Genotype):C(Treatment)' in anova_results.index:
                interaction_p = anova_results.loc['C(Genotype):C(Treatment)', 'PR(>F)']
                interaction_status = "SIGNIFICANT" if interaction_p < alpha else "NOT SIGNIFICANT"
                report += f"- Genotype × Treatment Interaction: {interaction_status} (p = {interaction_p:.6f})\n"
        
        report += f"""
===============================================================================

## PERFORMANCE SUMMARY

### BEST PERFORMING COMBINATION
- Genotype: {best_combo[0]}
- Treatment: {best_combo[1]}
- Mean Response: {best_mean:.3f}

### POOREST PERFORMING COMBINATION
- Genotype: {worst_combo[0]}
- Treatment: {worst_combo[1]}
- Mean Response: {worst_mean:.3f}

### PERFORMANCE RANGE
- Difference between best and worst: {(best_mean - worst_mean):.3f} units
- Relative improvement: {((best_mean - worst_mean) / worst_mean * 100):.1f}%

===============================================================================

## DETAILED SUMMARY STATISTICS

"""
        
        # Add summary table
        if summary_table is not None:
            report += summary_table.to_string(index=False)
        
        report += f"""

===============================================================================

## RECOMMENDATIONS

### GENOTYPE SELECTION"""
        
        # Genotype recommendations
        genotype_means = df.groupby('Genotype')['Response'].mean().sort_values(ascending=False)
        top_3_genotypes = genotype_means.head(3)
        
        report += f"""
Top 3 performing genotypes:
"""
        for i, (genotype, mean_val) in enumerate(top_3_genotypes.items(), 1):
            report += f"{i}. {genotype}: {mean_val:.3f}\n"
        
        report += f"""
### TREATMENT SELECTION"""
        
        # Treatment recommendations
        treatment_means = df.groupby('Treatment')['Response'].mean().sort_values(ascending=False)
        
        report += f"""
Treatment ranking (best to worst):
"""
        for i, (treatment, mean_val) in enumerate(treatment_means.items(), 1):
            report += f"{i}. {treatment}: {mean_val:.3f}\n"
        
        # Interaction-based recommendations
        if 'PR(>F)' in anova_results.columns and 'C(Genotype):C(Treatment)' in anova_results.index:
            interaction_p = anova_results.loc['C(Genotype):C(Treatment)', 'PR(>F)']
            if interaction_p < 0.05:
                report += f"""
### INTERACTION CONSIDERATIONS
⚠️  IMPORTANT: Significant genotype × treatment interaction detected!
   This means treatment effectiveness varies by genotype.
   
   Genotype-specific recommendations:
"""
                # Get best treatment for each genotype
                for genotype in df['Genotype'].unique():
                    genotype_data = df[df['Genotype'] == genotype]
                    best_treatment = genotype_data.groupby('Treatment')['Response'].mean().idxmax()
                    best_response = genotype_data.groupby('Treatment')['Response'].mean().max()
                    report += f"   - {genotype}: Use {best_treatment} (Mean: {best_response:.3f})\n"
        
        report += f"""

===============================================================================

## STATISTICAL ASSUMPTIONS

### MODEL VALIDATION
- Split plot ANOVA assumes:
  1. Normality of residuals
  2. Homogeneity of variances
  3. Independence of observations
  4. Additivity of effects

### DATA QUALITY METRICS
- Coefficient of Variation: {(overall_std/overall_mean*100):.2f}%
- Data completeness: {(len(df)/n_observations*100):.1f}%

===============================================================================

## METHODOLOGY

### EXPERIMENTAL DESIGN
This analysis uses Split Plot Design where:
- Main plots (whole plots) contain genotypes
- Subplots contain treatments within each main plot
- This design is efficient for studying factor interactions

### STATISTICAL METHODS
- Analysis of Variance (ANOVA) for testing main effects and interactions
- Tukey's Honestly Significant Difference (HSD) for multiple comparisons
- Split plot model accounts for different error terms for main plot and subplot factors

### SOFTWARE
- Analysis performed using Python with scipy.stats and statsmodels
- Visualization created with plotly and matplotlib
- Report generated automatically by Split Plot Analyzer

===============================================================================

END OF REPORT
        """
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    main()
