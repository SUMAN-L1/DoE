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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
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

def wide_to_long_transform(df, main_plot_col, subplot_col, response_cols):
    """Transform wide format data to long format for split plot analysis"""
    try:
        # Identify ID columns (non-response columns)
        id_cols = [col for col in df.columns if col not in response_cols]
        
        # Melt the dataframe
        df_long = pd.melt(df, 
                         id_vars=id_cols, 
                         value_vars=response_cols,
                         var_name='subplot_treatment', 
                         value_name='response')
        
        # Rename columns for clarity
        if main_plot_col in df_long.columns:
            df_long = df_long.rename(columns={main_plot_col: 'main_plot'})
        
        return df_long
    except Exception as e:
        st.error(f"Error in data transformation: {str(e)}")
        return None

def perform_split_plot_anova(df, main_plot_factor, subplot_factor, response_var, block_factor=None):
    """Perform Split Plot ANOVA Analysis"""
    try:
        # Create the model formula
        if block_factor:
            formula = f"{response_var} ~ C({main_plot_factor}) * C({subplot_factor}) + C({block_factor})"
        else:
            formula = f"{response_var} ~ C({main_plot_factor}) * C({subplot_factor})"
        
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

def create_interaction_plot(df, main_plot, subplot, response):
    """Create interaction plot"""
    try:
        # Calculate means
        means = df.groupby([main_plot, subplot])[response].mean().reset_index()
        
        fig = px.line(means, x=subplot, y=response, color=main_plot,
                     title="Interaction Plot: Main Plot × Subplot Factors",
                     markers=True)
        
        fig.update_layout(
            xaxis_title="Subplot Factor",
            yaxis_title="Response Mean",
            legend_title="Main Plot Factor",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating interaction plot: {str(e)}")
        return None

def create_boxplot(df, factor_col, response_col, title):
    """Create boxplot for factor effects"""
    try:
        fig = px.box(df, x=factor_col, y=response_col, title=title)
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error creating boxplot: {str(e)}")
        return None

# Main application
def main():
    # Sidebar for data upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV, XLS, or XLSX files containing your split plot data"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Load data
            df = load_data(uploaded_file)
            
            if df is not None:
                st.markdown("### 📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Data configuration section
                st.header("⚙️ Analysis Configuration")
                
                # Check if data needs transformation
                data_format = st.radio(
                    "Data Format:",
                    ["Wide Format (needs transformation)", "Long Format (ready for analysis)"],
                    help="Wide format: Multiple response columns. Long format: Single response column."
                )
                
                if data_format == "Wide Format (needs transformation)":
                    st.subheader("Data Transformation Settings")
                    
                    # Select columns for transformation
                    main_plot_col = st.selectbox("Main Plot Factor Column:", df.columns)
                    block_col = st.selectbox("Block/Replication Column (optional):", 
                                           ["None"] + list(df.columns))
                    
                    # Select response columns
                    response_columns = st.multiselect(
                        "Response Columns (subplot treatments):",
                        [col for col in df.columns if col not in [main_plot_col, block_col]],
                        help="Select all columns that represent different subplot treatments"
                    )
                    
                    if response_columns:
                        if st.button("🔄 Transform Data"):
                            df_transformed = wide_to_long_transform(df, main_plot_col, None, response_columns)
                            
                            if df_transformed is not None:
                                st.success("Data transformed successfully!")
                                st.session_state['df_analysis'] = df_transformed
                                st.dataframe(df_transformed.head())
                else:
                    st.session_state['df_analysis'] = df
                
                # Analysis configuration (shown when data is ready)
                if 'df_analysis' in st.session_state:
                    df_analysis = st.session_state['df_analysis']
                    
                    st.subheader("Analysis Variables")
                    main_plot_factor = st.selectbox("Main Plot Factor:", df_analysis.columns)
                    subplot_factor = st.selectbox("Subplot Factor:", 
                                                [col for col in df_analysis.columns if col != main_plot_factor])
                    response_variable = st.selectbox("Response Variable:", 
                                                   [col for col in df_analysis.columns 
                                                    if col not in [main_plot_factor, subplot_factor]])
                    
                    block_factor = st.selectbox("Block Factor (optional):", 
                                              ["None"] + [col for col in df_analysis.columns 
                                                         if col not in [main_plot_factor, subplot_factor, response_variable]])
                    
                    if block_factor == "None":
                        block_factor = None
                    
                    # Store configuration in session state
                    st.session_state['config'] = {
                        'main_plot': main_plot_factor,
                        'subplot': subplot_factor,
                        'response': response_variable,
                        'block': block_factor
                    }

    # Main content area
    if uploaded_file is None:
        st.info("👆 Please upload a data file to begin analysis")
        
        # Show example data format
        st.markdown("### 📋 Expected Data Format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Wide Format Example:**")
            wide_example = pd.DataFrame({
                'Block': [1, 2, 3, 1, 2, 3],
                'MainPlot': ['A', 'A', 'A', 'B', 'B', 'B'],
                'Treatment1': [23.5, 24.1, 22.8, 26.2, 25.9, 26.5],
                'Treatment2': [21.3, 22.0, 20.9, 24.1, 23.8, 24.6],
                'Treatment3': [19.8, 20.5, 19.2, 22.3, 21.9, 22.8]
            })
            st.dataframe(wide_example, use_container_width=True)
        
        with col2:
            st.markdown("**Long Format Example:**")
            long_example = pd.DataFrame({
                'Block': [1, 1, 1, 2, 2, 2],
                'MainPlot': ['A', 'A', 'B', 'A', 'A', 'B'],
                'Subplot': ['T1', 'T2', 'T1', 'T1', 'T2', 'T2'],
                'Response': [23.5, 21.3, 26.2, 24.1, 22.0, 23.8]
            })
            st.dataframe(long_example, use_container_width=True)
    
    elif 'df_analysis' in st.session_state and 'config' in st.session_state:
        df_analysis = st.session_state['df_analysis']
        config = st.session_state['config']
        
        # Analysis results
        st.header("📈 Split Plot Analysis Results")
        
        # Descriptive statistics
        with st.expander("📊 Descriptive Statistics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary by Main Plot Factor")
                main_summary = df_analysis.groupby(config['main_plot'])[config['response']].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                st.dataframe(main_summary, use_container_width=True)
            
            with col2:
                st.subheader("Summary by Subplot Factor")
                sub_summary = df_analysis.groupby(config['subplot'])[config['response']].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                st.dataframe(sub_summary, use_container_width=True)
        
        # ANOVA Analysis
        with st.expander("🔬 Split Plot ANOVA", expanded=True):
            model, anova_results = perform_split_plot_anova(
                df_analysis, config['main_plot'], config['subplot'], 
                config['response'], config['block']
            )
            
            if anova_results is not None:
                st.subheader("ANOVA Table")
                
                # Format ANOVA results
                anova_formatted = anova_results.round(6)
                anova_formatted['Significance'] = anova_formatted['PR(>F)'].apply(
                    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
                )
                
                st.dataframe(anova_formatted, use_container_width=True)
                
                # Interpretation
                st.subheader("🔍 Interpretation")
                alpha = 0.05
                
                for factor in anova_formatted.index:
                    if 'PR(>F)' in anova_formatted.columns:
                        p_value = anova_formatted.loc[factor, 'PR(>F)']
                        if pd.notna(p_value):
                            if p_value < alpha:
                                st.markdown(f"✅ **{factor}**: Significant effect (p = {p_value:.6f})")
                            else:
                                st.markdown(f"❌ **{factor}**: No significant effect (p = {p_value:.6f})")
        
        # Post-hoc tests
        with st.expander("🎯 Post-hoc Analysis (Tukey HSD)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Main Plot Factor Comparisons")
                tukey_main = perform_tukey_hsd(df_analysis, config['main_plot'], config['response'])
                if tukey_main is not None:
                    st.text(str(tukey_main))
            
            with col2:
                st.subheader("Subplot Factor Comparisons")
                tukey_sub = perform_tukey_hsd(df_analysis, config['subplot'], config['response'])
                if tukey_sub is not None:
                    st.text(str(tukey_sub))
        
        # Visualizations
        st.header("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Main plot boxplot
            fig1 = create_boxplot(df_analysis, config['main_plot'], config['response'], 
                                "Main Plot Factor Effect")
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Subplot boxplot
            fig2 = create_boxplot(df_analysis, config['subplot'], config['response'], 
                                "Subplot Factor Effect")
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Interaction plot
        st.subheader("Interaction Plot")
        interaction_fig = create_interaction_plot(df_analysis, config['main_plot'], 
                                                config['subplot'], config['response'])
        if interaction_fig:
            st.plotly_chart(interaction_fig, use_container_width=True)
        
        # Residual analysis
        if model is not None:
            with st.expander("🔍 Residual Analysis", expanded=False):
                residuals = model.resid
                fitted = model.fittedvalues
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residuals vs Fitted
                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers',
                                                 name='Residuals'))
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_resid.update_layout(title="Residuals vs Fitted Values",
                                          xaxis_title="Fitted Values",
                                          yaxis_title="Residuals")
                    st.plotly_chart(fig_resid, use_container_width=True)
                
                with col2:
                    # Q-Q plot
                    from scipy.stats import probplot
                    qq = probplot(residuals, dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                              name='Sample Quantiles'))
                    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                                              mode='lines', name='Theoretical Line'))
                    fig_qq.update_layout(title="Q-Q Plot",
                                       xaxis_title="Theoretical Quantiles",
                                       yaxis_title="Sample Quantiles")
                    st.plotly_chart(fig_qq, use_container_width=True)
        
        # Download results
        st.header("💾 Download Results")
        
        if st.button("📄 Generate Analysis Report"):
            # Create a comprehensive report
            report = f"""
# Split Plot Analysis Report
**Developed by Bhavya**

## Data Summary
- **Total Observations:** {len(df_analysis)}
- **Main Plot Levels:** {df_analysis[config['main_plot']].nunique()}
- **Subplot Levels:** {df_analysis[config['subplot']].nunique()}

## ANOVA Results
{anova_formatted.to_string()}

## Descriptive Statistics
### Main Plot Factor
{main_summary.to_string()}

### Subplot Factor
{sub_summary.to_string()}

## Post-hoc Test Results
### Main Plot Comparisons
{str(tukey_main) if tukey_main else 'No significant differences'}

### Subplot Comparisons
{str(tukey_sub) if tukey_sub else 'No significant differences'}
            """
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"split_plot_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
