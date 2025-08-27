import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="ANOVA & Tukey HSD Analysis", layout="wide")

# ------------------------------
# APP TITLE
# ------------------------------
st.title("ANOVA with Tukey HSD and Genotype Comparison")
st.write("""
This app performs **ANOVA, Tukey HSD test, and visualizations** for randomized block or factorial designs.  
Upload your dataset and explore the results with interpretations suitable for scientific publications.
""")

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload your dataset (.csv, .xls, .xlsx):", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.write(data.head())

    # ------------------------------
    # COLUMN SELECTION
    # ------------------------------
    st.sidebar.header("Select Columns for Analysis")
    main_factor = st.sidebar.selectbox("Select Main Plot Factor (e.g., Drought Levels):", data.columns)
    sub_factor = st.sidebar.selectbox("Select Sub Plot Factor (e.g., Genotypes):", data.columns)
    replication = st.sidebar.selectbox("Select Replication Column:", data.columns)
    response_var = st.sidebar.selectbox("Select Response Variable:", data.columns)

    if main_factor and sub_factor and replication and response_var:
        st.markdown("### Step 1: ANOVA Model Fitting")
        
        # ------------------------------
        # MODEL FITTING
        # ------------------------------
        formula = f"{response_var} ~ C({main_factor}) * C({sub_factor}) + C({replication})"
        model = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        st.write("**ANOVA Table:**")
        st.dataframe(anova_table)

        # Interpretation
        st.markdown("**Interpretation:**")
        st.write("""
        - If p-value < 0.05 for a factor, it means the factor has a statistically significant effect on the response variable.
        - Interaction term indicates whether the effect of one factor depends on the level of another.
        """)

        # ------------------------------
        # Tukey HSD for sub_factor
        # ------------------------------
        st.markdown("### Step 2: Tukey HSD Test for Genotypes")
        tukey = pairwise_tukeyhsd(endog=data[response_var], groups=data[sub_factor], alpha=0.05)
        st.text(tukey.summary())

        # Interpretation
        st.markdown("**Interpretation:**")
        st.write("""
        - Groups sharing the same letter are not significantly different at 5% significance level.
        - Significant differences indicate which genotypes perform differently under the given conditions.
        """)

        # ------------------------------
        # Group Means for Best Genotype
        # ------------------------------
        group_means = data.groupby(sub_factor)[response_var].mean().sort_values(ascending=False)
        best_genotype = group_means.index[0]
        second_best = group_means.index[1]
        best_value = group_means.iloc[0]
        second_value = group_means.iloc[1]
        advantage = ((best_value - second_value) / second_value) * 100

        st.markdown("### Step 3: Best Genotype Identification")
        st.write(f"**Best Genotype:** {best_genotype} with mean {response_var} = {best_value:.2f}")
        st.write(f"It performs **{advantage:.2f}% better** than the second-best genotype ({second_best}).")

        # ------------------------------
        # Visualization Section
        # ------------------------------
        st.markdown("### Step 4: Visualizations")

        # Boxplot
        st.subheader("Boxplot: Genotype vs Response")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=sub_factor, y=response_var, data=data, ax=ax1)
        ax1.set_title(f"{sub_factor} vs {response_var}")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.write("**Interpretation:** The spread of values for each genotype is shown. Narrow boxes indicate stability; higher medians indicate better performance.")

        # Interaction Plot
        st.subheader("Interaction Plot: Main Factor x Sub Factor")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        means = data.groupby([main_factor, sub_factor])[response_var].mean().unstack()
        means.T.plot(ax=ax2, marker='o')
        ax2.set_title("Interaction between Main Plot and Genotypes")
        ax2.set_ylabel(response_var)
        st.pyplot(fig2)

        st.write("**Interpretation:** Lines crossing indicate interaction effects. Parallel lines indicate no interaction.")

        # Bar Chart of Means
        st.subheader("Mean Performance of Genotypes")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        group_means.plot(kind='bar', ax=ax3)
        ax3.set_ylabel(f"Mean {response_var}")
        ax3.set_title("Genotype Performance")
        st.pyplot(fig3)

        st.write(f"**Interpretation:** {best_genotype} stands out as the top performer, with a clear margin over others.")

        # Download Results
        st.markdown("### Download Results")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, sheet_name="Raw Data", index=False)
            anova_table.to_excel(writer, sheet_name="ANOVA")
            pd.DataFrame(tukey.summary().data).to_excel(writer, sheet_name="TukeyHSD", index=False)
            pd.DataFrame(group_means).to_excel(writer, sheet_name="Genotype Means")
        st.download_button("Download Results as Excel", data=output.getvalue(), file_name="ANOVA_Tukey_Results.xlsx")

else:
    st.info("Please upload a dataset to start the analysis.")
