import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io

# ----------------------------
# Streamlit App Title
# ----------------------------
st.title("ANOVA and Tukey HSD Analysis for Genotypes")
st.write("Upload your dataset to perform ANOVA, Tukey HSD, and generate high-quality visualizations with interpretations.")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    # Detect file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📂 Preview of Uploaded Data")
    st.write(df.head())

    # ----------------------------
    # Data Columns Selection
    # ----------------------------
    st.subheader("🔍 Select Columns for Analysis")
    response_var = st.selectbox("Select the response variable (numeric)", df.columns)
    main_plot = st.selectbox("Select the main plot factor", df.columns)
    genotype = st.selectbox("Select the genotype factor", df.columns)
    replication = st.selectbox("Select the replication factor", df.columns)

    # Ensure categorical data type for factors
    df[main_plot] = df[main_plot].astype(str)
    df[genotype] = df[genotype].astype(str)
    df[replication] = df[replication].astype(str)

    st.write(f"**Selected Response Variable:** {response_var}")
    st.write(f"**Main Plot:** {main_plot}, **Genotype:** {genotype}, **Replication:** {replication}")

    # ----------------------------
    # ANOVA Model
    # ----------------------------
    formula = f"{response_var} ~ C({main_plot}) + C({genotype}) + C({replication})"
    model = ols(formula, data=df).fit()
    anova_results = anova_lm(model, typ=2)

    st.subheader("📊 ANOVA Table")
    st.write(anova_results)

    # ✅ Interpretation of ANOVA
    st.markdown("### 📌 Interpretation of ANOVA Results")
    for factor in [main_plot, genotype, replication]:
        p_value = anova_results.loc[f"C({factor})", "PR(>F)"]
        if p_value < 0.05:
            st.write(f"**{factor}** has a **significant effect** on {response_var} (p = {p_value:.4f}).")
        else:
            st.write(f"**{factor}** does **not have a significant effect** on {response_var} (p = {p_value:.4f}).")

    # ----------------------------
    # Tukey HSD Test
    # ----------------------------
    st.subheader("📌 Tukey HSD Test for Genotypes")
    tukey = pairwise_tukeyhsd(endog=df[response_var], groups=df[genotype], alpha=0.05)
    st.text(tukey)

    # ✅ Interpretation of Tukey
    st.markdown("### 📌 Interpretation of Tukey HSD")
    st.write("Groups marked as 'True' in the 'reject' column differ **significantly** in their means.")

    # ----------------------------
    # Identify Best Genotype
    # ----------------------------
    st.subheader("🏆 Best Genotype Identification")
    genotype_means = df.groupby(genotype)[response_var].mean().sort_values(ascending=False)
    best_genotype = genotype_means.index[0]
    best_value = genotype_means.iloc[0]
    second_best = genotype_means.iloc[1]
    improvement = ((best_value - second_best) / second_best) * 100

    st.write(f"**Best Genotype:** {best_genotype} with mean {best_value:.2f}")
    st.write(f"It is **{improvement:.2f}% better** than the second-best genotype.")

    # ----------------------------
    # Visualizations
    # ----------------------------
    st.subheader("📈 Visualizations")

    # Genotype Means Plot
    st.markdown("#### 🔹 Mean Comparison of Genotypes")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genotype_means.index, y=genotype_means.values, palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    plt.title("Mean Response for Each Genotype", fontsize=14)
    plt.ylabel(response_var)
    st.pyplot(fig)

    st.markdown("**Interpretation:** This plot shows the average performance of each genotype. The tallest bar corresponds to the best-performing genotype.")

    # Boxplot
    st.markdown("#### 🔹 Boxplot for Genotypes")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[genotype], y=df[response_var], palette="Set2", ax=ax)
    plt.xticks(rotation=45)
    plt.title("Distribution of Response Variable by Genotype", fontsize=14)
    st.pyplot(fig)

    st.markdown("**Interpretation:** The boxplot helps assess variability within each genotype. Smaller boxes indicate consistency, while wider boxes indicate more variation.")

    # Main Plot Effect
    st.markdown("#### 🔹 Effect of Main Plot Factor")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df[main_plot], y=df[response_var], estimator=np.mean, ci="sd", palette="coolwarm", ax=ax)
    plt.title(f"Effect of {main_plot} on {response_var}", fontsize=14)
    st.pyplot(fig)

    st.markdown(f"**Interpretation:** This shows the average effect of different {main_plot} levels on the response variable.")
