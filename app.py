import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Salaries Dashboard", page_icon="💰", layout="wide")

MAPPINGS = {
    "experience_level": {"EN": "Entry Level", "MI": "Mid Level", "SE": "Senior Level", "EX": "Executive Level"},
    "employment_type": {"FT": "Full Time", "PT": "Part Time", "CT": "Contract", "FL": "Freelance"},
    "company_size": {"S": "Small", "M": "Medium", "L": "Large"},
}


@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)

    # Drop exact duplicate rows (same as notebook)
    df = df.drop_duplicates()

    # Map coded columns to readable labels only if they still contain the codes
    for col, mapping in MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    return df


st.title("💰 Data Science Salaries Dashboard")
st.caption("Cleaning + EDA from the Colab notebook, turned into an interactive dashboard.")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload salaries CSV", type=["csv"])

default_path = "salaries (1).csv"
if uploaded_file is not None:
    df = load_and_clean(uploaded_file)
else:
    try:
        df = load_and_clean(default_path)
        st.sidebar.info(f"Using bundled file: {default_path}")
    except FileNotFoundError:
        st.warning("👈 Upload your salaries CSV from the sidebar to get started.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Filters")

# ---------------------------------------------------------------------------
# Sidebar filters (built dynamically based on available columns)
# ---------------------------------------------------------------------------
filtered = df.copy()

if "experience_level" in df.columns:
    exp_opts = sorted(df["experience_level"].dropna().unique())
    exp_sel = st.sidebar.multiselect("Experience Level", exp_opts, default=exp_opts)
    filtered = filtered[filtered["experience_level"].isin(exp_sel)]

if "employment_type" in df.columns:
    emp_opts = sorted(df["employment_type"].dropna().unique())
    emp_sel = st.sidebar.multiselect("Employment Type", emp_opts, default=emp_opts)
    filtered = filtered[filtered["employment_type"].isin(emp_sel)]

if "company_size" in df.columns:
    size_opts = sorted(df["company_size"].dropna().unique())
    size_sel = st.sidebar.multiselect("Company Size", size_opts, default=size_opts)
    filtered = filtered[filtered["company_size"].isin(size_sel)]

if "job_title" in df.columns:
    job_opts = sorted(df["job_title"].dropna().unique())
    job_sel = st.sidebar.multiselect("Job Title (optional)", job_opts, default=[])
    if job_sel:
        filtered = filtered[filtered["job_title"].isin(job_sel)]

if "salary_in_usd" in df.columns and not filtered.empty:
    min_sal, max_sal = int(df["salary_in_usd"].min()), int(df["salary_in_usd"].max())
    sal_range = st.sidebar.slider("Salary Range (USD)", min_sal, max_sal, (min_sal, max_sal))
    filtered = filtered[filtered["salary_in_usd"].between(*sal_range)]

if filtered.empty:
    st.warning("No rows match the current filters. Try widening your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", f"{len(filtered):,}")
col2.metric("Avg Salary (USD)", f"${filtered['salary_in_usd'].mean():,.0f}" if "salary_in_usd" in filtered.columns else "—")
col3.metric("Median Salary (USD)", f"${filtered['salary_in_usd'].median():,.0f}" if "salary_in_usd" in filtered.columns else "—")
col4.metric("Unique Job Titles", f"{filtered['job_title'].nunique():,}" if "job_title" in filtered.columns else "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Salary Distribution", "Experience Levels", "Top Jobs", "Correlation", "Raw Data"]
)

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=filtered, x="salary_in_usd", bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Salary in USD")
        ax.set_xlabel("Salary (USD)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.caption(
            "Most salaries are concentrated in the lower and middle ranges, "
            "with a smaller number of employees earning significantly higher salaries "
            "(right-skewed distribution)."
        )

    with c2:
        if "company_size" in filtered.columns:
            avg_salary = filtered.groupby("company_size")["salary_in_usd"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(data=avg_salary, x="company_size", y="salary_in_usd", ax=ax)
            ax.set_title("Average Salary by Company Size")
            ax.set_xlabel("Company Size")
            ax.set_ylabel("Average Salary (USD)")
            st.pyplot(fig)

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        if "experience_level" in filtered.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            order = filtered["experience_level"].value_counts().index
            sns.countplot(data=filtered, x="experience_level", order=order, ax=ax)
            ax.set_title("Distribution of Experience Levels")
            ax.set_xlabel("Experience Level")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.caption(
                "Senior and Mid-Level professionals typically represent the largest "
                "portion of the dataset, while Executive and Entry-Level are less common."
            )

    with c2:
        if "experience_level" in filtered.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=filtered, x="experience_level", y="salary_in_usd", ax=ax)
            ax.set_title("Salary Distribution by Experience Level")
            ax.set_xlabel("Experience Level")
            ax.set_ylabel("Salary (USD)")
            st.pyplot(fig)
            st.caption(
                "Executive-level professionals generally earn higher salaries, "
                "with several high-salary outliers present."
            )

with tab3:
    if "job_title" in filtered.columns:
        top_n = st.slider("Number of top job titles to show", 5, 20, 10)
        top_jobs = (
            filtered.groupby("job_title")["salary_in_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
        sns.barplot(x=top_jobs.values, y=top_jobs.index, ax=ax)
        ax.set_title(f"Top {top_n} Job Titles by Average Salary")
        ax.set_xlabel("Average Salary (USD)")
        ax.set_ylabel("Job Title")
        st.pyplot(fig)
        st.caption(
            "Some specialized AI and data-related roles have considerably higher "
            "average salaries, reflecting strong demand for advanced technical positions."
        )

with tab4:
    numeric_df = filtered.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        st.caption(
            "Most numerical variables show weak correlations with each other, "
            "suggesting salary is driven by multiple combined factors rather than "
            "a single numerical variable."
        )
    else:
        st.info("Not enough numeric columns to compute a correlation heatmap.")

with tab5:
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_salaries.csv",
        mime="text/csv",
    )

st.markdown("---")
with st.expander("📌 Insights Summary"):
    st.markdown(
        """
- **Salary Distribution:** Most salaries are concentrated in the lower and middle ranges, with a smaller number of employees earning very high salaries.
- **Experience Level:** Senior and Mid-Level professionals represent the largest portion of the dataset, while Executive and Entry-Level positions are less common.
- **Salary by Experience:** Salaries generally increase with experience level; Executive-level positions have the highest median salaries, with several high-salary outliers.
- **Top Job Titles:** Some specialized AI and data-related job titles have significantly higher average salaries than others.
- **Correlation:** Numerical variables show weak correlations overall, indicating salary is influenced by multiple factors rather than a single numerical variable.
        """
    )
