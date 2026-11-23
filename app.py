import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(page_title="Student Math Score Predictor", layout="wide")

st.title("📊 Student Performance EDA & Linear Regression")

# --- CSV File Automatic Load ---
# Yahan apni CSV file ka bilkul exact naam likhein jo GitHub par upload ki hai
CSV_FILE_NAME = "student_performance.csv"

try:
    df = pd.read_csv(CSV_FILE_NAME)
except FileNotFoundError:
    st.error(f"❌ '{CSV_FILE_NAME}' file nahi mili! Check karein ke GitHub par CSV file ka naam bilkul yahi hai.")
    st.stop()

# --- 1. Data Overview ---
st.subheader("1. Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Duplicate Rows", df.duplicated().sum())

with st.expander("View Raw Data & Summary Statistics"):
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())
    st.write("**Statistical Summary:**")
    st.dataframe(df.describe())
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

# --- 2. Exploratory Data Analysis (EDA) ---
st.subheader("2. Exploratory Data Analysis (EDA)")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.write("**Feature Histograms**")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df.hist(ax=ax1)
    plt.tight_layout()
    st.pyplot(fig1)
    
with viz_col2:
    st.write("**Correlation Heatmap**")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df.select_dtypes(include=np.number).corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax2
    )
    plt.title("Correlation Heatmap")
    st.pyplot(fig2)

# --- 3. Data Preprocessing & Model Training ---
df_clean = df.drop_duplicates().dropna()
df_encoded = pd.get_dummies(df_clean, drop_first=True)

X = df_encoded.drop("math_score", axis=1)
y = df_encoded["math_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 4. Model Evaluation ---
st.subheader("3. Linear Regression Model Performance")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.4f}")
m_col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.4f}")
m_col3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")

res_col1, res_col2 = st.columns(2)
with res_col1:
    st.write("**Actual vs Predicted Values (Table)**")
    comparison_df = pd.DataFrame({
        "Actual Math Score": y_test.values,
        "Predicted Math Score": np.round(y_pred, 2)
    })
    st.dataframe(comparison_df)
    
with res_col2:
    st.write("**Actual vs Predicted Scatter Plot**")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(y_test, y_pred, color="blue", alpha=0.7)
    ax3.set_xlabel("Actual Values")
    ax3.set_ylabel("Predicted Values")
    ax3.set_title("Actual vs Predicted Values")
    st.pyplot(fig3)

# --- 5. Interactive Live Prediction ---
st.subheader("4. Predict Math Score for a New Student")
with st.form("prediction_form"):
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        gender = st.selectbox("Gender", df_clean["gender"].unique())
        race = st.selectbox("Race/Ethnicity", df_clean["race_ethnicity"].unique())
    with p_col2:
        parent_edu = st.selectbox("Parental Education", df_clean["parental_education"].unique())
        lunch = st.selectbox("Lunch", df_clean["lunch"].unique())
    with p_col3:
        test_prep = st.selectbox("Test Preparation Course", df_clean["test_preparation_course"].unique())
        reading = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
        writing = st.number_input("Writing Score", min_value=0, max_value=100, value=70)
        
    submit_btn = st.form_submit_button("Predict Math Score")
    
    if submit_btn:
        input_data = pd.DataFrame({
            "gender": [gender],
            "race_ethnicity": [race],
            "parental_education": [parent_edu],
            "lunch": [lunch],
            "test_preparation_course": [test_prep],
            "reading_score": [reading],
            "writing_score": [writing]
        })
        
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
        
        prediction = model.predict(input_encoded)[0]
        st.success(f"🎯 Predicted Math Score: **{prediction:.2f} / 100**")
