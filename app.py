import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Page config
st.set_page_config(page_title="IBM HR Attrition Dashboard", layout="wide")

# Title
st.title("IBM HR Analytics – Employee Attrition Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = load_data()
@st.cache_data
def train_model(data):
    df_ml = data.copy()

    # Encode target variable
    df_ml['Attrition'] = df_ml['Attrition'].map({'Yes': 1, 'No': 0})

    # Select numeric features only
    X = df_ml.select_dtypes(include=['int64'])
    y = df_ml['Attrition']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns
model, feature_columns = train_model(df)


# KPI
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
st.metric("Overall Attrition Rate (%)", round(attrition_rate, 2))

st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")

department = st.sidebar.multiselect(
    "Select Department",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

filtered_df = df[
    (df['Department'].isin(department)) &
    (df['Gender'].isin(gender))
]

# Attrition by Gender
st.subheader("Attrition Rate by Gender")
gender_attr = filtered_df.groupby('Gender')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

st.bar_chart(gender_attr)

# Attrition by Work-Life Balance
st.subheader("Attrition Rate by Work-Life Balance")
wlb_attr = filtered_df.groupby('WorkLifeBalance')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

st.bar_chart(wlb_attr)

st.subheader("Attrition Rate by Department")

dept_attr = filtered_df.groupby('Department')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

st.bar_chart(dept_attr)

# Attrition Rate by Department
st.subheader("Attrition Rate by Department")

dept_attr = filtered_df.groupby('Department')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

st.bar_chart(dept_attr)


# Attrition Rate by Job Role
st.subheader("Attrition Rate by Job Role")

job_attr = filtered_df.groupby('JobRole')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

st.bar_chart(job_attr)

# Age distribution
st.subheader("Age Distribution of Employees Who Left")
fig, ax = plt.subplots()
sns.histplot(
    filtered_df[filtered_df['Attrition'] == 'Yes']['Age'],
    kde=True,
    ax=ax
)
st.pyplot(fig)
st.markdown("---")
st.header("Predict Employee Attrition")

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
years_at_company = st.number_input("Years at Company", 0, 40, 5)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
overtime = st.selectbox("OverTime", ["Yes", "No"])

overtime_val = 1 if overtime == "Yes" else 0

input_data = pd.DataFrame([{
    'Age': age,
    'MonthlyIncome': monthly_income,
    'YearsAtCompany': years_at_company,
    'JobLevel': job_level,
    'WorkLifeBalance': work_life_balance,
    'OverTime': overtime_val
}])

# Align columns
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[feature_columns]

if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")

