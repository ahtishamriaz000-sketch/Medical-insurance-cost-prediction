import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_squared_error,
    r2_score
)


# ================= STREAMLIT TITLE =================

st.title("Medical Insurance Cost Prediction")
st.write("Linear Regression Model")


# ================= LOAD DATA =================

df = pd.read_csv("insurance.csv")

st.subheader("Dataset")
st.write(df.head())


# ================= DATA INFO =================

st.subheader("Dataset Information")
st.write(df.shape)

nums_cols = df.select_dtypes(include=["int64", "float64"]).columns
text_cols = df.select_dtypes(include=["object", "string"]).columns


# ================= MISSING VALUES =================

df[nums_cols] = df[nums_cols].fillna(
    df[nums_cols].median()
)

for col in text_cols:
    df[col] = df[col].fillna(
        df[col].mode()[0]
    )


# ================= LABEL ENCODING =================

label_encoders = {}

for col in text_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# ================= FEATURES & TARGET =================

x = df.drop("charges", axis=1)
y = df["charges"]


# ================= TRAIN TEST SPLIT =================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)


# ================= SCALING =================

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ================= MODEL =================

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# ================= MODEL PERFORMANCE =================

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


st.subheader("Model Performance")

st.write("MAE:", mae)
st.write("MSE:", mse)
st.write("RMSE:", rmse)
st.write("R² Score:", r2)


# ================= GRAPH 1 =================

st.subheader("Age vs Charges")

fig, ax = plt.subplots()

ax.scatter(
    df["age"],
    df["charges"],
    marker="+",
    s=50,
    alpha=0.8,
    label="age vs charges"
)

ax.set_xlabel("Age")
ax.set_ylabel("Charges")
ax.set_title("Age vs Charges")
ax.grid()
ax.legend()
st.pyplot(fig)


# ================= GRAPH 2 =================

st.subheader("BMI vs Charges")

fig, ax = plt.subplots()

ax.scatter(
    df["bmi"],
    df["charges"],
    marker="o",
    s=50,
    alpha=0.8,
    label="bmi vs charges"
)

ax.set_xlabel("BMI")
ax.set_ylabel("Charges")
ax.set_title("BMI vs Charges")
ax.grid()
ax.legend()
st.pyplot(fig)


# ================= GRAPH 3 =================

st.subheader("Smoker vs Charges")

fig, ax = plt.subplots()

ax.scatter(
    df["smoker"],
    df["charges"],
    marker="+",
    s=50,
    alpha=0.8,
    label="smoker vs charges"
)

ax.set_xlabel("Smoker")
ax.set_ylabel("Charges")
ax.set_title("Smoker vs Charges")
ax.grid()
ax.legend()
st.pyplot(fig)


# ================= SAVE MODEL =================

joblib.dump(model, "insurance_model.pkl")
joblib.dump(scaler, "insurance_scaler.pkl")


# ================= LOAD MODEL =================

load_model = joblib.load("insurance_model.pkl")
load_scaler = joblib.load("insurance_scaler.pkl")


# ================= PREDICTION =================

st.subheader("💰 Insurance Charges Prediction")

age = st.number_input("Age", min_value=1, max_value=100, value=30)

sex = st.selectbox("Sex", ["male", "female"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)


if st.button("Predict Charges"):

    # Encode input values
    sex_value = label_encoders["sex"].transform([sex])[0]
    smoker_value = label_encoders["smoker"].transform([smoker])[0]
    region_value = label_encoders["region"].transform([region])[0]

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_value],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker_value],
        "region": [region_value]
    })

    input_scaled = load_scaler.transform(input_data)

    prediction = load_model.predict(input_scaled)

    st.success(
        f"Predicted Insurance Charges: ${prediction[0]:.2f}"
    )


#python -m streamlit run mediacal_insurance.py