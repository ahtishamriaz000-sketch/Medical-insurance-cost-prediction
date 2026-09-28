# 🏥 Medical Insurance Charges Prediction

A Machine Learning project that predicts **medical insurance charges** based on customer information such as age, gender, BMI, number of children, smoking status, and region.

The project uses **Python, Pandas, Scikit-learn, Matplotlib, Joblib, and Streamlit** to build an end-to-end Machine Learning application.

## 📌 Project Overview

Medical insurance charges can vary depending on different personal and lifestyle factors. This project uses **Linear Regression** to learn the relationship between these factors and insurance charges.

The trained model can then predict the expected insurance charges for a new customer through an easy-to-use **Streamlit web application**.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## 📊 Dataset Features

The dataset contains the following features:

| Feature    | Description                        |
| ---------- | ---------------------------------- |
| `age`      | Age of the customer                |
| `sex`      | Gender of the customer             |
| `bmi`      | Body Mass Index                    |
| `children` | Number of children                 |
| `smoker`   | Smoking status                     |
| `region`   | Customer's region                  |
| `charges`  | Medical insurance charges (Target) |

## 🔄 Machine Learning Workflow

The project follows these steps:

1. Load the insurance dataset
2. Inspect the dataset
3. Handle missing values
4. Separate numerical and categorical columns
5. Apply **Label Encoding** to categorical data
6. Separate features (`X`) and target (`y`)
7. Split data into training and testing sets
8. Apply **StandardScaler**
9. Train a **Linear Regression** model
10. Make predictions
11. Evaluate model performance
12. Visualize relationships between features and charges
13. Save the trained model using Joblib
14. Build a Streamlit prediction interface

## 📈 Model Evaluation

The model is evaluated using:

* **MAE — Mean Absolute Error**
* **MSE — Mean Squared Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**

These metrics help measure how accurately the Linear Regression model predicts insurance charges.

## 📊 Data Visualization

The project includes visualizations such as:

* Age vs Charges
* BMI vs Charges
* Smoker vs Charges

These graphs help understand the relationship between different features and medical insurance charges.

## 🌐 Streamlit Application

The project includes a Streamlit web interface where users can enter:

* Age
* Gender
* BMI
* Number of Children
* Smoking Status
* Region

After clicking **Predict Charges**, the trained model predicts the estimated medical insurance charges.

## 💾 Model Saving

The trained model and scaler are saved using Joblib:

```python
joblib.dump(model, "insurance_model.pkl")
joblib.dump(scaler, "insurance_scaler.pkl")
```

This allows the trained model to be reused without training it again.

## ▶️ How to Run the Project

### 1. Install required libraries

```bash
pip install pandas scikit-learn matplotlib joblib streamlit
```

### 2. Run the Streamlit application

```bash
python -m streamlit run mediacal_insurance.py
```

The application will open in your browser.

## 📁 Project Structure

```text
Medical-Insurance-Prediction/
│
├── insurance.csv
├── mediacal_insurance.py
├── insurance_model.pkl
├── insurance_scaler.pkl
└── README.md
```

## 🎯 Project Objective

The main objective of this project is to demonstrate an end-to-end **Machine Learning regression workflow**, from data preprocessing and model training to evaluation, visualization, model saving, and deployment through Streamlit.

## 👨‍💻 Skills Demonstrated

* Data Cleaning
* Data Preprocessing
* Label Encoding
* Feature Scaling
* Train-Test Split
* Linear Regression
* Model Evaluation
* Data Visualization
* Model Serialization with Joblib
* Streamlit Application Development

## 🚀 Future Improvements

Possible improvements include:

* Trying additional regression algorithms
* Comparing multiple models
* Improving model performance
* Adding more interactive visualizations
* Deploying the application online


To run it on streamlit or in terminal this is command
python -m streamlit run mediacal_insurance.py
