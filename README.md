# Student-performance-prediction
Student Performance Prediction Using Machine Learning and Streamlit
# 🎓 Student Performance Prediction Using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning-based application designed to analyse students' academic information and predict their expected final examination marks.

The project uses **Random Forest Regression** to identify the relationship between different academic factors and student performance.

A user-friendly interactive dashboard is developed using **Streamlit**, allowing users to explore datasets, visualise academic trends, evaluate the machine learning model, and predict the performance of new students.

---

## 🎯 Project Objectives

- Predict students' final examination marks using Machine Learning.
- Analyse the impact of study hours and attendance on academic performance.
- Understand the relationship between academic factors and student results.
- Visualise student data through interactive graphs and charts.
- Identify the most influential features affecting student performance.
- Provide an interactive interface for predicting new student results.

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning model |
| Random Forest Regressor | Performance prediction |
| Plotly | Interactive data visualisation |
| Streamlit | Web application interface |

---

## 📊 Dataset Description

The project uses a dataset containing 60 student records.

The dataset includes the following features:

| Feature | Description |
|---|---|
| Study_Hours | Daily study hours |
| Attendance | Student attendance percentage |
| Previous_Marks | Previous examination marks |
| Assignment_Score | Assignment performance |
| Sleep_Hours | Daily sleeping hours |
| Final_Marks | Final examination marks (Target Variable) |

**Note:** The generated dataset contains synthetic data created for educational and demonstration purposes.

---

## 🤖 Machine Learning Algorithm

### Random Forest Regression

Random Forest Regression is an ensemble learning algorithm that combines multiple decision trees to produce predictions.

In this project, Random Forest Regression is used to predict students' final examination marks based on their academic information.

### Model Configuration

- Algorithm: Random Forest Regression
- Number of Estimators: 100
- Maximum Depth: 5
- Training Data: 80%
- Testing Data: 20%

---

## 🔄 Project Workflow

1. Dataset Generation
2. Data Collection and Loading
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Train-Test Split
6. Random Forest Model Training
7. Model Evaluation
8. Feature Importance Analysis
9. Student Performance Prediction

---

## 📈 Model Evaluation Metrics

The model is evaluated using the following regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted marks.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the same unit as the target variable.
- **R² Score:** Measures how well the model explains the variation in student marks.

---

## 💻 Application Features

The Streamlit application provides the following modules:

### 1. Dashboard
Displays student statistics, project information and marks distribution.

### 2. Dataset & EDA
Allows users to upload CSV datasets, preview student records, explore statistical summaries and view interactive graphs.

### 3. Model Training
Trains the Random Forest Regression model using the selected dataset.

### 4. Model Results
Displays evaluation metrics, actual versus predicted marks and feature importance.

### 5. Predict Performance
Allows users to enter a student's academic details and obtain predicted final marks along with a performance category.

---

## 🛠️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
```

### Step 2: Open the Project Folder

```bash
cd student-performance-prediction
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```
