# ML Projects Portfolio

This repository contains beginner-friendly machine learning projects using simple linear regression.

## 📁 Projects

# 1. California Housing Price Prediction – Simple Linear Regression

This project demonstrates a **Simple Linear Regression** model using the **California Housing dataset** provided by `scikit-learn`. The goal is to predict the **median house value** based on **median income** in different districts of California.

📂 [Project Folder](Simple_Linear_Regression/simple_lr_housing_price)

---

## Project Objective

> To understand and implement Simple Linear Regression using one feature – **Median Income**, and evaluate its ability to predict the **Median House Price**.

---

## Dataset Details

- **Source**: `sklearn.datasets.fetch_california_housing`
- **Target**: Median House Value (in $100,000s)
- **Feature used**: Median Income (in $10,000s)

This is a clean, real-world dataset perfect for learning and practicing regression techniques.

---

## Libraries Used

- `scikit-learn`
- `matplotlib`

### 2. 📚 Student Score Prediction – Simple Linear Regression

This project applies a **Simple Linear Regression** model to a small dataset that records students' study hours and their corresponding scores. The goal is to predict the expected score based on the number of hours studied.

## 📌 Project Objective

> To build and evaluate a linear regression model that predicts student performance (scores) based on the number of hours they study.

## 🧪 Dataset Details

A simple synthetic dataset was created for this project:

| Hours | Scores |
| ----- | ------ |
| 1     | 35     |
| 2     | 45     |
| 3     | 55     |
| 4     | 60     |
| 5     | 70     |
| 6     | 80     |
| 7     | 85     |
| 8     | 90     |

- **Feature**: Hours Studied
- **Target**: Student Scores

## ⚙️ Model Description

- **Algorithm**: Simple Linear Regression
- **Train-Test Split**: 75% train, 25% test
- **Input Feature**: `Hours`
- **Target Variable**: `Scores`

📂 [Project Folder](Simple_Linear_Regression/simple_lr_hours_scores)

The model is trained on the input data and evaluated using standard metrics
