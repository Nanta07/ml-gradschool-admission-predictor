# 🎓 Graduate Admission Prediction
Machine Learning project to predict a student's likelihood of being admitted to a graduate program based on academic and profile-related factors using regression models.

## 📌 Project Overview
The goal of this project is to build a predictive model that estimates a candidate's `Chance of Admit` into graduate school based on historical data. The project uses supervised regression models and includes feature analysis, model comparison, and performance evaluation.

## 📊 Dataset
- **Title**: Graduate Admission 2  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)  
- **Reference**:  
  *Mohan S Acharya, Asfia Armaan, Aneeta S Antony: "A Comparison of Regression Models for Prediction of Graduate Admissions", IEEE International Conference on Computational Intelligence in Data Science, 2019.*

The dataset includes:
- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP) Strength
- Letter of Recommendation (LOR) Strength
- CGPA
- Research Experience
- Chance of Admit (Target Variable)

## 🎯 Business Objectives
- Assist universities in improving the efficiency and objectivity of admissions.
- Help applicants assess their likelihood of acceptance.
- Identify key factors that influence admission decisions.

## ❓ Problem Statements
1. How can we model the relationship between an applicant’s profile and their chance of admission?
2. Which features are the most influential in the admission decision?
3. Which regression algorithm performs best in predicting `Chance of Admit`?

## 🧠 ML Objectives
- Train and evaluate several regression models: Linear Regression, Random Forest, and Gradient Boosting.
- Analyze correlations and feature importance.
- Compare models using metrics: MAE, MSE, RMSE, and R² Score.
- Save the best-performing model for future use.

## 🛠️ Tech Stack
- Python
- Jupyter Notebook
- Pandas, Numpy
- Scikit-learn
- Matplotlib & Seaborn

## 📂 Project Structure
```

graduate-admission-prediction/
│
├── predictive\_analytics\_Ananta\_Boemi\_Adji.ipynb   # Jupyter notebook
├── predictive\_analytics\_ananta\_boemi\_adji.py      # Python script
├── laporan\_predictive-analytics\_Ananta Boemi Adi.md # Report in Markdown
└── README.md                                      # Project documentation

```

## 📈 Sample Output
The model successfully predicts admission chances with reasonable accuracy using regression algorithms, and highlights CGPA, GRE, and TOEFL scores as the top predictors.

## 🧪 How to Run
1. Clone the repository.
2. Install required libraries (`pip install -r requirements.txt` if available).
3. Run the Jupyter notebook or Python script to view predictions and visualizations.

Notes: The project report and also project markdown is still in Bahasa Indonesia, the English version will be available soon.

## 📬 Author
**Ananta Boemi Adji**  
Machine Learning Enthusiast | Computer Engineering Student  
Feel free to connect on [LinkedIn](https://www.linkedin.com) *https://www.linkedin.com/in/ananta-boemi-adji/*
