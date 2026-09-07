# CodSoft Machine Learning Internship Tasks

**Intern Name:** Giriraj Pande  
**Intern ID:** BY26RY229675  
**Domain:** Machine Learning Virtual Internship  
**Duration:** August 10, 2026 – September 10, 2026  

---

## 📌 Technical Environment

* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (`imbalanced-learn`)
* **Visualization Tools:** Matplotlib, Seaborn

---

## 🎬 Task 1: Movie Genre Classification

* **Objective:** Predict movie genres from plot descriptions using NLP and multi-class classification.
* **Dataset Specs:** `train_data.txt` (54,214 records) and `test_data.txt` (54,200 records) spanning 27 genres.
* **Preprocessing:** Text lowercasing, non-alphabetic filtering (`[^a-zA-Z\s]`), and whitespace cleaning.
* **Feature Extraction:** `TfidfVectorizer` (`max_features=20000`, `ngram_range=(1,2)`, `stop_words='english'`).
* **Class Imbalance Handling:** Applied `SMOTE` oversampling (`k_neighbors=3`) balancing all 27 classes to 13,613 samples each.

**Model Results:**

* Logistic Regression: 86.81% Cross-Validation Accuracy
* Multinomial Naive Bayes: 85.85% Cross-Validation Accuracy
* **Linear SVM (Best Model):** **91.20% Accuracy**

---

## 💳 Task 2: Credit Card Fraud Detection

* **Objective:** Detect fraudulent transactions in highly imbalanced credit card data.
* **Dataset Specs:** Sparkov dataset sampled to 50,000 training records (`fraudTrain.csv`) and 50,000 test records (`fraudTest.csv`).
* **Class Distribution:** 49,727 legitimate transactions vs. 273 fraudulent transactions.
* **Feature Engineering:** Calculated customer `age` from birth year (`2020 - dob.year`) and dropped non-predictive metadata.
* **Feature Encoding:** Applied `LabelEncoder` to `category` and `gender` attributes.

**Model Results:**

* Logistic Regression: 99.53% Accuracy
* Decision Tree Classifier: 99.61% Accuracy
* **Random Forest Classifier (Best Model):** **99.73% Accuracy**

---

## 📊 Task 3: Customer Churn Prediction

* **Objective:** Predict bank customer churn (`Exited`) and isolate key retention drivers.
* **Dataset Specs:** Bank Customer Churn dataset (`Churn_Modelling.csv`) containing 10,000 records and 14 attributes.
* **Data Preprocessing:** Handled missing values, encoded categorical features, applied 80/20 stratified split, and standardized numerical inputs via `StandardScaler`.
* **Feature Importance:** Identified `Age` as the single most influential attribute predicting churn.

**Model Results:**

* Logistic Regression: 80.50% Accuracy
* Random Forest Classifier: 86.40% Accuracy
* **Gradient Boosting Classifier (Best Model):** **86.75% Accuracy**

---

## 👤 Author & Acknowledgments

* **Author:** Giriraj Pande
* **Organization:** CodSoft
* **Program Manager:** Ruman Arshad
