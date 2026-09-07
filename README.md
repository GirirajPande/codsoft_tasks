Internship MetadataIntern 

Name: Giriraj Pande
Organization: CodSoftIntern 
ID: BY26RY229675
Domain: Machine Learning Virtual Internship
Duration: August 10, 2026 – September 10, 2026
Repository Name: codsoft_tasksCore 
Tech Stack: Python, Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (imbalanced-learn), Matplotlib, Seaborn


Task 1: Movie Genre ClassificationObjective: Multi-class text classification to predict movie genres from plot summaries.Dataset Specs: train_data.txt (54,214 records), test_data.txt (54,200 records) across 27 distinct genres.Class Distribution: Top genres are Drama (13,613), Documentary (13,096), and Comedy (7,447); bottom genres are War (132) and News (181).Text Preprocessing: Lowercased text, removed non-alphabetic characters ([^a-zA-Z\s]), and stripped excess whitespaces.

Feature Extraction: TfidfVectorizer configured with max_features=20000, ngram_range=(1,2), stop_words='english', min_df=2, and max_df=0.8.Imbalance Handling: SMOTE oversampling (k_neighbors=3) balancing all 27 classes to 13,613 samples each (367,551 total training instances).

Model Evaluation Metrics:Logistic Regression (class_weight='balanced'): 86.81% Cross-Validation AccuracyMultinomial Naive Bayes ($\alpha=0.5$): 85.85% Cross-Validation AccuracyLinear Support Vector Machine (LinearSVC): 91.20% Cross-Validation Accuracy (Best Model)Output Artifact: movie_predictions.csv containing final predictions on test plots.


Task 2: Credit Card Fraud DetectionObjective: Anomaly detection and imbalanced binary classification to flag fraudulent transactions.Dataset Specs: Sparkov credit card dataset sampled to 50,000 training records (fraudTrain.csv) and 50,000 testing records (fraudTest.csv).Class Distribution: Highly imbalanced with 49,727 legitimate transactions ($0$) vs. 273 fraudulent transactions ($1$).

Feature Engineering: Extracted customer age derived from birth year ($2020 - \text{dob.year}$).Feature Selection: Dropped non-informative columns (trans_date_trans_time, cc_num, merchant, first, last, street, city, state, zip, job, dob, trans_num, unix_time).Feature Encoding: Applied LabelEncoder to categorical attributes category and gender.Final Feature Set (9 Predictors): category, amt, gender, lat, long, city_pop, merch_lat, merch_long, age.

Model Evaluation Metrics:Logistic Regression: 99.53% AccuracyDecision Tree Classifier: 99.61% AccuracyRandom Forest Classifier: 99.73% Accuracy (Best Model)


Task 3: Customer Churn PredictionObjective: Binary classification predicting bank customer churn (Exited) and identifying churn drivers.Dataset Specs: Bank Customer Churn dataset (Churn_Modelling.csv) containing 10,000 records and 14 attributes.Class Distribution: 7,963 Stayed ($0$) vs. 2,037 Exited ($1$).

Data Cleaning & Preprocessing:Dropped non-predictive identifiers (RowNumber, CustomerId, Surname).Imputed numerical missing values with column medians and categorical missing values with modes.Encoded categorical attributes Geography and Gender using LabelEncoder.Performed an 80/20 stratified train-test split (8,000 train / 2,000 test).Scaled feature distributions using StandardScaler.

Model Evaluation Metrics:Logistic Regression: 80.50% AccuracyRandom Forest Classifier: 86.40% AccuracyGradient Boosting Classifier: 86.75% Accuracy (Best Model)Feature Importance Ranking (Random Forest): 1. Age, 2. EstimatedSalary, 3. CreditScore, 4. Balance, 5. NumOfProducts, 6. Tenure, 7. IsActiveMember, 8. Geography.
