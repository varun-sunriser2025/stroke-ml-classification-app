Stroke Prediction – Machine Learning Classification Project

a. Problem Statement
The objective of this project is to develop and compare multiple machine learning classification models to predict the likelihood of a patient experiencing a stroke based on demographic and medical attributes.
The goal is to evaluate model behavior using multiple performance metrics rather than relying solely on accuracy, especially in the presence of class imbalance.


b. Dataset Description
The dataset consists of 5110 patient records with 12 input features including:
	• Age
	• Hypertension
	• Heart Disease
	• BMI
	• Average Glucose Level
	• Smoking Status
	• Work Type
	• Residence Type
	• Gender
The target variable is binary:
	• 0 → No Stroke
	• 1 → Stroke
The dataset exhibits class imbalance, with significantly fewer stroke cases compared to non-stroke cases.


c. Models Used & Evaluation Metrics
The following classification models were implemented:
	1. Logistic Regression
	2. Decision Tree
	3. K-Nearest Neighbors (KNN)
	4. Naive Bayes
	5. Random Forest (Ensemble)
	6. XGBoost (Ensemble)
Evaluation Metrics Used:
	• Accuracy
	• AUC Score
	• Precision
	• Recall
	• F1 Score
Matthews Correlation Coefficient (MCC)


Model										Accuracy	  AUC		Precision	  Recall		F1		MCC
Logistic Regression				0.951		0.838			0.000			0.000		0.000		0.000
Decision Tree							0.917		0.567			0.170			0.180		0.175		0.131
KNN												0.948		0.614			0.200			0.020		0.036		0.049
Naive Bayes								0.868		0.803			0.165			0.420		0.237		0.203
Random Forest							0.948		0.809			0.200			0.020		0.036		0.049
XGBoost										0.946		0.797			0.273			0.060		0.098		0.108

Observations
Logistic Regression
	Although it achieved high accuracy and strong AUC, it failed to correctly classify stroke cases due to class imbalance and default decision threshold.
Decision Tree
Detected some stroke cases but showed moderate discrimination ability and possible instability.
KNN
Performed poorly on minority class detection due to imbalance and distance-based sensitivity.
Naive Bayes
Achieved the highest Recall (0.42) and MCC (0.203), indicating better minority detection capability despite lower overall accuracy.
Random Forest
Improved stability over Decision Tree but still struggled to detect minority stroke cases effectively.
XGBoost
Showed improved precision compared to other models but recall remained limited without imbalance handling techniques.

Conclusion
Accuracy alone is misleading for imbalanced medical datasets. Based on Recall and MCC, Naive Bayes demonstrated relatively better performance in identifying stroke cases.
Future improvements could include applying imbalance handling techniques such as SMOTE, class weighting, or threshold tuning.

Live Streamlit App
Deployed Application Link:
https://stroke-ml-classification-app-st5sk2gsgpokhzsotkorxj.streamlit.app/

