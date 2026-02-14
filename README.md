#**Stroke Prediction – Machine Learning Classification Project**

**a. Problem Statement**

The objective of this project is to develop and compare multiple machine learning classification models to predict the likelihood of a patient experiencing a stroke based on demographic and medical attributes.
The goal is to evaluate model behavior using multiple performance metrics rather than relying solely on accuracy, especially in the presence of class imbalance.


**b. Dataset Description**

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


**c. Models Used & Evaluation Metrics**

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
	• Matthews Correlation Coefficient (MCC)


| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.951 | 0.838 | 0.000 | 0.000 | 0.000 | 0.000 |
| Decision Tree | 0.917 | 0.567 | 0.170 | 0.180 | 0.175 | 0.131 |
| KNN | 0.948 | 0.614 | 0.200 | 0.020 | 0.036 | 0.049 |
| Naive Bayes | 0.868 | 0.803 | 0.165 | 0.420 | 0.237 | 0.203 |
| Random Forest | 0.948 | 0.809 | 0.200 | 0.020 | 0.036 | 0.049 |
| XGBoost | 0.946 | 0.797 | 0.273 | 0.060 | 0.098 | 0.108 |


**Observations**

**1.Logistic Regression**
	
Logistic Regression achieved high overall accuracy (95.1%) and strong AUC (0.838), indicating good probability ranking capability. However, it failed to correctly classify stroke cases (Recall = 0). This suggests that due to severe class imbalance, the default decision threshold (0.5) is not appropriate. Although the model separates classes probabilistically, it struggles with minority class detection.
	
**2.Decision Tree**

Decision Tree detected some stroke cases (Recall = 0.18) but showed lower AUC (0.567), indicating weaker discrimination ability. Being a single-tree model, it is sensitive to data variations and may overfit to dominant class patterns.

**3.KNN**

KNN achieved high accuracy but extremely low recall (0.02), meaning it rarely predicts stroke cases. Since KNN is distance-based, class imbalance significantly affects its local neighborhood voting mechanism. The low MCC indicates poor balanced classification performance.

**4.Naive Bayes**

Naive Bayes achieved the highest Recall (0.42) and highest MCC (0.203) among all models. Although its overall accuracy is lower (86.8%), it performs better in identifying minority stroke cases. The probabilistic independence assumption appears to help in handling class imbalance more effectively than other models.

**5.Random Forest**

Random Forest achieved high accuracy (94.8%) and good AUC (0.809), but like Logistic Regression, it struggled with minority detection (Recall = 0.02). While ensemble averaging reduces variance, it does not automatically solve class imbalance issues without parameter tuning.

**6.XGBoost**

XGBoost achieved improved precision (0.273) compared to other models and moderate AUC (0.797). However, recall remains limited (0.06), indicating that boosting improves confidence in predictions but still requires imbalance handling techniques for better minority detection

**Overall Conclusion**

Although several models achieved high accuracy, accuracy alone is misleading for imbalanced medical datasets. Based on Recall and MCC, Naive Bayes demonstrated relatively better capability in identifying stroke cases.

Live Streamlit App
Deployed Application Link:
https://stroke-ml-classification-app-st5sk2gsgpokhzsotkorxj.streamlit.app/

