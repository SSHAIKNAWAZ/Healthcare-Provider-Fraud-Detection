# Healthcare Provider Fraud Detection

## Project Overview

This project aims to develop a robust machine learning solution for detecting potential fraudulent activities among healthcare providers. Healthcare fraud poses a significant financial burden on insurance systems and governments, diverting resources and compromising patient care. This initiative leverages machine learning to proactively identify suspicious patterns in claims and beneficiary data, enabling targeted investigations and mitigating financial losses.

## Problem Statement

The core challenge addressed is the accurate identification of fraudulent providers within a highly imbalanced dataset, where fraudulent cases are rare. The primary objective is to build a model that maximizes **Recall** – the ability to correctly identify as many actual fraudulent providers as possible – without an excessive compromise to **Precision** – minimizing the number of legitimate providers incorrectly flagged as fraudulent, which can lead to inefficient investigations.

## Dataset

The project utilizes a simulated healthcare claims dataset, structured across multiple interconnected CSV files, for both training and unseen prediction:

### Training Data Set

* `Train_Beneficiarydata-1542865627584.csv`: Contains demographic and health-related information for various beneficiaries (e.g., `BeneID`, `Age`, `Gender`, `ChronicCond_*`).
* `Train_Inpatientdata-1542865627584.csv`: Records of inpatient claims, including details like `ClaimID`, `Provider`, `AttendingPhysician`, `AdmissionDt`, `DischargeDt`, `ClmDiagnosisCode_*`, `ClmProcedureCode_*`, `InscClaimAmtReimbursed`, and `DeductibleAmtPaid`.
* `Train_Outpatientdata-1542865627584.csv`: Similar to inpatient data but for outpatient claims.
* `Train-1542865627584.csv`: The target file containing `Provider` IDs and their corresponding `PotentialFraud` status ('Yes' or 'No').

### Test/Unseen Data Set

* `Unseen_Beneficiarydata-1542969243754.csv`
* `Unseen_Inpatientdata-1542969243754.csv`
* `Unseen_Outpatientdata-1542969243754.csv`
* `Unseen-1542969243754.csv`: Contains `Provider` IDs for which fraud predictions are to be submitted.

## Project Goals

1.  **Develop a Robust Predictive Model:** Build a machine learning model capable of accurately classifying healthcare providers as fraudulent or non-fraudulent.
2.  **Optimize for Recall:** Enhance the model's ability to identify actual fraud cases (high recall) while maintaining an acceptable level of precision.
3.  **Generate Predictions:** Apply the trained model to unseen data and provide a submission file with predicted fraud probabilities and classes.
4.  **Provide Actionable Recommendations:** Offer business-oriented insights and strategies for utilizing the model and for future improvements in fraud detection.

## Methodology and Approach

The project followed a comprehensive machine learning pipeline, emphasizing data preparation, feature engineering, advanced modeling, and rigorous evaluation.

### 1. Data Management and Preprocessing

* **Data Loading:** All training and unseen datasets were loaded using Pandas.
* **Data Merging:**
    * Inpatient and Outpatient claims were combined into a single claims dataset.
    * This combined claims data was then merged with the beneficiary data (`BeneID`) to enrich claim records with beneficiary demographics and chronic conditions.
    * Finally, this merged claims-beneficiary data was linked to the `Train.csv` (or `Unseen.csv`) file based on `Provider` ID, which served as the aggregation key and target identifier.
* **Initial Cleaning:** Ensured data types were appropriate for numerical operations and date fields were correctly parsed.

### 2. Feature Engineering

The core of the model's predictive power lies in aggregated and engineered features at the **Provider level**. This transformation captures behavioral patterns indicative of fraud.

* **Aggregated Features:**
    * Counts of total claims, inpatient claims, and outpatient claims (`TotalClaims`, `TotalInpatientClaims`, `TotalOutpatientClaims`).
    * Sum and average of reimbursed amounts and deductible paid (`SumInscClaimAmtReimbursed`, `AvgInscClaimAmtReimbursed`, `SumDeductibleAmtPaid`, `AvgDeductibleAmtPaid`).
    * Average demographic and health indicators of beneficiaries associated with the provider (`AvgAge`, `AvgGender`, `AvgRace`, `AvgChronicCond_*`, `AvgRenalDiseaseIndicator`).
    * Counts of unique beneficiaries, attending, operating, and other physicians (`UniqueBeneIDs`, `UniqueAttendingPhysicians`, `UniqueOperatingPhysicians`, `UniqueOtherPhysicians`).
    * Counts of unique diagnosis and procedure codes (`UniqueClmDiagnosisCode_*`, `UniqueClmProcedureCode_*`, `UniqueClmAdmitDiagnosisCode`, `UniqueDiagnosisGroupCode`).
    * Average claim duration (`AvgClaimDuration`) and average inpatient stay duration (`AvgInpatientStayDuration`).
    * Proportion of claims with missing physician information (`PropMissingAttendingPhysician`, `PropMissingOperatingPhysician`, `PropMissingOtherPhysician`).
* **Advanced, Domain-Specific Features:** These were engineered to capture more subtle fraud indicators:
    * **`ClaimsWithManyDiagnosisCodes`**: Counts claims with at least 4 diagnosis codes, potentially signaling 'upcoding' or unnecessary complexity.
    * **`InpatientToOutpatientRatio`**: The ratio of inpatient to outpatient claims, which can reveal unusual service distribution by a provider.
    * **`ReimburseToDeductibleRatio`**: The ratio of total reimbursed amount to total deductible paid, flagging abnormal financial patterns.
    * **`PropClaimsWithOperatingPhysician`**: Proportion of claims where an operating physician is listed, which might indicate higher volumes of complex or invasive procedures.
* **Missing Value Handling:** Missing values, particularly those resulting from aggregations where a certain type of claim/data was absent for a provider (e.g., `AvgInpatientStayDuration` for providers with no inpatient claims), were imputed using the median value of their respective columns.

### 3. Modelling and Evaluation

The problem was treated as a binary classification task. Due to the inherent class imbalance (fewer fraudulent providers), specific strategies were employed.

* **Baseline Model: Random Forest Classifier**
    * An initial Random Forest model was trained to establish a performance baseline. While it showed high overall accuracy and decent precision, its **recall (51%)** was relatively low, indicating a significant number of fraudulent providers were being missed.
* **Optimization 1: Threshold Tuning for Random Forest**
    * To address the low recall, the classification threshold for the Random Forest model's predicted probabilities was tuned. By lowering the threshold from the default 0.5, the model became more sensitive to the positive class (fraud).
    * **Result:** This successfully boosted **Recall to 67.33%** while maintaining a reasonable precision (64.76%) and an improved F1-score (0.6602).
* **Optimization 2: XGBoost Classifier**
    * XGBoost, a powerful gradient boosting framework, was chosen for its superior performance, ability to handle complex relationships, and built-in mechanisms for class imbalance.
    * The `scale_pos_weight` parameter was used during training to give more weight to the minority class (fraud), directly addressing the imbalance.
    * **Threshold Tuning for XGBoost:** Similar to Random Forest, the classification threshold for XGBoost was further optimized to maximize the F1-score, providing the best balance.

### Performance Summary

The table below summarizes the performance metrics across different model iterations:

| Metric        | Initial RF | RF (Tuned Thresh) | XGBoost (Default Thresh) | **XGBoost (Tuned Thresh)** |
| :------------ | :--------- | :---------------- | :----------------------- | :--------------------------- |
| Accuracy      | 0.9400     | 0.9353            | 0.9335                   | **0.9344** |
| Precision     | 0.7700     | 0.6476            | 0.6261                   | **0.6293** |
| **Recall** | 0.5100     | 0.6733            | 0.7129                   | **0.7228** |
| F1-Score      | 0.6200     | 0.6602            | 0.6667                   | **0.6728** |
| ROC AUC Score | 0.9480     | 0.9488            | 0.9495                   | **0.9495** |

**Conclusion:** The **XGBoost Classifier with a tuned threshold of approximately 0.4986** achieved the best overall performance, significantly increasing the **recall to 72.28%** while maintaining a precision of 62.93% and a high F1-Score of 0.6728. This model effectively balances the trade-off, enabling the identification of a large proportion of fraudulent providers for investigation.

## How to Run the Project

To execute this project and generate predictions on unseen data, follow these steps:

### Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

* `pandas` (for data manipulation)
* `numpy` (for numerical operations)
* `scikit-learn` (for machine learning utilities and metrics)
* `xgboost` (for the final model)
* `matplotlib` (for plotting)
* `seaborn` (for enhanced visualizations)
* `joblib` (for saving and loading models)

You can install these libraries using pip:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
