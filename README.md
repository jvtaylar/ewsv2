# Engineering Mathematics – Student Early Warning System

## New upload and validation capability
The Streamlit app now accepts CSV and Excel test datasets. In the **Upload test dataset / validate model** tab, users can:

1. Upload a test dataset and select the worksheet when using Excel.
2. Select Logistic Regression, Random Forest, or XGBoost.
3. Set a classification threshold.
4. Optionally select an actual outcome column (`At_Risk`, `Target`, or a final-status field).
5. Generate risk probabilities, binary predictions, risk bands, and recommended actions.
6. If actual labels are supplied, calculate accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, a confusion matrix, and a classification report.
7. Download scored results in CSV or Excel format.

If no actual outcome labels are supplied, the app displays predictions only. It does not report predictive accuracy without ground-truth outcomes.

## Run locally
1. Use Python 3.10 or later.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

The `models/` directory must contain the three fitted pipeline files:
- `logistic_regression.joblib`
- `random_forest.joblib`
- `xgboost.joblib`

## Input format
The model was trained with these 25 input features:
`Course_Name`, `Semester`, `Program`, `Quiz_Average`, `Assignment_Average`, `Prelim_Score`, `Current_Grade`, `Assessment_Completion_Rate`, `Login_Frequency`, `Total_LMS_Time_Minutes`, `Learning_Material_Access`, `Video_Engagement_Time_Minutes`, `Assignment_Submission_Rate`, `Late_Submission_Count`, `Activity_Completion_Rate`, `Discussion_Forum_Activity`, `Total_Classes_To_Date`, `Classes_Attended_To_Date`, `Absence_Count_To_Date`, `Attendance_Rate`, `Tardiness_Count_To_Date`, `Quiz_Performance_Trend`, `Attendance_Trend`, `Academic_Performance_Decline`, and `Learning_Activity_Gap_Days`.

Columns absent from the uploaded file are supplied as missing values and processed by the model's fitted imputation pipeline. This preserves technical compatibility but can reduce reliability. Review the missing-feature warning and prefer datasets with the complete feature set.

## Outcome labels
- Binary `At_Risk`: 1 = at risk / unsuccessful outcome; 0 = not at risk / successful outcome.
- Text outcomes: Fail, Failed, Failing, or Withdrawn map to 1; Pass, Passed, or Distinction map to 0.
- Do not use a model-generated prediction as the actual outcome label.

## Research limitations
The fitted models were trained on synthetic data. Predictions on institutional records are exploratory and are not evidence of validated institutional accuracy. Validation metrics are meaningful only when the uploaded file includes observed outcomes from the same records. A separate institutional training/recalibration study is recommended before operational use. Risk bands are provisional and should be calibrated with actual data and instructor input.
