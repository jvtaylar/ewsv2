import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

st.set_page_config(
    page_title="Engineering Mathematics Early Warning System",
    page_icon="📊",
    layout="wide"
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}
RISK_CUTOFFS = (0.30, 0.60)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def read_uploaded_file(uploaded_file, sheet_name=None):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)
    raise ValueError("Upload a CSV or Excel (.xlsx/.xls) file.")

def risk_band(probability):
    if probability >= RISK_CUTOFFS[1]:
        return "High"
    if probability >= RISK_CUTOFFS[0]:
        return "Moderate"
    return "Low"

def normalize_target(series):
    """Convert common binary outcome encodings to 0/1; return None if ambiguous."""
    if pd.api.types.is_numeric_dtype(series):
        vals = set(pd.Series(series.dropna()).unique().tolist())
        if vals.issubset({0, 1, 0.0, 1.0}):
            return series.astype(float).astype("Int64")
        return None
    mapping = {
        "1": 1, "at risk": 1, "at_risk": 1, "risk": 1,
        "fail": 1, "failed": 1, "failing": 1, "withdrawn": 1,
        "0": 0, "not at risk": 0, "not_at_risk": 0, "pass": 0,
        "passed": 0, "distinction": 0, "complete": 0, "completed": 0,
    }
    mapped = series.astype(str).str.strip().str.lower().map(mapping)
    if mapped.notna().sum() == series.notna().sum() and mapped.notna().any():
        return mapped.astype("Int64")
    return None

def get_feature_names(model):
    return list(getattr(model, "feature_names_in_", []))

def prepare_features(raw_df, feature_names):
    """Select model features and perform conservative column-name normalization."""
    df = raw_df.copy()
    # Common aliases from LMS exports / institutional files.
    aliases = {
        "Course": "Course_Name", "Course Name": "Course_Name",
        "Subject": "Course_Name", "Term": "Semester",
        "Section/Block": "Section", "Quiz Average": "Quiz_Average",
        "Assignment Average": "Assignment_Average",
        "Preliminary_Score": "Prelim_Score", "Prelim Score": "Prelim_Score",
        "Current Grade": "Current_Grade", "Attendance Rate": "Attendance_Rate",
        "Assignment Submission Rate": "Assignment_Submission_Rate",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = [c for c in feature_names if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    X = df[feature_names].copy()

    # Ensure numeric model inputs remain numeric; categorical inputs remain strings.
    categorical = {"Course_Name", "Semester", "Program"}
    for col in X.columns:
        if col not in categorical:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].where(X[col].notna(), np.nan)
    return X, missing

def score_dataset(raw_df, model_name, threshold):
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)
    feature_names = get_feature_names(model)
    if not feature_names:
        raise ValueError("The selected model does not expose its expected input feature names.")
    X, missing = prepare_features(raw_df, feature_names)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    result = raw_df.copy()
    result["Predicted_Risk_Probability"] = probabilities
    result["Predicted_At_Risk"] = predictions
    result["Risk_Level"] = [risk_band(p) for p in probabilities]
    result["Recommended_Action"] = result["Risk_Level"].map({
        "High": "Priority instructor outreach; review assessment gaps and arrange individualized support.",
        "Moderate": "Schedule an instructor check-in; review quiz results and incomplete requirements.",
        "Low": "Continue routine monitoring and reinforce effective learning practices.",
    })
    return result, missing, X

def find_label_column(df):
    for c in ["At_Risk", "at_risk", "Target", "target", "Final_Status", "Final_Status_Label"]:
        if c in df.columns:
            return c
    return None

st.title("Engineering Mathematics – Student Early Warning System")
st.caption("Research prototype. Predictions are decision-support indicators and must be reviewed by instructors.")

# Load existing research outputs when available.
def load_csv_optional(filename):
    path = os.path.join(BASE, filename)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

metrics = load_csv_optional("model_metrics.csv")
risk_register = load_csv_optional("instructor_risk_register.csv")
importance = load_csv_optional("feature_importance.csv")
shap_global = load_csv_optional("shap_global_importance.csv")

main_tab, upload_tab = st.tabs(["Existing demonstration dashboard", "Upload test dataset / validate model"])

with upload_tab:
    st.header("Upload a test dataset")
    st.write(
        "Upload a CSV or Excel file containing one row per student-course enrollment or prediction record. "
        "The app will align its columns with the selected trained model, generate risk probabilities, "
        "and calculate validation metrics only when actual outcome labels are supplied."
    )
    uploaded = st.file_uploader("Choose test dataset", type=["csv", "xlsx", "xls"], key="test_upload")
    raw = None
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Select worksheet", xls.sheet_names, index=0)
                raw = pd.read_excel(uploaded, sheet_name=sheet)
            else:
                raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(raw):,} records and {len(raw.columns):,} columns.")
            with st.expander("Preview uploaded data"):
                st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")

    if raw is not None and not raw.empty:
        st.subheader("Model settings")
        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            selected_model = st.selectbox("Trained model", list(MODEL_FILES.keys()), key="upload_model")
        with col2:
            threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05,
                                  help="Probability at or above this value is classified as at risk.")
        with col3:
            st.metric("Uploaded records", f"{len(raw):,}")

        label_col = find_label_column(raw)
        possible_labels = ["None – predictions only"] + [c for c in raw.columns if c.lower() in
            {"at_risk", "target", "final_status", "final_status_label", "outcome", "actual_outcome"}]
        chosen_label = st.selectbox(
            "Actual outcome column (optional)", possible_labels,
            index=(possible_labels.index(label_col) if label_col in possible_labels else 0),
            help="Choose a column containing actual outcomes to calculate validation metrics. Do not select a predictor or a model-generated prediction."
        )

        if st.button("Run model on uploaded dataset", type="primary", use_container_width=True):
            try:
                scored, missing_features, X = score_dataset(raw, selected_model, threshold)
                st.session_state["uploaded_scored"] = scored
                st.session_state["uploaded_filename"] = uploaded.name
                st.session_state["uploaded_model"] = selected_model
                st.session_state["uploaded_threshold"] = threshold
                st.session_state["uploaded_missing"] = missing_features
                st.session_state["uploaded_label"] = chosen_label
                st.success(f"Scoring completed using {selected_model}.")
            except Exception as exc:
                st.error(f"Model execution failed: {exc}")

        if ("uploaded_scored" in st.session_state
                and st.session_state.get("uploaded_filename") == uploaded.name):
            scored = st.session_state["uploaded_scored"]
            if len(scored) == len(raw):
                st.subheader("Prediction summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Records scored", f"{len(scored):,}")
                c2.metric("High risk", int((scored.Risk_Level == "High").sum()))
                c3.metric("Moderate risk", int((scored.Risk_Level == "Moderate").sum()))
                c4.metric("Low risk", int((scored.Risk_Level == "Low").sum()))

                if st.session_state.get("uploaded_missing"):
                    missing = st.session_state["uploaded_missing"]
                    st.warning(
                        f"The uploaded file did not contain {len(missing)} model input column(s): "
                        + ", ".join(missing) + ". These were supplied as missing values and handled by the model's fitted preprocessing pipeline. "
                        "Predictions may be less reliable when important inputs are absent."
                    )
                st.caption(
                    f"Model: {st.session_state['uploaded_model']} | "
                    f"At-risk threshold: {st.session_state['uploaded_threshold']:.2f} | "
                    "Risk bands: Low <0.30; Moderate 0.30–<0.60; High ≥0.60."
                )

                dist = scored["Risk_Level"].value_counts().reindex(["High", "Moderate", "Low"], fill_value=0)
                fig = px.bar(x=dist.index, y=dist.values, labels={"x": "Risk level", "y": "Records"},
                             title="Predicted risk distribution", color=dist.index)
                st.plotly_chart(fig, use_container_width=True)

                chosen_label = st.session_state.get("uploaded_label", "None – predictions only")
                if chosen_label != "None – predictions only" and chosen_label in scored.columns:
                    actual = normalize_target(scored[chosen_label])
                    if actual is None:
                        st.error("The selected outcome column could not be reliably converted to binary labels. Use At_Risk coded 0/1, or status values Pass/Passed and Fail/Failed/Withdrawn.")
                    else:
                        valid = actual.notna()
                        y_true = actual[valid].astype(int)
                        y_pred = scored.loc[valid, "Predicted_At_Risk"].astype(int)
                        y_prob = scored.loc[valid, "Predicted_Risk_Probability"].astype(float)
                        if y_true.nunique() < 2:
                            st.warning("Validation metrics requiring both outcome classes cannot be calculated because the selected records contain only one actual class.")
                        else:
                            st.subheader("Validation results against actual outcomes")
                            vals = [
                                ("Accuracy", accuracy_score(y_true, y_pred)),
                                ("Precision", precision_score(y_true, y_pred, zero_division=0)),
                                ("Recall", recall_score(y_true, y_pred, zero_division=0)),
                                ("F1-score", f1_score(y_true, y_pred, zero_division=0)),
                                ("ROC-AUC", roc_auc_score(y_true, y_prob)),
                                ("PR-AUC", average_precision_score(y_true, y_prob)),
                            ]
                            cols = st.columns(6)
                            for c, (label, value) in zip(cols, vals):
                                c.metric(label, f"{value:.3f}")
                            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                            cm_fig = px.imshow(cm, text_auto=True, x=["Predicted not at risk", "Predicted at risk"],
                                               y=["Actual not at risk", "Actual at risk"],
                                               labels=dict(x="Prediction", y="Actual", color="Count"),
                                               title="Confusion matrix")
                            st.plotly_chart(cm_fig, use_container_width=True)
                            fpr, tpr, _ = roc_curve(y_true, y_prob)
                            roc_fig = go.Figure()
                            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
                            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
                            roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
                            st.plotly_chart(roc_fig, use_container_width=True)
                            with st.expander("Detailed classification report"):
                                st.code(classification_report(y_true, y_pred, target_names=["Not at risk", "At risk"], zero_division=0))
                else:
                    st.info("No actual outcome label selected. These are prediction results only; accuracy, recall, F1-score, and ROC-AUC cannot be calculated without observed outcomes.")

                st.subheader("Student-level predictions")
                preferred = [c for c in ["Student_ID", "Course_Code", "Course_Name", "Section", "Semester", "Academic_Year"] if c in scored.columns]
                show_cols = preferred + ["Predicted_Risk_Probability", "Predicted_At_Risk", "Risk_Level", "Recommended_Action"]
                extra = [c for c in ["Quiz_Average", "Assignment_Average", "Current_Grade", "Attendance_Rate"] if c in scored.columns]
                show_cols += extra
                display = scored[show_cols].sort_values("Predicted_Risk_Probability", ascending=False)
                st.dataframe(display.style.format({"Predicted_Risk_Probability": "{:.1%}"}),
                             use_container_width=True, hide_index=True)
                csv_bytes = scored.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Download scored test dataset (CSV)", csv_bytes,
                                   file_name="early_warning_test_predictions.csv", mime="text/csv")
                xlsx_buffer = io.BytesIO()
                with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                    scored.to_excel(writer, sheet_name="Predictions", index=False)
                    pd.DataFrame({"Missing_Model_Feature": st.session_state.get("uploaded_missing", [])}).to_excel(
                        writer, sheet_name="Missing_Features", index=False)
                st.download_button("Download scored test dataset (Excel)", xlsx_buffer.getvalue(),
                                   file_name="early_warning_test_predictions.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with main_tab:
    st.header("Existing synthetic demonstration dashboard")
    if risk_register.empty:
        st.info("The demonstration risk register is not present in this project folder. Use the upload tab to score a test dataset.")
    else:
        with st.sidebar:
            st.header("Demonstration dashboard filters")
            courses = ["All courses"] + sorted(risk_register["Course_Name"].dropna().unique().tolist())
            course = st.selectbox("Course", courses)
            terms = ["All terms"] + sorted(risk_register["Semester"].dropna().unique().tolist())
            term = st.selectbox("Semester", terms)
            risk_filter = st.multiselect("Risk category", ["High", "Moderate", "Low"], default=["High", "Moderate", "Low"])
            model_name = st.selectbox("Feature-importance model", metrics["Model"].tolist() if not metrics.empty else list(MODEL_FILES))
        view = risk_register.copy()
        if course != "All courses":
            view = view[view.Course_Name == course]
        if term != "All terms":
            view = view[view.Semester == term]
        view = view[view.Risk_Level.isin(risk_filter)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Student-course records", f"{len(view):,}")
        c2.metric("High risk", int((view.Risk_Level == "High").sum()))
        c3.metric("Moderate risk", int((view.Risk_Level == "Moderate").sum()))
        c4.metric("Low risk", int((view.Risk_Level == "Low").sum()))
        dist = view["Risk_Level"].value_counts().reindex(["High", "Moderate", "Low"], fill_value=0).rename_axis("Risk level").reset_index(name="Students")
        st.plotly_chart(px.bar(dist, x="Risk level", y="Students", color="Risk level", category_orders={"Risk level": ["High", "Moderate", "Low"]}), use_container_width=True)
        if not metrics.empty:
            st.subheader("Synthetic holdout model performance")
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            st.caption("These are results from the synthetic-data experiment, not validation metrics for uploaded institutional records.")
        display_cols = [c for c in ["Student_ID", "Course_Name", "Semester", "Week_Number", "Predicted_Risk_Probability", "Risk_Level", "Quiz_Average", "Assignment_Average", "Attendance_Rate", "Activity_Completion_Rate", "Recommended_Action"] if c in view.columns]
        st.subheader("Students requiring attention")
        st.dataframe(view[display_cols].sort_values("Predicted_Risk_Probability", ascending=False).style.format({"Predicted_Risk_Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
        if not importance.empty:
            st.subheader("Predictor importance")
            imp = importance[importance.Model == model_name].nlargest(15, "Importance").sort_values("Importance")
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title=f"Top predictors – {model_name}"), use_container_width=True)
        if not shap_global.empty:
            shap_models = sorted(shap_global.Model.unique())
            shap_model = st.selectbox("SHAP model", shap_models)
            si = shap_global[shap_global.Model == shap_model].nlargest(15, "Mean_Absolute_SHAP").sort_values("Mean_Absolute_SHAP")
            st.plotly_chart(px.bar(si, x="Mean_Absolute_SHAP", y="Feature", orientation="h", title=f"Mean absolute SHAP values – {shap_model}"), use_container_width=True)

st.warning(
    "Research-use limitation: the original models were trained on synthetic data. Uploaded-record predictions are exploratory until "
    "validated against actual outcomes from an appropriate institutional cohort. Missing predictors are imputed by the fitted pipeline, "
    "and risk bands are provisional. Do not use automated predictions as the sole basis for academic decisions."
)
