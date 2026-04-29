import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Academic Risk Predictor — Nepal Class 9",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("data/rf_model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_model()
model = artifacts["model"]
FEATURE_COLS = artifacts["feature_cols"]
risk_labels = artifacts["risk_labels"]

RISK_COLORS = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}

st.title("📚 Academic Risk Prediction System")
st.markdown("**Early Warning System for Class 9 Students — Nepal Secondary Schools**")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Model:** Random Forest (ensemble)  \n"
        "**Features:** 14 engineered from EMIS + questionnaire  \n"
        "**Classes:** Low / Medium / High Risk  \n"
        "**Method:** SMOTE + 5-fold CV  "
    )
    st.markdown("---")
    st.markdown("*For educator use only. Results support, not replace, professional judgment.*")

tab1, tab2, tab3 = st.tabs(["🎯 Single Student Prediction", "📊 Batch Prediction (CSV)", "📈 Model Insights"])

# ─── TAB 1: SINGLE STUDENT ───────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Student Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📖 Academic & Attendance**")
        attendance = st.slider("Attendance Percentage (%)", 0.0, 100.0, 75.0, 0.5)
        st.markdown("**📝 Study Habits**")
        study_hours = st.selectbox("Daily Study Hours", options=[0,1,2,3,4],
            format_func=lambda x: ["<30 min","30–60 min","1–2 hrs","2–3 hrs",">3 hrs"][x])
        assignment = st.selectbox("Assignment Completion", options=[0,1,2,3,4],
            format_func=lambda x: ["Never","Rarely","Sometimes","Usually","Always"][x])
        tutoring = st.radio("Private Tutoring", options=[0,1],
            format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

    with col2:
        st.markdown("**🌙 Lifestyle**")
        sleep = st.selectbox("Sleep Hours (school nights)", options=[0,1,2,3],
            format_func=lambda x: ["<5 hrs","5–6 hrs","7–8 hrs",">8 hrs"][x], index=2)
        mobile = st.selectbox("Daily Mobile Usage", options=[0,1,2,3,4],
            format_func=lambda x: ["<1 hr","1–2 hrs","2–4 hrs","4–6 hrs",">6 hrs"][x])
        st.markdown("**🧠 Psychological**")
        motivation = st.selectbox("Motivation Level", options=[0,1,2,3,4],
            format_func=lambda x: ["None","Slight","Somewhat","Motivated","Very High"][x], index=2)
        stress = st.selectbox("Stress Level", options=[0,1,2,3,4],
            format_func=lambda x: ["Never","Rarely","Sometimes","Often","Almost Always"][x])

    with col3:
        st.markdown("**🏠 Environment & Background**")
        study_env = st.selectbox("Study Environment", options=[1,2,3,4],
            format_func=lambda x: {1:"No proper space",2:"Shared common",3:"Shared room",4:"Own quiet room"}[x])
        parental_edu = st.selectbox("Parental Education", options=[0,1,2,3,4],
            format_func=lambda x: ["None","Primary","Lower Sec","Secondary","Higher+"][x])
        family_income = st.selectbox("Family Monthly Income", options=[0,1,2,3],
            format_func=lambda x: ["<NPR 10k","NPR 10k–25k","NPR 25k–50k",">NPR 50k"][x])
        st.markdown("**👥 Social & Health**")
        peer = st.selectbox("Peer Influence", options=[1,2,3,4,5],
            format_func=lambda x: {1:"Very Negative",2:"Mostly Distract",3:"Neutral",
                                    4:"Mostly Encourage",5:"Very Positive"}[x], index=2)
        health = st.selectbox("Physical Health", options=[0,1,2,3,4],
            format_func=lambda x: ["Very Poor","Poor","Fair","Good","Excellent"][x], index=3)
        chronic = st.selectbox("Chronic Illness Impact", options=[0,1,2],
            format_func=lambda x: ["None","Occasional","Regular"][x])

    if st.button("🔍 Predict Academic Risk", type="primary"):
        input_data = np.array([[attendance, study_hours, assignment, tutoring,
                                 sleep, mobile, motivation, stress,
                                 study_env, parental_edu, family_income,
                                 peer, health, chronic]])
        pred_class = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        label = risk_labels[pred_class]
        color = RISK_COLORS[label]

        st.markdown("---")
        st.subheader("Prediction Result")

        result_col, prob_col = st.columns([1, 2])
        with result_col:
            st.markdown(
                f"<div style='background-color:{color};padding:20px;border-radius:10px;text-align:center;'>"
                f"<h2 style='color:white;margin:0;'>{label}</h2>"
                f"<p style='color:white;font-size:16px;margin:5px 0 0;'>"
                f"Confidence: {pred_proba[pred_class]*100:.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True
            )

        with prob_col:
            fig, ax = plt.subplots(figsize=(8, 3))
            bar_cols = [RISK_COLORS[l] for l in risk_labels]
            bars = ax.barh(risk_labels, pred_proba * 100, color=bar_cols, edgecolor="black", linewidth=0.7)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Class Probabilities")
            ax.set_xlim(0, 110)
            for bar, val in zip(bars, pred_proba * 100):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**💡 Suggested Actions:**")
        if pred_class == 2:
            st.error("🚨 Urgent intervention recommended. Alert class teacher and counsellor. "
                     "Review attendance records. Consider parental meeting.")
        elif pred_class == 1:
            st.warning("⚠️ Monitor closely. Encourage study habit improvement. "
                       "Check mobile usage and sleep patterns.")
        else:
            st.success("✅ Student appears on track. Continue regular monitoring.")

# ─── TAB 2: BATCH PREDICTION ─────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown(f"Upload a CSV with columns: `{', '.join(FEATURE_COLS)}`")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        missing_cols = [c for c in FEATURE_COLS if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            X_batch = batch_df[FEATURE_COLS].fillna(batch_df[FEATURE_COLS].median())
            preds = model.predict(X_batch)
            probas = model.predict_proba(X_batch)
            batch_df["Predicted_Risk"] = [risk_labels[p] for p in preds]
            batch_df["Low_Risk_Prob"]    = probas[:, 0].round(3)
            batch_df["Medium_Risk_Prob"] = probas[:, 1].round(3)
            batch_df["High_Risk_Prob"]   = probas[:, 2].round(3)

            st.success(f"Predicted {len(batch_df)} students.")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Low Risk",    (batch_df["Predicted_Risk"]=="Low Risk").sum())
            col_b.metric("Medium Risk", (batch_df["Predicted_Risk"]=="Medium Risk").sum())
            col_c.metric("High Risk",   (batch_df["Predicted_Risk"]=="High Risk").sum())
            st.dataframe(batch_df)

            csv_out = batch_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv_out, "risk_predictions.csv", "text/csv")
    else:
        template = pd.DataFrame(columns=FEATURE_COLS)
        st.download_button("⬇️ Download Template CSV", template.to_csv(index=False),
                           "template.csv", "text/csv")

# ─── TAB 3: MODEL INSIGHTS ───────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Importance & Model Information")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_names_clean = [f.replace("_", " ") for f in FEATURE_COLS]
    sorted_names = [feat_names_clean[i] for i in indices]
    sorted_imp   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(sorted_names)), sorted_imp[::-1], color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(list(reversed(sorted_names)), fontsize=11)
    ax.set_xlabel("Mean Decrease in Gini Impurity", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**Model Parameters:**")
    params = model.get_params()
    param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
    st.dataframe(param_df, use_container_width=True)

    for img_name, title in [
        ("data/confusion_matrices.png", "Confusion Matrices — All Models"),
        ("data/smote_comparison.png", "SMOTE Class Balancing"),
        ("data/model_comparison_chart.png", "Model Performance Comparison"),
        ("data/shap_importance_high_risk.png", "SHAP — High Risk Feature Importance"),
    ]:
        if os.path.exists(img_name):
            st.markdown(f"**{title}**")
            st.image(img_name)
