import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Diabetesmodel.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

# Risk level boundaries on the predicted probability of diabetes
MODERATE_RISK = 0.30
HIGH_RISK = 0.60

STATUS_ICON = {"ok": "🟢", "warn": "🟡", "high": "🔴"}

st.set_page_config(page_title="AI Diabetes Risk Assessment", page_icon="🩺")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def glucose_category(value):
    if value < 140:
        return "ok", "Normal"
    if value < 200:
        return "warn", "Pre-diabetic range"
    return "high", "Diabetic range"


def blood_pressure_category(value):
    if value < 60:
        return "warn", "Low"
    if value < 80:
        return "ok", "Normal"
    if value < 90:
        return "warn", "Elevated"
    return "high", "High"


def bmi_category(value):
    if value < 18.5:
        return "warn", "Underweight"
    if value < 25:
        return "ok", "Healthy weight"
    if value < 30:
        return "warn", "Overweight"
    return "high", "Obese"


def age_category(value):
    if value < 35:
        return "ok", "Lower-risk age group"
    if value < 45:
        return "warn", "Screening recommended"
    return "high", "Higher-risk age group"


def show_risk_level(probability):
    if probability >= HIGH_RISK:
        st.error("🔴 **High risk** — you are **likely** to have diabetes.")
        advice = (
            "Please see a doctor soon for a proper diabetes test, such as a "
            "fasting glucose or HbA1c test."
        )
    elif probability >= MODERATE_RISK:
        st.warning("🟡 **Moderate risk** — you **may be at risk** of diabetes.")
        advice = (
            "Consider a check-up with a healthcare professional, and look at "
            "diet, physical activity and weight management."
        )
    else:
        st.success("🟢 **Low risk** — you are **not likely** to have diabetes.")
        advice = "Keep up a healthy lifestyle and continue with routine health checks."

    st.metric("Estimated risk", f"{probability * 100:.0f} %")
    st.progress(float(probability))
    st.write(f"**Suggestion:** {advice}")


def show_value_breakdown(glucose, blood_pressure, bmi, age):
    st.subheader("How your values compare")
    rows = [
        ("Glucose", f"{glucose:g} mg/dL", glucose_category(glucose), "Below 140 mg/dL"),
        (
            "Blood Pressure",
            f"{blood_pressure:g} mm Hg",
            blood_pressure_category(blood_pressure),
            "60 – 79 mm Hg",
        ),
        ("BMI", f"{bmi:.1f} kg/m²", bmi_category(bmi), "18.5 – 24.9 kg/m²"),
        ("Age", f"{age} years", age_category(age), "Risk rises with age"),
    ]
    table = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Your value": value,
                "Category": f"{STATUS_ICON[status]} {label}",
                "Healthy reference": reference,
            }
            for metric, value, (status, label), reference in rows
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Glucose is the 2-hour reading from an oral glucose tolerance test and "
        "blood pressure is the diastolic (lower) reading, matching the data the "
        "model was trained on. The reference ranges are general guidelines; the "
        "risk estimate comes from the model, not from these ranges."
    )


def show_assessment(model):
    st.write(
        "Enter the patient's health metrics below, then press **Predict** to "
        "estimate the likelihood of diabetes."
    )

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.number_input(
                "Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=None,
                placeholder="e.g. 120",
            )
            bmi = st.number_input(
                "BMI (kg/m²)", min_value=0.0, max_value=80.0, value=None,
                step=0.1, format="%.1f", placeholder="e.g. 25.0",
            )
        with col2:
            blood_pressure = st.number_input(
                "Blood Pressure (diastolic, mm Hg)", min_value=0.0, max_value=250.0,
                value=None, placeholder="e.g. 70",
            )
            age = st.number_input(
                "Age (years)", min_value=0, max_value=120, value=None, step=1,
                placeholder="e.g. 30",
            )
        submitted = st.form_submit_button("Predict", type="primary")

    with st.expander("How the risk levels work"):
        st.markdown(
            f"- 🟢 **Low** — estimated risk below {MODERATE_RISK * 100:.0f} %\n"
            f"- 🟡 **Moderate** — {MODERATE_RISK * 100:.0f} % to {HIGH_RISK * 100 - 1:.0f} %\n"
            f"- 🔴 **High** — {HIGH_RISK * 100:.0f} % and above\n\n"
            "After a prediction, a table also shows the category of each value "
            "you entered and its healthy reference range."
        )

    if not submitted:
        return

    inputs = {
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "BMI": bmi,
        "Age": age,
    }
    missing = [name for name, value in inputs.items() if value is None]
    not_positive = [name for name, value in inputs.items() if value is not None and value <= 0]

    if missing:
        st.error(f"Please enter a value for: {', '.join(missing)}.")
        return
    if not_positive:
        st.error(f"{', '.join(not_positive)} must be greater than zero.")
        return

    input_data = np.asarray(list(inputs.values()), dtype=float).reshape(1, -1)
    probability = model.predict_proba(input_data)[0][1]

    show_risk_level(probability)
    show_value_breakdown(glucose, blood_pressure, bmi, age)


def show_model_comparison():
    if not METRICS_PATH.exists():
        st.info("model_metrics.json was not found. Run `python train.py` to create it.")
        return

    summary = load_metrics()
    models = summary["models"]
    deployed = summary["deployed_model"]

    st.write(
        f"Four classification algorithms were trained on the same "
        f"{summary['train_size']} patient records and evaluated on "
        f"{summary['test_size']} records the models had never seen. "
        f"The app uses **{deployed}**, which scored the highest accuracy."
    )

    table = pd.DataFrame(
        [
            {
                "Model": f"{name} ⭐" if name == deployed else name,
                "Accuracy": f"{m['accuracy'] * 100:.1f} %",
                "Precision": f"{m['precision'] * 100:.1f} %",
                "Recall": f"{m['recall'] * 100:.1f} %",
                "F1 score": f"{m['f1'] * 100:.1f} %",
            }
            for name, m in models.items()
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    names = list(models)
    accuracies = [models[name]["accuracy"] * 100 for name in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#ff4b4b" if name == deployed else "#9aa5b1" for name in names]
    bars = ax.bar(names, accuracies, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test accuracy by algorithm")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model details")
    selected = st.selectbox("Choose a model to inspect", names, index=names.index(deployed))
    scores = models[selected]
    matrix = np.array(scores["confusion_matrix"])
    (tn, fp), (fn, tp) = matrix.tolist()
    total = tn + fp + fn + tp

    for column, (label, key) in zip(
        st.columns(4),
        [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1")],
    ):
        column.metric(label, f"{scores[key] * 100:.1f} %")

    matrix_col, explain_col = st.columns(2)
    with matrix_col:
        st.markdown("**Confusion matrix**")
        labels = ["Not diabetic", "Diabetic"]
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with explain_col:
        st.markdown(f"**What these numbers mean for {selected}**")
        st.markdown(
            f"- **Accuracy {scores['accuracy'] * 100:.1f} %** — {tp + tn} of the "
            f"{total} test patients were classified correctly.\n"
            f"- **Precision {scores['precision'] * 100:.1f} %** — of the {tp + fp} "
            f"patients it flagged as diabetic, {tp} really were.\n"
            f"- **Recall {scores['recall'] * 100:.1f} %** — of the {tp + fn} patients "
            f"who really were diabetic, it caught {tp} and missed {fn}.\n"
            f"- **F1 score {scores['f1'] * 100:.1f} %** — a single score that "
            f"balances precision and recall; it is only high when both are."
        )

    st.write(
        f"In the matrix, the diagonal holds the correct predictions: {tn} "
        f"non-diabetic and {tp} diabetic patients. The other two cells are the "
        f"mistakes: {fn} diabetic patients missed and {fp} healthy patients "
        f"wrongly flagged."
    )

    st.subheader("Things to know")
    deployed_matrix = models[deployed]["confusion_matrix"]
    caught = deployed_matrix[1][1]
    diabetic_total = deployed_matrix[1][0] + deployed_matrix[1][1]
    st.markdown(
        f"- **Recall is weak.** {deployed} catches only about half of the truly "
        f"diabetic patients ({caught} of {diabetic_total} in the test set). This is "
        f"a real limitation of the model on these four features.\n"
        "- **Glucose means a 2-hour glucose tolerance test.** That is what the "
        "dataset records, so a fasting glucose reading would be judged against "
        "the wrong range.\n"
        "- **The dataset only contains women aged 21 and over** (Pima Indians "
        "Diabetes dataset), so predictions for other groups are less reliable."
    )
    st.caption(f"Features used: {', '.join(summary['features'])}.")


st.title("AI Diabetes Risk Assessment")

if not MODEL_PATH.exists():
    st.error("Diabetesmodel.pkl was not found. Run `python train.py` first to create it.")
    st.stop()

assessment_tab, comparison_tab = st.tabs(["🩺 Risk Assessment", "📊 Model Comparison"])
with assessment_tab:
    show_assessment(load_model())
with comparison_tab:
    show_model_comparison()

st.caption(
    "This tool is a classroom project for educational purposes only. It is not "
    "a medical diagnosis — please consult a healthcare professional."
)
