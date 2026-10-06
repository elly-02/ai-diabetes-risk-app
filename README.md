# 🩺 AI Diabetes Risk Assessment

A Streamlit web app that uses a machine learning model to predict whether a person is likely to have diabetes, based on four health metrics.

Built for the **Rocheston RCAI Classroom Project (Module 40 – AI Driven Healthcare)**.

## ✨ Features

- 📝 Four simple inputs: **Glucose**, **Blood Pressure**, **BMI** and **Age**
- 🤖 Prediction from a pre-trained K-Nearest Neighbors model (`Diabetesmodel.pkl`)
- 📊 Clear result ("likely" / "not likely" to have diabetes) with an estimated risk percentage
- 🛡️ Friendly error messages for empty or invalid inputs

## 🧠 The Model

| | |
|---|---|
| 📂 Dataset | Pima Indians Diabetes dataset (768 patient records) |
| 🔢 Features | Glucose, Blood Pressure, BMI, Age |
| ⚙️ Algorithm | K-Nearest Neighbors (24 neighbours) with MinMax scaling |
| 🎯 Test accuracy | ~74 % (80/20 train/test split) |

The training steps follow the notebook from the Kaggle model [gsaha123/diabetes-risk-assessment](https://www.kaggle.com/models/gsaha123/diabetes-risk-assessment): zero values in the medical columns are treated as missing and replaced with the column mean, then the features are scaled to the 0–1 range.

Two changes were made from the original notebook:

- 🔄 The notebook trains on Glucose, *Insulin*, BMI and Age. This project uses **Blood Pressure** instead of Insulin to match the project brief.
- 📦 The scaler is saved together with the model in one pipeline, so the app can pass raw values straight to the model.

## 📁 Project Structure

```
ai-diabetes-risk-app/
├── app.py                 # 🖥️ Streamlit frontend
├── train.py               # 🏋️ Trains the model and saves Diabetesmodel.pkl
├── Diabetesmodel.pkl      # 🤖 Trained model (scaler + KNN)
├── requirements.txt       # 📦 Python dependencies
└── diabetes-risk-assessment-scikitlearn-2gd-v1/
    ├── Diabetes_prediction (1).ipynb   # 📓 Original Kaggle notebook
    └── diabetes.csv                    # 📂 Dataset
```

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/elly-02/ai-diabetes-risk-app.git
cd ai-diabetes-risk-app
```

**2. Create a virtual environment and install the dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run the app**

```bash
streamlit run app.py
```

Then open the URL that Streamlit prints (usually `http://localhost:8501`). 🌐

> 💡 On WSL, if `localhost` does not open from Windows, use the **Network URL** that Streamlit prints instead.

## 🏋️ Retraining the Model (optional)

`Diabetesmodel.pkl` is already included. To rebuild it from the dataset:

```bash
python train.py
```

This prints the test accuracy and overwrites `Diabetesmodel.pkl`.

## 🧪 Example

| Glucose | Blood Pressure | BMI | Age | Result |
|---|---|---|---|---|
| 120 | 70 | 25.0 | 30 | ✅ Not likely to have diabetes (17 % risk) |
| 180 | 90 | 38.0 | 55 | ⚠️ Likely to have diabetes (88 % risk) |

## 🛠️ Tech Stack

🐍 Python · 🎈 Streamlit · 🔬 scikit-learn · 🐼 pandas · 🔢 NumPy

## ⚠️ Disclaimer

This app is a classroom project for **educational purposes only**. It is not a medical diagnosis. Please consult a healthcare professional for any health concerns.

## 🙏 Credits

- Model notebook and dataset: [gsaha123/diabetes-risk-assessment](https://www.kaggle.com/models/gsaha123/diabetes-risk-assessment) on Kaggle (Apache 2.0)
- Project brief: Rocheston RCAI – Module 40
