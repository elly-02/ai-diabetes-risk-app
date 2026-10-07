# 🩺 AI Diabetes Risk Assessment

A Streamlit web app that uses a machine learning model to predict whether a person is likely to have diabetes, based on four health metrics.

Built for the **Rocheston RCAI Classroom Project (Module 40 – AI Driven Healthcare)**.

## ✨ Features

- 📝 Four simple inputs: **Glucose**, **Blood Pressure**, **BMI** and **Age**
- 🤖 Prediction from a pre-trained K-Nearest Neighbors model (`Diabetesmodel.pkl`)
- 🚦 Three risk levels (**Low**, **Moderate**, **High**) shown on a result card with an estimated risk percentage, a segmented risk bar and a suggestion for each
- 🔍 A breakdown of how each entered value compares with healthy reference ranges
- 📊 A **Model Comparison** tab with a metrics table and accuracy chart for four algorithms, plus a model picker that shows each model's confusion matrix and explains its accuracy, precision, recall and F1 score
- 🛡️ Friendly error messages for empty or invalid inputs
- 🎮 A pixel-art sunset design with a retro game feel

## 🎮 Design

The app uses a pixel-art theme inspired by retro game title screens:

- 🌅 A sunset sky in hard colour bands (purple, magenta and deep indigo) with a dotted dither pattern
- ☀️ A hero banner with a striped pixel sun, clouds and stars, drawn entirely in CSS
- 🔤 Pixel fonts: **Press Start 2P** for headings and buttons, **VT323** for body text
- 🟨 Square panels, thick borders and hard drop shadows on the form, tables, tabs and metric cards
- 📊 Charts recoloured to match the palette

| Colour | Hex | Used for |
|---|---|---|
| 🟣 Sky purple | `#8f6fe3` | Background sky |
| 💗 Magenta | `#e326d3` | Background horizon, dither dots |
| 🔵 Deep indigo | `#2b1d73` | Main panel |
| 🌑 Night | `#1b1150` | Cards, form and shadows |
| 🟡 Sun yellow | `#ffc400` | Buttons, highlights, moderate risk |
| 🌸 Cloud pink | `#ff8fb0` | Borders, labels, chart bars |
| 🟢 Mint | `#5cf2a6` | Low risk |
| 🔴 Coral red | `#ff5c7a` | High risk |

> 💡 The pixel fonts are loaded from Google Fonts, so they need an internet connection. Offline, the app falls back to a plain monospace font.

## 🧠 The Model

| | |
|---|---|
| 📂 Dataset | Pima Indians Diabetes dataset (768 patient records) |
| 🔢 Features | Glucose, Blood Pressure, BMI, Age |
| ⚙️ Algorithm | K-Nearest Neighbors (24 neighbours) with MinMax scaling |
| 🎯 Test accuracy | ~74 % (80/20 train/test split) |

### 📊 Model comparison

`train.py` trains the four algorithms from the notebook on the same data. KNN scored the highest accuracy, so it is the model the app uses.

| Model | Accuracy | Precision | Recall | F1 score |
|---|---|---|---|---|
| Logistic Regression | 73.4 % | 65.1 % | 51.9 % | 57.7 % |
| **KNN** ⭐ | **74.0 %** | 66.7 % | 51.9 % | 58.3 % |
| Random Forest | 72.7 % | 63.0 % | 53.7 % | 58.0 % |
| Decision Tree | 66.2 % | 52.1 % | 46.3 % | 49.0 % |

The training steps follow the notebook from the Kaggle model [gsaha123/diabetes-risk-assessment](https://www.kaggle.com/models/gsaha123/diabetes-risk-assessment): zero values in the medical columns are treated as missing and replaced with the column mean, then the features are scaled to the 0–1 range.

Two changes were made from the original notebook:

- 🔄 The notebook trains on Glucose, *Insulin*, BMI and Age. This project uses **Blood Pressure** instead of Insulin to match the project brief.
- 📦 The scaler is saved together with the model in one pipeline, so the app can pass raw values straight to the model.

## 📁 Project Structure

```
ai-diabetes-risk-app/
├── app.py                 # 🖥️ Streamlit frontend
├── style.css              # 🎨 Pixel-art theme (fonts, colours, hero banner)
├── .streamlit/
│   └── config.toml        # 🌈 Streamlit theme colours
├── train.py               # 🏋️ Trains the models and saves Diabetesmodel.pkl
├── Diabetesmodel.pkl      # 🤖 Trained model (scaler + KNN)
├── model_metrics.json     # 📊 Test metrics of the four models
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

`Diabetesmodel.pkl` and `model_metrics.json` are already included. To rebuild them from the dataset:

```bash
python train.py
```

This prints the test metrics of all four models and overwrites both files.

## 🧪 Example

| Glucose | Blood Pressure | BMI | Age | Result |
|---|---|---|---|---|
| 120 | 70 | 25.0 | 30 | 🟢 Low risk (17 %) |
| 150 | 80 | 27.0 | 40 | 🟡 Moderate risk (54 %) |
| 180 | 90 | 38.0 | 55 | 🔴 High risk (88 %) |

Risk levels: 🟢 Low is below 30 %, 🟡 Moderate is 30 – 59 %, 🔴 High is 60 % and above.

## 🔍 Result Explanation

After a prediction, a table shows each value entered, its category (for example "Pre-diabetic range" or "Overweight") and the healthy reference range.

| Metric | Healthy reference |
|---|---|
| Glucose | Below 140 mg/dL |
| Blood Pressure | 60 – 79 mm Hg (diastolic) |
| BMI | 18.5 – 24.9 kg/m² |
| Age | Risk rises with age |

## 📌 Things to Know

- 📉 **Recall is weak.** KNN catches only about half of the truly diabetic patients (28 of 54 in the test set). This is a real limitation of the model on these four features.
- 🧪 **Glucose means a 2-hour glucose tolerance test**, because that is what the dataset records. A fasting glucose reading would be judged against the wrong range.
- 👩 **The dataset only contains women aged 21 and over**, so predictions for other groups are less reliable.

## 🛠️ Tech Stack

🐍 Python · 🎈 Streamlit · 🔬 scikit-learn · 🐼 pandas · 🔢 NumPy · 📈 Matplotlib · 🎨 CSS

## ⚠️ Disclaimer

This app is a classroom project for **educational purposes only**. It is not a medical diagnosis. Please consult a healthcare professional for any health concerns.

## 🙏 Credits

- Model notebook and dataset: [gsaha123/diabetes-risk-assessment](https://www.kaggle.com/models/gsaha123/diabetes-risk-assessment) on Kaggle (Apache 2.0)
- Project brief: Rocheston RCAI – Module 40
