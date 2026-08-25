
from pathlib import Path
import json
import warnings
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL TENSORFLOW IMPORT
# ============================================================

TF_AVAILABLE = False
tf = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PCOS AI | Smart Women's Health Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS - ROBUST FOR ROOT OR SRC/app.py
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Your earlier project used parent.parent. This version checks both
# the current folder and the project root automatically.
CANDIDATE_ROOTS = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
    CURRENT_DIR.parent.parent,
]

PROJECT_ROOT = CURRENT_DIR

for candidate in CANDIDATE_ROOTS:
    if (candidate / "models").exists() or (candidate / "data").exists():
        PROJECT_ROOT = candidate
        break

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

CSV_PATH = DATA_DIR / "PCOS_extended_dataset.csv"

ML_MODEL_PATH = MODEL_DIR / "pcos_ml_model.joblib"
FEATURES_PATH = MODEL_DIR / "pcos_features.joblib"
INFO_PATH = MODEL_DIR / "model_info.joblib"

CNN_MODEL_PATH = MODEL_DIR / "pcos_ultrasound_cnn.keras"
CLASS_PATH = MODEL_DIR / "ultrasound_classes.json"


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(244,114,182,.13), transparent 25%),
            radial-gradient(circle at 92% 8%, rgba(167,139,250,.13), transparent 25%),
            linear-gradient(135deg, #fff8fc 0%, #f8f7ff 50%, #f4fbff 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #20102f 0%, #351448 52%, #211633 100%);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        border-radius: 12px;
        padding: 7px 10px;
        margin: 2px 0;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.6px;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(82,35,104,.98), rgba(184,82,144,.92));
        color: white;
        box-shadow: 0 18px 45px rgba(75, 32, 94, .20);
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 2.25rem;
        margin: 0;
    }

    .hero p {
        color: #fceef8;
        font-size: 1.02rem;
        margin: 8px 0 0 0;
    }

    .eyebrow {
        color: #ffd6ef;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: .76rem;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .glass-card {
        background: rgba(255,255,255,.76);
        border: 1px solid rgba(255,255,255,.95);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(57, 31, 69, .08);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }

    .glass-card h3 {
        margin-top: 0;
        color: #3b174b;
    }

    .soft-card {
        background: linear-gradient(135deg, #ffffff, #fff5fb);
        border: 1px solid #f0d9e8;
        border-radius: 20px;
        padding: 20px;
        min-height: 150px;
        box-shadow: 0 10px 24px rgba(74, 37, 84, .06);
    }

    .soft-card h3 {
        color: #542264;
        margin: 0 0 8px 0;
    }

    .big-number {
        font-size: 2rem;
        font-weight: 800;
        color: #64266f;
    }

    .pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: #f6e5f1;
        color: #65225f;
        font-size: .78rem;
        font-weight: 700;
        margin: 3px 4px 3px 0;
    }

    .chat-user {
        background: #eadcf8;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 15px;
        margin: 8px 0 8px 18%;
    }

    .chat-bot {
        background: #ffffff;
        border: 1px solid #eadfea;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 15px;
        margin: 8px 18% 8px 0;
        box-shadow: 0 5px 15px rgba(65, 35, 75, .05);
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #41174f;
        margin: 8px 0 14px 0;
    }

    .disclaimer {
        padding: 15px 18px;
        border-radius: 15px;
        background: #fff8e8;
        border: 1px solid #f4dfaa;
        color: #654d1e;
        font-size: .88rem;
    }

    .footer {
        text-align: center;
        color: #806b83;
        padding: 25px 0 10px;
        font-size: .82rem;
    }

    div.stButton > button {
        border-radius: 13px;
        font-weight: 700;
        border: 0;
        padding: 10px 18px;
    }

    .stProgress > div > div > div > div {
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def hero(title, subtitle, eyebrow="PCOS AI • SMART HEALTH SCREENING"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title, body, icon="🌸"):
    st.markdown(
        f"""
        <div class="soft-card">
            <h3>{icon} {title}</h3>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


def get_risk_level(probability):
    if probability >= 0.70:
        return "Higher Risk", "🔴"
    if probability >= 0.50:
        return "Moderate Risk", "🟠"
    return "Lower Risk", "🟢"


def wellness_recommendations(risk, features):
    # General educational wellness suggestions only.

    fast_food = bool(features.get("Fast food (Y/N)", 0))
    regular_exercise = bool(features.get("Reg.Exercise(Y/N)", 0))
    weight_gain = bool(features.get("Weight gain(Y/N)", 0))

    foods = [
        "More vegetables, salads and high-fibre foods",
        "Protein sources such as eggs, dal, beans and paneer",
        "Whole grains such as oats, brown rice and whole-wheat foods",
        "Nuts and seeds in moderate portions",
        "Whole fruits instead of sugary drinks or packaged sweets",
    ]

    limit = [
        "Sugary drinks and excess added sugar",
        "Frequent fried and highly processed fast food",
        "Large portions of sweets and refined snacks",
    ]

    exercise_plan = [
        "20–30 minutes of comfortable walking most days",
        "Light-to-moderate strength training 2–3 times per week",
        "Gentle stretching, yoga or relaxation exercises",
        "Keep a regular sleep schedule and aim for good sleep",
    ]

    if fast_food:
        limit.insert(
            0,
            "Try reducing frequent fast-food meals"
        )

    if not regular_exercise:
        exercise_plan.insert(
            0,
            "Start slowly with short daily walks"
        )

    if weight_gain:
        exercise_plan.append(
            "Focus on consistent healthy habits rather than crash dieting"
        )

    if risk == "Higher Risk":
        note = (
            "Because the model shows a higher screening risk, "
            "consider discussing the result with a qualified "
            "healthcare professional."
        )

    elif risk == "Moderate Risk":
        note = (
            "The result is in a moderate screening range. "
            "Track symptoms and consider professional medical advice."
        )

    else:
        note = (
            "The model shows a lower screening probability, "
            "but symptoms should still be discussed with a "
            "healthcare professional when needed."
        )

    return foods, limit, exercise_plan, note

# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_ml_model():
    if not ML_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(ML_MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def load_ml_features():
    if not FEATURES_PATH.exists():
        return []
    try:
        return joblib.load(FEATURES_PATH)
    except Exception:
        return []


@st.cache_data
def load_ml_info():
    if not INFO_PATH.exists():
        return {}
    try:
        return joblib.load(INFO_PATH)
    except Exception:
        return {}


@st.cache_data
def load_dataset():
    if not CSV_PATH.exists():
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception:
        return None


@st.cache_resource
def load_cnn_model():
    if not TF_AVAILABLE or not CNN_MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def load_class_names():
    if CLASS_PATH.exists():
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                # Common Keras class-index format
                ordered = [None] * len(data)
                for name, idx in data.items():
                    if isinstance(idx, int) and idx < len(ordered):
                        ordered[idx] = str(name)
                if all(x is not None for x in ordered):
                    return ordered
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    return ["infected", "noninfected"]


ml_model = load_ml_model()
ml_features = load_ml_features()
ml_info = load_ml_info()
dataset = load_dataset()
cnn_model = load_cnn_model()
class_names = load_class_names()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "ml_result": None,
    "cnn_result": None,
    "health_report": None,
    "chat_history": [],
    "lmp_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌸 PCOS AI")
    st.caption("Smart screening • wellness • education")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "ML Risk Assessment",
            "Ultrasound DL Analysis",
            "Health Report",
            "LMP & Wellness",
            "AI Patient Chat",
            "AI Summary",
            "Model Information",
            "About",
        ],
    )

    st.divider()
    st.markdown("### SYSTEM STATUS")

    if ml_model is not None:
        st.success("✓ ML Model Loaded")
    else:
        st.error("✕ ML Model Missing")

    if TF_AVAILABLE and cnn_model is not None:
        st.success("✓ CNN Model Loaded")
    elif not TF_AVAILABLE:
        st.warning("! TensorFlow unavailable")
    else:
        st.warning("! CNN Model Missing")

    st.caption("ML • Random Forest")
    st.caption("DL • CNN")
    if TF_AVAILABLE:
        st.caption(f"TensorFlow • {tf.__version__}")

    st.divider()
    st.caption("Academic project only. Not a medical diagnosis.")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    hero(
        "PCOS AI Smart Health Dashboard",
        "One place for clinical risk screening, ultrasound analysis, wellness guidance and patient Q&A.",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="glass-card"><div class="pill">ML</div><div class="big-number">Random Forest</div><div>Clinical risk screening</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="glass-card"><div class="pill">DL</div><div class="big-number">CNN</div><div>Ultrasound analysis</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="glass-card"><div class="pill">DATA</div><div class="big-number">{len(ml_features)}</div><div>ML features</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="glass-card"><div class="pill">APP</div><div class="big-number">9 Modules</div><div>Interactive dashboard</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">✨ What can you do here?</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("PCOS Risk", "Enter patient information and calculate the model probability.", "🧠")
    with c2:
        card("Ultrasound AI", "Upload an image and run the trained CNN analysis.", "🩻")
    with c3:
        card("Health Report", "Get general food, activity and wellness suggestions based on the screening result.", "💗")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("LMP Predictor", "Estimate the next period date from the last menstrual period and cycle length.", "📅")
    with c2:
        card("Patient Chat", "Ask simple PCOS questions using the built-in educational assistant.", "💬")
    with c3:
        card("AI Summary", "View the ML and ultrasound results together without mixing their meanings.", "📊")

    st.markdown('<div class="section-title">🔬 Project pipeline</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    for col, number, title, desc in [
        (p1, "01", "Patient Data", "Clinical & symptom inputs"),
        (p2, "02", "ML Screening", "Random Forest probability"),
        (p3, "03", "Ultrasound", "CNN image classification"),
        (p4, "04", "Wellness", "Report + education"),
    ]:
        with col:
            st.markdown(
                f'<div class="soft-card"><div class="pill">{number}</div><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="disclaimer">⚠️ <b>Important:</b> This application is an academic screening project. Its predictions are not a diagnosis and should not replace a qualified healthcare professional.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ML RISK ASSESSMENT
# ============================================================

elif page == "ML Risk Assessment":

    hero(
        "PCOS Risk Assessment",
        "Enter patient information and use the trained Random Forest model for screening support.",
        "01 • MACHINE LEARNING",
    )

    if ml_model is None:
        st.error("ML model not found.")
        st.code(str(ML_MODEL_PATH))
        st.stop()

    if not ml_features:
        st.error("ML feature list not found.")
        st.stop()

    st.info("The model uses the exact saved feature list from training. Default numeric values are taken from the project dataset.")

    yes_no_features = [
        "Pregnant(Y/N)",
        "Weight gain(Y/N)",
        "hair growth(Y/N)",
        "Skin darkening (Y/N)",
        "Hair loss(Y/N)",
        "Pimples(Y/N)",
        "Fast food (Y/N)",
        "Reg.Exercise(Y/N)",
    ]
    cycle_feature = "Cycle(R/I)"

    numeric_defaults = {}

    if dataset is not None:
        temp_df = dataset.copy()
        temp_df.columns = temp_df.columns.astype(str).str.strip()

        for column in yes_no_features:
            if column in temp_df.columns:
                temp_df[column] = (
                    temp_df[column].astype(str).str.strip().str.upper().map(
                        {"Y": 1, "YES": 1, "1": 1, "N": 0, "NO": 0, "0": 0}
                    )
                )

        if cycle_feature in temp_df.columns:
            temp_df[cycle_feature] = (
                temp_df[cycle_feature].astype(str).str.strip().str.upper().map(
                    {"R": 0, "REGULAR": 0, "I": 1, "IRREGULAR": 1}
                )
            )

        for column in ml_features:
            if column in temp_df.columns:
                converted = pd.to_numeric(temp_df[column], errors="coerce")
                if converted.notna().sum() > 0:
                    numeric_defaults[column] = float(converted.median())

    with st.form("ml_prediction_form"):
        st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

        input_values = {}
        cols = st.columns(2)

        for index, feature in enumerate(ml_features):
            default = numeric_defaults.get(feature, 0.0)
            label = feature
            container = cols[index % 2]

            with container:
                if feature in yes_no_features:
                    input_values[feature] = 1 if st.checkbox(
                        label, value=(default >= 0.5), key=f"ml_{index}_{feature}"
                    ) else 0
                elif feature == cycle_feature:
                    value = st.selectbox(
                        label,
                        ["Regular", "Irregular"],
                        index=(1 if default >= 0.5 else 0),
                        key=f"ml_cycle_{index}",
                    )
                    input_values[feature] = 1 if value == "Irregular" else 0
                else:
                    input_values[feature] = st.number_input(
                        label,
                        value=float(default),
                        format="%.2f",
                        key=f"ml_num_{index}_{feature}",
                    )

        st.divider()
        submitted = st.form_submit_button("✨ Calculate PCOS Risk", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([input_values], columns=ml_features)
            prediction = int(ml_model.predict(input_df)[0])
            probability = float(ml_model.predict_proba(input_df)[0][1])

            risk, icon = get_risk_level(probability)

            st.session_state.ml_result = {
                "prediction": prediction,
                "probability": probability,
                "risk": risk,
                "inputs": input_values,
            }

            st.markdown('<div class="section-title">📋 Screening Result</div>', unsafe_allow_html=True)

            a, b, c = st.columns(3)
            with a:
                st.metric("PCOS Probability", safe_percent(probability))
            with b:
                st.metric("Risk Category", f"{icon} {risk}")
            with c:
                st.metric("Model", "Random Forest")

            st.progress(max(0.0, min(1.0, probability)))

            if prediction == 1:
                st.error(f"Model result: PCOS positive class • {safe_percent(probability)} probability")
            else:
                st.success(f"Model result: PCOS negative class • {safe_percent(probability)} positive-class probability")

            foods, limit, exercise_plan, note = wellness_recommendations(risk, input_values)

            st.markdown('<div class="section-title">💗 Instant Wellness Snapshot</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown("### 🥗 Prefer")
                for item in foods[:4]:
                    st.write(item)

            with r2:
                st.markdown("### 🚫 Limit")
                for item in limit:
                    st.write(item)

            with r3:
                st.markdown("### 🏃 Move")
                for item in exercise_plan[:4]:
                    st.write(item)

            st.info(note)

            st.markdown("### 🔎 Input data used")
            display_df = input_df.T.reset_index()
            display_df.columns = ["Feature", "Value"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="disclaimer">The risk category is a model output for screening support. It does not confirm or rule out PCOS.</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error("ML prediction could not be completed.")
            st.exception(e)


# ============================================================
# ULTRASOUND DL ANALYSIS
# ============================================================

elif page == "Ultrasound DL Analysis":

    hero(
        "Ultrasound Image Analysis",
        "Upload an ultrasound image and run the trained CNN model.",
        "02 • DEEP LEARNING",
    )

    if not TF_AVAILABLE:
        st.error("TensorFlow is not available in the current Python environment.")
        st.info("The ML section can still be used. Run the app with the project environment containing TensorFlow.")
        st.stop()

    if cnn_model is None:
        st.error("CNN model not found.")
        st.code(str(CNN_MODEL_PATH))
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG or PNG ultrasound image.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Unable to read the uploaded image.")
            st.stop()

        left, right = st.columns(2)

        with left:
            st.markdown("### 🩻 Uploaded Ultrasound")
            st.image(image, use_container_width=True)
            st.caption(f"{uploaded_file.name} • {image.width} × {image.height}")

        resized_image = image.resize((160, 160))
        image_array = np.asarray(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("CNN is analysing the ultrasound..."):
            raw_prediction = float(cnn_model.predict(image_array, verbose=0)[0][0])

        # Existing project convention:
        # class 0 = infected, class 1 = noninfected.
        infected_probability = 1.0 - raw_prediction
        noninfected_probability = raw_prediction

        if raw_prediction >= 0.5:
            final_prediction = "NON-INFECTED"
            confidence = noninfected_probability
        else:
            final_prediction = "INFECTED"
            confidence = infected_probability

        st.session_state.cnn_result = {
            "prediction": final_prediction,
            "infected_probability": infected_probability,
            "noninfected_probability": noninfected_probability,
            "confidence": confidence,
            "raw_prediction": raw_prediction,
        }

        with right:
            st.markdown("### 🤖 CNN Result")
            if final_prediction == "INFECTED":
                st.error("INFECTED")
            else:
                st.success("NON-INFECTED")
            st.metric("Model Confidence", safe_percent(confidence))
            st.caption("Decision threshold: 0.50")

        st.markdown('<div class="section-title">📊 Probability Analysis</div>', unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            st.metric("Infected", safe_percent(infected_probability))
        with b:
            st.metric("Non-Infected", safe_percent(noninfected_probability))
        with c:
            st.metric("Confidence", safe_percent(confidence))

        st.write(f"Infected — {safe_percent(infected_probability)}")
        st.progress(infected_probability)

        st.write(f"Non-Infected — {safe_percent(noninfected_probability)}")
        st.progress(noninfected_probability)

        chart_df = pd.DataFrame(
            {"Probability (%)": [infected_probability * 100, noninfected_probability * 100]},
            index=["Infected", "Non-Infected"],
        )
        st.bar_chart(chart_df)

        st.markdown("### 🔬 Technical Analysis")
        technical_df = pd.DataFrame({
            "Parameter": [
                "Input Size", "Output Shape", "Activation",
                "Raw Sigmoid Output", "Threshold", "Class 0", "Class 1",
                "Final Prediction"
            ],
            "Value": [
                "160 × 160 × 3", "(None, 1)", "Sigmoid",
                f"{raw_prediction:.6f}", "0.50",
                "infected", "noninfected", final_prediction
            ]
        })
        st.dataframe(technical_df, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="disclaimer">Academic project only. Ultrasound classification is not a medical diagnosis.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# HEALTH REPORT
# ============================================================

elif page == "Health Report":

    hero(
        "Personalized Wellness Report",
        "A simple educational health snapshot based on the latest ML screening result.",
        "03 • WELLNESS REPORT",
    )

    result = st.session_state.ml_result

    if result is None:
        st.info("Please complete the ML Risk Assessment first. Your wellness report will appear here.")
    else:
        probability = result["probability"]
        risk = result["risk"]
        inputs = result.get("inputs", {})

        foods, limit, exercise_plan, note = wellness_recommendations(risk, inputs)

        a, b, c = st.columns(3)
        with a:
            st.metric("Model Probability", safe_percent(probability))
        with b:
            st.metric("Screening Level", risk)
        with c:
            st.metric("Report Status", "Ready")

        st.markdown('<div class="section-title">🥗 Food Guide</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)

        with f1:
            st.markdown("### ✅ Better choices")
            for item in foods:
                st.write(item)

        with f2:
            st.markdown("### ⚠️ Try to limit")
            for item in limit:
                st.write(item)

        st.markdown('<div class="section-title">🏃 Activity & Lifestyle</div>', unsafe_allow_html=True)
        for item in exercise_plan:
            st.write(item)

        st.markdown('<div class="section-title">🧘 Simple daily routine</div>', unsafe_allow_html=True)
        routine = [
            ("Morning", "Water + balanced breakfast + a few minutes of movement"),
            ("Day", "Regular meals, vegetables/protein, hydration and short movement breaks"),
            ("Evening", "Walk or light exercise + balanced dinner"),
            ("Night", "Relaxation and a consistent sleep routine"),
        ]
        rr = st.columns(4)
        for col, (title, desc) in zip(rr, routine):
            with col:
                card(title, desc, "🌷")

        st.info(note)

        st.markdown(
            '<div class="disclaimer">This report gives general wellness education only. Food, exercise and lifestyle advice should be personalized by a qualified healthcare professional, especially when symptoms or other medical conditions are present.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# LMP & WELLNESS
# ============================================================

elif page == "LMP & Wellness":

    hero(
        "LMP & Cycle Planner",
        "Estimate the next period date from the last menstrual period and average cycle length.",
        "04 • MENSTRUAL CYCLE",
    )

    st.info("LMP means Last Menstrual Period. This is a simple date estimate, not a medical prediction.")

    c1, c2 = st.columns(2)

    with c1:
        lmp_date = st.date_input(
            "Last Menstrual Period (LMP)",
            value=date.today() - timedelta(days=28),
            max_value=date.today(),
        )

    with c2:
        cycle_length = st.number_input(
            "Average Cycle Length (days)",
            min_value=21,
            max_value=45,
            value=28,
            step=1,
        )

    if st.button("📅 Predict Next LMP", use_container_width=True):
        next_lmp = lmp_date + timedelta(days=int(cycle_length))
        fertile_start = next_lmp - timedelta(days=16)
        fertile_end = next_lmp - timedelta(days=12)

        st.session_state.lmp_result = {
            "next_lmp": next_lmp,
            "fertile_start": fertile_start,
            "fertile_end": fertile_end,
        }

    if st.session_state.lmp_result:
        result = st.session_state.lmp_result

        a, b, c = st.columns(3)
        with a:
            st.metric("Estimated Next Period", result["next_lmp"].strftime("%d %b %Y"))
        with b:
            st.metric("Cycle Length", f"{cycle_length} days")
        with c:
            st.metric("Estimated Fertile Window", f"{result['fertile_start'].strftime('%d %b')} – {result['fertile_end'].strftime('%d %b')}")

        st.markdown(
            '<div class="disclaimer">Cycle dates can vary naturally, especially with irregular cycles or hormonal conditions. Do not use this estimate alone for contraception or medical decisions.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">🌸 Healthy Cycle Habits</div>', unsafe_allow_html=True)
    habits = [
        "Track period dates and cycle changes regularly.",
        "Note symptoms such as unusual bleeding, severe pain or major cycle changes.",
        "Keep regular meals, movement, sleep and hydration habits.",
        "Seek medical advice for persistent or concerning symptoms.",
    ]
    for habit in habits:
        st.write("• " + habit)


# ============================================================
# AI PATIENT CHAT
# ============================================================

def chatbot_answer(message):
    q = message.lower().strip()

    if not q:
        return "Please type a question. I can help with basic PCOS education, food, exercise, periods, LMP and this app."

    if any(x in q for x in ["pcos", "polycystic"]):
        return (
            "PCOS stands for Polycystic Ovary Syndrome. It is a hormonal condition. "
            "Common concerns can include irregular periods, acne, excess hair growth and changes in weight. "
            "A healthcare professional should make the final diagnosis."
        )

    if any(x in q for x in ["food", "eat", "diet", "meal", "vegetable"]):
        return (
            "A balanced pattern is generally a good starting point: vegetables, fibre-rich whole grains, "
            "protein such as dal/beans/eggs/paneer, nuts or seeds in moderate portions, and whole fruits. "
            "Try to limit frequent sugary drinks, highly processed foods and excess added sugar."
        )

    if any(x in q for x in ["exercise", "workout", "walk", "gym", "yoga"]):
        return (
            "Regular movement can be helpful. A simple starting plan is comfortable walking most days, "
            "plus strength training 2–3 times per week if appropriate for you. Start gradually and choose "
            "activities you can continue consistently."
        )

    if any(x in q for x in ["period", "periods", "cycle", "menstrual", "lmp", "next period"]):
        return (
            "You can use the LMP & Wellness page to estimate your next period from your last period date "
            "and average cycle length. Cycle estimates are approximate, especially when cycles are irregular."
        )

    if any(x in q for x in ["ultrasound", "scan", "image", "cnn"]):
        return (
            "The Ultrasound DL Analysis page uses the trained CNN model to classify the uploaded ultrasound "
            "image according to the classes used during training. The output is an academic screening result, not a diagnosis."
        )

    if any(x in q for x in ["risk", "prediction", "probability", "result"]):
        return (
            "The ML Risk Assessment page uses the trained Random Forest model. The probability is a model "
            "estimate for the positive PCOS class. It should be interpreted as screening support, not a diagnosis."
        )

    if any(x in q for x in ["doctor", "diagnosis", "diagnose", "medicine", "tablet"]):
        return (
            "I can provide general educational information, but I cannot diagnose a condition or prescribe medicine. "
            "For diagnosis or treatment, please consult a qualified healthcare professional."
        )

    if any(x in q for x in ["hello", "hi", "hey"]):
        return "Hi! 🌸 Ask me about PCOS, food, exercise, periods, LMP, ultrasound analysis or the screening result."

    return (
        "I can help with basic questions about PCOS, food, exercise, periods/LMP, ultrasound analysis and "
        "your screening result. Try asking one of those topics."
    )


if page == "AI Patient Chat":

    hero(
        "PCOS AI Patient Assistant",
        "Ask simple questions about PCOS, food, exercise, periods and this project.",
        "05 • PATIENT Q&A",
    )

    st.info("This is an offline educational assistant built into the app. It does not replace a doctor.")

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="chat-user"><b>You</b><br>{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot"><b>🌸 PCOS AI</b><br>{message}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Ask your question",
            placeholder="Example: What food is better for PCOS?",
        )
        send = st.form_submit_button("Send 💬", use_container_width=True)

    if send and question.strip():
        answer = chatbot_answer(question)
        st.session_state.chat_history.append(("user", question.strip()))
        st.session_state.chat_history.append(("bot", answer))
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("### 💡 Try asking")
    suggestions = [
        "What is PCOS?",
        "What food is better?",
        "What exercise can I do?",
        "How can I predict my next period?",
        "What does the ML risk mean?",
        "How does the CNN work?",
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            st.markdown(f'<span class="pill">{suggestion}</span>', unsafe_allow_html=True)


# ============================================================
# AI SUMMARY
# ============================================================

elif page == "AI Summary":

    hero(
        "Multi-Modal AI Summary",
        "See the ML and ultrasound outputs together while keeping their meanings separate.",
        "06 • COMBINED AI VIEW",
    )

    if st.session_state.ml_result is not None:
        result = st.session_state.ml_result
        probability = result["probability"]
        risk = result["risk"]

        st.markdown("### 🧠 Machine Learning")
        a, b, c = st.columns(3)
        with a:
            st.metric("PCOS Probability", safe_percent(probability))
        with b:
            st.metric("Risk Category", risk)
        with c:
            st.metric("Model", "Random Forest")
    else:
        st.info("No ML prediction available yet.")

    st.divider()

    if st.session_state.cnn_result is not None:
        result = st.session_state.cnn_result

        st.markdown("### 🩻 Deep Learning")
        a, b, c = st.columns(3)
        with a:
            st.metric("Infected", safe_percent(result["infected_probability"]))
        with b:
            st.metric("Non-Infected", safe_percent(result["noninfected_probability"]))
        with c:
            st.metric("CNN Confidence", safe_percent(result["confidence"]))

        if result["prediction"] == "INFECTED":
            st.error("DL: INFECTED")
        else:
            st.success("DL: NON-INFECTED")
    else:
        st.info("No ultrasound CNN prediction available yet.")

    st.markdown("### 🔗 How the two components fit together")
    st.write(
        "The ML component analyzes structured patient information. "
        "The DL component analyzes the ultrasound image. "
        "They are separate model outputs and should not be treated as one medical diagnosis."
    )

    if st.session_state.ml_result is not None:
        st.markdown("### 💗 Wellness shortcut")
        st.info("Open **Health Report** for food, activity and general wellness guidance based on the latest screening result.")


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    hero(
        "AI Model Information",
        "Technical details of the trained models used by the application.",
        "07 • TECHNICAL DETAILS",
    )

    st.markdown("### 🧠 Machine Learning Model")

    ml_table = pd.DataFrame({
        "Parameter": [
            "Algorithm", "Estimator", "Imputation",
            "Number of Features", "Class Weight", "Saved Model"
        ],
        "Value": [
            "Random Forest", "300 trees", "Median",
            str(len(ml_features)), "Balanced", "pcos_ml_model.joblib"
        ]
    })
    st.dataframe(ml_table, use_container_width=True, hide_index=True)

    if ml_info:
        a, b, c = st.columns(3)
        with a:
            if "accuracy" in ml_info:
                st.metric("Accuracy", f"{ml_info['accuracy'] * 100:.2f}%")
        with b:
            if "roc_auc" in ml_info:
                st.metric("ROC-AUC", f"{ml_info['roc_auc']:.4f}")
        with c:
            if "total_rows" in ml_info:
                st.metric("Training Rows", str(ml_info["total_rows"]))

    st.divider()
    st.markdown("### 🩻 Deep Learning Model")

    dl_table = pd.DataFrame({
        "Parameter": [
            "Architecture", "Input", "Output", "Activation",
            "Class 0", "Class 1", "Saved Model"
        ],
        "Value": [
            "Custom CNN", "160 × 160 × 3", "(None, 1)", "Sigmoid",
            "infected", "noninfected", "pcos_ultrasound_cnn.keras"
        ]
    })
    st.dataframe(dl_table, use_container_width=True, hide_index=True)

    if cnn_model is not None:
        st.markdown("### CNN Architecture")
        architecture = pd.DataFrame({
            "Layer": [
                "Input", "Data Augmentation", "Rescaling", "Conv2D",
                "Batch Normalization", "MaxPooling", "Conv2D",
                "Batch Normalization", "MaxPooling", "Conv2D",
                "Batch Normalization", "MaxPooling", "Conv2D",
                "Batch Normalization", "MaxPooling", "Global Average Pooling",
                "Dropout", "Dense", "Dropout", "Sigmoid Output"
            ],
            "Configuration": [
                "160 × 160 × 3", "Flip + Rotation + Zoom + Contrast", "1/255",
                "32 filters", "Yes", "2 × 2", "64 filters", "Yes", "2 × 2",
                "128 filters", "Yes", "2 × 2", "256 filters", "Yes", "2 × 2",
                "256 features", "40%", "128 neurons", "30%", "1 neuron"
            ]
        })
        st.dataframe(architecture, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📁 Project Model Files")
    files_df = pd.DataFrame({
        "Component": [
            "ML Model", "ML Features", "ML Information",
            "CNN Model", "CNN Classes"
        ],
        "File": [
            str(ML_MODEL_PATH), str(FEATURES_PATH), str(INFO_PATH),
            str(CNN_MODEL_PATH), str(CLASS_PATH)
        ]
    })
    st.dataframe(files_df, use_container_width=True, hide_index=True)


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    hero(
        "About PCOS AI",
        "An academic project combining Machine Learning, Deep Learning and an interactive Streamlit dashboard.",
        "08 • PROJECT",
    )

    c1, c2 = st.columns(2)

    with c1:
        card(
            "Project Objective",
            "Use AI to demonstrate PCOS screening from structured patient information and ultrasound images.",
            "🎯",
        )

    with c2:
        card(
            "Technology Stack",
            "Python • Pandas • NumPy • Scikit-learn • Random Forest • TensorFlow • Keras • Pillow • Streamlit",
            "💻",
        )

    st.markdown("### 🧠 Machine Learning")
    st.write(
        "The ML component uses a Random Forest classifier. "
        "The training pipeline performs data cleaning, numerical conversion, median imputation and classification."
    )

    st.markdown("### 🩻 Deep Learning")
    st.write(
        "The DL component uses a Convolutional Neural Network to classify ultrasound images according to the trained classes."
    )

    st.markdown("### ✨ New dashboard features")
    for item in [
        "Personalized wellness report",
        "Food and lifestyle guidance",
        "LMP / next-period estimator",
        "Offline patient Q&A chatbot",
        "Modern responsive dashboard design",
        "Combined AI summary",
    ]:
        st.write("• " + item)

    st.markdown(
        '<div class="disclaimer">⚠️ <b>Academic Disclaimer:</b> This project is intended for educational and academic purposes. Model outputs should not be considered medical diagnoses or treatment advice.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">🌸 PCOS AI Smart Health Dashboard • ML + DL + Wellness Education<br>Built for academic demonstration</div>',
    unsafe_allow_html=True,
)
