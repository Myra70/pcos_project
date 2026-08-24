from pathlib import Path
import json
import warnings

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
    page_title="PCOS AI Screening System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown("""
<style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #16324F;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 20px;
    }


    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #16324F;
        margin-top: 15px;
        margin-bottom: 15px;
    }


    /* ======================================================
       KPI CARDS
       ====================================================== */

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e4eaf2;
        box-shadow: 0 6px 20px rgba(22, 50, 79, 0.08);
        min-height: 120px;
    }

    .metric-title {
        color: #667085;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #16324F;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-description {
        color: #667085;
        font-size: 13px;
        margin-top: 8px;
    }


    /* ======================================================
       AI CARDS
       ====================================================== */

    .ai-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        border: 1px solid #e5eaf1;
        box-shadow: 0 8px 25px rgba(22, 50, 79, 0.08);
        min-height: 260px;
    }

    .ai-card h3 {
        color: #16324F;
        margin-bottom: 15px;
        font-size: 22px;
    }

    .ai-card p {
        color: #667085;
        line-height: 1.7;
        font-size: 15px;
    }


    /* ======================================================
       PROCESS CARDS
       ====================================================== */

    .process-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e4eaf2;
        box-shadow: 0 5px 18px rgba(22, 50, 79, 0.06);
        min-height: 150px;
    }

    .process-number {
        font-size: 12px;
        font-weight: 700;
        color: #667085;
        letter-spacing: 1px;
    }

    .process-title {
        font-size: 19px;
        font-weight: 700;
        color: #16324F;
        margin-top: 8px;
    }

    .process-text {
        font-size: 13px;
        color: #667085;
        margin-top: 8px;
        line-height: 1.5;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #102A43 0%,
            #16324F 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: none;
        padding: 10px 20px;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        padding: 10px;
        border: 1px solid #dce4ee;
    }


    /* ======================================================
       DATAFRAMES
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }


    /* ======================================================
       RESULT CARD
       ====================================================== */

    .result-card {
        background: white;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #e4eaf2;
        box-shadow: 0 6px 20px rgba(22, 50, 79, 0.08);
        text-align: center;
    }

    .result-title {
        color: #667085;
        font-size: 14px;
        font-weight: 600;
    }

    .result-value {
        color: #16324F;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }


    /* ======================================================
       DISCLAIMER
       ====================================================== */

    .disclaimer {
        background: #fff8e6;
        border-left: 5px solid #f59e0b;
        padding: 16px;
        border-radius: 10px;
        color: #7a5a00;
        margin-top: 20px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #667085;
        font-size: 13px;
        padding: 25px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

CSV_PATH = DATA_DIR / "PCOS_extended_dataset.csv"

ML_MODEL_PATH = MODEL_DIR / "pcos_ml_model.joblib"
FEATURES_PATH = MODEL_DIR / "pcos_features.joblib"
INFO_PATH = MODEL_DIR / "model_info.joblib"

CNN_MODEL_PATH = MODEL_DIR / "pcos_ultrasound_cnn.keras"
CLASS_PATH = MODEL_DIR / "ultrasound_classes.json"


# ============================================================
# LOAD ML MODEL
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


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not CSV_PATH.exists():
        return None

    try:

        df = pd.read_csv(CSV_PATH)

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        return df

    except Exception:
        return None


# ============================================================
# LOAD CNN
# ============================================================

@st.cache_resource
def load_cnn_model():

    if not TF_AVAILABLE:
        return None

    if not CNN_MODEL_PATH.exists():
        return None

    try:

        return tf.keras.models.load_model(
            CNN_MODEL_PATH
        )

    except Exception:

        return None


@st.cache_data
def load_class_names():

    if CLASS_PATH.exists():

        try:

            with open(
                CLASS_PATH,
                "r"
            ) as file:

                return json.load(file)

        except Exception:
            pass

    return [
        "infected",
        "noninfected"
    ]


# ============================================================
# INITIALIZE MODELS
# ============================================================

ml_model = load_ml_model()
ml_features = load_ml_features()
ml_info = load_ml_info()

dataset = load_dataset()

cnn_model = load_cnn_model()
class_names = load_class_names()


# ============================================================
# SESSION STATE
# ============================================================

if "ml_result" not in st.session_state:
    st.session_state.ml_result = None

if "cnn_result" not in st.session_state:
    st.session_state.cnn_result = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("PCOS AI SYSTEM")

st.sidebar.write(
    "AI-Based PCOS Risk Assessment "
    "and Ultrasound Image Analysis"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "ML Risk Assessment",
        "Ultrasound DL Analysis",
        "AI Summary",
        "Model Information",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("System Status")


if ml_model is not None:

    st.sidebar.success(
        "ML Model Loaded"
    )

else:

    st.sidebar.error(
        "ML Model Missing"
    )


if TF_AVAILABLE:

    if cnn_model is not None:

        st.sidebar.success(
            "CNN Model Loaded"
        )

    else:

        st.sidebar.warning(
            "CNN Model Missing"
        )

else:

    st.sidebar.warning(
        "TensorFlow unavailable"
    )


if TF_AVAILABLE:

    st.sidebar.caption(
        f"TensorFlow: {tf.__version__}"
    )

st.sidebar.caption(
    "ML: Random Forest"
)

st.sidebar.caption(
    "DL: CNN"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'PCOS AI Screening & Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered PCOS risk assessment using Machine Learning '
    'and ultrasound image analysis using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">'
        'AI Screening Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This system demonstrates a two-component artificial
        intelligence workflow for PCOS screening research.
        The first component analyses structured patient
        information using Machine Learning, while the second
        component analyses ultrasound images using Deep Learning.
        """
    )

    st.divider()

    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        MACHINE LEARNING
        </div>

        <div class="metric-value">
        Random Forest
        </div>

        <div class="metric-description">
        Structured data classification
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        DEEP LEARNING
        </div>

        <div class="metric-value">
        CNN
        </div>

        <div class="metric-description">
        Ultrasound image classification
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-title">
        ML FEATURES
        </div>

        <div class="metric-value">
        {len(ml_features)}
        </div>

        <div class="metric-description">
        Patient-related features
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        IMAGE INPUT
        </div>

        <div class="metric-value">
        160 × 160
        </div>

        <div class="metric-description">
        CNN image resolution
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================================================
    # AI MODULES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'AI Analysis Modules'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="ai-card">

        <h3>Machine Learning</h3>

        <p>
        <b>Input:</b> Clinical, physical and symptom-related
        information.
        </p>

        <p>
        <b>Algorithm:</b> Random Forest Classifier.
        </p>

        <p>
        <b>Analysis:</b> Estimates PCOS probability using
        structured patient information.
        </p>

        <p>
        <b>Output:</b> PCOS probability and risk category.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="ai-card">

        <h3>Deep Learning</h3>

        <p>
        <b>Input:</b> Ultrasound image.
        </p>

        <p>
        <b>Algorithm:</b> Convolutional Neural Network.
        </p>

        <p>
        <b>Analysis:</b> Learns visual patterns from ultrasound
        images.
        </p>

        <p>
        <b>Output:</b> Infected / Non-Infected probability.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================================================
    # WORKFLOW
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'System Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    step1, step2, step3, step4 = st.columns(4)

    with step1:

        st.markdown("""
        <div class="process-card">

        <div class="process-number">
        STEP 01
        </div>

        <div class="process-title">
        Patient Data
        </div>

        <div class="process-text">
        Clinical and symptom-related information
        </div>

        </div>
        """, unsafe_allow_html=True)

    with step2:

        st.markdown("""
        <div class="process-card">

        <div class="process-number">
        STEP 02
        </div>

        <div class="process-title">
        ML Analysis
        </div>

        <div class="process-text">
        Random Forest estimates PCOS probability
        </div>

        </div>
        """, unsafe_allow_html=True)

    with step3:

        st.markdown("""
        <div class="process-card">

        <div class="process-number">
        STEP 03
        </div>

        <div class="process-title">
        CNN Analysis
        </div>

        <div class="process-text">
        Ultrasound image classification
        </div>

        </div>
        """, unsafe_allow_html=True)

    with step4:

        st.markdown("""
        <div class="process-card">

        <div class="process-number">
        STEP 04
        </div>

        <div class="process-title">
        AI Summary
        </div>

        <div class="process-text">
        Separate ML and DL results for interpretation
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'ML Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    if ml_info:

        col1, col2, col3 = st.columns(3)

        with col1:

            accuracy = ml_info.get(
                "accuracy",
                None
            )

            if accuracy is not None:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

        with col2:

            auc = ml_info.get(
                "roc_auc",
                None
            )

            if auc is not None:

                st.metric(
                    "ROC-AUC",
                    f"{auc:.4f}"
                )

        with col3:

            rows = ml_info.get(
                "total_rows",
                None
            )

            if rows is not None:

                st.metric(
                    "Dataset Records",
                    str(rows)
                )

    else:

        st.info(
            "ML model performance information is not available."
        )

    st.divider()

    st.success(
        "Select a module from the navigation panel to begin analysis."
    )


# ============================================================
# ML RISK ASSESSMENT
# ============================================================

elif page == "ML Risk Assessment":

    st.markdown(
        '<div class="section-title">'
        'Machine Learning — PCOS Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Enter patient-related clinical and symptom information.
        The trained Random Forest model will estimate the
        probability of PCOS.
        """
    )

    if ml_model is None:

        st.error(
            "ML model not found."
        )

        st.code(
            str(ML_MODEL_PATH)
        )

        st.stop()

    if not ml_features:

        st.error(
            "ML feature list not found."
        )

        st.stop()

    st.divider()

    st.info(
        """
        The model uses the same feature list saved during
        training. Default values are derived from the project
        dataset and can be changed before prediction.
        """
    )

    with st.form(
        "ml_prediction_form"
    ):

        st.subheader(
            "Patient Information"
        )

        input_values = {}

        yes_no_features = [
            "Pregnant(Y/N)",
            "Weight gain(Y/N)",
            "hair growth(Y/N)",
            "Skin darkening (Y/N)",
            "Hair loss(Y/N)",
            "Pimples(Y/N)",
            "Fast food (Y/N)",
            "Reg.Exercise(Y/N)"
        ]

        cycle_feature = "Cycle(R/I)"

        numeric_defaults = {}

        if dataset is not None:

            temp_df = dataset.copy()

            temp_df.columns = (
                temp_df.columns
                .astype(str)
                .str.strip()
            )

            for column in yes_no_features:

                if column in temp_df.columns:

                    temp_df[column] = (
                        temp_df[column]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .map(
                            {
                                "Y": 1,
                                "YES": 1,
                                "1": 1,
                                "N": 0,
                                "NO": 0,
                                "0": 0
                            }
                        )
                    )

            if cycle_feature in temp_df.columns:

                temp_df[cycle_feature] = (
                    temp_df[cycle_feature]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map(
                        {
                            "R": 0,
                            "REGULAR": 0,
                            "I": 1,
                            "IRREGULAR": 1
                        }
                    )
                )

            for column in ml_features:

                if column in temp_df.columns:

                    converted = pd.to_numeric(
                        temp_df[column],
                        errors="coerce"
                    )

                    if converted.notna().sum() > 0:

                        numeric_defaults[column] = float(
                            converted.median()
                        )

        for index, feature in enumerate(
            ml_features
        ):

            default = numeric_defaults.get(
                feature,
                0.0
            )

            label = feature

            if feature in yes_no_features:

                default_bool = (
                    default >= 0.5
                )

                value = st.checkbox(
                    label,
                    value=default_bool
                )

                input_values[feature] = (
                    1 if value else 0
                )

            elif feature == cycle_feature:

                options = [
                    "Regular",
                    "Irregular"
                ]

                default_index = (
                    1 if default >= 0.5
                    else 0
                )

                value = st.selectbox(
                    label,
                    options,
                    index=default_index
                )

                input_values[feature] = (
                    1
                    if value == "Irregular"
                    else 0
                )

            else:

                value = st.number_input(
                    label,
                    value=float(default),
                    format="%.2f"
                )

                input_values[feature] = value

        st.divider()

        submitted = st.form_submit_button(
            "Calculate PCOS Risk",
            use_container_width=True
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [input_values],
                columns=ml_features
            )

            prediction = ml_model.predict(
                input_df
            )[0]

            probability = ml_model.predict_proba(
                input_df
            )[0][1]

            probability = float(
                probability
            )

            st.session_state.ml_result = {
                "prediction": int(prediction),
                "probability": probability
            }

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Prediction Overview'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "PCOS Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:

                if probability >= 0.5:

                    st.metric(
                        "Risk Category",
                        "Higher Risk"
                    )

                else:

                    st.metric(
                        "Risk Category",
                        "Lower Risk"
                    )

            with col3:

                st.metric(
                    "Model",
                    "Random Forest"
                )

            st.divider()

            if prediction == 1:

                st.error(
                    "ML Prediction: PCOS Positive"
                )

                st.warning(
                    f"""
                    The Random Forest model estimated a
                    {probability * 100:.2f}% probability for
                    the positive PCOS class.
                    """
                )

            else:

                st.success(
                    "ML Prediction: PCOS Negative"
                )

                st.success(
                    f"""
                    The Random Forest model estimated a
                    {probability * 100:.2f}% probability for
                    the positive PCOS class.
                    """
                )

            st.subheader(
                "Risk Probability"
            )

            st.progress(
                probability
            )

            st.caption(
                "0% represents lower model probability and "
                "100% represents higher model probability."
            )

            st.divider()

            st.subheader(
                "Input Data Used by ML Model"
            )

            display_df = input_df.T.reset_index()

            display_df.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                "ML prediction could not be completed."
            )

            st.exception(e)


# ============================================================
# ULTRASOUND DEEP LEARNING
# ============================================================

elif page == "Ultrasound DL Analysis":

    st.markdown(
        '<div class="section-title">'
        'Deep Learning — Ultrasound Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Upload an ultrasound image. The trained CNN analyses
        the image and produces probabilities for the two
        ultrasound classes.
        """
    )

    if not TF_AVAILABLE:

        st.error(
            "TensorFlow is not available in the current Python environment."
        )

        st.info(
            "Run this dashboard with the project virtual environment "
            "that contains TensorFlow."
        )

        st.stop()

    if cnn_model is None:

        st.error(
            "CNN model not found."
        )

        st.code(
            str(CNN_MODEL_PATH)
        )

        st.stop()

    st.divider()

    st.info(
        """
        Upload a clear ultrasound image. The CNN resizes the
        image to 160 × 160 pixels before prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Ultrasound Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is None:

        st.info(
            "Upload a JPG, JPEG or PNG ultrasound image."
        )

    else:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "Unable to read the uploaded image."
            )

            st.stop()

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.subheader(
                "Uploaded Ultrasound"
            )

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"File: {uploaded_file.name}"
            )

            st.caption(
                f"Original size: "
                f"{image.width} × {image.height}"
            )

        resized_image = image.resize(
            (160, 160)
        )

        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        with st.spinner(
            "CNN is analysing the ultrasound..."
        ):

            raw_prediction = cnn_model.predict(
                image_array,
                verbose=0
            )[0][0]

        raw_prediction = float(
            raw_prediction
        )

        # ====================================================
        # CLASS MAPPING
        # class 0 = infected
        # class 1 = noninfected
        # sigmoid = probability of class 1
        # ====================================================

        infected_probability = (
            1.0 - raw_prediction
        )

        noninfected_probability = (
            raw_prediction
        )

        if raw_prediction >= 0.5:

            final_prediction = "NON-INFECTED"

            confidence = (
                noninfected_probability
            )

        else:

            final_prediction = "INFECTED"

            confidence = (
                infected_probability
            )

        st.session_state.cnn_result = {
            "prediction": final_prediction,
            "infected_probability":
                infected_probability,
            "noninfected_probability":
                noninfected_probability,
            "confidence": confidence,
            "raw_prediction":
                raw_prediction
        }

        with col2:

            st.subheader(
                "CNN Result"
            )

            if final_prediction == "INFECTED":

                st.error(
                    "Prediction: INFECTED"
                )

            else:

                st.success(
                    "Prediction: NON-INFECTED"
                )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Decision threshold: 0.50"
            )

        st.divider()

        st.subheader(
            "CNN Probability Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Infected",
                f"{infected_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Non-Infected",
                f"{noninfected_probability * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.divider()

        st.subheader(
            "Probability Distribution"
        )

        st.write(
            f"Infected — "
            f"{infected_probability * 100:.2f}%"
        )

        st.progress(
            infected_probability
        )

        st.write(
            f"Non-Infected — "
            f"{noninfected_probability * 100:.2f}%"
        )

        st.progress(
            noninfected_probability
        )

        st.divider()

        chart_df = pd.DataFrame(
            {
                "Probability (%)": [
                    infected_probability * 100,
                    noninfected_probability * 100
                ]
            },
            index=[
                "Infected",
                "Non-Infected"
            ]
        )

        st.subheader(
            "Class Probability Comparison"
        )

        st.bar_chart(
            chart_df
        )

        st.divider()

        st.subheader(
            "Technical Analysis"
        )

        technical_df = pd.DataFrame(
            {
                "Parameter": [
                    "Input Size",
                    "Output Shape",
                    "Activation",
                    "Raw Sigmoid Output",
                    "Threshold",
                    "Class 0",
                    "Class 1",
                    "Final Prediction"
                ],
                "Value": [
                    "160 × 160 × 3",
                    "(None, 1)",
                    "Sigmoid",
                    f"{raw_prediction:.6f}",
                    "0.50",
                    "infected",
                    "noninfected",
                    final_prediction
                ]
            }
        )

        st.dataframe(
            technical_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader(
            "CNN Interpretation"
        )

        st.info(
            f"""
            The CNN produced a sigmoid output of
            {raw_prediction:.4f}.

            Class 0 represents infected and class 1 represents
            noninfected.

            Infected probability:
            {(1 - raw_prediction) * 100:.2f}%

            Non-Infected probability:
            {raw_prediction * 100:.2f}%

            The class with the higher probability is displayed
            as the final prediction.
            """
        )

        st.markdown("""
        <div class="disclaimer">

        <b>Academic Disclaimer</b><br><br>

        This ultrasound prediction is intended for academic
        and educational demonstration only. It is not a medical
        diagnosis and should not replace professional clinical
        evaluation.

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# AI SUMMARY
# ============================================================

elif page == "AI Summary":

    st.markdown(
        '<div class="section-title">'
        'Multi-Modal AI Summary'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This page presents the outputs of the Machine Learning
        and Deep Learning components separately.
        """
    )

    st.divider()

    # ========================================================
    # ML RESULT
    # ========================================================

    st.subheader(
        "Machine Learning Result"
    )

    if st.session_state.ml_result is not None:

        ml_result = st.session_state.ml_result

        ml_probability = ml_result[
            "probability"
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "PCOS Probability",
                f"{ml_probability * 100:.2f}%"
            )

        with col2:

            if ml_result["prediction"] == 1:

                st.error(
                    "ML: PCOS Positive"
                )

            else:

                st.success(
                    "ML: PCOS Negative"
                )

    else:

        st.info(
            "No ML prediction available yet."
        )

    st.divider()

    # ========================================================
    # DL RESULT
    # ========================================================

    st.subheader(
        "Deep Learning Result"
    )

    if st.session_state.cnn_result is not None:

        cnn_result = (
            st.session_state.cnn_result
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Infected",
                f"{cnn_result['infected_probability'] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Non-Infected",
                f"{cnn_result['noninfected_probability'] * 100:.2f}%"
            )

        with col3:

            st.metric(
                "CNN Confidence",
                f"{cnn_result['confidence'] * 100:.2f}%"
            )

        if (
            cnn_result["prediction"]
            == "INFECTED"
        ):

            st.error(
                "DL: INFECTED"
            )

        else:

            st.success(
                "DL: NON-INFECTED"
            )

    else:

        st.info(
            "No ultrasound CNN prediction available yet."
        )

    st.divider()

    st.subheader(
        "Interpretation of AI Components"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="ai-card">

        <h3>Machine Learning</h3>

        <p>
        Uses structured patient information such as clinical,
        physical and symptom-related features to estimate
        PCOS probability.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="ai-card">

        <h3>Deep Learning</h3>

        <p>
        Uses an ultrasound image and a CNN to classify the image
        into the trained ultrasound classes.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.warning(
        """
        Important: The ML and CNN models analyse different
        types of information. Their outputs are presented
        separately and are not mathematically combined into
        one diagnostic score.
        """
    )

    if (
        st.session_state.ml_result is not None
        and st.session_state.cnn_result is not None
    ):

        st.success(
            """
            Both ML and DL analyses have been completed.
            The results can be reviewed independently in this
            summary section.
            """
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="section-title">'
        'AI Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # ML MODEL
    # ========================================================

    st.subheader(
        "Machine Learning Model"
    )

    ml_table = pd.DataFrame(
        {
            "Parameter": [
                "Algorithm",
                "Estimator",
                "Imputation",
                "Number of Features",
                "Class Weight",
                "Saved Model"
            ],
            "Value": [
                "Random Forest",
                "300 trees",
                "Median",
                str(len(ml_features)),
                "Balanced",
                "pcos_ml_model.joblib"
            ]
        }
    )

    st.dataframe(
        ml_table,
        use_container_width=True,
        hide_index=True
    )

    if ml_info:

        st.subheader(
            "ML Performance"
        )

        col1, col2 = st.columns(2)

        with col1:

            if "accuracy" in ml_info:

                st.metric(
                    "Accuracy",
                    f"{ml_info['accuracy'] * 100:.2f}%"
                )

        with col2:

            if "roc_auc" in ml_info:

                st.metric(
                    "ROC-AUC",
                    f"{ml_info['roc_auc']:.4f}"
                )

    st.divider()

    # ========================================================
    # DL MODEL
    # ========================================================

    st.subheader(
        "Deep Learning Model"
    )

    dl_table = pd.DataFrame(
        {
            "Parameter": [
                "Architecture",
                "Input",
                "Output",
                "Activation",
                "Class 0",
                "Class 1",
                "Saved Model"
            ],
            "Value": [
                "Custom CNN",
                "160 × 160 × 3",
                "(None, 1)",
                "Sigmoid",
                "infected",
                "noninfected",
                "pcos_ultrasound_cnn.keras"
            ]
        }
    )

    st.dataframe(
        dl_table,
        use_container_width=True,
        hide_index=True
    )

    if cnn_model is not None:

        st.divider()

        st.subheader(
            "CNN Architecture"
        )

        architecture = pd.DataFrame(
            {
                "Layer": [
                    "Input",
                    "Data Augmentation",
                    "Rescaling",
                    "Conv2D",
                    "Batch Normalization",
                    "MaxPooling",
                    "Conv2D",
                    "Batch Normalization",
                    "MaxPooling",
                    "Conv2D",
                    "Batch Normalization",
                    "MaxPooling",
                    "Conv2D",
                    "Batch Normalization",
                    "MaxPooling",
                    "Global Average Pooling",
                    "Dropout",
                    "Dense",
                    "Dropout",
                    "Sigmoid Output"
                ],
                "Configuration": [
                    "160 × 160 × 3",
                    "Flip + Rotation + Zoom + Contrast",
                    "1/255",
                    "32 filters",
                    "Yes",
                    "2 × 2",
                    "64 filters",
                    "Yes",
                    "2 × 2",
                    "128 filters",
                    "Yes",
                    "2 × 2",
                    "256 filters",
                    "Yes",
                    "2 × 2",
                    "256 features",
                    "40%",
                    "128 neurons",
                    "30%",
                    "1 neuron"
                ]
            }
        )

        st.dataframe(
            architecture,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Project Model Files"
    )

    files_df = pd.DataFrame(
        {
            "Component": [
                "ML Model",
                "ML Features",
                "ML Information",
                "CNN Model",
                "CNN Classes"
            ],
            "File": [
                str(ML_MODEL_PATH),
                str(FEATURES_PATH),
                str(INFO_PATH),
                str(CNN_MODEL_PATH),
                str(CLASS_PATH)
            ]
        }
    )

    st.dataframe(
        files_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="section-title">'
        'About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Project Title"
    )

    st.write(
        """
        AI-Based PCOS Risk Assessment and Ultrasound Image
        Analysis Using Machine Learning and Deep Learning
        """
    )

    st.divider()

    st.subheader(
        "Objective"
    )

    st.write(
        """
        The objective of this project is to demonstrate an
        AI-based PCOS screening workflow using two complementary
        approaches:

        1. Machine Learning for structured patient data.
        2. Deep Learning for ultrasound image analysis.
        """
    )

    st.divider()

    st.subheader(
        "Machine Learning"
    )

    st.write(
        """
        The ML component uses a Random Forest classifier.
        The training pipeline performs data cleaning, numerical
        conversion, median imputation and classification.
        """
    )

    st.divider()

    st.subheader(
        "Deep Learning"
    )

    st.write(
        """
        The DL component uses a Convolutional Neural Network
        to classify ultrasound images.
        """
    )

    st.divider()

    st.subheader(
        "System Architecture"
    )

    architecture_flow = pd.DataFrame(
        {
            "Stage": [
                "Patient Data",
                "ML Preprocessing",
                "Random Forest",
                "PCOS Risk",
                "Ultrasound Image",
                "Image Preprocessing",
                "CNN",
                "Image Classification"
            ],
            "Component": [
                "Structured Dataset",
                "Cleaning + Encoding + Imputation",
                "Machine Learning",
                "Probability + Risk Category",
                "JPG / PNG",
                "Resize 160 × 160",
                "Deep Learning",
                "Infected / Non-Infected"
            ]
        }
    )

    st.dataframe(
        architecture_flow,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Technologies"
    )

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "NumPy",
                "Scikit-learn",
                "Joblib",
                "TensorFlow",
                "Keras",
                "Pillow",
                "Streamlit"
            ],
            "Purpose": [
                "Programming",
                "Data Processing",
                "Numerical Processing",
                "Machine Learning",
                "ML Model Storage",
                "Deep Learning",
                "CNN",
                "Image Processing",
                "Dashboard"
            ]
        }
    )

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="disclaimer">

    <b>Academic Disclaimer</b><br><br>

    This project is intended for educational and academic
    purposes. Model predictions should not be considered
    a medical diagnosis.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">

<b>PCOS AI Screening & Analysis System</b><br>

Machine Learning • Deep Learning • Computer Vision • Streamlit

<br><br>

Academic Capstone Project | Educational Purpose Only

</div>
""", unsafe_allow_html=True)