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
    layout="wide",
    initial_sidebar_state="expanded"
)


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
# LOAD DATASET FOR DEFAULT VALUES
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


# ============================================================
# SYSTEM STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("System Status")


if ml_model is not None:
    st.sidebar.success("ML Model Loaded")
else:
    st.sidebar.error("ML Model Missing")


if TF_AVAILABLE:

    if cnn_model is not None:
        st.sidebar.success("CNN Model Loaded")
    else:
        st.sidebar.warning("CNN Model Missing")

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

st.title(
    "PCOS AI Screening & Analysis System"
)

st.caption(
    "Machine Learning + Deep Learning + Streamlit"
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header(
        "AI Screening Dashboard"
    )

    st.write(
        """
        This project uses **two AI approaches**:

        **Machine Learning**
        → Clinical, physical and symptom-related information

        **Deep Learning**
        → Ultrasound image analysis
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT KPIs
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "ML Algorithm",
            "Random Forest"
        )

    with col2:

        st.metric(
            "DL Algorithm",
            "CNN"
        )

    with col3:

        st.metric(
            "ML Features",
            str(len(ml_features))
        )

    with col4:

        st.metric(
            "Ultrasound Input",
            "160 × 160"
        )

    st.divider()

    # --------------------------------------------------------
    # TWO AI COMPONENTS
    # --------------------------------------------------------

    st.subheader(
        "What does this system analyse?"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            ### Machine Learning

            **Input**

            Clinical and symptom-related data

            **Algorithm**

            Random Forest Classifier

            **Analysis**

            PCOS risk based on structured
            patient information

            **Output**

            PCOS probability + risk category
            """
        )

    with col2:

        st.info(
            """
            ### Deep Learning

            **Input**

            Ultrasound image

            **Algorithm**

            Convolutional Neural Network

            **Analysis**

            Visual patterns in ultrasound image

            **Output**

            Infected / Non-infected probability
            """
        )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "ML Model Performance"
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
                    "ML Accuracy",
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
                    "Training Dataset",
                    str(rows)
                )

    else:

        st.info(
            "ML model information is not available."
        )

    st.divider()

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "Complete AI Pipeline"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.write("### Step 1")
        st.write("Patient Data")

    with p2:

        st.write("### Step 2")
        st.write("ML Risk")

    with p3:

        st.write("### Step 3")
        st.write("Ultrasound CNN")

    with p4:

        st.write("### Step 4")
        st.write("AI Summary")

    st.divider()

    st.success(
        "Use the sidebar to perform ML risk assessment "
        "or ultrasound deep-learning analysis."
    )


# ============================================================
# ML RISK ASSESSMENT
# ============================================================

elif page == "ML Risk Assessment":

    st.header(
        "Machine Learning — PCOS Risk Assessment"
    )

    st.write(
        """
        Enter patient-related clinical and symptom information.
        The trained **Random Forest model** will estimate the
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
        The model was trained using the same feature list
        saved during ML training. Default values are taken
        from the project dataset; change them according to
        the patient information before prediction.
        """
    )

    # --------------------------------------------------------
    # CREATE INPUT FORM
    # --------------------------------------------------------

    with st.form(
        "ml_prediction_form"
    ):

        st.subheader(
            "Patient Information"
        )

        input_values = {}

        # ----------------------------------------------------
        # GROUP FEATURES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GET DATASET DEFAULTS
        # ----------------------------------------------------

        numeric_defaults = {}

        if dataset is not None:

            temp_df = dataset.copy()

            temp_df.columns = (
                temp_df.columns
                .astype(str)
                .str.strip()
            )

            # Same cleaning approach as training

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

        # ----------------------------------------------------
        # DISPLAY INPUTS
        # ----------------------------------------------------

        for index, feature in enumerate(
            ml_features
        ):

            default = numeric_defaults.get(
                feature,
                0.0
            )

            label = feature

            # -----------------------------------------------
            # YES / NO
            # -----------------------------------------------

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

            # -----------------------------------------------
            # CYCLE
            # -----------------------------------------------

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

            # -----------------------------------------------
            # NUMERIC
            # -----------------------------------------------

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

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

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

            # Save result

            st.session_state.ml_result = {
                "prediction": int(prediction),
                "probability": probability
            }

            st.divider()

            st.subheader(
                "ML Prediction Result"
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

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == 1:

                st.error(
                    "ML Prediction: PCOS Positive"
                )

                st.warning(
                    f"""
                    The Random Forest model estimated a
                    **{probability * 100:.2f}% probability**
                    for the positive PCOS class.
                    """
                )

            else:

                st.success(
                    "ML Prediction: PCOS Negative"
                )

                st.success(
                    f"""
                    The Random Forest model estimated a
                    **{probability * 100:.2f}% probability**
                    for the positive PCOS class.
                    """
                )

            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            st.subheader(
                "Risk Probability"
            )

            st.progress(
                probability
            )

            st.caption(
                "0% = lower model probability | "
                "100% = higher model probability"
            )

            st.divider()

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

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

    st.header(
        "Deep Learning — Ultrasound Analysis"
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
            "The ML section can still be used. "
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

        # ----------------------------------------------------
        # IMAGE DISPLAY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

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

        # IMPORTANT:
        # CNN already contains Rescaling(1/255)
        # so do NOT divide image_array by 255 here.

        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

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

        # Your trained class mapping:
        #
        # class 0 = infected
        # class 1 = noninfected
        #
        # sigmoid output = probability of class 1

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

        # Save result

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

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "CNN Result"
            )

            if final_prediction == "INFECTED":

                st.error(
                    "INFECTED"
                )

            else:

                st.success(
                    "NON-INFECTED"
                )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Decision threshold: 0.50"
            )

        st.divider()

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TECHNICAL ANALYSIS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.subheader(
            "What did the CNN analyse?"
        )

        st.info(
            f"""
            The CNN produced a sigmoid output of
            **{raw_prediction:.4f}**.

            Because class 0 is **infected** and class 1 is
            **noninfected**, the system calculates:

            **Infected probability:**
            {(1 - raw_prediction) * 100:.2f}%

            **Non-infected probability:**
            {raw_prediction * 100:.2f}%

            The class with the higher probability becomes
            the displayed prediction.
            """
        )

        st.warning(
            """
            Academic project only.

            This ultrasound prediction is not a medical
            diagnosis and should not replace professional
            clinical evaluation.
            """
        )


# ============================================================
# AI SUMMARY
# ============================================================

elif page == "AI Summary":

    st.header(
        "Multi-Modal AI Summary"
    )

    st.write(
        """
        This page brings the outputs of both components
        together so you can understand what the project
        analyses.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # ML RESULT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DL RESULT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "How to interpret the two AI components"
    )

    st.write(
        """
        **ML component**

        Uses structured patient information such as
        clinical and symptom-related features to estimate
        PCOS probability.

        **DL component**

        Uses an ultrasound image and a CNN to classify
        the image into the trained ultrasound classes.

        These are two separate model outputs and should
        not be treated as a single medical diagnosis.
        """
    )

    if (
        st.session_state.ml_result is not None
        and st.session_state.cnn_result is not None
    ):

        st.success(
            """
            Both ML and DL analyses have been completed.

            You can use this page during your project
            demonstration to explain how the two AI
            components work together.
            """
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header(
        "AI Model Information"
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DL
    # --------------------------------------------------------

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
        "Project Models"
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

    st.header(
        "About the Project"
    )

    st.subheader(
        "Project Title"
    )

    st.write(
        """
        **AI-Based PCOS Risk Assessment and
        Ultrasound Image Analysis Using Machine
        Learning and Deep Learning**
        """
    )

    st.divider()

    st.subheader(
        "Objective"
    )

    st.write(
        """
        The objective of this project is to demonstrate
        an AI-based PCOS screening workflow using two
        complementary approaches:

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
        The training pipeline performs data cleaning,
        numerical conversion, median imputation and
        classification.
        """
    )

    st.divider()

    st.subheader(
        "Deep Learning"
    )

    st.write(
        """
        The DL component uses a Convolutional Neural
        Network to classify ultrasound images.
        """
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

    st.warning(
        """
        Academic Disclaimer

        This project is intended for educational and
        academic purposes. Model predictions should not
        be considered a medical diagnosis.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PCOS AI Screening System • ML + DL Capstone Project"
)