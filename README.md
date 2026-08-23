# 🌸 PCOS AI — Risk Assessment & Ultrasound Image Analysis

An AI-based screening system for Polycystic Ovary Syndrome (PCOS) that combines **Machine Learning, Deep Learning, and Streamlit** to provide risk assessment from clinical data and analyze ultrasound images.

> ⚠️ **Disclaimer:** This project is developed for educational and screening-support purposes only. It is not intended to provide medical diagnosis or replace professional medical advice.

---

## 📌 Project Overview

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder that can affect women's reproductive and metabolic health.

This project aims to build an AI-powered screening system that uses two different Artificial Intelligence approaches:

- 🤖 **Machine Learning** for PCOS risk assessment using structured/clinical data.
- 🧠 **Deep Learning (CNN)** for classification of ultrasound images.
- 📊 **Streamlit** for an interactive web-based dashboard.

The system provides a simple interface where users can enter relevant health information and/or upload an ultrasound image to obtain an AI-based screening result.

---

## 🎯 Objectives

1. To develop a Machine Learning model for PCOS risk assessment.
2. To develop a CNN-based Deep Learning model for ultrasound image classification.
3. To integrate both models into a single Streamlit application.
4. To provide an easy-to-use interactive dashboard.
5. To demonstrate the practical application of AI in women's health screening.

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Random Forest | PCOS risk prediction |
| TensorFlow | Deep Learning |
| Keras | CNN model development |
| CNN | Ultrasound image classification |
| Streamlit | Web application |
| Matplotlib / Seaborn | Data visualization |

---

## 🏗️ System Architecture

```text
                    PCOS AI SYSTEM
                          |
             +------------+------------+
             |                         |
       Clinical Data              Ultrasound Image
             |                         |
             ↓                         ↓
      Data Preprocessing        Image Preprocessing
             |                         |
             ↓                         ↓
      Random Forest ML             CNN Model
             |                         |
             ↓                         ↓
       PCOS Risk Score        Image Classification
             |                         |
             +------------+------------+
                          |
                          ↓
                Streamlit Dashboard
                          |
                          ↓
                 Screening Results
