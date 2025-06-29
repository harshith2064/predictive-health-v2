import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
from datetime import datetime
import re
from utils.explain import get_shap_values, generate_summary

# Load model
model = joblib.load("model/heart_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

# Professional CSS Styling - Same as Diabetes Predictor
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #f8fbff 0%, #e8f4fd 50%, #deedf7 100%);
            min-height: 100vh;
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            position: relative;
        }

        /* Background Pattern */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(26, 117, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(77, 166, 255, 0.08) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Header Section */
        .header-section {
            text-align: center;
            padding: 3rem 2rem 2rem 2rem;
            margin-bottom: 3rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,251,255,0.98) 100%);
            border-radius: 24px;
            box-shadow: 
                0 8px 32px rgba(0,31,63,0.08),
                0 1px 0px rgba(255,255,255,0.5) inset;
            border: 1px solid rgba(255,255,255,0.3);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }

        .header-section::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(from 0deg at 50% 50%, transparent 0deg, rgba(26, 117, 255, 0.12) 90deg, transparent 180deg, rgba(77, 166, 255, 0.12) 270deg, transparent 360deg);
            animation: rotate 20s linear infinite;
            z-index: -1;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .page-title {
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            font-size: 1.2rem;
            color: #dc2626;
            margin-bottom: 1.5rem;
            font-weight: 500;
            line-height: 1.5;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }

        .divider {
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            margin: 1.5rem auto;
            border-radius: 2px;
        }

        /* Section Cards */
        .section-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 
                0 4px 20px rgba(0,0,0,0.08),
                0 1px 0px rgba(255,255,255,0.8) inset;
            border: 1px solid rgba(226, 232, 240, 0.8);
            position: relative;
            overflow: hidden;
        }

        .section-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #dc2626;
            margin-bottom: 1.5rem;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .section-subtitle {
            font-size: 1rem;
            color: #dc2626;
            margin-bottom: 2rem;
            line-height: 1.6;
            font-weight: 500;
        }

        /* Medical inputs vertical layout */
        .medical-inputs-vertical {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-width: 500px;
            margin: 0 auto;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        /* Medical inputs grid layout */
        .medical-inputs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 0 auto;
        }

        /* Streamlit input styling */
        .stNumberInput > div > div > input {
            background-color: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #dc2626 !important;
            font-weight: 500 !important;
        }

        .stNumberInput > div > div > input:focus {
            border-color: #dc2626 !important;
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
        }

        .stTextInput > div > div > input {
            background-color: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #dc2626 !important;
            font-weight: 500 !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #dc2626 !important;
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
        }

        .stSelectbox > div > div > div {
            background-color: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #dc2626 !important;
            font-weight: 500 !important;
        }

        .stSelectbox > div > div > div:focus-within {
            border-color: #dc2626 !important;
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
        }

        .stSlider > div > div > div > div {
            background-color: #dc2626 !important;
        }

        /* Button Styling */
        .predict-button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
            width: 100% !important;
            margin-top: 1rem !important;
        }

        .predict-button:hover {
            background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(220, 38, 38, 0.4) !important;
        }

        /* Results Section */
        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 
                0 8px 32px rgba(0,0,0,0.1),
                0 1px 0px rgba(255,255,255,0.8) inset;
            border: 1px solid rgba(226, 232, 240, 0.8);
            position: relative;
            overflow: hidden;
        }

        .result-card.high-risk::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
        }

        .result-card.no-risk::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        }

        /* Success/Error Messages */
        .custom-success {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: #065f46;
            font-weight: 600;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .custom-error {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 2px solid #ef4444;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: #991b1b;
            font-weight: 600;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Chart Container */
        .chart-container {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid rgba(226, 232, 240, 0.8);
            margin: 1.5rem 0;
        }

        /* Download Button */
        .download-section {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(14, 165, 233, 0.2);
            margin-top: 2rem;
        }

        .download-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #0369a1;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        /* Warning Messages */
        .custom-warning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 2px solid #f59e0b;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: #92400e;
            font-weight: 600;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Reasoning text styling */
        .reasoning-text {
            color: #dc2626 !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            font-weight: 500 !important;
            margin-top: 1rem !important;
        }

        .reasoning-text ul {
            list-style-type: disc !important;
            padding-left: 1.5rem !important;
            margin: 0 !important;
        }

        .reasoning-text li {
            color: #dc2626 !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            font-weight: 500 !important;
            margin-bottom: 0.5rem !important;
        }

        /* Download section text */
        .download-section p {
            color: #dc2626 !important;
            font-weight: 500 !important;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .page-title {
                font-size: 2.5rem;
            }
            
            .section-card {
                padding: 2rem 1.5rem;
            }
            
            .header-section {
                padding: 2rem 1.5rem;
            }

            .medical-inputs-vertical {
                max-width: 100%;
            }

            .medical-inputs-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Hide Streamlit elements */
        .stDeployButton {
            display: none;
        }
        
        #MainMenu {
            visibility: hidden;
        }
        
        .stAppHeader {
            display: none;
        }

        /* Input labels */
        .stNumberInput label,
        .stTextInput label,
        .stSelectbox label,
        .stSlider label {
            color: #dc2626 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }

        /* Additional text visibility fixes */
        .section-card p,
        .section-card div,
        .section-card span {
            color: #dc2626 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-section">
    <div class="page-title">❤️ Heart Disease Risk Predictor</div>
    <div class="page-subtitle">
        Advanced AI-powered analysis for early heart disease detection and comprehensive cardiovascular health assessment
    </div>
    <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# Patient Information Section
st.markdown("""
<div class="section-card">
    <div class="section-title">
        👤 Patient Information
    </div>
    <div class="section-subtitle">
        Please provide accurate patient details for personalized health assessment and report generation
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input("👤 Patient Name", placeholder="Enter full name")

with col2:
    phone = st.text_input("📱 Phone Number", max_chars=10, placeholder="10-digit number")
    valid_phone = phone.isdigit() and len(phone) == 10
    if phone and not valid_phone:
        st.markdown('<div class="custom-warning">⚠ Phone number must be exactly 10 digits.</div>', unsafe_allow_html=True)

with col3:
    email = st.text_input("📧 Email (Optional)", placeholder="user@gmail.com")
    email_regex = r"^[\w\.-]+@gmail\.com$"
    valid_email = (not email) or re.match(email_regex, email)
    if email and not valid_email:
        st.markdown('<div class="custom-warning">⚠ Please enter a valid Gmail address.</div>', unsafe_allow_html=True)

# Medical Information Section
st.markdown("""
<div class="section-card">
    <div class="section-title">
        🏥 Medical Assessment
    </div>
    <div class="section-subtitle">
        Enter comprehensive medical data for accurate AI-powered heart disease risk prediction
    </div>
</div>
""", unsafe_allow_html=True)

# Medical inputs in a grid layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("👶 Age", 20, 90, 50, help="Patient's age in years")
    sex = st.selectbox("👥 Sex", ["Male", "Female"], help="Patient's biological sex")
    
    cp_map = {
        "Typical Angina (TA)": "TA",
        "Atypical Angina (ATA)": "ATA",
        "Non-Anginal Pain (NAP)": "NAP",
        "Asymptomatic (ASY)": "ASY"
    }
    cp_display = st.selectbox("💔 Chest Pain Type", list(cp_map.keys()), help="Type of chest pain experienced")
    cp = cp_map[cp_display]
    
    bp = st.number_input("💓 Resting Blood Pressure (mm Hg)", 60, 200, 120, help="Resting blood pressure")
    chol = st.number_input("🧪 Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol level")
    fbs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"], help="Fasting blood sugar greater than 120 mg/dL")

with col2:
    ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST [Segment]", "LVH [Left Ventricular Hypertrophy]"], help="Resting electrocardiogram results")
    maxhr = st.slider("❤️ Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
    angina = st.selectbox("⚡ Exercise-Induced Angina", ["Yes", "No"], help="Exercise-induced angina")
    oldpeak = st.number_input("📉 Oldpeak (ST Depression)", value=1.0, help="ST depression induced by exercise")
    slope = st.selectbox("📊 ST Segment Slope", ["Up", "Flat", "Down"], help="Slope of peak exercise ST segment")

# Prediction Button
if st.button("❤️ Analyze Heart Disease Risk", key="predict_btn"):
    if not patient_name.strip():
        st.markdown('<div class="custom-warning">⚠ Please enter patient name.</div>', unsafe_allow_html=True)
    elif not valid_phone:
        st.markdown('<div class="custom-warning">⚠ Please enter a valid 10-digit phone number.</div>', unsafe_allow_html=True)
    elif email and not valid_email:
        st.markdown('<div class="custom-warning">⚠ Please enter a valid Gmail address.</div>', unsafe_allow_html=True)
    else:
        data = {
            "Age": age,
            "Sex": 1 if sex == "Male" else 0,
            "ChestPainType": cp,
            "RestingBP": bp,
            "Cholesterol": chol,
            "FastingBS": 1 if fbs == "Yes" else 0,
            "RestingECG": ecg,
            "MaxHR": maxhr,
            "ExerciseAngina": 'Y' if angina == "Yes" else 'N',
            "Oldpeak": oldpeak,
            "ST_Slope": slope
        }

        input_df = pd.DataFrame([data])
        input_encoded = pd.get_dummies(input_df)
        for col in model.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_].astype(float)

        prediction = model.predict(input_encoded)[0]
        result_text = "High Risk of Heart Disease" if prediction == 1 else "No Heart Disease Detected"

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card high-risk">
                <div class="custom-error">🔺 High Risk of Heart Disease Detected</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card no-risk">
                <div class="custom-success">✅ No Heart Disease Risk Detected</div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP Feature Impact Analysis
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                📊 AI Feature Impact Analysis
            </div>
            <div class="section-subtitle">
                Understanding which factors contributed most to the heart disease risk prediction
            </div>
        """, unsafe_allow_html=True)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_encoded)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # Class 1
            else:
                shap_vals = shap_values[0]

            shap_vals = np.array(shap_vals).flatten()[:input_encoded.shape[1]]

            if not np.any(np.abs(shap_vals) > 1e-6):
                st.markdown('<div class="custom-warning">⚠ SHAP values are all near zero.</div>', unsafe_allow_html=True)
            elif len(shap_vals) != len(input_encoded.columns):
                st.markdown('<div class="custom-warning">⚠ SHAP mismatch: input features and SHAP values don\'t align.</div>', unsafe_allow_html=True)
                st.text(f"Features: {len(input_encoded.columns)}, SHAP: {len(shap_vals)}")
            else:
                shap_df = pd.DataFrame({
                    "Feature": input_encoded.columns,
                    "SHAP Value": shap_vals
                }).sort_values("SHAP Value", key=abs, ascending=True)

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#ef4444' if x > 0 else '#10b981' for x in shap_df["SHAP Value"]]
                ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, alpha=0.8)
                ax.set_xlabel("SHAP Value (Impact on Prediction)", fontsize=12, fontweight='bold')
                ax.set_title("Feature Importance in Heart Disease Risk Prediction", fontsize=14, fontweight='bold', pad=20)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                
                # Customize appearance
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div class="custom-warning">⚠ Could not generate feature impact chart.</div>', unsafe_allow_html=True)
            st.text(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

        # Model Explanation Section
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                🧠 AI Model Reasoning
            </div>
            <div class="section-subtitle">
                Detailed explanation of why this prediction was made
            </div>
        """, unsafe_allow_html=True)

        reasons = generate_summary(input_encoded.iloc[0], input_encoded.columns)
        
        # Simple bullet points with dark red text
        st.markdown('<div class="reasoning-text">', unsafe_allow_html=True)
        reason_html = "<ul>"
        for reason in reasons:
            reason_html += f"<li>{reason}</li>"
        reason_html += "</ul>"
        
        st.markdown(reason_html, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # PDF Report Generation
        st.markdown("""
        <div class="download-section">
            <div class="download-title">
                📄 Generate Comprehensive Report
            </div>
            <p style="color: #dc2626; margin-bottom: 1.5rem; font-weight: 500;">
                Download a detailed PDF report with complete analysis, recommendations, and cardiovascular health insights
            </p>
        """, unsafe_allow_html=True)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=1, align="C")
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.ln(3)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Patient Details:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 8, txt=f"Name: {patient_name}", ln=1)
        pdf.cell(200, 8, txt=f"Phone: {phone}", ln=1)
        if email:
            pdf.cell(200, 8, txt=f"Email: {email}", ln=1)
        pdf.ln(3)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Prediction Result:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, txt=result_text)
        pdf.ln(2)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Entered Medical Data:", ln=1)
        pdf.set_font("Arial", size=11)
        for key, val in data.items():
            pdf.cell(200, 8, txt=f"{key}: {val}", ln=1)
        pdf.ln(3)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Model Reasoning:", ln=1)
        pdf.set_font("Arial", size=11)
        for reason in reasons:
            pdf.multi_cell(0, 8, txt=f"- {reason}")
        pdf.ln(3)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Doctor's Advice:", ln=1)
        pdf.set_font("Arial", size=11)
        if prediction == 1:
            pdf.multi_cell(0, 8, txt=(
                "Recommendations:\n"
                "- Visit a cardiologist immediately.\n"
                "- Get ECG, ECHO, and comprehensive blood tests.\n"
                "- Avoid high-fat diets and smoking.\n"
                "- Exercise regularly and monitor cardiovascular symptoms."
            ))
        else:
            pdf.multi_cell(0, 8, txt=(
                "To maintain a healthy heart:\n"
                "- Eat low-cholesterol, heart-healthy foods.\n"
                "- Walk or jog at least 30 minutes daily.\n"
                "- Manage stress and monitor blood pressure regularly.\n"
                "- Avoid alcohol and smoking completely."
            ))

        pdf.ln(3)
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="General Heart Health Tips:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, txt=(
            "- Stay physically active and maintain a healthy weight\n"
            "- Reduce sodium and sugar intake significantly\n"
            "- Monitor cholesterol levels regularly\n"
            "- Get 7-8 hours of quality sleep per night\n"
            "- Manage stress effectively and stay well-hydrated"
        ))

        pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="ignore")
        buffer = io.BytesIO(pdf_bytes)

        st.download_button(
            label="📥 Download PDF Report",
            data=buffer,
            file_name=f"heart_report_{patient_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)