import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import io
from fpdf import FPDF
from datetime import datetime
import re
from utils.explain import get_shap_values, generate_summary

# Load model
model = joblib.load("model/diabetes_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🩺")

# Professional CSS Styling (matching home page)
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
            background: linear-gradient(135deg, #1a75ff 0%, #4da6ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            font-size: 1.2rem;
            color: #1e40af;
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
            background: linear-gradient(90deg, #1a75ff 0%, #4da6ff 100%);
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
            background: linear-gradient(90deg, #1a75ff 0%, #4da6ff 100%);
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 1.5rem;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .section-subtitle {
            font-size: 1rem;
            color: #1e40af;
            margin-bottom: 2rem;
            line-height: 1.6;
            font-weight: 500;
        }

        /* Medical inputs vertical layout */
        .medical-inputs-vertical {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            max-width: 450px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,251,255,0.95) 100%);
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        /* Streamlit input styling */
        .stNumberInput > div > div > input {
            background-color: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 12px !important;
            color: #1e40af !important;
            font-weight: 500 !important;
            padding: 0.8rem 1rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
        }

        .stNumberInput > div > div > input:focus {
            border-color: #1a75ff !important;
            box-shadow: 0 0 0 3px rgba(26, 117, 255, 0.1) !important;
            transform: translateY(-1px) !important;
        }

        .stTextInput > div > div > input {
            background-color: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 12px !important;
            color: #1e40af !important;
            font-weight: 500 !important;
            padding: 0.8rem 1rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #1a75ff !important;
            box-shadow: 0 0 0 3px rgba(26, 117, 255, 0.1) !important;
            transform: translateY(-1px) !important;
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #1a75ff 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 1.2rem 2.5rem !important;
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 6px 20px rgba(26, 117, 255, 0.3) !important;
            width: 100% !important;
            margin: 2rem 0 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            position: relative !important;
            overflow: hidden !important;
        }

        .stButton > button:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .stButton > button:hover:before {
            left: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #0056cc 0%, #1d4ed8 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(26, 117, 255, 0.4) !important;
        }

        .stButton > button:active {
            transform: translateY(-1px) !important;
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

        .diabetes-type-card {
            background: linear-gradient(135deg, #fff0f5 0%, #fce7f3 100%);
            border: 2px solid #ec4899;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            position: relative;
        }

        .diabetes-type-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #be185d;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
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

        /* Reasoning List */
        .reasoning-list {
            background: #f8fafc;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
        }

        .reasoning-list ul {
            margin: 0;
            padding-left: 1.5rem;
        }

        .reasoning-list li {
            margin-bottom: 0.5rem;
            line-height: 1.6;
            color: #1e40af;
            font-weight: 500;
        }

        /* Download section text */
        .download-section p {
            color: #1e40af !important;
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
        .stTextInput label {
            color: #1e40af !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            margin-bottom: 0.5rem !important;
            display: block !important;
        }

        /* Input help text */
        .stNumberInput .help,
        .stTextInput .help {
            color: #64748b !important;
            font-size: 0.875rem !important;
            margin-top: 0.25rem !important;
        }

        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #1a75ff !important;
        }

        /* Better spacing for sections */
        .section-card + .section-card {
            margin-top: 3rem;
        }

        /* Improve chart readability */
        .chart-container {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(226, 232, 240, 0.8);
            margin: 2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-section">
    <div class="page-title">🩺 Diabetes Risk Predictor</div>
    <div class="page-subtitle">
        Advanced AI-powered analysis for early diabetes detection and comprehensive health assessment
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
        st.markdown('<div class="custom-warning">⚠️ Phone number must be exactly 10 digits.</div>', unsafe_allow_html=True)

with col3:
    email = st.text_input("📧 Email (Optional)", placeholder="user@gmail.com")
    email_regex = r"^[\w\.-]+@gmail\.com$"
    valid_email = (not email) or re.match(email_regex, email)
    if email and not valid_email:
        st.markdown('<div class="custom-warning">⚠️ Please enter a valid Gmail address.</div>', unsafe_allow_html=True)

# Medical Information Section - Vertical Layout
st.markdown("""
<div class="section-card">
    <div class="section-title">
        🏥 Medical Assessment
    </div>
    <div class="section-subtitle">
        Enter comprehensive medical data for accurate AI-powered diabetes risk prediction
    </div>
</div>
""", unsafe_allow_html=True)

# Create a centered column for vertical inputs
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="medical-inputs-vertical">', unsafe_allow_html=True)
    
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, help="Number of pregnancies")
    glucose = st.number_input("🩸 Glucose Level", min_value=0, help="Blood glucose level (mg/dL)")
    bp = st.number_input("💓 Blood Pressure", min_value=0, help="Systolic blood pressure (mmHg)")
    skin = st.number_input("📏 Skin Thickness", min_value=0, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input("💉 Insulin", min_value=0, help="Insulin level (μU/mL)")
    bmi = st.number_input("⚖️ BMI", min_value=0.0, format="%.1f", help="Body Mass Index")
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, format="%.3f", help="Genetic predisposition factor")
    age = st.number_input("👶 Age", min_value=1, help="Age in years")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Button - Centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔬 Analyze Diabetes Risk", key="predict_btn", use_container_width=True):
    if not patient_name.strip():
        st.markdown('<div class="custom-warning">⚠️ Please enter patient name.</div>', unsafe_allow_html=True)
    elif not valid_phone:
        st.markdown('<div class="custom-warning">⚠️ Please enter a valid 10-digit phone number.</div>', unsafe_allow_html=True)
    elif email and not valid_email:
        st.markdown('<div class="custom-warning">⚠️ Please enter a valid Gmail address.</div>', unsafe_allow_html=True)
    else:
        # Run Prediction
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]

        result_text = "High Risk of Diabetes" if prediction == 1 else "No Diabetes Detected"

        # Diabetes Type Estimation
        diabetes_type = ""
        if prediction == 1:
            if glucose > 180 and insulin < 50:
                diabetes_type = "Likely Type 1 Diabetes"
            elif glucose > 130 and bmi > 30 and dpf > 0.5:
                diabetes_type = "Likely Type 2 Diabetes"
            elif age < 20:
                diabetes_type = "Possible Juvenile Diabetes"
            else:
                diabetes_type = "Uncertain - Clinical diagnosis required"

            # Results Section for High Risk
            st.markdown(f"""
            <div class="result-card high-risk">
                <div class="diabetes-type-card">
                    <h3 class="diabetes-type-title">🧬 AI Prediction: {diabetes_type}</h3>
                </div>
                <div class="custom-error">🔺 High Risk of Diabetes Detected</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card no-risk">
                <div class="custom-success">✅ No Diabetes Risk Detected</div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP Feature Impact Analysis
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

        shap_vals = get_shap_values(input_data[0], model, feature_names)
        shap_vals = np.array(shap_vals).flatten()[:8]

        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                📊 AI Feature Impact Analysis
            </div>
            <div class="section-subtitle">
                Understanding which factors contributed most to the prediction result
            </div>
        """, unsafe_allow_html=True)

        if len(shap_vals) == len(feature_names):
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_vals
            }).sort_values("SHAP Value", key=abs, ascending=True)

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ef4444' if x > 0 else '#10b981' for x in shap_df["SHAP Value"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, alpha=0.8)
            ax.set_xlabel("SHAP Value (Impact on Prediction)", fontsize=12, fontweight='bold')
            ax.set_title("Feature Importance in Diabetes Risk Prediction", fontsize=14, fontweight='bold', pad=20)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Customize appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="custom-warning">⚠️ Could not generate feature impact chart due to data mismatch.</div>', unsafe_allow_html=True)

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
            <div class="reasoning-list">
        """, unsafe_allow_html=True)

        reasons = generate_summary(input_data[0], feature_names)
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
            <p style="color: #1e40af; margin-bottom: 1.5rem; font-weight: 500;">
                Download a detailed PDF report with complete analysis, recommendations, and health insights
            </p>
        """, unsafe_allow_html=True)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=1, align="C")
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.ln(3)

        if prediction == 1:
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(200, 10, txt="Estimated Diabetes Type:", ln=1)
            pdf.set_font("Arial", size=11)
            pdf.cell(200, 8, txt=diabetes_type, ln=1)
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
        pdf.cell(200, 10, txt="Entered Patient Data:", ln=1)
        pdf.set_font("Arial", size=11)
        for name, val in zip(feature_names, input_data[0]):
            pdf.cell(200, 8, txt=f"{name}: {val}", ln=1)
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
                "Based on the prediction, we recommend:\n"
                "- Consult a diabetologist.\n"
                "- Conduct a full blood sugar and HbA1c test.\n"
                "- Follow a low-sugar, high-fiber diet.\n"
                "- Increase physical activity and monitor glucose levels regularly."
            ))
        else:
            pdf.multi_cell(0, 8, txt=(
                "No current risk detected. To stay healthy:\n"
                "- Continue with healthy habits.\n"
                "- Schedule regular check-ups.\n"
                "- Stay alert to symptoms like fatigue, thirst, or frequent urination."
            ))
        pdf.ln(3)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="General Health Tips:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, txt=(
            "- Eat fiber-rich, low glycemic foods\n"
            "- Exercise at least 30 minutes daily\n"
            "- Drink plenty of water\n"
            "- Avoid sugary and processed foods\n"
            "- Sleep 6-8 hours each night"
        ))

        pdf_bytes = pdf.output(dest="S").encode("latin1")
        buffer = io.BytesIO(pdf_bytes)

        st.download_button(
            label="📥 Download PDF Report",
            data=buffer,
            file_name=f"diabetes_report_{patient_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)