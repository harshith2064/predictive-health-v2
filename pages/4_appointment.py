import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import tempfile

# ✅ Email credentials
sender_email = "harshithkm2004@gmail.com"
sender_password = "cuvbnjnqhlyauapj"
doctor_email = "jnanesh229@gmail.com"

# 🚨 Emergency keywords
emergency_keywords = [
    "chest pain", "shortness of breath", "unconscious",
    "severe headache", "difficulty breathing"
]

# ---------------------------
# Send email (with optional file)
# ---------------------------
def send_emergency_email(name, phone, symptoms, duration, appointment_time, uploaded_file):
    subject = "🚨 Emergency Alert: Patient Needs Immediate Attention"
    
    body = f"""
    EMERGENCY ALERT - AUTO GENERATED

    Patient Name: {name}
    Phone Number: {phone}
    Symptoms: {symptoms}
    Duration: {duration}
    Preferred Appointment Slot: {appointment_time}
    """

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = doctor_email
    msg.attach(MIMEText(body))

    # Attach uploaded file if present
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=uploaded_file.name)
            part['Content-Disposition'] = f'attachment; filename="{uploaded_file.name}"'
            msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        return f"❌ Email error: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="HealthBot 1 - Emergency Checker", layout="wide")
st.title("🩺 Emergency Medical Symptom Checker")

st.subheader("👤 Patient Information")
patient_name = st.text_input("Full Name")
patient_phone = st.text_input("Phone Number")

symptoms = [
    "Chest Pain", "Headache", "Stomach Pain", "Cough", "Fever",
    "Dizziness", "Shortness of Breath", "Fatigue", "Vomiting", "Nausea",
    "Body Pain", "Joint Pain", "Leg Pain", "Back Pain", "Neck Pain",
    "Sore Throat", "Eye Pain", "Ear Pain", "Muscle Ache", "Skin Rash"
]
durations = ["1 day", "1–2 days", "2–3 days", "3–5 days", "More than 5 days"]
appointment_options = [
    "Today Morning (9am–11am)", "Today Afternoon (2pm–4pm)",
    "Today Evening (5pm–7pm)", "Tomorrow Morning",
    "Tomorrow Afternoon", "Tomorrow Evening",
    "Urgent - Any Earliest Slot"
]

selected_symptoms = st.multiselect("🤒 Select symptoms:", symptoms)
selected_duration = st.selectbox("📆 Duration of symptoms:", durations)
appointment_time = st.selectbox("📅 Preferred Appointment Time:", appointment_options)

# 📎 File upload
uploaded_file = st.file_uploader("📁 Upload any reports/images for the doctor (optional)", type=["pdf", "jpg", "png", "jpeg", "docx"])

# ---------------------------
# Submit
# ---------------------------
if st.button("📩 Send Emergency Email to Doctor"):
    if not patient_name or not patient_phone:
        st.warning("Please enter your name and phone number.")
    elif not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        symptom_list = ", ".join(sym.lower() for sym in selected_symptoms)

        is_emergency = (
            any(
                keyword in symptom.lower()
                for symptom in selected_symptoms
                for keyword in emergency_keywords
            ) or len(selected_symptoms) >= 5
        )

        if is_emergency:
            with st.spinner("Sending emergency email to doctor..."):
                result = send_emergency_email(
                    name=patient_name,
                    phone=patient_phone,
                    symptoms=symptom_list,
                    duration=selected_duration,
                    appointment_time=appointment_time,
                    uploaded_file=uploaded_file
                )
                if result is True:
                    st.success("✅ Emergency email sent to doctor.")
                else:
                    st.error(result)
        else:
            st.info("⚠️ No emergency detected, but details are available for review.")