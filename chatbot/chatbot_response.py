import pandas as pd
from chatbot.disease_advice import get_advice
from chatbot.faq_answers import get_faq_response
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for extracting digital PDF text
import streamlit as st
import re
import wikipedia
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
from datetime import datetime

# ---------- DISCLAIMER ----------
DISCLAIMER = """
---
⚠️ **Disclaimer:**  
This assistant provides preliminary health information based on input and known data.  
It is *not* a substitute for professional medical advice, diagnosis, or treatment.  
Please consult a licensed medical professional for any health concerns.
"""

# ---------- LOAD DATA & MODELS ----------
model = SentenceTransformer('all-MiniLM-L6-v2')
df_raw = pd.read_csv("data/symptom_disease_dataset.csv", encoding='utf-8-sig')
symptom_cols = [col for col in df_raw.columns if col.lower().startswith("symptom")]
df = df_raw.melt(id_vars=["Disease"], value_vars=symptom_cols,
                 var_name="Symptom_#", value_name="Symptom")
df.dropna(subset=["Symptom"], inplace=True)
df["Symptom"] = df["Symptom"].str.strip().str.lower()
df["Disease"] = df["Disease"].str.strip()

all_known_symptoms = df["Symptom"].unique()
symptom_vectors = model.encode(all_known_symptoms, convert_to_tensor=True)

# ---------- FOLLOW-UP & NORMALIZATION MAPS ----------
follow_up = {
    "chest pain": "Do you also feel pain in your left arm or dizziness?",
    "fever": "Do you have chills or night sweats?",
    "fatigue": "Are you also feeling muscle weakness or drowsiness?",
    "cough": "Is the cough dry or with mucus?",
    "nausea": "Are you experiencing vomiting as well?"
}

alias_map = {
    "heart pain": "chest pain", "pain in chest": "chest pain", "my heart hurts": "chest pain",
    "blurry vision": "blurred vision", "throat hurting": "sore throat", "tummy pain": "abdominal pain",
    "pee a lot": "frequent urination", "hard to breathe": "shortness of breath",
    "i feel fine": "no symptoms", "healthy": "no symptoms", "can't breathe": "shortness of breath",
    "sweating a lot": "sweating", "left arm pain": "shoulder pain", "short of breath": "shortness of breath",
    "light headed": "dizziness", "tired all the time": "always tired", "pee frequently": "frequent urination"
}

disease_priority_boost = {
    "heart attack": ["chest pain", "shoulder pain", "left arm pain", "breathlessness"],
    "stroke": ["slurred speech", "numbness", "dizziness"],
    "diabetes": ["frequent urination", "blurred vision", "thirst"]
}

report_disease_map = {
    "high risk of diabetes": "Diabetes",
    "no heart disease detected": "Heart Disease",
    "heart disease": "Heart Disease",
    "diabetes": "Diabetes",
    "stroke": "Stroke",
    "risk of stroke": "Stroke"
}

# ---------- ENHANCED MENTAL HEALTH DETECTION ----------
distress_keywords = [
    "depressed", "hopeless", "want to die", "suicidal", "suicide",
    "no reason to live", "kill myself", "anxious", "panic", "overwhelmed",
    "worthless", "alone", "scared", "help me", "cant cope", "breaking down",
    "mental breakdown", "losing my mind", "going crazy", "hate myself",
    "self harm", "hurt myself", "end it all", "give up", "no point",
    "stressed out", "burnout", "exhausted mentally", "cant handle",
    "feel empty", "numb", "isolation", "lonely", "abandoned",
    "crisis", "despair", "dark thoughts", "nothing matters", "no energy",
    "crying constantly", "emotional pain", "unbearable", "feeling lost",
    "my life is over", "no escape", "i am broken", "hurting inside",
    "cant go on", "dead inside", "not okay", "need help", "mental pain",
    "helpless", "hopelessness", "drained", "inner pain", "i feel like dying"
]


anxiety_keywords = [
    "anxiety", "panic attack", "nervous", "worried", "restless",
    "heart racing", "can't breathe", "sweating", "trembling", "fear"
]

depression_keywords = [
    "sad", "crying", "tears", "empty", "hopeless", "worthless",
    "tired", "no energy", "sleep problems", "insomnia", "appetite"
]

def detect_mental_distress(text):
    lowered = text.lower()
    
    # Check for direct distress keywords
    distress_detected = any(keyword in lowered for keyword in distress_keywords)
    
    # Check for anxiety-related terms
    anxiety_detected = any(keyword in lowered for keyword in anxiety_keywords)
    
    # Check for depression-related terms  
    depression_detected = any(keyword in lowered for keyword in depression_keywords)
    
    # Check for urgent/crisis language
    crisis_phrases = ["want to die", "kill myself", "end it all", "suicidal", "suicide"]
    crisis_detected = any(phrase in lowered for phrase in crisis_phrases)
    
    return {
        'has_distress': distress_detected or anxiety_detected or depression_detected,
        'crisis_level': crisis_detected,
        'type': 'crisis' if crisis_detected else 'anxiety' if anxiety_detected else 'depression' if depression_detected else 'general'
    }

def mental_health_support_message(distress_info):
    base_message = "💙 *I notice you may be going through a difficult time.*\n"
    
    if distress_info['crisis_level']:
        return f"""
🚨 **URGENT - Crisis Support Available** 🚨

{base_message}
**If you're having thoughts of suicide or self-harm, please reach out immediately:**

🆘 **EMERGENCY CONTACTS:**
• **India**: National Suicide Prevention Helpline: **91-9152987821**
• **US**: 988 Suicide & Crisis Lifeline: **988**
• **UK**: Samaritans: **116 123**
• **Emergency Services**: **Call 911/112/108**

💬 **24/7 Online Support:**
• [iCall India](https://icallhelpline.org/)
• [Crisis Text Line](https://www.crisistextline.org/) - Text HOME to 741741

🏥 **You can also go to your nearest emergency room**

💝 **Remember**: You matter, your life has value, and help is available. These feelings can change with proper support.
"""
    
    elif distress_info['type'] == 'anxiety':
        return f"""
💙 **Anxiety Support** 💙

{base_message}
**For anxiety and panic-related concerns:**

📞 **Helplines:**
• **India**: iCall - **9152987821** (Mon-Sat, 10am-8pm)
• **US**: Anxiety & Depression Association - **240-485-1001**
• **Anxiety UK**: **03444 775 774**

🧘 **Immediate Anxiety Relief:**
• Try deep breathing: 4 counts in, hold for 4, out for 4
• Ground yourself: Name 5 things you can see, 4 you can hear, 3 you can touch
• Remember: This feeling will pass

💬 **Online Resources:**
• [Anxiety India](https://anxietyindia.com/)
• [ADAA Resources](https://adaa.org/)
"""
    
    elif distress_info['type'] == 'depression':
        return f"""
💙 **Depression Support** 💙

{base_message}
**For depression and low mood:**

📞 **Helplines:**
• **India**: iCall - **9152987821**
• **US**: SAMHSA National Helpline - **1-800-662-4357**
• **Beyond Blue (Australia)**: **1300 22 4636**

💡 **Small Steps That Help:**
• Reach out to one trusted person today
• Try to get some sunlight and fresh air
• Even 5 minutes of gentle movement can help
• You don't have to face this alone

💬 **Online Support:**
• [iCall India](https://icallhelpline.org/)
• [7 Cups - Free Emotional Support](https://www.7cups.com/)
"""
    
    else:
        return f"""
💙 **Mental Health Support** 💙

{base_message}
**Support is available when you need it:**

📞 **General Mental Health Support:**
• **India**: iCall - **9152987821**
• **US**: SAMHSA - **1-800-662-4357**
• **Mental Health America**: **Text MHA to 741741**

🏥 **Professional Help:**
• Consider speaking with a counselor or therapist
• Your doctor can provide mental health referrals
• Many employers offer Employee Assistance Programs (EAP)

💪 **Self-Care Reminders:**
• It's okay to not be okay sometimes
• Seeking help is a sign of strength, not weakness
• Small steps forward still count as progress

💬 **24/7 Online Support:**
• [iCall India](https://icallhelpline.org/)
• [BetterHelp](https://www.betterhelp.com/) (Professional counseling)
• [7 Cups](https://www.7cups.com/) (Peer support)
"""

# ---------- URGENCY DETECTION ----------
def detect_urgency(symptoms):
    emergency = ["chest pain", "shortness of breath", "left arm pain", "slurred speech", "loss of consciousness"]
    moderate = ["fever", "fatigue", "cough", "nausea", "dizziness", "headache"]

    score = 0
    for symptom in symptoms:
        if symptom in emergency:
            score += 2
        elif symptom in moderate:
            score += 1

    if score >= 3:
        return "🔴 *Urgency Level:* Emergency"
    elif score == 2:
        return "🟡 *Urgency Level:* Moderate"
    else:
        return "🟢 *Urgency Level:* Mild"

# ---------- PDF GENERATION ----------
def generate_health_summary_pdf(analysis_data):
    """
    Generate a PDF health summary report
    analysis_data should contain: disease, confidence, urgency, symptoms, advice, summary, alternatives
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        textColor=colors.darkred
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        alignment=TA_JUSTIFY
    )
    
    # Title
    title = Paragraph("AI Health Summary Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 20))
    
    # Generation date
    date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    date_para = Paragraph(date_text, normal_style)
    elements.append(date_para)
    elements.append(Spacer(1, 20))
    
    # Most Likely Disease
    if analysis_data.get('disease'):
        disease_heading = Paragraph("Most Likely Condition", heading_style)
        elements.append(disease_heading)
        disease_text = Paragraph(f"<b>{analysis_data['disease']}</b>", normal_style)
        elements.append(disease_text)
        elements.append(Spacer(1, 10))
    
    # Confidence Score
    if analysis_data.get('confidence'):
        conf_heading = Paragraph("Confidence Score", heading_style)
        elements.append(conf_heading)
        conf_text = Paragraph(f"{analysis_data['confidence']:.2f} ({analysis_data['confidence']*100:.1f}%)", normal_style)
        elements.append(conf_text)
        elements.append(Spacer(1, 10))
    
    # Urgency Level
    if analysis_data.get('urgency'):
        urgency_heading = Paragraph("Urgency Assessment", heading_style)
        elements.append(urgency_heading)
        urgency_clean = analysis_data['urgency'].replace('🔴', 'RED').replace('🟡', 'YELLOW').replace('🟢', 'GREEN').replace('*', '')
        urgency_text = Paragraph(urgency_clean, normal_style)
        elements.append(urgency_text)
        elements.append(Spacer(1, 10))
    
    # Detected Symptoms
    if analysis_data.get('symptoms'):
        symptoms_heading = Paragraph("Detected Symptoms", heading_style)
        elements.append(symptoms_heading)
        symptoms_text = Paragraph(', '.join(analysis_data['symptoms']), normal_style)
        elements.append(symptoms_text)
        elements.append(Spacer(1, 10))
    
    # Medical Description
    if analysis_data.get('summary'):
        desc_heading = Paragraph("Medical Description", heading_style)
        elements.append(desc_heading)
        desc_text = Paragraph(analysis_data['summary'], normal_style)
        elements.append(desc_text)
        elements.append(Spacer(1, 10))
    
    # Treatment Advice
    if analysis_data.get('advice'):
        advice_heading = Paragraph("Treatment Recommendations", heading_style)
        elements.append(advice_heading)
        advice_text = Paragraph(analysis_data['advice'], normal_style)
        elements.append(advice_text)
        elements.append(Spacer(1, 10))
    
    # Alternative Conditions
    if analysis_data.get('alternatives'):
        alt_heading = Paragraph("Other Possible Conditions", heading_style)
        elements.append(alt_heading)
        alt_text = Paragraph(', '.join(analysis_data['alternatives']), normal_style)
        elements.append(alt_text)
        elements.append(Spacer(1, 10))
    
    # Follow-up Question
    if analysis_data.get('follow_up'):
        followup_heading = Paragraph("Follow-up Question", heading_style)
        elements.append(followup_heading)
        followup_text = Paragraph(analysis_data['follow_up'], normal_style)
        elements.append(followup_text)
        elements.append(Spacer(1, 10))
    
    # Disclaimer
    elements.append(Spacer(1, 20))
    disclaimer_heading = Paragraph("Important Disclaimer", heading_style)
    elements.append(disclaimer_heading)
    disclaimer_text = """This assistant provides preliminary health information based on input and known data. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Please consult a licensed medical professional for any health concerns."""
    disclaimer_para = Paragraph(disclaimer_text, normal_style)
    elements.append(disclaimer_para)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- CORE FUNCTIONS ----------
def correct_input(text):
    return str(TextBlob(text).correct())

def normalize_text(text):
    for alias, standard in alias_map.items():
        if alias in text:
            text = text.replace(alias, standard)
    return text

def detect_alias_symptoms(user_input):
    return [standard for alias, standard in alias_map.items() if alias in user_input]

def extract_symptoms_with_similarity(user_input, threshold=0.45):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(input_embedding, symptom_vectors)[0]
    matched = [(all_known_symptoms[i], float(score)) for i, score in enumerate(scores) if score >= threshold]
    matched.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, score in matched[:10]]

def extract_text_from_pdf(file_obj):
    try:
        doc = fitz.open(stream=file_obj.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text if len(text.strip()) > 10 else ""
    except Exception:
        return ""

def extract_metrics_from_text(text):
    metrics = {}
    ignore_keys = ["Name", "Phone", "Email"]
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) == 2:
            key, val = parts
            key = key.strip()
            if key in ignore_keys:
                continue
            try:
                val = float(val.strip())
                metrics[key] = val
            except:
                continue
    return metrics

def plot_metrics_chart(metrics):
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(metrics.keys())
    values = [metrics[k] for k in keys]
    ax.barh(keys, values, color="#1f77b4")
    ax.set_xlabel("Values")
    ax.set_title("📊 Extracted Health Metrics")
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}", color='black', va='center', fontsize=8)
    st.pyplot(fig)

# ---------- REPORT ANALYSIS ----------
def analyze_report(file_obj):
    with st.spinner("🔍 Analyzing report and extracting insights..."):
        text = extract_text_from_pdf(file_obj)
        if not text:
            return "⚠ Couldn't extract readable text from the PDF report. Please try a different file." + DISCLAIMER

        lower_text = text.lower()
        for key_phrase, disease in report_disease_map.items():
            if key_phrase in lower_text:
                advice = get_advice(disease)
                try:
                    summary = wikipedia.summary(disease, sentences=2)
                except:
                    summary = "This condition may require further medical attention. Please consult a specialist."

                status_msg = (
                    "✅ Your report shows no signs of this disease."
                    if "no " in key_phrase else
                    "⚠ Your report shows indicators of this condition."
                )

                proactive_tip = {
                    "Diabetes": "🩺 Tip: Maintain blood sugar control and have an HbA1c test every 3 months.",
                    "Heart Disease": "💓 Tip: Monitor your blood pressure and avoid high-cholesterol food.",
                    "Stroke": "🧠 Tip: Avoid smoking, and stay on top of blood pressure and cholesterol."
                }.get(disease, "💡 General health tip: Regular checkups and exercise go a long way.")

                metrics = extract_metrics_from_text(text)
                plot_metrics_chart(metrics)

                # Prepare data for PDF generation
                analysis_data = {
                    'disease': disease,
                    'summary': summary,
                    'advice': advice,
                    'urgency': status_msg,
                    'symptoms': [],
                    'alternatives': [],
                    'confidence': 0.85  # Default confidence for report analysis
                }

                # Generate PDF download button
                pdf_buffer = generate_health_summary_pdf(analysis_data)
                st.download_button(
                    label="📄 Download AI Summary PDF",
                    data=pdf_buffer,
                    file_name=f"health_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Download a comprehensive PDF summary of your health analysis"
                )

                return f"""
📄 *Disease Mentioned in Report*:
{disease}

💬 *What is it?*
{summary}

{status_msg}

🩺 *Doctor's Advice & Treatment Tips:*
{advice}

🧠 *Health Tip:*
{proactive_tip}

🧾 *Note:* This was inferred based on patterns in the uploaded medical report.
{DISCLAIMER}
"""
        return get_bot_response(text)

# ---------- CHAT RESPONSE ----------
def get_bot_response(user_input):
    with st.spinner("💬 Analyzing your symptoms and generating response..."):
        user_input = normalize_text(correct_input(user_input.lower()))

        # Enhanced mental health detection
        distress_info = detect_mental_distress(user_input)
        if distress_info['has_distress']:
            return mental_health_support_message(distress_info) + DISCLAIMER

        if len(user_input.split()) < 2:
            return "⚠ Please describe more symptoms for accurate prediction." + DISCLAIMER

        if "no symptoms" in user_input:
            return "✅ You appear to be in good health. No signs of any disease." + DISCLAIMER

        alias_matched = set(detect_alias_symptoms(user_input))
        semantic_matched = set(extract_symptoms_with_similarity(user_input))
        matched_symptoms = list(alias_matched.union(semantic_matched))

        if not matched_symptoms:
            try:
                summary = wikipedia.summary(user_input, sentences=2)
                return f"""
🧾 *Health Info*:
{summary}
"""
            except:
                return "❗ Sorry, I couldn't find info about that condition. Please rephrase or ask something else." + DISCLAIMER

        cleaned_symptoms = sorted(set(symptom.replace("_", " ") for symptom in matched_symptoms))

        disease_scores = {}
        for disease in df["Disease"].unique():
            disease_symptoms = df[df["Disease"] == disease]["Symptom"].tolist()
            intersection = set(matched_symptoms).intersection(set(disease_symptoms))
            score = len(intersection) / len(set(disease_symptoms))

            for boost_symptom in disease_priority_boost.get(disease.lower(), []):
                if boost_symptom in matched_symptoms:
                    score += 0.3
            if alias_matched.intersection(disease_symptoms):
                score += 0.2
            if score > 0:
                disease_scores[disease] = score

        if not disease_scores:
            return "✅ Based on your symptoms, no clear disease risk detected. You may be in good health." + DISCLAIMER

        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        top_disease, confidence = sorted_diseases[0]

        if confidence < 0.3:
            return "⚠ Your symptoms are a bit unclear or mild. I recommend checking with a doctor for confirmation." + DISCLAIMER

        alt_diseases = [d[0] for d in sorted_diseases[1:4]]
        advice = get_advice(top_disease)
        try:
            summary = wikipedia.summary(top_disease, sentences=2)
        except:
            summary = "This condition may require further medical attention. Please consult a specialist."

        urgency_level = detect_urgency(matched_symptoms)

        # Prepare data for PDF generation
        analysis_data = {
            'disease': top_disease,
            'confidence': confidence,
            'urgency': urgency_level,
            'symptoms': cleaned_symptoms,
            'summary': summary,
            'advice': advice,
            'alternatives': alt_diseases
        }

        # Check for follow-up questions
        follow_up_question = None
        for symptom in matched_symptoms:
            if symptom in follow_up:
                follow_up_question = follow_up[symptom]
                analysis_data['follow_up'] = follow_up_question
                break

        # Generate PDF download button
        pdf_buffer = generate_health_summary_pdf(analysis_data)
        st.download_button(
            label="📄 Download AI Summary PDF",
            data=pdf_buffer,
            file_name=f"health_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            help="Download a comprehensive PDF summary of your health analysis"
        )

        response = f"""
🩺 *Most Likely Disease*:
{top_disease}
📊 *Confidence*: {confidence:.2f}
{urgency_level}

💬 *What is it?*
{summary}

💡 *Doctor's Advice & Treatment Tips:*
{advice}

🧠 *Detected Symptoms:*
{', '.join(cleaned_symptoms)}

📋 *Other Possible Conditions:*
{', '.join(alt_diseases)}
{DISCLAIMER}
"""

        if follow_up_question:
            response = f"""
🩺 *Most Likely Disease*:
{top_disease}
📊 *Confidence*: {confidence:.2f}
{urgency_level}

💬 *Follow-up Question:*
{follow_up_question}

💬 *What is it?*
{summary}

💡 *Doctor's Advice & Treatment Tips:*
{advice}

🧠 *Detected Symptoms:*
{', '.join(cleaned_symptoms)}

📋 *Other Possible Conditions:*
{', '.join(alt_diseases)}
{DISCLAIMER}
"""

        return response