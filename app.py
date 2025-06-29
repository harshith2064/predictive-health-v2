import streamlit as st

# --- Set page config ---
st.set_page_config(
    page_title="PredictiveHealthAI",
    page_icon="🧠",
    layout="wide"
)

# --- Routing logic using query parameters ---
query_params = st.query_params

if query_params.get("page") == "diabetes":
    st.switch_page("pages/1_Diabetes_Predictor.py")
elif query_params.get("page") == "heart":
    st.switch_page("pages/2_Heart_Predictor.py")
elif query_params.get("page") == "healthbot":
    st.switch_page("pages/3_HealthBot.py")
elif query_params.get("page") == "appointment":
    st.switch_page("pages/4_appointment.py")

# --- Professional CSS Styling ---
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

        /* Hero Section */
        .hero-section {
            text-align: center;
            padding: 5rem 2rem 4rem 2rem;
            margin-bottom: 4rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,251,255,0.98) 100%);
            border-radius: 32px;
            box-shadow: 
                0 8px 32px rgba(0,31,63,0.08),
                0 1px 0px rgba(255,255,255,0.5) inset;
            border: 1px solid rgba(255,255,255,0.3);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }

        .hero-section::before {
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

        .main-title {
            font-size: 4.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #001F3F 0%, #1a4d80 50%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }

        .subtitle {
            font-size: 1.4rem;
            color: #4a5568;
            margin-bottom: 2rem;
            font-weight: 400;
            line-height: 1.5;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }

        .divider {
            width: 120px;
            height: 4px;
            background: linear-gradient(90deg, #1a75ff 0%, #4da6ff 100%);
            margin: 2rem auto;
            border-radius: 2px;
        }

        /* Section Styling */
        .content-section {
            margin: 4rem 0;
            padding: 0 2rem;
        }

        .section-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a202c;
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: -0.01em;
        }

        .section-subtitle {
            font-size: 1.1rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 3rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        /* Prediction Modules Grid */
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 2rem;
            max-width: 800px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        .module-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            text-align: center;
            box-shadow: 
                0 4px 20px rgba(0,0,0,0.08),
                0 1px 0px rgba(255,255,255,0.8) inset;
            border: 1px solid rgba(226, 232, 240, 0.8);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .module-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #1a75ff 0%, #4da6ff 100%);
        }

        .module-card::after {
            content: '';
            position: absolute;
            top: 1rem;
            right: 1rem;
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        }

        .module-card:hover {
            transform: translateY(-12px) scale(1.02);
            box-shadow: 
                0 20px 60px rgba(0,0,0,0.15),
                0 1px 0px rgba(255,255,255,0.8) inset;
        }

        .module-icon {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            display: block;
        }

        .module-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1a202c;
            margin-bottom: 1rem;
            line-height: 1.3;
        }

        .module-description {
            font-size: 1rem;
            color: #64748b;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        .module-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 24px;
            background: linear-gradient(135deg, #1a75ff 0%, #2563eb 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            text-decoration: none !important;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(26, 117, 255, 0.3);
        }

        .module-button:hover {
            background: linear-gradient(135deg, #0056cc 0%, #1d4ed8 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(26, 117, 255, 0.4);
            color: white !important;
            text-decoration: none !important;
        }

        .module-button:visited,
        .module-button:active,
        .module-button:focus {
            color: white !important;
            text-decoration: none !important;
        }

        /* ChatBot Section */
        .chatbot-section {
            margin: 4rem 0;
            padding: 0 2rem;
        }

        .chatbot-container {
            max-width: 600px;
            margin: 0 auto;
        }

        .chatbot-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            text-align: center;
            box-shadow: 
                0 8px 32px rgba(0,0,0,0.1),
                0 1px 0px rgba(255,255,255,0.8) inset;
            border: 1px solid rgba(226, 232, 240, 0.8);
            position: relative;
            overflow: hidden;
        }

        .chatbot-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4da6ff 0%, #66b3ff 100%);
        }

        .chatbot-card::after {
            content: '🔴 LIVE';
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            font-size: 0.7rem;
            font-weight: 600;
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
            padding: 0.3rem 0.6rem;
            border-radius: 12px;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .chatbot-card .module-icon {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
        }

        .chatbot-card .module-title {
            font-size: 1.8rem;
            color: #1a202c;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }

        .chatbot-card .module-description {
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 2.5rem;
            line-height: 1.6;
        }

        .chatbot-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #4da6ff 0%, #66b3ff 100%);
            color: white !important;
            padding: 16px 32px;
            border-radius: 12px;
            text-decoration: none !important;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(77, 166, 255, 0.3);
            border: none;
            cursor: pointer;
        }

        .chatbot-button:hover {
            background: linear-gradient(135deg, #1a75ff 0%, #4da6ff 100%);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(77, 166, 255, 0.4);
            color: white !important;
            text-decoration: none !important;
        }

        .chatbot-button:visited,
        .chatbot-button:active,
        .chatbot-button:focus {
            color: white !important;
            text-decoration: none !important;
        }

        /* Appointment Section */
        .appointment-section {
            margin: 4rem 0;
            padding: 0 2rem;
        }

        .appointment-container {
            max-width: 600px;
            margin: 0 auto;
        }

        .appointment-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            text-align: center;
            box-shadow: 
                0 8px 32px rgba(0,0,0,0.1),
                0 1px 0px rgba(255,255,255,0.8) inset;
            border: 1px solid rgba(226, 232, 240, 0.8);
            position: relative;
            overflow: hidden;
        }

        .appointment-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        }

        .appointment-card::after {
            content: '🟢 URGENT';
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            font-size: 0.7rem;
            font-weight: 600;
            color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            padding: 0.3rem 0.6rem;
            border-radius: 12px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .appointment-card .module-icon {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
        }

        .appointment-card .module-title {
            font-size: 1.8rem;
            color: #1a202c;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }

        .appointment-card .module-description {
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 2.5rem;
            line-height: 1.6;
        }

        .appointment-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            color: white !important;
            padding: 16px 32px;
            border-radius: 12px;
            text-decoration: none !important;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
            border: none;
            cursor: pointer;
        }

        .appointment-button:hover {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
            color: white !important;
            text-decoration: none !important;
        }

        .appointment-button:visited,
        .appointment-button:active,
        .appointment-button:focus {
            color: white !important;
            text-decoration: none !important;
        }

        /* Features Section */
        .features-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 24px;
            padding: 4rem 3rem;
            margin: 4rem 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }

        .features-title {
            font-size: 2.2rem;
            color: #1a202c;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        .features-subtitle {
            font-size: 1.1rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 3rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2.5rem;
            max-width: 1000px;
            margin: 0 auto;
        }

        .feature-item {
            text-align: center;
            padding: 1.5rem;
        }

        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            display: block;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1a202c;
            margin-bottom: 1rem;
            line-height: 1.3;
        }

        .feature-text {
            font-size: 1rem;
            color: #64748b;
            line-height: 1.6;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 3rem 2rem 2rem;
            margin-top: 4rem;
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            color: #e2e8f0;
            border-radius: 24px 24px 0 0;
            margin-left: 2rem;
            margin-right: 2rem;
        }

        .footer-content {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .footer-heart {
            color: #ef4444;
            animation: heartbeat 2s ease-in-out infinite;
        }

        .footer-small {
            font-size: 0.95rem;
            color: #a0aec0;
            font-weight: 400;
        }

        @keyframes heartbeat {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-title {
                font-size: 3rem;
            }
            
            .subtitle {
                font-size: 1.2rem;
            }
            
            .section-title {
                font-size: 1.8rem;
            }
            
            .hero-section {
                padding: 3rem 1.5rem 2rem;
            }
            
            .features-section {
                padding: 3rem 2rem;
                margin: 3rem 1rem;
            }
            
            .prediction-grid {
                grid-template-columns: 1fr;
                padding: 0;
            }
            
            .module-card {
                padding: 2rem 1.5rem;
            }
            
            .chatbot-card {
                padding: 2.5rem 2rem;
            }

            .appointment-card {
                padding: 2.5rem 2rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="main-title">🧠 PredictiveHealthAI</div>
    <div class="subtitle">
        Advanced AI-powered platform for early disease risk detection and comprehensive health insights
    </div>
    <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# Prediction Modules Section
st.markdown("""
<div class="content-section">
    <div class="section-title">🔬 AI Disease Prediction Models</div>
    <div class="section-subtitle">
        Leverage cutting-edge machine learning algorithms to assess your health risks with clinical-grade accuracy
    </div>
    <div class="prediction-grid">
        <div class="module-card">
            <span class="module-icon">🩺</span>
            <div class="module-title">Diabetes Risk Predictor</div>
            <div class="module-description">
                Advanced machine learning model that analyzes multiple health indicators to predict diabetes risk with high precision and actionable insights.
            </div>
            <div style="display: flex; justify-content: center; gap: 1rem; margin: 1rem 0; font-size: 0.8rem; color: #64748b;">
                <span>⚡ Fast Analysis</span>
                <span>🎯 AI-Powered</span>
                <span>📊 Multi-Factor</span>
            </div>
            <a href="/?page=diabetes" class="module-button">Start Prediction →</a>
        </div>
        <div class="module-card">
            <span class="module-icon">❤</span>
            <div class="module-title">Heart Disease Predictor</div>
            <div class="module-description">
                Comprehensive cardiovascular risk assessment using AI algorithms trained on extensive cardiac health datasets for reliable predictions.
            </div>
            <div style="display: flex; justify-content: center; gap: 1rem; margin: 1rem 0; font-size: 0.8rem; color: #64748b;">
                <span>⚡ Quick Results</span>
                <span>🎯 Advanced AI</span>
                <span>📊 Comprehensive</span>
            </div>
            <a href="/?page=heart" class="module-button">Start Prediction →</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ChatBot Section
st.markdown("""
<div class="chatbot-section">
    <div class="section-title">💬 AI Health Assistant</div>
    <div class="section-subtitle">
        Get personalized medical insights and health guidance from our intelligent AI doctor
    </div>
    <div class="chatbot-container">
        <div class="chatbot-card">
            <span class="module-icon">🤖</span>
            <div class="module-title">AI Health ChatBot</div>
            <div class="module-description">
                Intelligent medical assistant providing comprehensive health insights, symptom analysis, 
                medical report interpretation, and personalized health recommendations.
            </div>
            <a href="/?page=healthbot" class="chatbot-button">🤖 Chat with AI Doctor</a>
            <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; font-size: 0.85rem; color: #64748b;">
                <span>⚡ Instant responses</span>
                <span>🔒 Private & Secure</span>
                <span>🏥 Health-focused AI</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Appointment Booking Section
st.markdown("""
<div class="appointment-section">
    <div class="section-title">🩺 Emergency Medical Services</div>
    <div class="section-subtitle">
        Quick symptom assessment and direct doctor appointment booking for urgent medical needs
    </div>
    <div class="appointment-container">
        <div class="appointment-card">
            <span class="module-icon">📅</span>
            <div class="module-title">Emergency Appointment Booking</div>
            <div class="module-description">
                Fast-track medical symptom checker with automatic doctor notification for emergency cases. 
                Get immediate medical attention when you need it most.
            </div>
            <a href="/?page=appointment" class="appointment-button">📅 Book Emergency Appointment</a>
            <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; font-size: 0.85rem; color: #64748b;">
                <span>🚨 Emergency alerts</span>
                <span>📧 Direct doctor contact</span>
                <span>⏰ Urgent care</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("""
<div class="features-section">
    <div class="features-title">Why Choose PredictiveHealthAI?</div>
    <div class="features-subtitle">
        Experience the future of healthcare with our state-of-the-art AI technology and comprehensive health solutions
    </div>
    <div class="features-grid">
        <div class="feature-item">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Precision AI Algorithms</div>
            <div class="feature-text">
                Advanced machine learning models trained on extensive medical datasets for accurate health predictions and insights
            </div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">🔐</span>
            <div class="feature-title">Secure & Confidential</div>
            <div class="feature-text">
                Enterprise-grade security ensuring your health data remains private and protected at all times
            </div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Instant Results</div>
            <div class="feature-text">
                Get immediate health assessments with detailed insights and actionable recommendations in seconds
            </div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">🏥</span>
            <div class="feature-title">Clinical-Grade Accuracy</div>
            <div class="feature-text">
                Validated algorithms delivering hospital-quality predictions you can trust for important health decisions
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-content">
        © 2025 PredictiveHealthAI | Built with <span class="footer-heart">❤</span> by Dev Synergy
    </div>
    <div class="footer-small">
        Empowering proactive healthcare through artificial intelligence
    </div>
</div>
""", unsafe_allow_html=True)