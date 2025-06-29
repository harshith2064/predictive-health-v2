import streamlit as st
from chatbot.chatbot_response import get_bot_response, analyze_report
from PyPDF2 import PdfReader
import speech_recognition as sr
import streamlit.components.v1 as components
from io import BytesIO
import re
import emoji

st.set_page_config(page_title="AI HealthBot", page_icon="🩺")
st.title("🩺 AI Health ChatBot")

# Updated styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5faff;
    }
    h1, h2, h3, h4, h5, h6,
    .markdown-text-container,
    .stTextInput > label,
    .stFileUploader > label,
    .stTextArea > label,
    .stChatMessage .stMarkdown,
    .stMarkdown p {
        color: #001F3F !important;
    }
    .stAlert > div {
        color: #001F3F !important;  /* Fixes info/warning text like "Listening..." */
    }
    .stSpinner > div {
        color: #001F3F !important;  /* Fixes spinner loading text */
    }
    .blue-label > label, .blue-label {
        color: #003366 !important;
        font-weight: bold;
    }
    .stTextInput > div > input,
    .stTextArea textarea,
    .stChatInput input {
        background-color: #ffffff;
        color: #003366;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #002244;
    }
    .speak-btn {
        background-color: #004080;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        font-size: 15px;
        font-weight: bold;
        cursor: pointer;
        margin-top: 10px;
    }
    .speak-btn:hover {
        background-color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("Talk about symptoms or ask health questions. Upload reports or use voice!")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# PDF Upload
st.markdown("<div class='blue-label'>Upload Medical Report (PDF)</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"], key="pdf_uploader_main")
if uploaded_file:
    report_text = "\n".join(
        page.extract_text() for page in PdfReader(BytesIO(uploaded_file.read())).pages if page.extract_text()
    )
    st.text_area("Extracted Report", report_text, height=200)
    uploaded_file.seek(0)
    response = analyze_report(uploaded_file)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_response = response

# Text input
if prompt := st.chat_input("Type your symptoms or question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_bot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_response = response
    st.session_state.initialized = True

# Display chat
for idx, msg in enumerate(st.session_state.messages):
    with st.container():
        st.chat_message(msg["role"]).markdown(msg["content"])
        if msg["role"] == "assistant":
            def clean_response(text):
                no_emoji = emoji.replace_emoji(text, replace="")
                no_md = re.sub(r"[*_`~#>\\[\\](){}|\\\\]", "", no_emoji)
                return no_md

            clean_text = clean_response(msg['content'])
            components.html(f"""
                <div style='text-align: right;'>
                    <button id='read-btn-{idx}' class='speak-btn'>🎧 Read Aloud</button>
                </div>
                <script>
                    const btn{idx} = document.getElementById('read-btn-{idx}');
                    let isSpeaking{idx} = false;
                    let utterance{idx} = new SpeechSynthesisUtterance({repr(clean_text)});

                    btn{idx}.addEventListener('click', () => {{
                        if (!isSpeaking{idx}) {{
                            speechSynthesis.speak(utterance{idx});
                            isSpeaking{idx} = true;
                            utterance{idx}.onend = () => {{ isSpeaking{idx} = false; }};
                        }} else {{
                            speechSynthesis.cancel();
                            isSpeaking{idx} = false;
                        }}
                    }});
                </script>
            """, height=100)

# Voice input
if st.button("🎧 Speak Symptoms"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")
        st.session_state.messages.append({"role": "user", "content": text})
        response = get_bot_response(text)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_response = response
        st.rerun()
    except Exception as e:
        st.error("Sorry, couldn't recognize your voice: " + str(e))
        
