import streamlit as st
import sqlite3
import os
import pandas as pd
import streamlit.components.v1 as components
import json
import time
from google import genai
from google.genai import types

# --- 1. CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "edgarvich_research_span107_gemini.db")
st.set_page_config(page_title="👨‍🏫 SPAN 107: Edgarvich virtual tutor", layout="wide", page_icon="🌎")


# --- 2. BASE DE DATOS (MODO WAL) ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS diagnostics 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       student_name TEXT, student_input TEXT, ai_feedback TEXT, 
                       used_pdf TEXT, used_kolibri TEXT,
                       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

init_db()

def save_log(name, inp, feedback, used_pdf, used_kolibri):
    try:
        conn = get_db_connection()
        with conn:
            conn.execute("INSERT INTO diagnostics (student_name, student_input, ai_feedback, used_pdf, used_kolibri) VALUES (?,?,?,?,?)",
                          (name, inp, feedback, used_pdf, used_kolibri))
        conn.close()
    except sqlite3.OperationalError:
        pass


# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("👨‍🏫 SPAN 107: Edgarvich virtual tutor")
    
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key_input = st.text_input("🔑 Gemini API Key:", type="password", help="Obtenla gratis en aistudio.google.com")
        gemini_api_key = api_key_input or os.environ.get("GEMINI_API_KEY")
    
    student_name = st.text_input("Student Name:", placeholder="e.g. Alex Smith...")
    
    st.divider()
    st.subheader("📋 Quick Paste Zone")
    st.info("Paste a Spanish exercise, sentence, or syllabus snippet here.")
    pasted_exercise = st.text_area("Paste Spanish exercise here:", placeholder="Ctrl + V here...", height=100)
    enviar_pegado = st.button("🚀 Submit Pasted Exercise")
    
    st.divider()
    st.subheader("🌐 Offline Resources")
    st.link_button("🚀 Open Kolibri Library", "http://127.0.0.1:8080")
    
    st.divider()
    st.subheader("📚 Course Reference Material")
    uploaded_file = st.file_uploader("Upload Syllabus / Notes (PDF, TXT, PNG, JPG)", type=["pdf", "txt", "md", "png", "jpg", "jpeg"])
    
    # Procesamiento multimodal nativo: extrae bytes directos
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name.lower()
        
        mime_type = "application/pdf"
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            mime_type = "image/png" if file_name.endswith('.png') else "image/jpeg"
        elif file_name.endswith(('.txt', '.md')):
            mime_type = "text/plain"
            
        st.session_state.file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        st.session_state.loaded_file_name = uploaded_file.name
        st.sidebar.success(f"✅ Document attached: {uploaded_file.name}")
    else:
        st.session_state.file_part = None
        st.session_state.loaded_file_name = None
            
    st.divider()
    if st.text_input("Instructor Access:", type="password") == "peru2026":
        st.subheader("📊 Research Data Logs")
        with st.expander("Click to view interaction logs"):
            conn = get_db_connection()
            df = pd.read_sql_query("SELECT * FROM diagnostics ORDER BY timestamp DESC", conn)
            conn.close()
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "span107_gemini_research.csv")


# --- 4. GESTIÓN DE SESIÓN ---
st.title("👨‍🏫 SPAN 107: Edgarvich virtual tutor")

if not gemini_api_key:
    st.warning("👈 Por favor, configura tu GEMINI_API_KEY en .streamlit/secrets.toml o ingrésala en la barra lateral para comenzar.")
    st.stop()

if not student_name:
    st.warning("👈 Please enter your name in the sidebar to begin.")
    st.stop()

@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)

client = get_client(gemini_api_key)

if 'messages' not in st.session_state: 
    st.session_state.messages = []

# Asistente de dictado por voz
st.write("### 🎙️ Speech-to-Text Assistant")
if st.button("🔴 Click for Voice Typing Instructions"):
    components.html("""
        <script>
        alert("Voice Typing Instructions for Students:\\n1. Click inside the chat box below.\\n2. Press 'Windows + H' on your keyboard.\\n3. Start speaking your Spanish or English question!");
        </script>
    """, height=0)

for m in st.session_state.messages:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])

st.info("⌨️ **Classroom Mode Active:** Type below, use 'Win + H' to dictate, or use the Quick Paste Zone.")


# --- 5. CHAT Y RESPUESTA INTELIGENTE ---
user_input = st.chat_input("Ask Coach Edgarvich about Spanish grammar, verbs, vocabulary, or the attached document...")

if enviar_pegado and pasted_exercise:
    final_query = f"Please help me analyze and solve this Spanish exercise/text: {pasted_exercise}"
else:
    final_query = user_input

if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): 
        st.markdown(final_query)

    file_is_attached = st.session_state.get('file_part') is not None
    doc_name = st.session_state.get('loaded_file_name', '')

    if file_is_attached:
        doc_info_prompt = (
            f"SYSTEM NOTICE: The student HAS uploaded the file '{doc_name}'. "
            f"The raw bytes are provided in this multimodal prompt. You CAN view, inspect, and read its entire content (scanned or digital). "
            f"NEVER tell the student that you cannot read the document or that they have not uploaded it. "
            f"Confirm that you see '{doc_name}' and answer their questions based directly on it."
        )
    else:
        doc_info_prompt = (
            "SYSTEM NOTICE: No document is currently attached. If the student asks about a file, kindly remind them "
            "to attach it using the sidebar file uploader or paste the text directly."
        )

    system_instruction = (
        f"You are Coach Edgarvich, an encouraging, patient, and highly skilled Spanish language tutor. "
        f"Student Name: {student_name}. "
        f"{doc_info_prompt}\n\n"
        "Pedagogical Guidelines:\n"
        "- Explain grammatical rules and breakdowns clearly in English.\n"
        "- Format Spanish examples and vocabulary in bold with immediate English translations in parentheses (e.g., **el libro** (the book)).\n"
        "- If the student makes an error, explain why kindly in English and provide the correct Spanish version.\n"
        "- If an uploaded document is present, reference its content directly and accurately.\n"
        "- Conclude naturally with a single interactive practice question or translation challenge in Spanish directly addressed to the student."
    )

    with st.chat_message("assistant"):
        try:
            # Construir historial de mensajes (ventana de los últimos 6 turnos)
            contents = []
            for msg in st.session_state.messages[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

            # Inyectar el documento de forma multimodal en el primer mensaje de usuario
            if file_is_attached and contents:
                for content in contents:
                    if content.role == "user":
                        content.parts.insert(0, st.session_state.file_part)
                        break

            import time

            def stream_response():
                models_to_try = ['gemini-3.6-flash', 'gemini-2.5-flash']
                last_error = None

                for target_model in models_to_try:
                    for attempt in range(2):
                        try:
                            response = client.models.generate_content_stream(
                                model=target_model,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    temperature=0.3,
                                    max_output_tokens=1500,
                                )
                            )
                            for chunk in response:
                                if chunk.text:
                                    yield chunk.text
                            return
                        except Exception as err:
                            last_error = err
                            if "503" in str(err) or "429" in str(err):
                                time.sleep(1.5)
                                continue
                            break
                if last_error:
                    raise last_error

            ai_text = st.write_stream(stream_response)
            
            save_log(student_name, final_query, ai_text, "YES" if uploaded_file else "NO", "YES")
            st.session_state.messages.append({"role": "assistant", "content": ai_text})
            
            # --- MOTOR DE AUDIO ---
            voice_text = ai_text.replace('*', '').replace('#', '').replace('\n', ' ')
            js_safe_voice = json.dumps(voice_text) 
            
            html_script = f"""
            <script>
            function speak() {{
                if (typeof window.speechSynthesis === 'undefined') return;
                window.speechSynthesis.cancel(); 
                
                var text = {js_safe_voice};
                var msg = new SpeechSynthesisUtterance(text);
                var voices = window.speechSynthesis.getVoices();
                
                var naturalVoice = voices.find(v => v.lang.includes("en") && (v.name.toLowerCase().includes("natural") || v.name.toLowerCase().includes("neural")));
                var englishVoice = voices.find(v => v.lang.includes("en-US") || v.lang.includes("en-GB") || v.lang.includes("en"));
                
                if (naturalVoice) {{
                    msg.voice = naturalVoice;
                }} else if (englishVoice) {{
                    msg.voice = englishVoice;
                }} else if (voices.length > 0) {{
                    msg.voice = voices[0];
                }}
                
                msg.rate = 1.0; 
                msg.pitch = 1.0;
                window.speechSynthesis.speak(msg);
            }}
            if (typeof window.speechSynthesis !== 'undefined') {{
                if (window.speechSynthesis.onvoiceschanged !== undefined) {{
                    window.speechSynthesis.onvoiceschanged = speak;
                }}
                speak();
            }}
            </script>
            """
            components.html(html_script, height=0)

        except Exception as e:
            st.error(f"🚨 El servidor de IA está recibiendo alta demanda en este momento. Por favor, reenvía tu mensaje en 5 segundos. Detalle: {e}")
