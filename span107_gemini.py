import streamlit as st
import sqlite3
import os
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import json
from google import genai
from google.genai import types

# --- 1. CONFIGURACIÓN DE LIBRERÍAS DE LECTURA (RAG) ---
PDF_SUPPORT = True
IMPORT_ERRORS = []

try:
    import faiss
except ImportError:
    PDF_SUPPORT = False
    IMPORT_ERRORS.append("Falta instalar 'faiss-cpu'. Ejecuta: pip install faiss-cpu")

try:
    from PyPDF2 import PdfReader
except ImportError:
    PDF_SUPPORT = False
    IMPORT_ERRORS.append("Falta instalar 'PyPDF2'. Ejecuta: pip install PyPDF2")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    PDF_SUPPORT = False
    IMPORT_ERRORS.append("Falta instalar 'sentence-transformers'. Ejecuta: pip install sentence-transformers")


# --- 2. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "edgarvich_research_span107_gemini.db")
st.set_page_config(page_title="SPAN 107: Spanish Tutor (Google AI Studio)", layout="wide", page_icon="🌎")

# --- 3. DATABASE ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS diagnostics 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       student_name TEXT, student_input TEXT, ai_feedback TEXT, 
                       used_pdf TEXT, used_kolibri TEXT,
                       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
init_db()

def save_log(name, inp, feedback, used_pdf, used_kolibri):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO diagnostics (student_name, student_input, ai_feedback, used_pdf, used_kolibri) VALUES (?,?,?,?,?)",
                      (name, inp, feedback, used_pdf, used_kolibri))

# --- 4. OPTIMIZED RAG ENGINE ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2') if PDF_SUPPORT else None

def process_document(file):
    if not PDF_SUPPORT: 
        return None, None
    try:
        file_name = file.name.lower()
        text = ""
        
        if file_name.endswith('.pdf'):
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file_name.endswith('.txt') or file_name.endswith('.md'):
            text = file.read().decode("utf-8", errors="ignore")
        else:
            st.error("❌ Formato de archivo no soportado.")
            return None, None
        
        if not text.strip():
            st.error("⚠️ El documento no contiene texto legible.")
            return None, None
            
        chunks = [text[i:i+600] for i in range(0, len(text), 500)]
        embedder = load_embedder()
        embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return chunks, index
    except Exception as e:
        st.error(f"🚨 Error al procesar documento: {e}")
        return None, None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🌎 SPAN 107 Portal (Gemini)")
    
    # Lectura automática de API Key (desde secrets.toml o variable de entorno)
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
    uploaded_file = st.file_uploader("Upload Syllabus / Notes (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    
    if not PDF_SUPPORT:
        st.error("⚠️ Document Reader Disabled:")
        for err in IMPORT_ERRORS:
            st.caption(err)
            
    st.divider()
    if st.text_input("Instructor Access:", type="password") == "peru2026":
        st.subheader("📊 Research Data Logs")
        with st.expander("Click to view interaction logs"):
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql_query("SELECT * FROM diagnostics ORDER BY timestamp DESC", conn)
                st.dataframe(df)
                st.download_button("Download CSV", df.to_csv(index=False), "span107_gemini_research.csv")

# --- 6. TUTORING LOGIC ---
st.title("👨‍🏫 SPAN 107: Spanish Language Coach (Powered by Gemini)")

if not gemini_api_key:
    st.warning("👈 Por favor, configura tu GEMINI_API_KEY en .streamlit/secrets.toml o ingrésala en la barra lateral para comenzar.")
    st.stop()

if not student_name:
    st.warning("👈 Please enter your name in the sidebar to begin.")
    st.stop()

# Inicializar cliente de Google GenAI
client = genai.Client(api_key=gemini_api_key)

if 'messages' not in st.session_state: 
    st.session_state.messages = []

# Procesamiento de documentos
if uploaded_file and PDF_SUPPORT:
    if 'current_pdf_name' not in st.session_state or st.session_state.current_pdf_name != uploaded_file.name:
        with st.spinner("Analyzing and indexing course document..."):
            chunks, index = process_document(uploaded_file)
            if chunks and index:
                st.session_state.pdf_data = (chunks, index)
                st.session_state.current_pdf_name = uploaded_file.name
                st.success("Course reference material loaded successfully!")
else:
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None

# --- 🎙️ VOICE DICTATION HELPER ---
st.write("### 🎙️ Speech-to-Text Assistant")

if st.button("🔴 Click for Voice Typing Instructions"):
    components.html("""
        <script>
        alert("Voice Typing Instructions for Students:\\n1. Click inside the chat box below.\\n2. Press 'Windows + H' on your keyboard.\\n3. Start speaking your Spanish or English question!");
        </script>
    """, height=0)

# Renderizar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])

st.info("⌨️ **Classroom Mode Active:** Type below, use 'Win + H' to dictate, or use the Quick Paste Zone.")

# --- 7. CHAT & AI RESPONSE ---
user_input = st.chat_input("Ask Coach Edgarvich about Spanish grammar, verbs, vocabulary, or culture...")

if enviar_pegado and pasted_exercise:
    final_query = f"Please help me analyze and solve this Spanish exercise/text from my course material: {pasted_exercise}"
else:
    final_query = user_input

if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): 
        st.markdown(final_query)

    context_text = ""
    if st.session_state.get('pdf_data'):
        chunks, index = st.session_state.pdf_data
        query_emb = load_embedder().encode([final_query])
        D, I = index.search(np.array(query_emb), 1)
        if I[0][0] != -1:
            context_text = chunks[I[0][0]]

    # System instruction para Gemini
    system_instruction = (
        f"You are Coach Edgarvich, an encouraging, patient, and highly skilled Spanish language tutor. "
        f"Student Name: {student_name}. "
        f"Course Reference Material: {context_text}. "
        f"MANDATORY INSTRUCTIONS: "
        f"1. Provide explanations and grammatical breakdowns primarily in CLEAR ENGLISH. "
        f"2. Provide Spanish examples and vocabulary in bold with immediate English translations in parentheses (e.g., **el libro** (the book)). "
        f"3. If the student makes a mistake in Spanish, gently explain why the error occurred in English and show the correct Spanish version. "
        f"4. If course reference material is present, prioritize vocabulary and rules aligned with it. "
        f"5. Always end your response with exactly ONE engaging practice question or translation challenge for the student in Spanish."
    )

    with st.chat_message("assistant"):
        try:
            # Construir historial de conversación para Gemini
            contents = []
            for msg in st.session_state.messages[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                )
            )
            
            ai_text = response.text
            st.markdown(ai_text)
            
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
            st.error(f"🚨 Error al conectar con Google AI Studio: {e}")