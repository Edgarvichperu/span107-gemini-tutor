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
st.set_page_config(page_title="👨‍🏫 SPAN 107: Edgarvich virtual tutor", layout="wide", page_icon="🌎")


# --- 3. DATABASE (CONCURRENCIA OPTIMIZADA CON MODO WAL) ---
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


# --- 4. OPTIMIZED RAG ENGINE (CON CACHÉ GLOBAL) ---
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
    st.title("👨‍🏫 SPAN 107: Edgarvich virtual tutor")
    
    # Lectura automática de API Key
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
            conn = get_db_connection()
            df = pd.read_sql_query("SELECT * FROM diagnostics ORDER BY timestamp DESC", conn)
            conn.close()
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "span107_gemini_research.csv")


# --- 6. TUTORING LOGIC ---
st.title("👨‍🏫 SPAN 107: Edgarvich virtual tutor")

if not gemini_api_key:
    st.warning("👈 Por favor, configura tu GEMINI_API_KEY en .streamlit/secrets.toml o ingrésala en la barra lateral para comenzar.")
    st.stop()

if not student_name:
    st.warning("👈 Please enter your name in the sidebar to begin.")
    st.stop()

# Cliente de Google GenAI en caché global
@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)

client = get_client(gemini_api_key)

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

# Renderizar historial previo en la interfaz visual
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
        embedder = load_embedder()
        if embedder:
            query_emb = embedder.encode([final_query])
            D, I = index.search(np.array(query_emb), 1)
            if I[0][0] != -1:
                context_text = chunks[I[0][0]]

    # System instruction pedagógica
    system_instruction = (
        f"You are Coach Edgarvich, an encouraging, patient, and highly skilled Spanish language tutor. "
        f"Student Name: {student_name}. "
        f"Course Reference Material: {context_text}. "
        "Pedagogical Guidelines:\n"
        "- Explain grammatical rules and breakdowns clearly in English.\n"
        "- Format Spanish examples and vocabulary in bold with immediate English translations in parentheses (e.g., **el libro** (the book)).\n"
        "- If the student makes an error, explain why kindly in English and provide the correct Spanish version.\n"
        "- Align explanations with the course reference material when available.\n"
        "- Conclude naturally with a single interactive practice question or translation challenge in Spanish directly addressed to the student."
    )

    with st.chat_message("assistant"):
        try:
            # Ventana deslizante (Sliding Window): solo envía los últimos 6 turnos a la API para velocidad óptima
            contents = []
            for msg in st.session_state.messages[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

            def stream_response():
                response = client.models.generate_content_stream(
                    model='gemini-3.6-flash',
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
            st.error(f"🚨 Error al conectar con Google AI Studio: {e}")
