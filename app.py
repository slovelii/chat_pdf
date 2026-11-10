import io
import json
import platform
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuizMaster IA — Tu Entrenador de Exámenes",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS (ESTÉTICA DE ESTUDIO / MENTA)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background-color: #52663D !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #A8BA93 !important;
        border-right: 1px solid #D1EAD9;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Tarjetas de Contenido */
    .study-card {
        background: #29331F !important;
        border: 1px solid #3E4D2F;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        color: #FFFFFF !important;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #29331F 0%, #1E2617 100%);
        border: 1px solid #00D18F;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 209, 143, 0.15);
        color: #FFFFFF !important;
    }

    /* Tags / Badges */
    .badge-tag {
        background-color: #E2F5EA !important;
        color: #007A52 !important;
        font-size: 0.75rem !important;
        font-weight: 800 !important;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    p, label, span {
        color: #E0E0E0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GESTIÓN DEL ESTADO DE LA SESIÓN (SESSION STATE)
# ─────────────────────────────────────────────
if "quiz_preguntas" not in st.session_state:
    st.session_state.quiz_preguntas = []
if "respuestas_usuario" not in st.session_state:
    st.session_state.respuestas_usuario = {}
if "quiz_enviado" not in st.session_state:
    st.session_state.quiz_enviado = False

# ─────────────────────────────────────────────
# BARRA LATERAL (SIDEBAR)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#000000 !important; font-weight:800;'>🎯 Panel de Estudio</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#222222 !important; font-size:0.9rem;'>Sube tu material de lectura y prepárate para el examen.</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<b style='color:#000000 !important;'>🔑 Credenciales</b>", unsafe_allow_html=True)
    api_key = st.text_input("Clave API OpenAI", type="password", placeholder="sk-...")

    st.divider()
    
    st.markdown("<b style='color:#000000 !important;'>📄 Tu Apunte o PDF</b>", unsafe_allow_html=True)
    pdf = st.file_uploader("Sube el PDF de estudio", type="pdf", label_visibility="collapsed")

    if pdf and st.button("🔄 Reiniciar / Generar Nuevo Quiz", use_container_width=True):
        st.session_state.quiz_preguntas = []
        st.session_state.respuestas_usuario = {}
        st.session_state.quiz_enviado = False
        st.rerun()

    st.divider()
    st.markdown(f"<p style='color:#444444 !important; font-size:0.75rem;'>QuizMaster Engine v2.0</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER / ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <span class="badge-tag">🎓 Estudio Activo & Gamificado</span>
    <h1 style="margin: 4px 0 10px 0; color: #FFFFFF !important; font-weight: 800;">QuizMaster: Tu Entrenador de Exámenes 🎯</h1>
    <p style="font-size: 1.05rem; color: #CCCCCC !important; margin-bottom: 0px; line-height: 1.5;">
        Sube las lecturas de tu clase. La inteligencia artificial creará un simulacro de examen personalizado para evaluar tus conocimientos antes de la prueba oficial.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNCIONES DE PROCESAMIENTO Y GENERACIÓN DE QUIZ
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Procesando documento...")
def extraer_texto_pdf(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
    return texto

def generar_preguntas_quiz(texto: str, api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
    
    prompt = f"""
    A partir del siguiente texto de estudio, genera exactamente 5 preguntas de opción múltiple diseñadas para evaluar a un estudiante.
    Devuelve la respuesta ÚNICAMENTE en formato JSON válido, sin bloques de código markdown adicionales (nada de ```json), con la siguiente estructura de lista:
    [
      {{
        "pregunta": "Texto de la pregunta",
        "opciones": ["Opción A", "Opción B", "Opción C", "Opción D"],
        "respuesta_correcta": "Opción exacta que es correcta",
        "explicacion": "Breve explicación de por qué es la respuesta correcta"
      }}
    ]

    Texto de estudio:
    {texto[:12000]}
    """
    
    respuesta = llm.invoke(prompt)
    try:
        # Limpieza por si el modelo incluye marcas de código
        texto_limpio = respuesta.content.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
    except Exception:
        return []

# ─────────────────────────────────────────────
# LÓGICA PRINCIPAL DE LA VISTA
# ─────────────────────────────────────────────
if not api_key:
    st.markdown("""
    <div class="study-card" style="text-align: center; padding: 40px;">
        <h3 style="color: #00D18F !important;">🔒 Ingresa tu Clave de OpenAI</h3>
        <p>Escribe tu clave en la barra lateral para activar tu simulador de exámenes.</p>
    </div>
    """, unsafe_allow_html=True)

elif pdf is None:
    st.markdown("""
    <div class="study-card" style="text-align: center; padding: 40px;">
        <h3 style="color: #00D18F !important;">📂 Sube tu PDF de Estudio</h3>
        <p>Arrastra tu guía, lectura o documento de clase en la barra lateral para generar tu quiz personalizado.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        texto_pdf = extraer_texto_pdf(pdf.getvalue())
        
        if not texto_pdf.strip():
            st.error("El PDF no contiene texto legible.")
            st.stop()

        # Generar preguntas si aún no existen en la sesión
        if not st.session_state.quiz_preguntas:
            with st.spinner("🧠 Leyendo apuntes y diseñando el examen simulado..."):
                st.session_state.quiz_preguntas = generar_preguntas_quiz(texto_pdf, api_key)
        
        preguntas = st.session_state.quiz_preguntas

        if not preguntas:
            st.error("No se pudieron generar preguntas a partir de este documento. Intenta con otro archivo.")
            st.stop()

        # Interfaz del Examen
        st.markdown("""
        <div class="study-card">
            <h3 style="color: #00D18F !important; margin-top:0;">📝 Simulacro de Examen Activo</h3>
            <p>Responde las siguientes preguntas basadas en tu documento. Al finalizar, califica tu desempeño.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_quiz"):
            for i, q in enumerate(preguntas):
                st.markdown(f"#### **Pregunta {i+1}:** {q['pregunta']}")
                
                # Widget de selección
                opcion_elegida = st.radio(
                    f"Selecciona una opción (P{i+1})",
                    q['opciones'],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                st.session_state.respuestas_usuario[i] = opcion_elegida
                st.markdown("<hr style='border-color: #3E4D2F;'>", unsafe_allow_html=True)

            enviar_quiz = st.form_submit_button("🚀 Evaluar Mis Respuestas", use_container_width=True)

        if enviar_quiz:
            st.session_state.quiz_enviado = True

        # Mostrar resultados si ya se envió el quiz
        if st.session_state.quiz_enviado:
            puntaje = 0
            st.markdown("---")
            st.markdown("<h2 style='color: #00D18F;'>📊 Resultados de tu Evaluación</h2>", unsafe_allow_html=True)
            
            for i, q in enumerate(preguntas):
                resp_usuario = st.session_state.respuestas_usuario.get(i)
                es_correcta = (resp_usuario == q['respuesta_correcta'])
                
                if es_correcta:
                    puntaje += 1
                    icono = "✅"
                    color_borde = "#00D18F"
                else:
                    icono = "❌"
                    color_borde = "#FF6B6B"

                st.markdown(f"""
                <div class="study-card" style="border-left: 5px solid {color_borde};">
                    <p style="font-weight: 700; color: #FFFFFF;">Pregunta {i+1}: {q['pregunta']}</p>
                    <p>Tu respuesta: <b>{resp_usuario}</b> {icono}</p>
                    <p>Respuesta correcta: <b>{q['respuesta_correcta']}</b></p>
                    <p style="font-size: 0.9rem; color: #CCCCCC; font-style: italic;">💡 Explicación: {q['explicacion']}</p>
                </div>
                """, unsafe_allow_html=True)

            nota_final = (puntaje / len(preguntas)) * 100
            st.markdown(f"""
            <div class="hero-card" style="text-align: center;">
                <h2>Tu Calificación Final: {puntaje} / {len(preguntas)} ({nota_final}%)</h2>
                <p>{'🎉 ¡Excelente trabajo, estás listo para el examen!' if nota_final >= 80 else '💡 ¡Buen intento! Repasa los conceptos marcados en rojo antes de la prueba.'}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el examen: {e}")
