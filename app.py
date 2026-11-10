import io
import json
import platform
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Aula Virtual — Profesor IA",
    page_icon="👨‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS (ESTÉTICA DE PIZARRA Y CLASE)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background-color: #3b4d2e !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #93a67d !important;
        border-right: 1px solid #bce5cc;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Estilo Pizarra / Contenedor de Clase */
    .classroom-card {
        background: #242c1d !important;
        border: 2px solid #5a7346;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        color: #FFFFFF !important;
    }

    /* Burbujas de diálogo del Profesor */
    .prof-bubble {
        background: #2e3b24;
        border-left: 5px solid #00D18F;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 15px;
        color: #ffffff;
    }

    /* Burbujas del Estudiante */
    .student-bubble {
        background: #38482d;
        border-right: 5px solid #ffcc00;
        padding: 16px;
        border-radius: 12px 0 0 12px;
        margin-bottom: 15px;
        color: #ffffff;
        text-align: right;
    }

    p, label, span {
        color: #E0E0E0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GESTIÓN DEL ESTADO DE LA CLASE (SESSION STATE)
# ─────────────────────────────────────────────
if "historial_clase" not in st.session_state:
    st.session_state.historial_clase = [] # Lista de tuplas ("profesor"/"estudiante", texto)
if "preguntas_examen" not in st.session_state:
    st.session_state.preguntas_examen = []
if "indice_pregunta" not in st.session_state:
    st.session_state.indice_pregunta = 0
if "clase_iniciada" not in st.session_state:
    st.session_state.clase_iniciada = False

# ─────────────────────────────────────────────
# BARRA LATERAL (CONTROL DEL PROFESOR)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#000000 !important; font-weight:800;'>🏫 Dirección del Colegio</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#222222 !important; font-size:0.9rem;'>Prepara tu salón de clases cargando los apuntes.</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<b style='color:#000000 !important;'>🔑 Credenciales de Acceso</b>", unsafe_allow_html=True)
    api_key = st.text_input("Clave API OpenAI", type="password", placeholder="sk-...")

    st.divider()
    st.markdown("<b style='color:#000000 !important;'>📄 Material de Estudio (PDF)</b>", unsafe_allow_html=True)
    pdf_subido = st.file_uploader("Sube el PDF de la clase", type="pdf", label_visibility="collapsed")

    if pdf_subido and not st.session_state.clase_iniciada:
        if st.button("🔔 ¡Comenzar la Clase!", use_container_width=True):
            if not api_key:
                st.error("Por favor ingresa tu clave de OpenAI primero.")
            else:
                with st.spinner("El Profesor está leyendo el documento y preparando la lección..."):
                    # Extraer texto
                    reader = PdfReader(io.BytesIO(pdf_subido.getvalue()))
                    texto_pdf = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
                    
                    # Generar preguntas conversacionales con IA
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)
                    prompt = f"""
                    Eres un profesor experto, amigable pero exigente. A partir del siguiente texto, genera una lista de 4 preguntas de examen abiertas y estimulantes para evaluar a un estudiante en forma de diálogo.
                    Devuelve el resultado ÚNICAMENTE en formato JSON de lista de strings (ejemplo: ["Pregunta 1?", "Pregunta 2?"]):
                    
                    Texto:
                    {texto_pdf[:10000]}
                    """
                    resp = llm.invoke(prompt)
                    try:
                        limpio = resp.content.replace("```json", "").replace("```", "").strip()
                        preguntas = json.loads(limpio)
                        st.session_state.preguntas_examen = preguntas
                        st.session_state.indice_pregunta = 0
                        st.session_state.historial_clase = []
                        
                        # Saludo inicial del profesor y primera pregunta
                        primer_saludo = f"¡Hola! Toma asiento por favor. He leído detenidamente tu documento y hoy seré tu profesor personal. Vamos a ver qué tanto aprendiste. Comencemos con la primera pregunta:\n\n**{preguntas[0]}**"
                        st.session_state.historial_clase.append(("profesor", primer_saludo))
                        st.session_state.clase_iniciada = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al preparar las preguntas: {e}")

    if st.session_state.clase_iniciada:
        if st.button("🚪 Cambiar de Tema / Reiniciar Clase", use_container_width=True):
            st.session_state.clase_iniciada = False
            st.session_state.historial_clase = []
            st.session_state.preguntas_examen = []
            st.rerun()

    st.divider()
    st.markdown(f"<p style='color:#333333 !important; font-size:0.75rem;'>Aula Virtual v2.0 - Modo Interactivo</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PANTALLA PRINCIPAL: EL SALÓN DE CLASES
# ─────────────────────────────────────────────
st.markdown("""
<div class="classroom-card" style="text-align: center; border-color: #00D18F;">
    <span style="background-color: #00D18F; color: #000000; font-weight: 800; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; text-transform: uppercase;">👨‍🏫 Aula Interactiva en Vivo</span>
    <h1 style="margin: 10px 0 5px 0; color: #FFFFFF !important; font-weight: 800;">Clase Magistral con el Profesor IA</h1>
    <p style="font-size: 0.95rem; color: #CCCCCC !important; margin: 0;">
        Responde a las preguntas del profesor en el chat inferior. Él evaluará tu respuesta, te dará retroalimentación y te hará la siguiente pregunta.
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.clase_iniciada:
    st.markdown("""
    <div class="classroom-card" style="text-align: center; padding: 50px;">
        <h3 style="color: #00D18F;">🚪 El salón está vacío</h3>
        <p>Por favor, ingresa tu clave de OpenAI y sube un archivo PDF en la barra lateral izquierda, luego haz clic en <b>"¡Comenzar la Clase!"</b> para que el profesor te reciba.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Mostrar el pizarron / historial de la conversación
    for rol, mensaje in st.session_state.historial_clase:
        if rol == "profesor":
            st.markdown(f"""
            <div class="prof-bubble">
                <b>👨‍🏫 Profesor IA:</b><br>{mensaje}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="student-bubble">
                <b>🎓 Tú (Estudiante):</b><br>{mensaje}
            </div>
            """, unsafe_allow_html=True)

    # Si aún hay preguntas por hacer, mostrar el input de respuesta
    idx_actual = st.session_state.indice_pregunta
    total_preguntas = len(st.session_state.preguntas_examen)

    if idx_actual < total_preguntas:
        with st.form(key="form_respuesta_clase", clear_on_submit=True):
            respuesta_estudiante = st.text_input("💬 Levanta la mano y responde al profesor:", placeholder="Escribe tu respuesta aquí...")
            btn_enviar = st.form_submit_button("🙋‍♂️ Responder al Profesor", use_container_width=True)

            if btn_enviar and respuesta_estudiante.strip():
                # Agregar respuesta del estudiante al historial
                st.session_state.historial_clase.append(("estudiante", respuesta_estudiante))
                
                # Evaluar con IA
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
                pregunta_hecha = st.session_state.preguntas_examen[idx_actual]
                
                prompt_evaluacion = f"""
                Estás actuando como un profesor en un aula evaluando a un estudiante.
                Pregunta que hiciste: "{pregunta_hecha}"
                Respuesta del estudiante: "{respuesta_estudiante}"
                
                Evalúa su respuesta de forma constructiva (indícale si está bien, qué le faltó, corrígele si hay errores).
                Sé conversacional, cercano y motivador. 
                """
                
                with st.spinner("El Profesor está evaluando tu respuesta..."):
                    evaluacion = llm.invoke(prompt_evaluacion).content
                
                # Avanzar índice
                st.session_state.indice_pregunta += 1
                nuevo_idx = st.session_state.indice_pregunta
                
                if nuevo_idx < total_preguntas:
                    siguiente_pregunta = st.session_state.preguntas_examen[nuevo_idx]
                    respuesta_profesor = f"{evaluacion}\n\n--- \nPasemos a la siguiente cuestión:\n\n**{siguiente_pregunta}**"
                else:
                    respuesta_profesor = f"{evaluacion}\n\n🎉 **¡Clase terminada!** Has respondido todas las preguntas de la lección. ¡Excelente esfuerzo hoy, puedes descansar o reiniciar con otro documento!"
                
                st.session_state.historial_clase.append(("profesor", respuesta_profesor))
                st.rerun()
    else:
        st.markdown("""
        <div class="classroom-card" style="text-align: center; background: #2e3b24 !important; border-color: #00D18F;">
            <h3 style="color: #00D18F;">🌟 ¡Lección Concluida con Éxito!</h3>
            <p>Has completado todo el cuestionario oral del profesor. Usa el botón de la barra lateral si deseas estudiar otro documento.</p>
        </div>
        """, unsafe_allow_html=True)
