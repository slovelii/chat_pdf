import io
import json
import platform
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="El Salón de Clases Interactivo 🏫✏️",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS: SALÓN DE CLASES VIBRANTE Y COLORIDO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Quicksand:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif !important;
    }

    /* Fondo general estilo pared de aula moderna (Azul Enérgico) */
    .stApp {
        background-color: #1a2b4c !important;
        background-image: radial-gradient(#273d6b 15%, transparent 16%), radial-gradient(#273d6b 15%, transparent 16%);
        background-size: 60px 60px;
        background-position: 0 0, 30px 30px;
    }

    /* Barra lateral estilo casillero escolar (Amarillo Vibrante) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffde59 0%, #ff914d 100%) !important;
        border-right: 4px solid #ffffff;
    }
    
    [data-testid="stSidebar"] * {
        color: #1a2b4c !important;
        font-weight: 700 !important;
    }

    /* Estilo Pizarra del Profesor */
    .pizarra-card {
        background: #132210 !important;
        border: 6px solid #8B5A2B; /* Marco de madera */
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        color: #FFFFFF !important;
    }

    .pizarra-titulo {
        font-family: 'Fredoka', cursive !important;
        color: #ffde59 !important;
        font-size: 2.2rem;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px #000000;
    }

    /* Burbujas de diálogo estilo cuaderno escolar */
    .profesor-habla {
        background: #ffffff;
        border: 3px solid #ffde59;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 15px;
        color: #1a2b4c !important;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .estudiante-habla {
        background: #ff914d;
        border: 3px solid #ffffff;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 15px;
        color: #ffffff !important;
        font-size: 1.1rem;
        text-align: right;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Inputs y botones vibrantes */
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a2b4c !important;
        border-radius: 12px !important;
        border: 3px solid #ffde59 !important;
        font-weight: bold !important;
    }

    .stButton button {
        background: #00bf63 !important;
        color: white !important;
        font-family: 'Fredoka', cursive !important;
        font-size: 1.2rem !important;
        border-radius: 14px !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        background: #ffde59 !important;
        color: #1a2b4c !important;
        transform: scale(1.02);
    }

    h1, h2, h3, h4 {
        font-family: 'Fredoka', cursive !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ESTADO DE LA CLASE EN VIVO
# ─────────────────────────────────────────────
if "historial_clase" not in st.session_state:
    st.session_state.historial_clase = []
if "preguntas_examen" not in st.session_state:
    st.session_state.preguntas_examen = []
if "indice_pregunta" not in st.session_state:
    st.session_state.indice_pregunta = 0
if "clase_iniciada" not in st.session_state:
    st.session_state.clase_iniciada = False

# ─────────────────────────────────────────────
# BARRA LATERAL: ESCRITORIO DEL PROFESOR / MOCHILA
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1a2b4c;'>🎒 Mochila Escolar</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("🔑 **Pase de Entrada (OpenAI API):**")
    api_key = st.text_input("Clave API", type="password", placeholder="sk-...", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("📚 **Libro de Texto (PDF):**")
    pdf_subido = st.file_uploader("Sube tu archivo", type="pdf", label_visibility="collapsed")

    if pdf_subido and not st.session_state.clase_iniciada:
        if st.button("🔔 ¡Tocar el Timbre!", use_container_width=True):
            if not api_key:
                st.error("¡Falta la llave de acceso en la mochila!")
            else:
                with st.spinner("🍎 El profesor está preparando la pizarra..."):
                    reader = PdfReader(io.BytesIO(pdf_subido.getvalue()))
                    texto_pdf = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
                    
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)
                    prompt = f"""
                    Eres un profesor super divertido, dinámico y carismático de una escuela moderna. 
                    A partir del siguiente texto, genera 4 preguntas de estudio estimulantes y claras para evaluar a un estudiante en forma de un divertido examen oral.
                    Devuelve el resultado ÚNICAMENTE en formato JSON de lista de strings (ejemplo: ["Pregunta 1?", "Pregunta 2?"]):
                    
                    Texto de la clase:
                    {texto_pdf[:10000]}
                    """
                    resp = llm.invoke(prompt)
                    try:
                        limpio = resp.content.replace("```json", "").replace("```", "").strip()
                        preguntas = json.loads(limpio)
                        st.session_state.preguntas_examen = preguntas
                        st.session_state.indice_pregunta = 0
                        st.session_state.historial_clase = []
                        
                        saludo_profe = f"¡Hola, hola! 🎒 Pasa y toma asiento. He echado un vistazo al documento que trajiste hoy y tengo un par de retos para ti. ¡Empecemos con la pregunta número 1:\n\n✨ **{preguntas[0]}**"
                        st.session_state.historial_clase.append(("profesor", saludo_profe))
                        st.session_state.clase_iniciada = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"¡Rayos! Hubo un problema al leer el documento: {e}")

    if st.session_state.clase_iniciada:
        if st.button("🧹 Limpiar Pizarra / Salir", use_container_width=True):
            st.session_state.clase_iniciada = False
            st.session_state.historial_clase = []
            st.session_state.preguntas_examen = []
            st.rerun()

    st.markdown("<br><p style='text-align: center; font-size: 0.8rem;'>¡Aprender nunca fue tan divertido! 🎨</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PANTALLA PRINCIPAL: EL AULA DE CLASES
# ─────────────────────────────────────────────
st.markdown("""
<div class="pizarra-card">
    <div class="pizarra-titulo">🏫 ESCUELA SECRETA DE IA ✏️</div>
    <p style="text-align: center; color: #a3ffb1; font-size: 1.1rem; margin: 0;">
        Bienvenido a tu pupitre. Tu profesor virtual evaluará tus conocimientos lección por lección de forma interactiva. ¡Levanta la mano para responder!
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.clase_iniciada:
    st.markdown("""
    <div class="pizarra-card" style="text-align: center; padding: 50px; background: #1c3218 !important;">
        <h2 style="color: #ffde59;">🚪 El salón está listo y esperando...</h2>
        <p style="font-size: 1.2rem; color: #ffffff;">
            Para comenzar la clase, ingresa tu clave secreta de OpenAI y sube tu PDF de estudio en la <b>Mochila Escolar (barra lateral izquierda)</b>. ¡Luego toca el timbre! 🔔
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Mostramos la conversación a modo de pizarra interactiva / diálogo de clase
    for rol, mensaje in st.session_state.historial_clase:
        if rol == "profesor":
            st.markdown(f"""
            <div class="profesor-habla">
                <b>👨‍🏫 Profesor Alex:</b><br><br>{mensaje}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="estudiante-habla">
                <b>🙋‍♂️ Tú (Estudiante):</b><br><br>{mensaje}
            </div>
            """, unsafe_allow_html=True)

    idx_actual = st.session_state.indice_pregunta
    total_preguntas = len(st.session_state.preguntas_examen)

    if idx_actual < total_preguntas:
        with st.form(key="form_clase_vibrante", clear_on_submit=True):
            st.markdown("<h4 style='color: #ffde59;'>📝 Turno de participación en clase:</h4>", unsafe_allow_html=True)
            respuesta_estudiante = st.text_input("Escribe tu respuesta como si hablaras con el profesor:", placeholder="Ej: Pienso que la respuesta principal es...")
            btn_enviar = st.form_submit_button("🙋‍♂️ Levantar la Mano y Responder", use_container_width=True)

            if btn_enviar and respuesta_estudiante.strip():
                st.session_state.historial_clase.append(("estudiante", respuesta_estudiante))
                
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
                pregunta_hecha = st.session_state.preguntas_examen[idx_actual]
                
                prompt_evaluacion = f"""
                Estás actuando como un profesor súper amigable, carismático y dinámico en un salón de clases colorido.
                Pregunta de examen que hiciste al alumno: "{pregunta_hecha}"
                Lo que respondió el alumno: "{respuesta_estudiante}"
                
                Evalúa su respuesta de forma divertida, dale retroalimentación (10 partes de aliento, corrección si hace falta) y explícale con buena onda.
                """
                
                with st.spinner("🍎 El profesor está revisando tu libreta..."):
                    evaluacion = llm.invoke(prompt_evaluacion).content
                
                st.session_state.indice_pregunta += 1
                nuevo_idx = st.session_state.indice_pregunta
                
                if nuevo_idx < total_preguntas:
                    siguiente_pregunta = st.session_state.preguntas_examen[nuevo_idx]
                    respuesta_profesor = f"{evaluacion}\n\n--- \n¡Muy bien apuntado! Pasemos al siguiente desafío en la pizarra:\n\n✨ **{siguiente_pregunta}**"
                else:
                    respuesta_profesor = f"{evaluacion}\n\n🏆 **¡Increíble trabajo!** Has completado todas las preguntas de la lección de hoy. ¡Te haz ganado una estrella dorada en conducta y calificaciones! ⭐"
                
                st.session_state.historial_clase.append(("profesor", respuesta_profesor))
                st.rerun()
    else:
        st.markdown("""
        <div class="pizarra-card" style="text-align: center; background: #1c3218 !important; border-color: #00bf63;">
            <h2 style="color: #00bf63;">🌟 ¡Clase Concluida con Honores!</h2>
            <p style="font-size: 1.1rem; color: #ffffff;">
                Has terminado todo el cuestionario oral del día de hoy. Puedes guardar tus cosas y recargar la página o cambiar de documento en la mochila escolar cuando gustes.
            </p>
        </div>
        """, unsafe_allow_html=True)
