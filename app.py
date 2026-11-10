import io
import platform

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Oráculo de Documentos — RAG IA",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS (ESTÉTICA NATURAL & MENTA)
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
    .main-card {
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

    /* Ajustes generales de texto legible sobre fondo oscuro */
    p, label, span {
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BARRA LATERAL (SIDEBAR)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#000000 !important; font-weight:800;'>🍃 Panel de Control</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#222222 !important; font-size:0.9rem;'>Configura tus credenciales y carga tu archivo de lectura.</p>", unsafe_allow_html=True)
    st.divider()

    # Clave de API
    st.markdown("<b style='color:#000000 !important;'>🔑 Credenciales</b>", unsafe_allow_html=True)
    api_key = st.text_input("Ingresa tu Clave de OpenAI", type="password", placeholder="sk-...")

    st.divider()
    
    # Carga del PDF
    st.markdown("<b style='color:#000000 !important;'>📄 Archivo de Conocimiento</b>", unsafe_allow_html=True)
    pdf = st.file_uploader("Carga el archivo PDF", type="pdf", label_visibility="collapsed")

    st.divider()
    st.markdown(f"<p style='color:#444444 !important; font-size:0.75rem;'>Entorno Python: {platform.python_version()}</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER / ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <span class="badge-tag">✨ Inteligencia Artificial & RAG</span>
    <h1 style="margin: 4px 0 10px 0; color: #FFFFFF !important; font-weight: 800;">Oráculo Inteligente de Documentos PDF 💬</h1>
    <p style="font-size: 1.05rem; color: #CCCCCC !important; margin-bottom: 0px; line-height: 1.5;">
        Sube cualquier documento y conversa directamente con su contenido mediante recuperación semántica avanzada y modelos de lenguaje de última generación.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNCIÓN DE PROCESAMIENTO CON CACHÉ
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Creando la base de conocimiento...")
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    """Extrae el texto, lo divide en fragmentos y crea el índice FAISS."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    if not fragmentos:
        return None, len(texto), 0

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    base = FAISS.from_texts(fragmentos, embeddings)
    return base, len(texto), len(fragmentos)


# ─────────────────────────────────────────────
# LÓGICA PRINCIPAL DE LA APLICACIÓN
# ─────────────────────────────────────────────
if not api_key:
    st.markdown("""
    <div class="main-card" style="text-align: center; padding: 40px;">
        <h3 style="color: #00D18F !important;">🔒 Esperando Clave de API</h3>
        <p>Por favor ingresa tu clave secreta de OpenAI en la barra lateral izquierda para habilitar el sistema.</p>
    </div>
    """, unsafe_allow_html=True)

elif pdf is None:
    st.markdown("""
    <div class="main-card" style="text-align: center; padding: 40px;">
        <h3 style="color: #00D18F !important;">📂 Carga un Documento PDF</h3>
        <p>Selecciona o arrastra un archivo PDF en la barra lateral para comenzar la magia del análisis inteligente.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        with st.spinner("🧠 Analizando estructura e indexando fragmentos..."):
            base, n_caracteres, n_fragmentos = crear_base_conocimiento(pdf.getvalue(), api_key)

        if base is None:
            st.error("No se pudo extraer texto del PDF (¿es un PDF escaneado sin OCR?).")
            st.stop()

        # Métricas resumidas en tarjetas limpias
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div style="background: #29331F; padding: 15px; border-radius: 12px; border: 1px solid #3E4D2F; text-align: center;">
                <span style="font-size: 0.85rem; color: #00D18F; font-weight: 700;">TOTAL CARACTERES</span>
                <h3 style="color: #FFFFFF; margin: 5px 0 0 0;">{n_caracteres:,}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div style="background: #29331F; padding: 15px; border-radius: 12px; border: 1px solid #3E4D2F; text-align: center;">
                <span style="font-size: 0.85rem; color: #00D18F; font-weight: 700;">FRAGMENTOS INDEXADOS</span>
                <h3 style="color: #FFFFFF; margin: 5px 0 0 0;">{n_fragmentos}</h3>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sección de Preguntas
        st.markdown("""
        <div class="main-card">
            <h3 style="color: #00D18F !important; margin-top:0;">🤖 Conversa con tu PDF</h3>
            <p>Escribe cualquier duda o concepto que desees consultar dentro de las páginas del documento.</p>
        </div>
        """, unsafe_allow_html=True)

        pregunta = st.text_area(" ", placeholder="Ej: ¿Cuáles son las conclusiones principales del documento?", label_visibility="collapsed")

        if pregunta:
            # 1. Recuperación: fragmentos más parecidos a la pregunta
            docs = base.similarity_search(pregunta, k=4)
            contexto = "\n\n---\n\n".join(d.page_content for d in docs)

            # 2. Generación: el modelo responde usando solo ese contexto
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            mensajes = [
                ("system",
                 "Responde la pregunta usando únicamente el contexto proporcionado. "
                 "Si la respuesta no está en el contexto, dilo claramente. Responde en español."),
                ("human", f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"),
            ]
            
            with st.spinner("✨ El Oráculo está redactando la respuesta..."):
                respuesta = llm.invoke(mensajes)

            # Tarjeta de Respuesta
            st.markdown(f"""
            <div class="main-card" style="border-left: 5px solid #00D18F;">
                <h4 style="color: #00D18F !important; margin-top:0;">💡 Respuesta del Oráculo:</h4>
                <div style="color: #FFFFFF; font-size: 1.05rem; line-height: 1.6;">
                    {respuesta.content}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Expansor para los fragmentos de origen
            with st.expander("🔍 Ver fragmentos recuperados del documento (Contexto RAG)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Fragmento {i}**")
                    st.code(d.page_content, language="text")

    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
        import traceback
        st.code(traceback.format_exc())
