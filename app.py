import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ==========================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================

st.set_page_config(
    page_title="Chat PDF - RAG",
    page_icon="💬",
    layout="wide"
)


# ==========================================
# DISEÑO VISUAL: BLANCO Y AZUL
# ==========================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #F4F8FF;
    color: #193153;
}

/* Ocultar menú y pie de página */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Área principal */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Encabezado */
.chat-header {
    background: linear-gradient(135deg, #1264D8, #398BFA);
    padding: 30px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(28, 100, 216, 0.15);
}

.chat-header h1 {
    color: white;
    font-size: 30px;
    font-weight: 750;
    margin-bottom: 8px;
}

.chat-header p {
    color: #EAF3FF;
    font-size: 15px;
    margin-bottom: 0;
}

/* Paneles blancos */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE8F8;
    border-radius: 18px;
    padding: 16px;
}

/* Barra lateral */
section[data-testid="stSidebar"] {
    background: #EAF2FF;
    border-right: 1px solid #D5E5FC;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: #193D70;
}

/* Textos */
h2, h3, h4 {
    color: #174A91;
}

p, label {
    color: #294565;
}

/* Campos de texto */
.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #193153 !important;
    border: 1px solid #C8D9F1 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #2678E8 !important;
    box-shadow: 0 0 0 2px rgba(38, 120, 232, 0.12) !important;
}

/* Botones */
.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(135deg, #1769D2, #398BFA);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #1055B5;
    border: none;
    box-shadow: 0 5px 14px rgba(23, 105, 210, 0.2);
}

/* Cargador de archivos */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #8BB8F2;
    border-radius: 15px;
    padding: 14px;
}

/* Mensajes informativos */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Separadores */
hr {
    border-color: #DCE8F8;
}

/* Respuesta del asistente */
.assistant-message {
    background: #FFFFFF;
    border: 1px solid #D8E7FB;
    border-left: 5px solid #2879E8;
    border-radius: 5px 18px 18px 18px;
    padding: 20px 23px;
    margin: 12px 0 22px 0;
    color: #193153;
    box-shadow: 0 4px 14px rgba(36, 91, 157, 0.05);
}

/* Mensaje del usuario */
.user-message {
    background: #E1EEFF;
    border: 1px solid #C5DDFF;
    border-radius: 18px 5px 18px 18px;
    padding: 16px 20px;
    margin: 15px 0;
    color: #174A91;
    margin-left: 10%;
}

/* Etiquetas del chat */
.chat-label {
    color: #2678E8;
    font-size: 13px;
    font-weight: 750;
    margin-bottom: 8px;
}

/* Pie del chat */
.chat-footer {
    text-align: center;
    color: #7B91AD;
    font-size: 12px;
    padding: 18px 0 5px 0;
}

/* Expansores */
details {
    background: #FFFFFF;
    border-radius: 12px;
}

/* Adaptación a pantallas pequeñas */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .chat-header {
        padding: 22px;
    }

    .chat-header h1 {
        font-size: 24px;
    }

    .user-message {
        margin-left: 3%;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# ENCABEZADO
# ==========================================

st.markdown("""
<div class="chat-header">
    <h1>💬 Chat PDF</h1>
    <p>
        Generación Aumentada por Recuperación (RAG)
        <br>
        Conversa con tus documentos, consulta información
        y encuentra respuestas basadas en el contenido del PDF.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("Versión de Python:", platform.python_version())


# ==========================================
# CARGAR IMAGEN
# ==========================================

try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)
except Exception:
    st.warning("No se pudo cargar la imagen Chat_pdf.png.")


# ==========================================
# BARRA LATERAL
# ==========================================

with st.sidebar:
    st.subheader("📚 Tu asistente documental")

    st.write(
        "Este agente te ayudará a realizar análisis "
        "sobre el PDF cargado."
    )

    st.divider()

    st.markdown("### ✨ ¿Qué puedes hacer?")

    st.markdown("""
    - 📄 Cargar un documento PDF.
    - 🔎 Buscar información específica.
    - 💡 Identificar ideas principales.
    - 📖 Consultar fragmentos y páginas.
    """)

    st.divider()

    st.caption("Chat PDF · Sistema RAG")


# ==========================================
# CLAVE API DE OPENAI
# ==========================================

st.markdown("### 🔑 Conexión con OpenAI")

api_key = st.text_input(
    "Ingresa tu clave de API de OpenAI",
    type="password",
    placeholder="sk-...",
    help="Tu clave se utiliza para generar embeddings y respuestas."
)


# ==========================================
# CARGAR ARCHIVO PDF
# ==========================================

st.markdown("### 📄 Documento de consulta")

pdf = st.file_uploader(
    "Carga el archivo PDF",
    type=["pdf"],
    help="Selecciona el documento que quieres analizar."
)


# ==========================================
# VALIDAR ENTRADAS
# ==========================================

if not api_key:
    st.info("🔐 Ingresa tu clave de API de OpenAI para continuar.")

elif pdf is None:
    st.info("📂 Carga un archivo PDF para comenzar.")

else:
    try:
        # ==========================================
        # EXTRAER TEXTO DEL PDF
        # ==========================================

        lector_pdf = PdfReader(pdf)
        documentos = []

        for numero_pagina, pagina in enumerate(
            lector_pdf.pages,
            start=1
        ):
            texto_pagina = pagina.extract_text() or ""

            if texto_pagina.strip():
                documentos.append(
                    Document(
                        page_content=texto_pagina,
                        metadata={
                            "pagina": numero_pagina,
                            "archivo": pdf.name
                        }
                    )
                )

        if not documentos:
            st.error(
                "No se pudo extraer texto del PDF. "
                "Si es un documento escaneado, necesitas utilizar OCR."
            )
            st.stop()

        texto_completo = "\n".join(
            documento.page_content
            for documento in documentos
        )

        st.success(
            f"📄 Texto extraído: {len(texto_completo):,} caracteres"
        )

        # ==========================================
        # DIVIDIR TEXTO EN FRAGMENTOS
        # ==========================================

        divisor_texto = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        fragmentos = divisor_texto.split_documents(documentos)

        if not fragmentos:
            st.error("No se pudieron crear fragmentos del documento.")
            st.stop()

        st.success(
            f"🧩 Documento dividido en {len(fragmentos)} fragmentos."
        )

        # ==========================================
        # CREAR EMBEDDINGS Y BASE VECTORIAL
        #
